"""Run the training algorithms over actual subprocess IPC connections."""

from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from time import monotonic

import numpy as np

from .data import sample_worker_batch
from .network import cross_entropy_loss, model_forward
from .optim import (
    adamw_step,
    compute_outer_gradient,
    init_adamw_state,
    init_outer_optimizer,
    nesterov_param_update,
    update_outer_momentum,
)
from .parameters import average_params
from .training import loss_and_gradients
from .workload import character_features


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 320
    batch_size: int = 32
    eval_every: int = 80
    inner_lr: float = 0.003
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0001
    outer_lr: float = 0.7
    momentum: float = 0.9

    def adamw_args(self):
        return (self.inner_lr, self.beta1, self.beta2, self.eps, self.weight_decay)


@dataclass(frozen=True)
class TensorLayout:
    """Shared layout for one flat, little-endian float64 model payload."""

    entries: tuple
    nbytes: int

    @classmethod
    def from_params(cls, params):
        entries = tuple((key, array.shape, array.size) for key, array in params.items())
        return cls(entries, 8 * sum(size for _, _, size in entries))

    def pack(self, params):
        return b"".join(np.asarray(params[key], dtype="<f8").tobytes()
                        for key, _, _ in self.entries)

    def unpack(self, payload):
        if len(payload) != self.nbytes:
            raise ValueError(f"Expected {self.nbytes} model bytes, got {len(payload)}")
        values = np.frombuffer(payload, dtype="<f8")
        params, offset = {}, 0
        for key, shape, size in self.entries:
            params[key] = values[offset:offset + size].reshape(shape)
            offset += size
        return params


def _worker(channel, shard, vocabulary_size, layout, config, interval, seed, rank):
    rng = np.random.default_rng([seed, rank])
    local_steps = interval or 1
    with channel:
        for _ in range(config.steps // local_steps):
            params = layout.unpack(channel.recv_bytes(layout.nbytes))
            if interval is None:
                x, y = sample_worker_batch(*shard, config.batch_size, rng)
                _, response = loss_and_gradients(params, character_features(x, vocabulary_size), y)
            else:
                state = init_adamw_state(params)
                for _ in range(local_steps):
                    x, y = sample_worker_batch(*shard, config.batch_size, rng)
                    _, gradients = loss_and_gradients(params, character_features(x, vocabulary_size), y)
                    params = adamw_step(params, state, gradients, *config.adamw_args())
                response = params
            channel.send_bytes(layout.pack(response))


@contextmanager
def _worker_group(shards, vocabulary_size, layout, config, interval, seed):
    context = get_context("spawn")
    workers = []
    try:
        for rank, shard in enumerate(shards):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                name=f"diloco-worker-{rank}",
                target=_worker,
                args=(child, shard, vocabulary_size, layout, config, interval, seed, rank),
            )
            try:
                process.start()
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            workers.append((parent, process))
        yield workers

        deadline = monotonic() + 10
        for _, process in workers:
            process.join(timeout=max(0, deadline - monotonic()))
            if process.exitcode != 0:
                raise RuntimeError(f"{process.name} failed to exit cleanly ({process.exitcode})")
    finally:
        for channel, process in workers:
            channel.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()


def _evaluate(params, holdout, vocabulary_size):
    windows, targets = holdout
    loss_total, correct = 0.0, 0
    for start in range(0, len(targets), 256):
        tokens = windows[start:start + 256]
        labels = targets[start:start + 256]
        logits, _ = model_forward(params, character_features(tokens, vocabulary_size))
        loss_total += len(labels) * cross_entropy_loss(logits, labels)
        correct += np.count_nonzero(logits.argmax(axis=1) == labels)
    loss = loss_total / len(targets)
    return {"val_loss": loss, "perplexity": float(np.exp(loss)), "top1_accuracy": correct / len(targets)}


def train_distributed(initial, shards, holdout, vocabulary_size, config, interval, seed):
    """Compare equal per-worker minibatches, counting successful IPC model bytes."""
    local_steps = interval or 1
    if interval is not None and interval < 1:
        raise ValueError("A DiLoCo interval must be positive")
    if min(config.steps, config.batch_size, config.eval_every) < 1:
        raise ValueError("Steps, batch size, and evaluation frequency must be positive")
    if config.steps % local_steps or config.eval_every % local_steps:
        raise ValueError("Training and evaluation steps must align with every interval")
    if not shards or any(not len(labels) for _, labels in shards):
        raise ValueError("Each worker needs a nonempty shard")
    if not len(holdout[1]):
        raise ValueError("A nonempty held-out set is required")

    layout = TensorLayout.from_params(initial)
    params = initial
    outer_state = init_outer_optimizer(initial)
    sync_state = init_adamw_state(initial)
    sent_bytes = received_bytes = 0
    history = []

    def record(local_step, exchanges):
        history.append({
            "local_steps": local_step,
            "sync_rounds": exchanges,
            "examples_seen": local_step * config.batch_size * len(shards),
            "sent_model_bytes": sent_bytes,
            "received_model_bytes": received_bytes,
            "model_payload_bytes": sent_bytes + received_bytes,
            **_evaluate(params, holdout, vocabulary_size),
        })

    record(0, 0)
    with _worker_group(shards, vocabulary_size, layout, config, interval, seed) as workers:
        for exchange in range(1, config.steps // local_steps + 1):
            payload = layout.pack(params)
            for channel, _ in workers:
                channel.send_bytes(payload)
                sent_bytes += len(payload)

            responses = []
            for channel, process in workers:
                if not channel.poll(60):
                    raise TimeoutError(f"No reply from {process.name} during exchange {exchange}")
                try:
                    response = channel.recv_bytes(layout.nbytes)
                except EOFError as exc:
                    raise RuntimeError(f"{process.name} exited during exchange {exchange}") from exc
                received_bytes += len(response)
                responses.append(layout.unpack(response))

            if interval is None:
                gradients = average_params(responses)
                params = adamw_step(params, sync_state, gradients, *config.adamw_args())
            else:
                gradients = compute_outer_gradient(params, responses)
                update_outer_momentum(outer_state, gradients, config.momentum)
                params = nesterov_param_update(
                    params, outer_state, gradients, config.outer_lr, config.momentum
                )

            local_step = exchange * local_steps
            if local_step % config.eval_every == 0 or local_step == config.steps:
                record(local_step, exchange)
    return params, history
