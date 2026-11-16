"""In-process reference training loops for the NumPy model."""

import numpy as np

from .data import sample_worker_batch
from .network import cross_entropy_loss, model_backward, model_forward
from .optim import (
    adamw_step, compute_outer_gradient, init_adamw_state, init_outer_optimizer,
    nesterov_param_update, update_outer_momentum,
)
from .parameters import average_params, clone_params


def loss_and_gradients(params, features, labels):
    logits, cache = model_forward(params, features)
    return cross_entropy_loss(logits, labels), model_backward(params, cache, labels)


def local_train_step(params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay):
    loss, grads = loss_and_gradients(params, x_batch, y_batch)
    new_state = {"t": adam_state["t"], "m": dict(adam_state["m"]), "v": dict(adam_state["v"])}
    new_params = adamw_step(params, new_state, grads, lr, beta1, beta2, eps, weight_decay)
    return new_params, new_state, loss


def inner_train_worker(params, x_shard, y_shard, num_inner_steps, batch_size, lr, beta1, beta2, eps, weight_decay, seed):
    # Clone params so the original is not mutated
    worker_params = clone_params(params)
    adam_state = init_adamw_state(worker_params)

    rng = np.random.default_rng(seed)
    total_loss = 0.0

    for _ in range(num_inner_steps):
        x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)
        worker_params, adam_state, loss = local_train_step(
            worker_params, adam_state, x_batch, y_batch,
            lr, beta1, beta2, eps, weight_decay
        )
        total_loss += loss

    mean_loss = total_loss / max(1, num_inner_steps)
    return worker_params, float(mean_loss)


def run_diloco_round(global_params, outer_state, worker_shards, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed):
    if not worker_shards and num_inner_steps:
        raise ValueError("DiLoCo requires at least one worker")

    worker_params_list = []
    worker_losses = []

    lr = inner_hparams['lr']
    beta1 = inner_hparams['beta1']
    beta2 = inner_hparams['beta2']
    eps = inner_hparams['eps']
    weight_decay = inner_hparams['weight_decay']

    for i, (x_shard, y_shard) in enumerate(worker_shards):
        w_params, w_loss = inner_train_worker(
            global_params, x_shard, y_shard,
            num_inner_steps, batch_size,
            lr, beta1, beta2, eps, weight_decay,
            seed + i
        )
        worker_params_list.append(w_params)
        worker_losses.append(w_loss)

    outer_grad = compute_outer_gradient(global_params, worker_params_list)
    outer_state = update_outer_momentum(outer_state, outer_grad, momentum_coef)
    new_global_params = nesterov_param_update(global_params, outer_state, outer_grad, outer_lr, momentum_coef)

    return new_global_params, outer_state, worker_losses


def train_diloco(initial_params, worker_shards, num_outer_rounds, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed):
    params = clone_params(initial_params)
    outer_state = init_outer_optimizer(params)
    round_losses = []

    num_shards = len(worker_shards)

    for round_idx in range(num_outer_rounds):
        round_seed = seed + round_idx * max(1, num_shards)

        # If num_inner_steps is 0, inner_train_worker will cause a division by zero.
        # We can bypass inner training for 0 steps and manually return default worker states/losses.
        if num_inner_steps == 0:
            worker_losses = [0.0 for _ in worker_shards]
            # When inner steps are 0, parameters and outer state remain unchanged
        else:
            params, outer_state, worker_losses = run_diloco_round(
                params, outer_state, worker_shards,
                num_inner_steps, batch_size,
                inner_hparams, outer_lr, momentum_coef,
                round_seed
            )

        if worker_losses:
            mean_round_loss = float(sum(worker_losses) / len(worker_losses))
            round_losses.append(mean_round_loss)
        else:
            round_losses.append(0.0)

    return params, {'round_losses': round_losses}


def train_synchronous_baseline(init_params, worker_shards, num_steps, batch_size, inner_hparams, seed=0):
    params = clone_params(init_params)
    adam_state = init_adamw_state(params)
    rng = np.random.default_rng(seed)

    lr = inner_hparams['lr']
    beta1 = inner_hparams['beta1']
    beta2 = inner_hparams['beta2']
    eps = inner_hparams['eps']
    weight_decay = inner_hparams['weight_decay']

    step_losses = []
    num_workers = len(worker_shards)

    for _ in range(num_steps):
        worker_grads = []
        worker_losses = []

        for x_shard, y_shard in worker_shards:
            x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)
            loss, grads = loss_and_gradients(params, x_batch, y_batch)

            worker_grads.append(grads)
            worker_losses.append(loss)

        if worker_grads:
            avg_grads = average_params(worker_grads)
            mean_loss = float(sum(worker_losses) / num_workers)
        else:
            avg_grads = {k: np.zeros_like(v) for k, v in params.items()}
            mean_loss = 0.0

        step_losses.append(mean_loss)

        # AdamW step using the averaged gradients
        params = adamw_step(params, adam_state, avg_grads, lr, beta1, beta2, eps, weight_decay)

    return params, {'step_losses': step_losses}


def communication_savings(num_rounds, num_inner_steps, num_workers, param_count):
    # DiLoCo communicates once per round:
    # Each worker sends params (param_count) and receives updated global params (param_count) = 2 * param_count per worker.
    # Total per round across all workers: 2 * num_workers * param_count.
    # Over all num_rounds:
    diloco_scalars = num_rounds * (2 * num_workers * param_count)

    # Synchronous baseline communicates every local step:
    # Total synchronous steps = num_rounds * num_inner_steps.
    # Each step communicates 2 * num_workers * param_count scalars.
    total_sync_steps = num_rounds * num_inner_steps
    sync_scalars = total_sync_steps * (2 * num_workers * param_count)

    ratio = diloco_scalars / sync_scalars if sync_scalars > 0 else 0.0
    savings_factor = sync_scalars / diloco_scalars if diloco_scalars > 0 else 0.0

    return {
        'diloco_scalars': diloco_scalars,
        'sync_scalars': sync_scalars,
        'ratio': ratio,
        'savings_factor': savings_factor
    }
