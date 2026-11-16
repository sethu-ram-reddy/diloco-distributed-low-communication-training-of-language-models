"""Benchmark DiLoCo and synchronous AdamW with matched per-worker compute."""

import argparse
import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import iid_shard_dataset, noniid_shard_dataset
from .distributed import TrainingConfig, train_distributed
from .network import init_model_params
from .workload import load_character_data


def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="UTF-8 text corpus")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shards", choices=("iid", "label-skew"), nargs="+",
                        default=["iid", "label-skew"])
    parser.add_argument("--workers", type=_positive, default=4)
    parser.add_argument("--intervals", type=_positive, nargs="+", default=[5, 20, 80])
    parser.add_argument("--steps", type=_positive, default=320)
    parser.add_argument("--eval-every", type=_positive, default=80)
    parser.add_argument("--batch-size", type=_positive, default=32)
    parser.add_argument("--context", type=_positive, default=8)
    parser.add_argument("--hidden-size", type=_positive, default=64)
    parser.add_argument("--train-windows", type=_positive, default=65536)
    parser.add_argument("--holdout-windows", type=_positive, default=4096)
    args = parser.parse_args(argv)

    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if len(set(args.shards)) != len(args.shards) or len(set(args.intervals)) != len(args.intervals):
        parser.error("shard modes and intervals must be distinct")
    if any(args.steps % interval or args.eval_every % interval for interval in args.intervals):
        parser.error("--steps and --eval-every must be divisible by each interval")

    try:
        workload = load_character_data(
            args.corpus, args.context, args.train_windows, args.holdout_windows
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    config = TrainingConfig(steps=args.steps, batch_size=args.batch_size, eval_every=args.eval_every)
    dimension = args.context * workload.vocabulary_size
    checkpoints, summaries, distributions = [], [], []

    for seed in args.seeds:
        initial = init_model_params(dimension, args.hidden_size, workload.vocabulary_size, seed)
        for partition in args.shards:
            if partition == "iid":
                shards = iid_shard_dataset(*workload.train, args.workers, seed=seed)
            else:
                shards = noniid_shard_dataset(
                    *workload.train, args.workers, workload.vocabulary_size, seed=seed
                )
            if any(not len(y) for _, y in shards):
                parser.error(f"{partition} sharding yielded an empty shard")

            for rank, (_, y) in enumerate(shards):
                distributions.append({
                    "seed": seed,
                    "sharding": partition,
                    "worker": rank,
                    "windows": len(y),
                    "distinct_targets": len(np.unique(y)),
                })

            for interval in (None, *args.intervals):
                method = "synchronous" if interval is None else "diloco"
                tags = {"seed": seed, "sharding": partition, "method": method, "interval": interval or 1}
                _, history = train_distributed(
                    initial, shards, workload.holdout, workload.vocabulary_size, config, interval, seed
                )
                checkpoints.extend(tags | row for row in history)
                final = tags | history[-1]
                summaries.append(final)
                print(
                    f"seed={seed} {partition:<10} {method:<11} H={interval or 1:<3}"
                    f" loss={final['val_loss']:.4f} perplexity={final['perplexity']:.3f}"
                    f" payload={final['model_payload_bytes']:,} bytes",
                    flush=True,
                )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "summary.csv", summaries)
    _write_csv(args.output / "learning_curves.csv", checkpoints)
    _write_csv(args.output / "shards.csv", distributions)
    metadata = {
        "model": {
            "context_characters": args.context,
            "vocabulary_size_including_unknown": workload.vocabulary_size,
            "hidden_size": args.hidden_size,
            "parameters": sum(array.size for array in initial.values()),
            "payload_dtype": "float64 little-endian",
        },
        "corpus": {
            "sha256": workload.corpus_sha256,
            "training_characters": workload.training_characters,
            "held_out_characters": workload.held_out_characters,
            "training_windows": len(workload.train[1]),
            "held_out_windows": len(workload.holdout[1]),
        },
        "training": asdict(config),
        "workers": args.workers,
        "intervals": args.intervals,
        "seeds": args.seeds,
        "sharding": args.shards,
        "execution": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "worker_start_method": "spawn",
            "transport": "multiprocessing.Connection.send_bytes",
        },
    }
    (args.output / "config.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
