"""Reproducible worker data sharding and batch sampling."""

import numpy as np


def iid_shard_dataset(x, y, num_workers, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = x.shape[0]

    # Randomly permute indices reproducibly
    perm = rng.permutation(n_samples)

    # np.array_split automatically distributes remainder across the earliest shards
    index_shards = np.array_split(perm, num_workers)

    shards = []
    for idx in index_shards:
        shards.append((x[idx], y[idx]))

    return shards


def noniid_shard_dataset(x, y, num_workers, num_classes, seed=0):
    rng = np.random.default_rng(seed)
    shards = []

    for w in range(num_workers):
        # Collect indices where label c satisfies c % num_workers == w
        idx = np.where((y % num_workers) == w)[0]

        if len(idx) > 0:
            perm = rng.permutation(len(idx))
            idx = idx[perm]

        shards.append((x[idx], y[idx]))

    return shards


def sample_worker_batch(x_shard, y_shard, batch_size, rng):
    n = x_shard.shape[0]
    replace = batch_size > n
    indices = rng.choice(n, size=batch_size, replace=replace)
    return x_shard[indices], y_shard[indices]
