"""A fixed-context next-character workload with a chronological holdout."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CharacterData:
    train: tuple
    holdout: tuple
    vocabulary_size: int
    corpus_sha256: str
    training_characters: int
    held_out_characters: int


def _sample_windows(tokens, context, limit):
    windows = np.lib.stride_tricks.sliding_window_view(tokens, context + 1)
    if len(windows) > limit:
        indices = np.linspace(0, len(windows) - 1, limit, dtype=np.intp)
        windows = windows[indices]
    return windows[:, :-1].copy(), windows[:, -1].copy()


def load_character_data(path: Path, context: int, train_limit: int, holdout_limit: int):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    boundary = len(text) * 9 // 10
    training, holdout = text[:boundary], text[boundary:]
    if min(len(training), len(holdout)) <= context:
        raise ValueError("Both text splits must be longer than the context")

    symbols = {char: index for index, char in enumerate(sorted(set(training)), start=1)}
    # ID zero is reserved for characters that first appear in the held-out text.
    def tokenize(split):
        return np.fromiter((symbols.get(char, 0) for char in split), dtype=np.int32, count=len(split))

    return CharacterData(
        train=_sample_windows(tokenize(training), context, train_limit),
        holdout=_sample_windows(tokenize(holdout), context, holdout_limit),
        vocabulary_size=len(symbols) + 1,
        corpus_sha256=sha256(raw).hexdigest(),
        training_characters=len(training),
        held_out_characters=len(holdout),
    )


def character_features(windows, vocabulary_size):
    """One-hot encode only the requested batch, keeping the shards compact."""
    rows, context = windows.shape
    features = np.zeros((rows, context, vocabulary_size), dtype=np.float64)
    features[np.arange(rows)[:, None], np.arange(context), windows] = 1.0
    return features.reshape(rows, context * vocabulary_size)
