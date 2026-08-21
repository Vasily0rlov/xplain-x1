"""Splits, probe set, standardisation, and CPSS subsampler (S-#4).

Split seed is independent of model seeds; the probe set is identical across
restarts (matching signatures depend on it, M-#3.6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from ..util.seeding import rng
from .dataset import Dataset


@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    probe: np.ndarray          # indices into the full X, fixed across restarts
    mean: np.ndarray           # standardisation fit on TRAIN only
    std: np.ndarray

    def standardise(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


def make_splits(ds: Dataset, split_seed: int, probe_size: int = 2048,
                fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> Splits:
    idx = np.arange(ds.n)
    strat = ds.y if ds.task == "classification" else None
    f_train, f_val, f_test = fractions
    train, rest = train_test_split(
        idx, train_size=f_train, random_state=split_seed,
        stratify=strat if strat is not None else None)
    strat_rest = ds.y[rest] if strat is not None else None
    rel_val = f_val / (f_val + f_test)
    # tiny datasets can lack per-class members for stratified sub-split; degrade gracefully
    try:
        val, test = train_test_split(rest, train_size=rel_val,
                                     random_state=split_seed + 1, stratify=strat_rest)
    except ValueError:
        val, test = train_test_split(rest, train_size=rel_val,
                                     random_state=split_seed + 1)

    pool = np.concatenate([train, val])
    g = rng(split_seed, "probe")
    probe = (pool if len(pool) <= probe_size
             else g.choice(pool, size=probe_size, replace=False))

    Xtr = ds.X[train]
    mean = np.zeros(ds.d, dtype=np.float32)
    std = np.ones(ds.d, dtype=np.float32)
    cm = ds.continuous_mask
    mean[cm] = Xtr[:, cm].mean(axis=0)
    s = Xtr[:, cm].std(axis=0)
    std[cm] = np.where(s > 1e-8, s, 1.0)
    return Splits(train=np.sort(train), val=np.sort(val), test=np.sort(test),
                  probe=np.sort(probe), mean=mean, std=std)


def cpss_pairs(train_idx: np.ndarray, n_pairs: int, split_seed: int
               ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Complementary half/half partitions of the training set (M-#3.6)."""
    pairs = []
    for b in range(n_pairs):
        g = rng(split_seed, "cpss", b)
        perm = g.permutation(train_idx)
        half = len(perm) // 2
        pairs.append((np.sort(perm[:half]), np.sort(perm[half:half * 2])))
    return pairs
