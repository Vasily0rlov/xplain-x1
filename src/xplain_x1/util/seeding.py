"""Deterministic seeding: every stochastic step derives from one run seed (S-#2).

A `SeedSequence` spawned per named path guarantees independent, reproducible
streams for data splits, model init, batching, restarts, and CPSS subsampling.
"""
from __future__ import annotations

import numpy as np


def seed_seq(root_seed: int, *path: int | str) -> np.random.SeedSequence:
    """SeedSequence for a named sub-stream, e.g. seed_seq(7, "restart", 3)."""
    entropy = [root_seed] + [_component(p) for p in path]
    return np.random.SeedSequence(entropy)


def rng(root_seed: int, *path: int | str) -> np.random.Generator:
    return np.random.default_rng(seed_seq(root_seed, *path))


def torch_seed(root_seed: int, *path: int | str) -> int:
    """A 63-bit torch.manual_seed derived from the same tree."""
    return int(seed_seq(root_seed, *path).generate_state(1, np.uint64)[0] >> 1)


def _component(p: int | str) -> int:
    if isinstance(p, int):
        return p
    return int.from_bytes(p.encode("utf-8"), "little") % (2**63)
