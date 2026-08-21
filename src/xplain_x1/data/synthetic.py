"""Synthetic control suite: 6 pinned configs with known ground truth (S-#5).

Structures (d=10 standard-normal features, irrelevant features included):
  ADD    y = f1(x1)+f2(x2)+f3(x3)         honest depth 1
  COMP2  y = f1(x1) + x2*x3               one planted order-2 concept, depth 2
  COMP3  y = x1*x2*x3 + f4(x4)            one planted order-3 concept, depth 2
  NOISE  y independent of x               honest output: nothing certified

Configs: COMP2 in all three regimes (clean-8k, noisy-8k, noisy-2k power floor);
ADD and COMP3 noisy-8k; NOISE 8k.  All regression tasks.
"""
from __future__ import annotations

import numpy as np

from ..util.seeding import rng
from .dataset import Dataset

D = 10
NOISE_SIGMA = 0.3
DATA_SEED = 424242  # dataset generation is fixed, independent of model seeds

# monotone nonlinear main effects (distinct so surrogates can tell them apart)
_F = {
    1: lambda x: np.tanh(1.5 * x),
    2: lambda x: 0.5 * x + 0.3 * x**3 / (1 + x**2),
    3: lambda x: np.log1p(np.exp(x)) - 0.7,   # softplus, centred-ish
    4: lambda x: np.sign(x) * np.sqrt(np.abs(x)),
}

CONFIGS: dict[str, dict] = {
    "synthetic:add-noisy-8k":    {"structure": "ADD",   "n": 8000, "sigma": NOISE_SIGMA},
    "synthetic:comp2-clean-8k":  {"structure": "COMP2", "n": 8000, "sigma": 0.0},
    "synthetic:comp2-noisy-8k":  {"structure": "COMP2", "n": 8000, "sigma": NOISE_SIGMA},
    "synthetic:comp2-noisy-2k":  {"structure": "COMP2", "n": 2000, "sigma": NOISE_SIGMA},
    "synthetic:comp3-noisy-8k":  {"structure": "COMP3", "n": 8000, "sigma": NOISE_SIGMA},
    "synthetic:noise-8k":        {"structure": "NOISE", "n": 8000, "sigma": 1.0},
}


def _signal(structure: str, X: np.ndarray) -> tuple[np.ndarray, dict]:
    x = {j: X[:, j - 1] for j in range(1, 5)}
    if structure == "ADD":
        y = _F[1](x[1]) + _F[2](x[2]) + _F[3](x[3])
        gt = {"honest_depth": 1, "concepts": [
            {"support": [0], "order": 1}, {"support": [1], "order": 1},
            {"support": [2], "order": 1}]}
    elif structure == "COMP2":
        y = _F[1](x[1]) + x[2] * x[3]
        gt = {"honest_depth": 2, "concepts": [
            {"support": [0], "order": 1},
            {"support": [1, 2], "order": 2, "planted": True}]}
    elif structure == "COMP3":
        y = x[1] * x[2] * x[3] + _F[4](x[4])
        gt = {"honest_depth": 2, "concepts": [
            {"support": [0, 1, 2], "order": 3, "planted": True},
            {"support": [3], "order": 1}]}
    elif structure == "NOISE":
        y = np.zeros(X.shape[0])
        gt = {"honest_depth": 0, "concepts": []}
    else:
        raise ValueError(structure)
    return y, gt


def make_synthetic(name: str) -> Dataset:
    cfg = CONFIGS[name]
    g = rng(DATA_SEED, name)
    X = g.standard_normal((cfg["n"], D))
    y_sig, gt = _signal(cfg["structure"], X)
    y = y_sig + cfg["sigma"] * g.standard_normal(cfg["n"])
    # signal-vs-noise bookkeeping for honest ceilings
    var_sig, var_y = float(np.var(y_sig)), float(np.var(y))
    gt.update({"structure": cfg["structure"], "sigma": cfg["sigma"],
               "r2_ceiling": var_sig / var_y if var_y > 0 else 0.0})
    return Dataset(
        name=name, X=X, y=y, task="regression",
        feature_names=[f"x{j}" for j in range(1, D + 1)],
        ground_truth=gt, meta={"generator": "synthetic", "config": cfg})
