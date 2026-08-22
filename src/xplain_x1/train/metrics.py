"""Fidelity = held-out pseudo-R2 (M-#3.1): 1 - LL_model/LL_null (classification),
ordinary R2 vs the train-mean null (regression).  Accuracy reported separately."""
from __future__ import annotations

import numpy as np


def _clip(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(p, eps, 1.0)


def classification_fidelity(proba: np.ndarray, y: np.ndarray,
                            class_prior: np.ndarray) -> float:
    ll_model = -np.log(_clip(proba[np.arange(len(y)), y])).mean()
    ll_null = -np.log(_clip(class_prior[y])).mean()
    return float(1.0 - ll_model / max(ll_null, 1e-12))


def regression_fidelity(pred: np.ndarray, y: np.ndarray, y_train_mean: float) -> float:
    mse = float(np.mean((pred - y) ** 2))
    var = float(np.mean((y - y_train_mean) ** 2))
    return float(1.0 - mse / max(var, 1e-12))


def accuracy(proba_or_pred: np.ndarray, y: np.ndarray, task: str) -> float | None:
    if task != "classification":
        return None
    return float((proba_or_pred.argmax(axis=1) == y).mean())
