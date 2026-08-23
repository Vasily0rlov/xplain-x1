"""A plain, standard feed-forward neural network baseline.

This is deliberately NOT the XPLAIN-x1 method: it is an ordinary single-hidden-
layer dense MLP trained with vanilla MSE — no sparsity/selectivity pressures, no
constructive growth, no certification.  It is what a practitioner would build by
default, and it exists only as an interpretability contrast (a dense, entangled
"black box") for the side-by-side Bike comparison.

It reuses the shared, method-neutral substrate — the `MaskedMLP` container (with
all masks left at 1, i.e. fully dense), the `settle` training loop with pressures
switched OFF, the data loaders/splits, and the `run_audit` measurement — because
a fair comparison MUST measure both models with the same instrument.  The only
thing that differs from the method is that this net gets none of the method's
pressures, growth, or certification.

Kept in `baselines/` so it never mixes with the method's own code path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xplain_x1.audit.audit import run_audit                # noqa: E402
from xplain_x1.data.dataset import Dataset                 # noqa: E402
from xplain_x1.data.splits import Splits                   # noqa: E402
from xplain_x1.model.mlp import build_model                # noqa: E402
from xplain_x1.train.settle import _tensors, evaluate, null_statistics, settle  # noqa: E402


def train_standard_nn(ds: Dataset, splits: Splits, cfg: dict,
                      width: int = 32, seed: int = 0) -> dict:
    """Train a standard 1-hidden-layer dense MLP (no pressures, no growth)."""
    model = build_model(ds.d, [width], ds.task, ds.n_classes, seed=seed)
    # pressures=None  ->  plain task loss (MSE/CE) + weight decay only.
    settle(model, ds, splits, cfg, seed=seed, pressures=None)
    null_stats = null_statistics(ds, splits)
    audit = run_audit(model, ds, splits, cfg)
    val = evaluate(model, ds, splits, splits.val, null_stats)
    test = evaluate(model, ds, splits, splits.test, null_stats)
    return {"model": model, "audit": audit, "val": val, "test": test,
            "widths": model.widths}


def predict(model, ds: Dataset, splits: Splits, idx: np.ndarray) -> np.ndarray:
    """Model output on `idx`, in the model's native (scaled-y) space."""
    model.eval()
    X, _ = _tensors(ds, splits, idx)
    with torch.no_grad():
        out = model(X)
    if ds.task == "classification":
        return torch.softmax(out, dim=1).numpy()
    return out.squeeze(1).numpy()
