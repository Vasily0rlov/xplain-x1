"""Reality test (M-#3.6, S-#10): a concept is real only if ablating its unit
costs held-out fidelity, with a bootstrap CI excluding zero.

Per-row loss vectors are computed once for the base and ablated models; the
bootstrap resamples rows of those vectors (1000x) to get the CI on delta-fid.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.mlp import MaskedMLP
from ..train.settle import null_statistics
from ..util.seeding import rng


@torch.no_grad()
def _row_losses(model: MaskedMLP, ds: Dataset, splits: Splits,
                idx: np.ndarray, null_stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """(per-row model loss, per-row null loss) on rows idx."""
    X = torch.from_numpy(splits.standardise(ds.X[idx]))
    out = model(X)
    y = ds.y[idx]
    if ds.task == "classification":
        logp = F.log_softmax(out, dim=1).numpy()
        ll = -logp[np.arange(len(y)), y]
        ll_null = -np.log(np.clip(null_stats["class_prior"][y], 1e-12, 1))
        return ll, ll_null
    pred = out.squeeze(1).numpy()
    se = (pred - y) ** 2
    se_null = (y - null_stats["y_train_mean"]) ** 2
    return se, se_null


def _fid(ll: np.ndarray, ll_null: np.ndarray) -> float:
    return float(1.0 - ll.mean() / max(ll_null.mean(), 1e-12))


@torch.no_grad()
def reality_test(model: MaskedMLP, uid: str, ds: Dataset, splits: Splits,
                 n_bootstrap: int = 1000, seed: int = 0) -> dict:
    null_stats = null_statistics(ds, splits)
    ll_base, ll_null = _row_losses(model, ds, splits, splits.test, null_stats)

    li = next(j for j, ids in enumerate(model.unit_ids) if uid in ids)
    i = model.unit_ids[li].index(uid)
    out_mask = (model.mask(li + 1) if li + 1 < len(model.layers)
                else model.mask_head)
    saved = out_mask[:, i].clone()
    out_mask[:, i] = 0.0
    ll_abl, _ = _row_losses(model, ds, splits, splits.test, null_stats)
    out_mask[:, i] = saved

    delta = _fid(ll_base, ll_null) - _fid(ll_abl, ll_null)
    g = rng(seed, "reality", uid)
    n = len(ll_base)
    deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        bi = g.integers(0, n, n)
        deltas[b] = _fid(ll_base[bi], ll_null[bi]) - _fid(ll_abl[bi], ll_null[bi])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"uid": uid, "delta": float(delta),
            "ci_low": float(lo), "ci_high": float(hi)}
