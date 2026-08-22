"""Layer F: weighted functional-ANOVA decomposition of the learned function
(Build Plan P7; Lengerich et al. 2020 purification under the empirical measure,
estimation kept deliberately simple to prove the logic in principle).

Pipeline: quantile-bin features -> backfit truncated components (mains + pairs
+ screened triples) on (X, f(X)) -> mass-moving purification so every slice of
every tensor has zero weighted mean (the exact weighted-fANOVA condition for
piecewise-constant functions).  The declared measure w is the empirical
training distribution (cell counts).  Components are unique given w — the
carving-invariant certified objects.

Classification: the per-class logits are decomposed separately and a support's
share is the max over classes (a component matters if it drives any class).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.mlp import MaskedMLP

N_BINS = 8           # ONE quantile-bin tier for all orders: 16^3 cells starve
                     # the min-count guard and kill real order-3 effects (the
                     # comp3 lesson), and mixed tiers would need cross-resolution
                     # purification - complexity without need at proof stage
V_MIN = 0.01         # minimum variance share for a component to be "selected"
MAX_TRIPLES = 20     # screened order-3 candidates
BACKFIT_PASSES = 6


def _bin(ds: Dataset, splits: Splits, max_bins: int) -> tuple[np.ndarray, list[int]]:
    Xb = np.zeros_like(ds.X, dtype=np.int64)
    n_bins: list[int] = []
    for j in range(ds.d):
        col = ds.X[:, j]
        uniq = np.unique(ds.X[splits.train, j])
        if len(uniq) <= max_bins:                   # categorical/binary/ordinal
            edges = uniq
            Xb[:, j] = np.searchsorted(edges, col).clip(0, len(edges) - 1)
            n_bins.append(len(edges))
        else:
            qs = np.quantile(ds.X[splits.train, j],
                             np.linspace(0, 1, max_bins + 1)[1:-1])
            Xb[:, j] = np.searchsorted(np.unique(qs), col)
            n_bins.append(int(Xb[:, j].max()) + 1)
    return Xb, n_bins


def bin_features(ds: Dataset, splits: Splits) -> tuple[np.ndarray, list[int]]:
    """Bin all features on train-quantile edges -> integer bin ids [n, d]."""
    return _bin(ds, splits, N_BINS)


def _fit_component(resid: np.ndarray, cells: np.ndarray, size: int,
                   min_count: int = 5) -> np.ndarray:
    """Weighted (count) mean of residual per cell; sparse cells (< min_count
    train rows) get 0 — cheap shrinkage against interpolating noise."""
    sums = np.bincount(cells, weights=resid, minlength=size)
    cnts = np.bincount(cells, minlength=size)
    out = np.zeros(size)
    nz = cnts >= min_count
    out[nz] = sums[nz] / cnts[nz]
    return out


def _purify(tensors: dict, counts: dict, shapes: dict,
            intercept: float, n_iter: int = 25) -> tuple[dict, float]:
    """Mass-moving (Lengerich Alg. 1-2): make every slice of every tensor have
    zero WEIGHTED mean, cascading order-3 -> order-2 -> mains -> intercept.

    All tensors are SHAPED arrays over their supports' bin grids; counts[u] is
    the empirical measure w on u's grid.  Removing axis `ax` from sorted
    support u yields exactly shapes[sub], so the slice-mean array broadcasts
    into the sub-tensor directly.
    """
    order = sorted(tensors, key=len, reverse=True)
    for u in order:
        T = tensors[u]
        W = counts[u].reshape(shapes[u]).astype(float)
        for _ in range(n_iter):
            moved = 0.0
            for ax, _feat in enumerate(u):
                wsum = W.sum(axis=ax)
                m = np.where(wsum > 0,
                             (T * W).sum(axis=ax) / np.clip(wsum, 1e-12, None),
                             0.0)
                T = T - np.expand_dims(m, ax)
                moved = max(moved, float(np.abs(m).max()) if m.size else 0.0)
                sub = u[:ax] + u[ax + 1:]
                if sub:
                    tensors[sub] = tensors[sub] + m
                else:
                    intercept += float(m)
            if moved < 1e-10:
                break
        tensors[u] = T
    return tensors, intercept


def decompose(fvals: np.ndarray, Xb: np.ndarray, n_bins: list[int],
              train_idx: np.ndarray, val_idx: np.ndarray, K: int = 2,
              triples: list[tuple[int, int, int]] | None = None) -> dict:
    """Truncated weighted fANOVA of fvals: tensors FIT on train rows, shares
    scored OUT-OF-SAMPLE on val rows as cov(T_u(x), y)/var(y) — a noise-fitted
    tensor decorrelates on val, so spurious components vanish (the honest
    estimator; train-side shares hit r2=1.0 on pure noise).
    Returns {"components": {support: share}, "recon_r2": float (val)}.
    """
    d = Xb.shape[1]
    tr = train_idx
    y = fvals[tr].astype(float)
    intercept = float(y.mean())
    resid = y - intercept

    supports: list[tuple[int, ...]] = [(j,) for j in range(d)]
    supports += [tuple(s) for s in combinations(range(d), 2)] if K >= 2 else []
    supports += [tuple(t) for t in (triples or [])]

    shapes = {u: tuple(n_bins[f] for f in u) for u in supports}
    sizes = {u: int(np.prod(shapes[u])) for u in supports}
    cell_ids = {}
    for u in supports:
        cid = np.zeros(len(tr), dtype=np.int64)
        for f in u:
            cid = cid * n_bins[f] + Xb[tr, f]
        cell_ids[u] = cid
    counts = {u: np.bincount(cell_ids[u], minlength=sizes[u]) for u in supports}
    tensors = {u: np.zeros(sizes[u]) for u in supports}

    # cyclic backfitting: mains first, then pairs, then triples
    ordered = sorted(supports, key=len)
    for _ in range(BACKFIT_PASSES):
        for u in ordered:
            resid += tensors[u][cell_ids[u]]
            tensors[u] = _fit_component(resid, cell_ids[u], sizes[u])
            resid -= tensors[u][cell_ids[u]]

    # purification: exact zero-mean slices under the empirical measure
    tensors = {u: t.reshape(shapes[u]) for u, t in tensors.items()}
    tensors, intercept = _purify(tensors, counts, shapes, intercept)

    # out-of-sample scoring on val
    yv = fvals[val_idx].astype(float)
    yv_c = yv - float(y.mean())
    var_v = float(np.var(yv))
    comp = {}
    recon_v = np.full(len(val_idx), intercept)
    for u in supports:
        flat = tensors[u].reshape(-1)
        cid_v = np.zeros(len(val_idx), dtype=np.int64)
        for f in u:
            cid_v = cid_v * n_bins[f] + Xb[val_idx, f]
        vals_v = flat[cid_v]
        recon_v += vals_v
        share = float(np.mean(vals_v * yv_c) / max(var_v, 1e-12))
        if share >= V_MIN:
            comp[u] = round(share, 4)
    r2 = float(1 - np.mean((yv - recon_v) ** 2) / max(var_v, 1e-12))
    return {"components": comp, "recon_r2": round(r2, 4),
            "_tensors": tensors, "_intercept": intercept, "_n_bins": n_bins}


def predict_decomposition(dec: dict, Xb: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Evaluate the fitted (truncated) decomposition on rows idx."""
    n_bins = dec["_n_bins"]
    out = np.full(len(idx), dec["_intercept"])
    for u, T in dec["_tensors"].items():
        cid = np.zeros(len(idx), dtype=np.int64)
        for f in u:
            cid = cid * n_bins[f] + Xb[idx, f]
        out += T.reshape(-1)[cid]
    return out


def screen_triples(fvals: np.ndarray, Xb: np.ndarray, n_bins: list[int],
                   train_idx: np.ndarray, val_idx: np.ndarray,
                   max_triples: int = MAX_TRIPLES) -> list[tuple[int, int, int]]:
    """Screen ALL triples when feasible (a pure order-3 effect has ZERO
    pairwise projections, so pair-based candidate features would miss it);
    for large d restrict to the top features by main+pair evidence.  Scored by
    OUT-OF-SAMPLE covariance of the raw triple fit."""
    d = Xb.shape[1]
    two = decompose(fvals, Xb, n_bins, train_idx, val_idx, K=2)
    if d <= 12:
        cands = list(combinations(range(d), 3))
    else:
        feats = sorted({f for u in two["components"] for f in u})[:12]
        if len(feats) < 3:
            return []
        cands = list(combinations(feats, 3))
    # score on the RESIDUAL after K=2: raw-y scoring lets every triple that
    # contains a dominant main outrank a genuinely pure order-3 effect
    tr, va = train_idx, val_idx
    y = fvals[tr].astype(float) - predict_decomposition(two, Xb, tr)
    yv_c = fvals[va].astype(float) - predict_decomposition(two, Xb, va)
    yv_c = yv_c - float(y.mean())
    scores = []
    for t in cands:
        size = int(np.prod([n_bins[f] for f in t]))
        cid = np.zeros(len(tr), dtype=np.int64)
        cid_v = np.zeros(len(va), dtype=np.int64)
        for f in t:
            cid = cid * n_bins[f] + Xb[tr, f]
            cid_v = cid_v * n_bins[f] + Xb[va, f]
        m = _fit_component(y - y.mean(), cid, size)
        scores.append((float(np.mean(m[cid_v] * yv_c)), t))
    scores.sort(reverse=True)
    return [t for s, t in scores[:max_triples] if s > 0]


@torch.no_grad()
def model_fvals(model: MaskedMLP, ds: Dataset, splits: Splits) -> np.ndarray:
    """Scalar function values to decompose: regression output (scaled space) or
    per-class logits [n, C] for classification."""
    X = torch.from_numpy(splits.standardise(ds.X))
    out = model(X).numpy()
    return out if ds.task == "classification" else out[:, 0:1]


def component_shares(model: MaskedMLP, ds: Dataset, splits: Splits,
                     K: int = 2, with_triples: bool = True) -> dict:
    """Per-support max-over-outputs variance shares for one model."""
    fv = model_fvals(model, ds, splits)
    Xb, n_bins = bin_features(ds, splits)
    merged: dict[tuple[int, ...], float] = {}
    r2s = []
    for c in range(fv.shape[1]):
        triples = (screen_triples(fv[:, c], Xb, n_bins, splits.train, splits.val)
                   if with_triples else [])
        dec = decompose(fv[:, c], Xb, n_bins, splits.train, splits.val,
                        K=K, triples=triples)
        r2s.append(dec["recon_r2"])
        for u, share in dec["components"].items():
            merged[u] = max(merged.get(u, 0.0), share)
    return {"components": merged, "recon_r2": round(float(np.mean(r2s)), 4)}
