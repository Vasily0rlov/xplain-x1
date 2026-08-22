"""Feature grouping for route-level certification (Build Plan P6, owner decision
2026-08-22).

Collinearity groups are discovered from TRAIN DATA ONLY (model-independent,
auditable): greedy clustering of features with |Spearman rho| >= threshold
(pinned 0.8).  On near-orthogonal data every group is a singleton and
route-level certification reduces exactly to unit-level (E6.1 bar).

The certified object at route level is a GROUP-SUPPORT: the unit's input
support mapped feature -> group.  Units whose carvings differ only within
collinearity groups share a group-support — the Rashomon-invariant identity.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..data.dataset import Dataset
from ..data.splits import Splits


def discover_groups(ds: Dataset, splits: Splits, rho_min: float = 0.8
                    ) -> list[list[int]]:
    """Greedy |Spearman|-clustering of features on the train split.

    Deterministic: features scanned in index order; a feature joins the first
    existing group whose EVERY member correlates with it at |rho| >= rho_min
    (complete linkage — keeps groups tight), else starts a new group.
    """
    X = ds.X[splits.train]
    d = ds.d
    if d == 1:
        return [[0]]
    rho = spearmanr(X).statistic
    if np.ndim(rho) == 0:          # d == 2 returns a scalar
        rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    rho = np.nan_to_num(np.abs(rho), nan=0.0)

    groups: list[list[int]] = []
    for j in range(d):
        placed = False
        for g in groups:
            if all(rho[j, k] >= rho_min for k in g):
                g.append(j)
                placed = True
                break
        if not placed:
            groups.append([j])
    return groups


def group_index(groups: list[list[int]], d: int) -> np.ndarray:
    """feature index -> group id."""
    gi = np.full(d, -1, dtype=int)
    for g_id, g in enumerate(groups):
        for j in g:
            gi[j] = g_id
    return gi


def group_support(support: frozenset[int] | set[int],
                  gi: np.ndarray) -> frozenset[int]:
    """Map an input-feature support to its group-support."""
    return frozenset(int(gi[j]) for j in support)


def group_names(groups: list[list[int]], feature_names: list[str],
                max_show: int = 3) -> list[str]:
    """Human-readable group labels: singleton -> the feature name; else
    'G{id}{first members,...}'."""
    out = []
    for g_id, g in enumerate(groups):
        if len(g) == 1:
            out.append(feature_names[g[0]])
        else:
            shown = ", ".join(feature_names[j] for j in g[:max_show])
            more = f", +{len(g) - max_show}" if len(g) > max_show else ""
            out.append(f"G{g_id}{{{shown}{more}}}")
    return out
