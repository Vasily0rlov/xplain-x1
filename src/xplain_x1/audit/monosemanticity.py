"""Monosemanticity mu(u) (M-#3.2, S-#8): held-out R2 of the best simple surrogate
of <= F_max parents predicting the unit's activation.

Candidate supports: all subsets (size <= F_max) of the unit's top-`top_parents`
parents by edge contribution.  Surrogate classes (fixed, pre-registered):
  |S| = 1  : isotonic regression  OR  5-knot cubic spline ridge (best of)
  |S| = 2,3: depth-3 decision tree OR degree-2 polynomial ridge (best of)
mu = best val R2; the argmax support is S(u); the winning surrogate is form(u).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=LinAlgWarning)  # ill-conditioned ridge
                                                           # solves are expected on
                                                           # tiny/collinear supports


@dataclass
class UnitAudit:
    uid: str
    layer: int
    mu: float
    support: list[int]           # parent indices (layer-local)
    support_names: list[str]
    form: str                    # short human-readable descriptor
    effective_fanin: int
    contribution: float          # ablation drop
    act_std: float


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    ss = float(np.var(y))
    if ss < 1e-12:
        return 1.0
    return float(1.0 - np.mean((pred - y) ** 2) / ss)


def _fit_1d(x_tr, y_tr, x_va, y_va) -> tuple[float, str]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer

    best, form = -np.inf, "?"
    for inc in (True, False):
        iso = IsotonicRegression(increasing=inc, out_of_bounds="clip")
        iso.fit(x_tr[:, 0], y_tr)
        r = _r2(iso.predict(x_va[:, 0]), y_va)
        if r > best:
            best, form = r, f"monotone{'+' if inc else '-'}"
    try:
        sp = SplineTransformer(n_knots=5, degree=3)
        z_tr, z_va = sp.fit_transform(x_tr), sp.transform(x_va)
        ridge = Ridge(alpha=1e-3).fit(z_tr, y_tr)
        r = _r2(ridge.predict(z_va), y_va)
        if r > best:
            best, form = r, "smooth"
    except ValueError:
        pass
    return best, form


def _fit_nd(x_tr, y_tr, x_va, y_va) -> tuple[float, str]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.tree import DecisionTreeRegressor

    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(x_tr, y_tr)
    best, form = _r2(tree.predict(x_va), y_va), "tree(d3)"
    poly = PolynomialFeatures(degree=2, include_bias=False)
    z_tr, z_va = poly.fit_transform(x_tr), poly.transform(x_va)
    ridge = Ridge(alpha=1e-3).fit(z_tr, y_tr)
    r = _r2(ridge.predict(z_va), y_va)
    if r > best:
        best, form = r, "poly(2)"
    return best, form


def unit_mu(a_tr: np.ndarray, a_va: np.ndarray, P_tr: np.ndarray, P_va: np.ndarray,
            edge_contrib: np.ndarray, parent_names: list[str],
            f_max: int = 3, top_parents: int = 6) -> tuple[float, list[int], str]:
    """mu, support, form for one unit given its parent activations."""
    if float(a_va.std()) < 1e-6:
        return 1.0, [], "constant"
    order = np.argsort(edge_contrib)[::-1]
    pool = [int(j) for j in order[:top_parents] if edge_contrib[j] > 0]
    best_mu, best_S, best_form = -np.inf, [], "?"
    for k in range(1, f_max + 1):
        for S in combinations(pool, k):
            xs_tr, xs_va = P_tr[:, S], P_va[:, S]
            if k == 1:
                r, form = _fit_1d(xs_tr, a_tr, xs_va, a_va)
            else:
                r, form = _fit_nd(xs_tr, a_tr, xs_va, a_va)
            if r > best_mu + 1e-9:
                best_mu, best_S, best_form = r, list(S), form
    return max(0.0, min(1.0, best_mu)), best_S, best_form


def effective_fanin(edge_contrib_row: np.ndarray, eps_edge: float) -> int:
    return int((edge_contrib_row >= eps_edge).sum())
