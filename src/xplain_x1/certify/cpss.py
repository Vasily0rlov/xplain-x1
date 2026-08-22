"""Complementary-pairs stability selection (M-#3.6, S-#10).

The pipeline reruns on B complementary half/half partitions of the training set
(2B runs).  A concept is SELECTED in a run iff some unit matches its signature
(tau_match + activation correlation) with mu >= mu_min.  pi(c) = selection
fraction.  E[V] <= q^2 / ((2*pi_thr - 1) * p) over the STRUCTURE-LEVEL universe:
all supports of arity <= F_max per layer interface of the main model's topology
(a raw-feature universe makes the bound vacuous — v4 finding, M-#9).
"""
from __future__ import annotations

from math import comb

import numpy as np

from .matching import match_pair
from .signatures import UnitSignature


def structure_universe_size(d_in: int, widths: list[int], f_max: int) -> int:
    p = 0
    parents = [d_in] + widths[:-1]
    for n_par in parents:
        p += sum(comb(n_par, a) for a in range(1, min(f_max, n_par) + 1))
    return p


def selection_frequencies(main_sigs: list[UnitSignature],
                          half_run_results: list[dict],
                          tau: float, mu_min: float) -> dict[str, float]:
    """pi(uid of main concept) over the 2B half-runs.

    Each half-run result: {"sigs": [UnitSignature], "mu": {uid: mu}}.
    """
    counts = {s.uid: 0 for s in main_sigs}
    n_runs = len(half_run_results)
    for res in half_run_results:
        matched = match_pair(main_sigs, res["sigs"], tau)
        for uid_main, uid_half, _cos in matched:
            if res["mu"].get(uid_half, 0.0) >= mu_min:
                counts[uid_main] += 1
    return {uid: c / max(1, n_runs) for uid, c in counts.items()}


def ev_bound(q_mean: float, p_universe: int, pi_thr: float) -> float:
    """Meinshausen-Buhlmann expected false selections bound (exchangeability
    assumption reported, not silently assumed away — see certificate)."""
    denom = (2.0 * pi_thr - 1.0) * p_universe
    return float(q_mean ** 2 / denom) if denom > 0 else float("inf")
