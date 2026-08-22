"""Route-level certification (Build Plan P6): the Rashomon-invariant layer.

A ROUTE is identified by its GROUP-SUPPORT — the unit's recursive input support
mapped through the collinearity groups (groups.py).  Units whose carvings
differ only within groups share a route.  Identity across runs is exact
group-support equality, with the invariant-core chain rule applied at group
level (nested group-supports = boundary jitter of one route; incomparable =
genuinely alternative routes, still labelled multiplicitous).

Certification statistics mirror the unit level:
  Pi_route  — fraction of restarts in which the route is present
              (some unit with that group-support and mu >= mu_min)
  pi_route  — CPSS selection frequency, same presence test on half-runs
  Delta     — JOINT ablation of all main-run units carrying the route
  E[V]      — MB bound over the input-group universe (supports of arity <=
              F_max over groups); q = distinct routes selected per half-run
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np


@dataclass
class Route:
    rid: str
    certified_support: frozenset[int]   # MODAL variant (most runs agree on it);
    common_core: frozenset[int] = frozenset()   # intersection of variants —
    # the conservative "at least these are involved" claim.  Minimal-variant
    # certification was wrong for under-detection chains (a COMP3 run that only
    # detected {x1} of the product {x1,x2,x3} would drag the certified support
    # down to {x1}); modal certification names what the runs typically found.
    variants: list[list[int]] = field(default_factory=list)
    present_runs: set[int] = field(default_factory=set)
    Pi: float = 0.0
    pi: float = 0.0
    members_main: list[str] = field(default_factory=list)   # main-run unit ids
    best_mu_main: float | None = None
    multiplicitous: bool = False


def _presence(run_gs: dict[int, dict[frozenset[int], float]], mu_min: float
              ) -> dict[frozenset[int], set[int]]:
    """group-support -> set of runs where present with mu >= mu_min."""
    out: dict[frozenset[int], set[int]] = {}
    for r, gs_map in run_gs.items():
        for gs, mu in gs_map.items():
            if mu >= mu_min and gs:
                out.setdefault(gs, set()).add(r)
    return out


def build_routes(run_gs: dict[int, dict[frozenset[int], float]],
                 main_members: dict[frozenset[int], list[str]],
                 main_mu: dict[frozenset[int], float],
                 n_runs: int, mu_min: float) -> list[Route]:
    """run_gs: run index -> {group-support: best mu}; main_members/mu keyed by
    group-support for run 0."""
    presence = _presence(run_gs, mu_min)
    candidates = sorted(presence, key=len)            # smallest-first
    merged_into: dict[frozenset[int], frozenset[int]] = {}
    variants: dict[frozenset[int], set[frozenset[int]]] = {
        gs: {gs} for gs in candidates}

    # invariant-core chain rule at group level: a superset candidate whose
    # presence complements a smaller one is jitter of the same route
    for i, g in enumerate(candidates):
        if g in merged_into:
            continue
        for g2 in candidates[i + 1:]:
            if g2 in merged_into or not g < g2:
                continue
            if presence[g2] - presence[g]:            # complements: merge down
                presence[g] |= presence[g2]
                variants[g] |= variants.pop(g2)
                merged_into[g2] = g
            elif presence[g2] <= presence[g]:         # redundant superset
                variants[g] |= variants.pop(g2)
                merged_into[g2] = g

    # per-variant run counts, for modal-variant selection
    variant_runs: dict[frozenset[int], int] = {}
    for gs_map in run_gs.values():
        for gs in gs_map:
            variant_runs[gs] = variant_runs.get(gs, 0) + 1

    routes = []
    for i, g in enumerate(sorted(variants, key=lambda s: (len(s), sorted(s)))):
        vset = variants[g]
        var_list = sorted(sorted(v) for v in vset)
        modal = max(vset, key=lambda v: (variant_runs.get(v, 0), -len(v)))
        core = frozenset.intersection(*vset)
        members, mu = [], None
        for v in vset:
            members += main_members.get(v, [])
            m = main_mu.get(v)
            if m is not None:
                mu = max(mu, m) if mu is not None else m
        # Rashomon multiplicity = variants share NO common anchor.  Within a
        # chain-merged route every variant contains the base group, so the
        # core is non-empty by construction: companion variation around a
        # shared anchor is jitter/context, not rival explanations.  Genuine
        # alternatives (mushroom odor vs spore-print) appear as SEPARATE
        # parallel routes whose low Pi the stability bar already handles.
        routes.append(Route(
            rid=f"r{i}", certified_support=modal, common_core=core,
            variants=var_list,
            present_runs=presence[g], Pi=len(presence[g]) / n_runs,
            members_main=sorted(set(members)), best_mu_main=mu,
            multiplicitous=(len(core) == 0)))
    return routes


def route_selection(half_gs: list[dict[frozenset[int], float]],
                    routes: list[Route], mu_min: float) -> dict[str, float]:
    """pi per route over CPSS half-runs (presence of any variant)."""
    out = {}
    for route in routes:
        vset = {frozenset(v) for v in route.variants}
        hits = sum(
            1 for gs_map in half_gs
            if any(mu >= mu_min and gs in vset for gs, mu in gs_map.items()))
        out[route.rid] = hits / max(1, len(half_gs))
    return out


def route_universe_size(n_groups: int, f_max: int) -> int:
    return sum(comb(n_groups, a) for a in range(1, min(f_max, n_groups) + 1))


def route_q_mean(half_gs: list[dict[frozenset[int], float]],
                 mu_min: float) -> float:
    return float(np.mean([
        len({gs for gs, mu in gs_map.items() if mu >= mu_min and gs})
        for gs_map in half_gs])) if half_gs else 0.0
