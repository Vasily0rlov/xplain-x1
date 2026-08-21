"""Greedy cross-run matching and concept clustering (M-#3.6, S-#10).

Two units instantiate the same concept iff cosine(attribution) >= tau_match AND
|Pearson(probe activations)| >= 0.7.  Concepts are transitive clusters of
matched units across runs; Pi(c) = fraction of runs represented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .signatures import UnitSignature

ACT_CORR_MIN = 0.7


def match_pair(sigs_a: list[UnitSignature], sigs_b: list[UnitSignature],
               tau: float) -> list[tuple[str, str, float]]:
    """Greedy 1:1 matching between two runs: [(uid_a, uid_b, cosine)]."""
    if not sigs_a or not sigs_b:
        return []
    A = np.stack([s.attribution for s in sigs_a])
    B = np.stack([s.attribution for s in sigs_b])
    cos = A @ B.T
    order = np.dstack(np.unravel_index(np.argsort(-cos, axis=None), cos.shape))[0]
    used_a: set[int] = set()
    used_b: set[int] = set()
    out = []
    for i, j in order:
        if cos[i, j] < tau:
            break
        if i in used_a or j in used_b:
            continue
        r = np.corrcoef(sigs_a[i].probe_acts, sigs_b[j].probe_acts)[0, 1]
        if abs(float(r)) < ACT_CORR_MIN:
            continue
        used_a.add(int(i))
        used_b.add(int(j))
        out.append((sigs_a[i].uid, sigs_b[j].uid, float(cos[i, j])))
    return out


@dataclass
class Concept:
    cid: str
    members: dict[int, str] = field(default_factory=dict)   # run index -> uid
    Pi: float = 0.0
    modal_support: frozenset[int] = frozenset()
    support_variants: list[list[int]] = field(default_factory=list)
    multiplicitous: bool = False


def cluster_concepts(run_sigs: list[list[UnitSignature]],
                     run_supports: list[dict[str, frozenset[int]]],
                     tau: float) -> list[Concept]:
    """Union-find over pairwise matches vs run 0's frame (star clustering).

    Star topology (every run matched against run 0) keeps clusters 1:1 with the
    reference run's units; concepts present in other runs but absent from run 0
    are collected as their own clusters afterwards.
    """
    n_runs = len(run_sigs)
    clusters: dict[str, Concept] = {}
    matched_elsewhere: list[set[str]] = [set() for _ in range(n_runs)]

    for i, sig in enumerate(run_sigs[0]):
        clusters[sig.uid] = Concept(cid=f"c{i}", members={0: sig.uid})
    for r in range(1, n_runs):
        for uid0, uid_r, _cos in match_pair(run_sigs[0], run_sigs[r], tau):
            clusters[uid0].members[r] = uid_r
            matched_elsewhere[r].add(uid_r)

    # units in run r>0 with no run-0 partner become singleton concepts
    extra = 0
    for r in range(1, n_runs):
        for sig in run_sigs[r]:
            if sig.uid not in matched_elsewhere[r]:
                clusters[f"x{r}:{sig.uid}"] = Concept(
                    cid=f"e{extra}", members={r: sig.uid})
                extra += 1

    out = []
    for c in clusters.values():
        c.Pi = len(c.members) / n_runs
        variants: list[frozenset[int]] = []
        for r, uid in c.members.items():
            sup = run_supports[r].get(uid)
            if sup is not None:
                variants.append(sup)
        if variants:
            uniq = {}
            for v in variants:
                uniq[v] = uniq.get(v, 0) + 1
            c.support_variants = [sorted(v) for v in uniq]
            chain = all(a <= b or b <= a for a in uniq for b in uniq)
            if chain:
                # Nested variants = boundary jitter of ONE concept (a weak
                # rider present in some runs).  The certified support is the
                # minimal common variant — the run-invariant core (POSITIONING
                # "invariant core") — with the jitter on record above.
                c.modal_support = min(uniq, key=len)
                c.multiplicitous = False
            else:
                # Incomparable variants = genuinely interchangeable
                # alternatives (Rashomon multiplicity): not certifiable unique.
                c.modal_support = max(uniq, key=uniq.get)
                c.multiplicitous = True
        out.append(c)
    return out
