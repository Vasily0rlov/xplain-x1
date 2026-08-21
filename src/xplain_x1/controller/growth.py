"""The growth controller (M-#4.3, S-#9): settle -> audit -> act, from a small start.

Every accepted structural action is demanded by evidence: width when fidelity
lags AND monosemanticity stalls (true superposition), depth when width failed,
prune/merge/dissolve whenever they cost nothing.  Growth is accepted only if the
next settle improves val fidelity by >= delta_grow, else reverted (snapshots).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from ..audit.audit import mu_stalled, run_audit
from ..audit.contribution import edge_contributions
from ..audit.dissolve import dissolution_cost
from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.gauge import gauge_pass
from ..model.mlp import MaskedMLP, build_model
from ..model.ops import add_unit, insert_layer, merge_units, prune_edges, remove_units
from ..train.losses import make_pressures
from ..train.settle import evaluate, null_statistics, settle
from ..util.seeding import rng


@dataclass
class GrowthTrace:
    model: MaskedMLP
    audits: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    rounds: int = 0


def _probe_tensor(ds: Dataset, splits: Splits) -> torch.Tensor:
    return torch.from_numpy(splits.standardise(ds.X[splits.probe]))


def _least_mono_layer(audit: dict) -> int:
    """0-based index of the layer with lowest median live mu."""
    per_layer: dict[int, list[float]] = {}
    for u in audit["units"]:
        if u["act_std"] > 1e-6:
            per_layer.setdefault(u["layer"] - 1, []).append(u["mu"])
    if not per_layer:
        return 0
    return min(per_layer, key=lambda k: float(np.median(per_layer[k])))


def _least_mono_unit(audit: dict, li: int) -> str | None:
    cands = [u for u in audit["units"]
             if u["layer"] - 1 == li and u["act_std"] > 1e-6]
    if not cands:
        return None
    return min(cands, key=lambda u: u["mu"])["uid"]


def _prune_step(model: MaskedMLP, audit: dict, ds: Dataset, splits: Splits,
                cfg: dict, actions: list[dict]) -> MaskedMLP:
    acfg, ccfg = cfg["audit"], cfg["controller"]
    Xtr = splits.standardise(ds.X[splits.train])
    with torch.no_grad():
        acts = model.hidden(torch.from_numpy(Xtr))
    parent_acts = [Xtr] + [a.numpy() for a in acts[:-1]]

    cut = prune_edges(model, edge_contributions(model, parent_acts),
                      float(acfg["eps_edge"]))
    if cut:
        actions.append({"action": "prune_edges", "n": cut})

    # dead / no-effect units (keep >= 1 unit per layer)
    dead: set[str] = set()
    by_layer: dict[int, list[dict]] = {}
    for u in audit["units"]:
        by_layer.setdefault(u["layer"] - 1, []).append(u)
    for li, units in by_layer.items():
        removable = [u["uid"] for u in units
                     if u["act_std"] <= 1e-6
                     or abs(u["contribution"]) < float(ccfg["eps_prune"])
                     * max(abs(audit["fidelity"]), 1e-3)]
        keep_at_least = max(1, len(units) - len(removable))
        if keep_at_least < 1:
            removable = removable[:-1]
        if len(removable) == len(units):
            removable = removable[1:]
        dead.update(removable)
    if dead:
        model = remove_units(model, dead)
        actions.append({"action": "remove_units", "uids": sorted(dead)})

    # near-duplicate merge: incoming-weight cosine > tau AND val act corr > 0.95
    tau = float(cfg["certify"]["tau_match"])
    Xva = splits.standardise(ds.X[splits.val])
    with torch.no_grad():
        acts_va = [a.numpy() for a in model.hidden(torch.from_numpy(Xva))]
    merged = True
    while merged:
        merged = False
        for li in range(len(model.layers)):
            W = (model.layers[li].weight * model.mask(li)).detach().numpy()
            norms = np.linalg.norm(W, axis=1, keepdims=True).clip(1e-12)
            cos = (W / norms) @ (W / norms).T
            A = acts_va[li]
            for i in range(len(W)):
                for j in range(i + 1, len(W)):
                    if cos[i, j] > tau and A[:, i].std() > 1e-6 and A[:, j].std() > 1e-6:
                        corr = float(np.corrcoef(A[:, i], A[:, j])[0, 1])
                        if corr > 0.95:
                            keep = model.unit_ids[li][i]
                            drop = model.unit_ids[li][j]
                            model = merge_units(model, keep, drop)
                            actions.append({"action": "merge", "keep": keep,
                                            "drop": drop})
                            with torch.no_grad():
                                acts_va = [a.numpy() for a in
                                           model.hidden(torch.from_numpy(Xva))]
                            merged = True
                            break
                if merged:
                    break
            if merged:
                break
    return model


def grow(ds: Dataset, splits: Splits, cfg: dict, seed: int,
         fid_ref: float) -> GrowthTrace:
    mcfg, ccfg = cfg["model"], cfg["controller"]
    null_stats = null_statistics(ds, splits)
    probe = _probe_tensor(ds, splits)
    pressures = make_pressures(cfg)

    model = build_model(ds.d, [int(mcfg["init_width"])] * int(mcfg["init_layers"]),
                        ds.task, ds.n_classes, seed=seed)
    trace = GrowthTrace(model=model)
    pending: dict | None = None      # {"kind", "snapshot", "fid_before"}
    failed_kinds: set[str] = set()
    delta_grow = float(ccfg["delta_grow"])
    eps_depth = float(ccfg["eps_depth"])

    for rnd in range(int(ccfg["max_rounds"])):
        trace.rounds = rnd + 1
        settle(model, ds, splits, cfg, seed=seed, pressures=pressures)
        gauge_pass(model, probe)
        audit = run_audit(model, ds, splits, cfg)
        trace.audits.append(audit)
        fid = audit["fidelity"]

        if pending is not None:
            if fid >= pending["fid_before"] + delta_grow:
                trace.actions.append({"action": f"accept_{pending['kind']}",
                                      "round": rnd, "fid": fid})
                failed_kinds.clear()
            else:
                model = pending["snapshot"]
                failed_kinds.add(pending["kind"])
                trace.actions.append({"action": f"revert_{pending['kind']}",
                                      "round": rnd, "fid": fid})
                audit = run_audit(model, ds, splits, cfg)
                trace.audits[-1] = audit
                fid = audit["fidelity"]
            pending = None

        model = _prune_step(model, audit, ds, splits, cfg, trace.actions)

        # dissolve unearned depth
        if len(model.layers) >= 2:
            costs = [(dissolution_cost(model, li, ds, splits, seed), li)
                     for li in range(len(model.layers) - 1)]
            (best_cost, cand), best_li = min(costs, key=lambda t: t[0][0])
            if best_cost <= eps_depth:
                model = cand
                trace.actions.append({"action": "dissolve", "layer": best_li,
                                      "cost": round(best_cost, 5), "round": rnd})
                continue

        fid_now = evaluate(model, ds, splits, splits.val, null_stats)["fidelity"]
        gap = fid_ref - fid_now
        if gap <= float(ccfg["delta_stop"]):
            trace.actions.append({"action": "stop_at_ceiling", "round": rnd,
                                  "gap": round(gap, 5)})
            break

        stalled = mu_stalled(trace.audits) or len(trace.audits) == 1
        if not (stalled and gap > delta_grow):
            continue                       # pressures still working: keep settling

        can_widen = (sum(model.widths) + 2 <= int(mcfg["max_total_units"])
                     and "width" not in failed_kinds)
        can_deepen = (len(model.layers) + 1 <= int(mcfg["max_layers"])
                      and len(model.layers) >= 1 and "depth" not in failed_kinds)
        li = _least_mono_layer(audit)
        if can_widen:
            snapshot = copy.deepcopy(model)
            uid = _least_mono_unit(audit, li)
            model = add_unit(model, li, f"split:{uid}" if uid else "fresh", seed)
            model = add_unit(model, li, "fresh", seed + 7919)
            pending = {"kind": "width", "snapshot": snapshot, "fid_before": fid_now}
            trace.actions.append({"action": "grow_width", "layer": li, "round": rnd})
        elif can_deepen:
            snapshot = copy.deepcopy(model)
            pos = max(1, li + 1)           # identity insert needs post-ReLU input
            model = insert_layer(model, pos)
            pending = {"kind": "depth", "snapshot": snapshot, "fid_before": fid_now}
            trace.actions.append({"action": "grow_depth", "pos": pos, "round": rnd})
        else:
            trace.actions.append({"action": "stop_no_moves", "round": rnd,
                                  "gap": round(gap, 5)})
            break

    # final: settle -> gauge -> audit -> prune
    settle(model, ds, splits, cfg, seed=seed, pressures=pressures)
    gauge_pass(model, _probe_tensor(ds, splits))
    audit = run_audit(model, ds, splits, cfg)
    model = _prune_step(model, audit, ds, splits, cfg, trace.actions)
    trace.audits.append(run_audit(model, ds, splits, cfg))
    trace.model = model
    return trace
