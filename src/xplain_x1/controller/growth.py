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
from ..audit.contribution import edge_contributions, unit_ablation_drops
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
    """Every structural mutation here is BUDGETED (S-#9: pruning is accepted
    only if it costs <= eps_prune).  Unbudgeted edge-masking/merging measurably
    wrecked near-separable models (mushroom: pseudo-R2 is calibration-sensitive,
    a few flipped rows explode log-loss)."""
    acfg, ccfg = cfg["audit"], cfg["controller"]
    null_stats = null_statistics(ds, splits)
    eps = float(ccfg["eps_prune"]) * max(abs(audit["fidelity"]), 1e-3)
    fid0 = evaluate(model, ds, splits, splits.val, null_stats)["fidelity"]
    Xtr = splits.standardise(ds.X[splits.train])
    with torch.no_grad():
        acts = model.hidden(torch.from_numpy(Xtr))
    parent_acts = [Xtr] + [a.numpy() for a in acts[:-1]]

    saved_masks = [model.mask(li).clone() for li in range(len(model.layers))]
    cut = prune_edges(model, edge_contributions(model, parent_acts),
                      float(acfg["eps_edge"]))
    if cut:
        fid1 = evaluate(model, ds, splits, splits.val, null_stats)["fidelity"]
        if fid1 < fid0 - eps:                    # jointly too costly: revert
            with torch.no_grad():
                for li, m in enumerate(saved_masks):
                    model.mask(li).copy_(m)
            actions.append({"action": "prune_edges_reverted", "n": cut,
                            "cost": round(fid0 - fid1, 4)})
        else:
            actions.append({"action": "prune_edges", "n": cut})

    # Truly dead units (never activate): jointly safe to remove in one batch.
    by_layer: dict[int, list[dict]] = {}
    for u in audit["units"]:
        by_layer.setdefault(u["layer"] - 1, []).append(u)
    dead = {u["uid"] for units in by_layer.values() for u in units
            if u["act_std"] <= 1e-6}
    for li, units in by_layer.items():           # never empty a layer
        if all(u["uid"] in dead for u in units):
            dead.discard(units[0]["uid"])
    if dead:
        model = remove_units(model, dead)
        actions.append({"action": "remove_units", "uids": sorted(dead)})

    # Low-contribution units: SEQUENTIAL removal under a fidelity budget.
    # Batch removal by marginal ablation is unsound on redundant structure
    # (mushroom: backup rules each ablate cheaply alone, jointly load-bearing).
    fid_start = evaluate(model, ds, splits, splits.val, null_stats)["fidelity"]
    removed: list[str] = []
    for _ in range(sum(model.widths)):
        cands = sorted(
            ((u, d) for u, d in
             unit_ablation_drops(model, ds, splits, null_stats).items()
             if abs(d) < eps),
            key=lambda t: abs(t[1]))
        cands = [(u, d) for u, d in cands
                 if any(u in ids and len(ids) > 1 for ids in model.unit_ids)]
        if not cands:
            break
        uid = cands[0][0]
        trial = remove_units(model, {uid})
        fid_now = evaluate(trial, ds, splits, splits.val, null_stats)["fidelity"]
        if fid_now < fid_start - eps:            # joint cost exceeded budget
            break
        model = trial
        removed.append(uid)
    if removed:
        actions.append({"action": "remove_units_seq", "uids": removed})

    # Near-duplicate merge, BUDGETED per merge: summing outgoing weights is
    # exact only for identical activations; corr 0.95 is not identity, so each
    # merge must prove itself on val or be rejected.
    tau = float(cfg["certify"]["tau_match"])
    Xva = splits.standardise(ds.X[splits.val])
    with torch.no_grad():
        acts_va = [a.numpy() for a in model.hidden(torch.from_numpy(Xva))]
    fid_cur = evaluate(model, ds, splits, splits.val, null_stats)["fidelity"]
    tried: set[tuple[str, str]] = set()
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
                    keep = model.unit_ids[li][i]
                    drop = model.unit_ids[li][j]
                    if (keep, drop) in tried:
                        continue
                    if cos[i, j] > tau and A[:, i].std() > 1e-6 and A[:, j].std() > 1e-6:
                        corr = float(np.corrcoef(A[:, i], A[:, j])[0, 1])
                        if corr > 0.95:
                            tried.add((keep, drop))
                            trial = merge_units(model, keep, drop)
                            fid_t = evaluate(trial, ds, splits, splits.val,
                                             null_stats)["fidelity"]
                            if fid_t < fid_cur - eps:
                                actions.append({"action": "merge_rejected",
                                                "keep": keep, "drop": drop,
                                                "cost": round(fid_cur - fid_t, 4)})
                                continue
                            model, fid_cur = trial, fid_t
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
    pressures = make_pressures(cfg, ds.d)

    model = build_model(ds.d, [int(mcfg["init_width"])] * int(mcfg["init_layers"]),
                        ds.task, ds.n_classes, seed=seed)
    trace = GrowthTrace(model=model)
    pending: dict | None = None      # {"kind", "snapshot", "fid_before"}
    failed_kinds: set[str] = set()
    delta_grow = float(ccfg["delta_grow"])
    eps_depth = float(ccfg["eps_depth"])

    discovery_scale = float(ccfg.get("discovery_scale", 0.3))
    pressures.scale = discovery_scale       # start in discovery mode
    prev_fid: float | None = None

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
        # discovery/cleanup gate: full pressure only near the ceiling (M-C4)
        pressures.scale = (1.0 if gap <= 2 * float(ccfg["delta_stop"])
                           else discovery_scale)
        # progress unlocks retrying a previously reverted growth kind
        if prev_fid is not None and fid_now >= prev_fid + delta_grow:
            failed_kinds.clear()
        prev_fid = fid_now
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
        li = min(_least_mono_layer(audit), len(model.layers) - 1)
        if can_widen:
            snapshot = copy.deepcopy(model)
            uid = _least_mono_unit(audit, li)
            if uid is not None and uid not in model.unit_ids[li]:
                uid = None                 # audit is pre-prune; unit may be gone
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

    # final: settle -> gauge -> recalibrate -> audit -> prune
    settle(model, ds, splits, cfg, seed=seed, pressures=pressures)
    gauge_pass(model, _probe_tensor(ds, splits))
    if ds.task == "regression":
        # Closed-form linear head recalibration on val: regression runs can
        # walk into calibration-broken territory (near-perfect correlation,
        # exploded amplitude, fid -14) because accept/revert compares RELATIVE
        # improvements.  a*f+b is a gauge-legal readout scaling - structure,
        # units, and legibility are untouched; it folds into the head exactly.
        from ..train.settle import _tensors
        with torch.no_grad():
            Xv, yv = _tensors(ds, splits, splits.val)
            pred = model(Xv).squeeze(1)
            yv = yv.squeeze(1)
            vp = float(pred.var())
            if vp > 1e-12:
                a = float(((pred - pred.mean()) * (yv - yv.mean())).mean() / vp)
                b = float(yv.mean() - a * pred.mean())
                if abs(a - 1.0) > 1e-3 or abs(b) > 1e-3:
                    model.head.weight.mul_(a)
                    model.head.bias.mul_(a)
                    model.head.bias.add_(b)
                    trace.actions.append({"action": "recalibrate_head",
                                          "a": round(a, 4), "b": round(b, 4)})
    audit = run_audit(model, ds, splits, cfg)
    model = _prune_step(model, audit, ds, splits, cfg, trace.actions)
    trace.audits.append(run_audit(model, ds, splits, cfg))
    trace.model = model
    return trace
