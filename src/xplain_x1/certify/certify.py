"""Certification orchestration (M-#4.4, S-#10): restarts -> matching -> CPSS ->
reality -> CORE/PERIPHERY labels.  Run 0 (first dev seed) anchors the concept
frame; the deliverable model is run 0's model.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from ..data.registry import get_dataset
from ..data.splits import cpss_pairs, make_splits
from ..pipeline import run_pipeline
from ..util.box import pin_threads
from .cpss import ev_bound, selection_frequencies, structure_universe_size
from .matching import cluster_concepts
from .reality import reality_test
from .signatures import input_support, unit_signatures


def _one_restart(dataset: str, cfg: dict, seed: int, n_workers: int) -> dict:
    pin_threads(n_workers=n_workers)
    r = run_pipeline(dataset, cfg, seed)
    ds = get_dataset(dataset)
    sigs = unit_signatures(r["model"], ds, r["splits"])
    units_by_id = {u["uid"]: u for u in r["final_audit"]["units"]}
    supports = {s.uid: input_support(s.uid, units_by_id) for s in sigs}
    from .fanova import V_MIN, component_shares
    fdec = component_shares(r["model"], ds, r["splits"])
    # Shares are re-based to TASK variance: raw shares are relative to var(f),
    # and a near-useless model's internal wiggle can decompose "stably" because
    # restarts share training data (the NOISE x5 lesson).  share_task =
    # share_f x max(fid, 0): fraction of target variance the component carries.
    scale = max(0.0, float(r["final_audit"]["fidelity"]))
    comps = {u: round(s * scale, 4) for u, s in fdec["components"].items()
             if s * scale >= V_MIN}
    return {"seed": seed, "result": r, "sigs": sigs, "supports": supports,
            "mu": {u["uid"]: u["mu"] for u in r["final_audit"]["units"]},
            "components": comps, "fanova_r2": fdec["recon_r2"]}


def _one_half(dataset: str, cfg: dict, seed: int, half, fid_ref: float,
              n_workers: int) -> dict:
    pin_threads(n_workers=n_workers)
    r = run_pipeline(dataset, cfg, seed, train_override=half, fid_ref=fid_ref)
    ds = get_dataset(dataset)
    sigs = unit_signatures(r["model"], ds, r["splits"])
    units_by_id = {u["uid"]: u for u in r["final_audit"]["units"]}
    from .fanova import V_MIN, component_shares
    fdec = component_shares(r["model"], ds, r["splits"])
    scale = max(0.0, float(r["final_audit"]["fidelity"]))
    comps = {u: round(s * scale, 4) for u, s in fdec["components"].items()
             if s * scale >= V_MIN}
    return {"sigs": sigs,
            "mu": {u["uid"]: u["mu"] for u in r["final_audit"]["units"]},
            "supports": {s.uid: input_support(s.uid, units_by_id) for s in sigs},
            "components": comps}


def certify(dataset: str, cfg: dict, n_workers: int | None = None) -> dict:
    ccfg = cfg["certify"]
    R = int(ccfg["restarts"])
    B = int(ccfg["cpss_pairs"])
    tau = float(ccfg["tau_match"])
    mu_min = float(ccfg["mu_min"])
    pi_thr = float(ccfg["pi_thr"])
    workers = n_workers or int(cfg["compute"]["n_workers"])

    # 1) restarts (dev seeds 0..R-1; run 0 anchors)
    restarts = Parallel(n_jobs=min(workers, R), backend="loky")(
        delayed(_one_restart)(dataset, cfg, s, min(workers, R)) for s in range(R))
    main = restarts[0]
    concepts = cluster_concepts([r["sigs"] for r in restarts],
                                [r["supports"] for r in restarts], tau)

    # E3.2 instrumentation: Pi stability of the concept set across tau values
    tau_sweep = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        cs = cluster_concepts([r["sigs"] for r in restarts],
                              [r["supports"] for r in restarts], t)
        tau_sweep.append({
            "tau": t, "n_concepts": len(cs),
            "n_stable": sum(1 for c in cs if c.Pi >= float(ccfg["Pi_min"])),
            "supports_stable": sorted(
                sorted(c.modal_support) for c in cs
                if c.Pi >= float(ccfg["Pi_min"]) and c.members.get(0)),
        })

    # 2) CPSS half-runs (cpss seed offset 100 — outside dev and confirmatory pools)
    ds = get_dataset(dataset)
    splits = make_splits(ds, int(cfg["data"]["split_seed"]),
                         int(cfg["data"]["probe_size"]),
                         tuple(cfg["data"]["fractions"]))
    pairs = cpss_pairs(splits.train, B, int(cfg["data"]["split_seed"]))
    halves = [h for pair in pairs for h in pair]
    fid_ref = main["result"]["fid_ref"]
    half_results = Parallel(n_jobs=min(workers, len(halves)), backend="loky")(
        delayed(_one_half)(dataset, cfg, 100 + i, h, fid_ref,
                           min(workers, len(halves)))
        for i, h in enumerate(halves))
    pi = selection_frequencies(main["sigs"], half_results, tau, mu_min)
    # q = distinct SUPPORTS selected per subsample (the universe is supports,
    # M-#3.6): several units carrying the same support are one selection.
    q_mean = float(sum(
        len({tuple(sorted(hr["supports"][uid]))
             for uid, m in hr["mu"].items()
             if m >= mu_min and uid in hr["supports"]})
        for hr in half_results) / max(1, len(half_results)))
    model = main["result"]["model"]
    p_universe = structure_universe_size(model.d_in, model.widths,
                                        int(cfg["audit"]["f_max"]))
    ev = ev_bound(q_mean, p_universe, pi_thr)
    # E3.3 instrumentation: raw-feature universe comparison (v4 vacuity finding)
    p_raw = structure_universe_size(model.d_in, [1], 1)   # d singleton features
    ev_raw = ev_bound(q_mean, p_raw, pi_thr)

    # 3) reality tests on main-run units + 4) labels
    units_by_id = {u["uid"]: u for u in main["result"]["final_audit"]["units"]}
    concept_rows = []
    for c in concepts:
        uid0 = c.members.get(0)
        row = {"cid": c.cid, "Pi": c.Pi, "members": c.members,
               "modal_support": sorted(c.modal_support),
               "support_variants": c.support_variants,
               "multiplicitous": c.multiplicitous,
               "uid_main": uid0}
        if uid0 is not None and uid0 in units_by_id:
            u = units_by_id[uid0]
            rt = reality_test(model, uid0, ds, splits,
                              int(ccfg["bootstrap"]),
                              seed=int(cfg["data"]["split_seed"]))
            row.update({
                "mu": u["mu"], "form": u["form"],
                "support_names": u["support_names"],
                "layer": u["layer"], "contribution": u["contribution"],
                "pi": pi.get(uid0, 0.0),
                "delta": rt["delta"], "ci_low": rt["ci_low"],
                "ci_high": rt["ci_high"]})
            reasons = []
            if u["mu"] < mu_min:
                reasons.append("polysemantic")
            if c.Pi < float(ccfg["Pi_min"]):
                reasons.append("unstable")
            if row["pi"] < pi_thr:
                reasons.append("infrequent")
            if not (rt["delta"] >= float(ccfg["delta_min"]) and rt["ci_low"] > 0):
                reasons.append("no_effect")
            if c.multiplicitous:
                reasons.append("multiplicitous")
            row["label"] = "CORE" if not reasons else "PERIPHERY"
            row["reasons"] = reasons
        else:
            row.update({"label": "PERIPHERY", "reasons": ["absent_in_main"]})
        concept_rows.append(row)

    # ---- route level (P6): certify at the Rashomon-invariant layer --------
    from .groups import discover_groups, group_index, group_names, group_support
    from .routes import (build_routes, route_q_mean, route_selection,
                         route_universe_size)

    groups = discover_groups(ds, splits)
    gi = group_index(groups, ds.d)

    def gs_map_of(supports: dict, mus: dict) -> dict:
        out: dict[frozenset[int], float] = {}
        for u_id, sup in supports.items():
            gs = group_support(sup, gi)
            if not gs:
                continue
            m = float(mus.get(u_id, 0.0))
            if gs not in out or m > out[gs]:
                out[gs] = m
        return out

    run_gs = {ri: gs_map_of(r["supports"], r["mu"])
              for ri, r in enumerate(restarts)}
    main_members: dict[frozenset[int], list[str]] = {}
    for u_id, sup in main["supports"].items():
        main_members.setdefault(group_support(sup, gi), []).append(u_id)
    routes = build_routes(run_gs, main_members, run_gs[0], R, mu_min)
    half_gs = [gs_map_of(hr["supports"], hr["mu"]) for hr in half_results]
    pi_routes = route_selection(half_gs, routes, mu_min)
    p_routes = route_universe_size(len(groups), int(cfg["audit"]["f_max"]))
    ev_routes = ev_bound(route_q_mean(half_gs, mu_min), p_routes, pi_thr)
    gnames = group_names(groups, ds.feature_names)

    route_rows = []
    for rt in routes:
        row = {"rid": rt.rid,
               "support_groups": sorted(rt.certified_support),
               "support_names": [gnames[g] for g in sorted(rt.certified_support)],
               "common_core": sorted(rt.common_core),
               "common_core_names": [gnames[g] for g in sorted(rt.common_core)],
               "variants": rt.variants,
               "Pi": rt.Pi, "pi": pi_routes[rt.rid],
               "members_main": rt.members_main, "mu": rt.best_mu_main,
               "multiplicitous": rt.multiplicitous}
        if rt.members_main:
            rtst = reality_test(model, rt.members_main, ds, splits,
                                int(ccfg["bootstrap"]),
                                seed=int(cfg["data"]["split_seed"]))
            row.update({"delta": rtst["delta"], "ci_low": rtst["ci_low"],
                        "ci_high": rtst["ci_high"]})
            reasons = []
            if (rt.best_mu_main or 0) < mu_min:
                reasons.append("polysemantic")
            if rt.Pi < float(ccfg["Pi_min"]):
                reasons.append("unstable")
            if row["pi"] < pi_thr:
                reasons.append("infrequent")
            if not (rtst["delta"] >= float(ccfg["delta_min"])
                    and rtst["ci_low"] > 0):
                reasons.append("no_effect")
            if rt.multiplicitous:
                reasons.append("multiplicitous")
            row["label"] = "CORE" if not reasons else "PERIPHERY"
            row["reasons"] = reasons
        else:
            row.update({"label": "PERIPHERY", "reasons": ["absent_in_main"]})
        route_rows.append(row)

    # ---- Layer F (P7): certified fANOVA components of the learned function --
    from math import comb as _comb

    from .fanova import V_MIN
    all_supports: set[tuple] = set()
    for rr in restarts:
        all_supports |= set(rr["components"])
    comp_rows = []
    half_comp = [hr.get("components", {}) for hr in half_results]
    for u in sorted(all_supports, key=lambda s: (len(s), s)):
        share_main = restarts[0]["components"].get(u, 0.0)
        Pi_F = sum(1 for rr in restarts if u in rr["components"]) / R
        pi_F = (sum(1 for hc in half_comp if u in hc) / max(1, len(half_comp)))
        gs = sorted({int(gi[f]) for f in u})
        reasons = []
        if share_main < V_MIN:
            reasons.append("absent_in_main")
        if Pi_F < float(ccfg["Pi_min"]):
            reasons.append("unstable")
        if pi_F < pi_thr:
            reasons.append("infrequent")
        comp_rows.append({
            "support": list(u),
            "support_names": [ds.feature_names[f] for f in u],
            "group_support": gs,
            "group_names": [gnames[g] for g in gs],
            "share_main": share_main,
            "shares_all": [rr["components"].get(u, 0.0) for rr in restarts],
            "Pi": round(Pi_F, 3), "pi": round(pi_F, 3),
            "label": "CORE" if not reasons else "PERIPHERY",
            "reasons": reasons})
    q_comp = float(np.mean([len(hc) for hc in half_comp])) if half_comp else 0.0
    p_comp = sum(_comb(ds.d, a) for a in range(1, min(3, ds.d) + 1))
    ev_comp = ev_bound(q_comp, p_comp, pi_thr)
    # group-aggregated certified shares (collinearity-robust claims)
    group_shares: dict[tuple, float] = {}
    for row in comp_rows:
        if row["label"] == "CORE":
            key = tuple(row["group_support"])
            group_shares[key] = group_shares.get(key, 0.0) + row["share_main"]

    # ---- Layer R: portfolio range statements (MCR-style group reliance) ----
    # "every restart relies on group G by >= min_reliance": fidelity drop when
    # the group's features are jointly permuted on val, minimised over restarts.
    from ..train.settle import evaluate as _eval
    from ..train.settle import null_statistics as _nulls
    reliance_rows = []
    rng_r = np.random.default_rng(int(cfg["data"]["split_seed"]))
    perm = rng_r.permutation(len(splits.val))
    null_stats_r = _nulls(ds, splits)
    for g_id, group in enumerate(groups):
        drops = []
        X_orig = ds.X
        Xp = ds.X.copy()
        for f in group:
            Xp[splits.val, f] = Xp[splits.val[perm], f]
        for rr in restarts:
            mdl = rr["result"]["model"]
            base = rr["result"]["final_audit"]["fidelity"]
            try:
                ds.X = Xp
                fid_p = _eval(mdl, ds, splits, splits.val,
                              null_stats_r)["fidelity"]
            finally:
                ds.X = X_orig
            drops.append(base - fid_p)
        reliance_rows.append({
            "group": gnames[g_id],
            "min_reliance": round(float(min(drops)), 4),
            "max_reliance": round(float(max(drops)), 4)})

    return {
        "dataset": dataset, "R": R, "B": B,
        "concepts": concept_rows,
        "components": comp_rows,
        "n_core_components": sum(1 for c in comp_rows if c["label"] == "CORE"),
        "group_shares": {",".join(map(str, k)): round(v, 4)
                         for k, v in group_shares.items()},
        "ev_bound_components": ev_comp, "p_components_universe": p_comp,
        "fanova_r2": restarts[0].get("fanova_r2"),
        "reliance": reliance_rows,
        "routes": route_rows,
        "n_core_routes": sum(1 for r in route_rows if r["label"] == "CORE"),
        "groups": [[ds.feature_names[j] for j in g] for g in groups],
        "n_groups": len(groups),
        "ev_bound_routes": ev_routes, "p_routes_universe": p_routes,
        "n_core": sum(1 for c in concept_rows if c["label"] == "CORE"),
        "ev_bound": ev, "q_mean": q_mean, "p_universe": p_universe,
        "ev_bound_raw_universe": ev_raw, "p_raw_universe": p_raw,
        "tau_sweep": tau_sweep,
        "main": {"widths": model.widths,
                 "fidelity": main["result"]["final_audit"]["fidelity"],
                 "fid_ref": fid_ref,
                 "val": main["result"]["val"], "test": main["result"]["test"],
                 "actions": main["result"]["actions"]},
        "model": model, "splits": splits,
        "restart_fids": [r["result"]["final_audit"]["fidelity"] for r in restarts],
    }
