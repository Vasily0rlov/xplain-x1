"""Certification orchestration (M-#4.4, S-#10): restarts -> matching -> CPSS ->
reality -> CORE/PERIPHERY labels.  Run 0 (first dev seed) anchors the concept
frame; the deliverable model is run 0's model.
"""
from __future__ import annotations

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
    return {"seed": seed, "result": r, "sigs": sigs, "supports": supports,
            "mu": {u["uid"]: u["mu"] for u in r["final_audit"]["units"]}}


def _one_half(dataset: str, cfg: dict, seed: int, half, fid_ref: float,
              n_workers: int) -> dict:
    pin_threads(n_workers=n_workers)
    r = run_pipeline(dataset, cfg, seed, train_override=half, fid_ref=fid_ref)
    ds = get_dataset(dataset)
    sigs = unit_signatures(r["model"], ds, r["splits"])
    return {"sigs": sigs,
            "mu": {u["uid"]: u["mu"] for u in r["final_audit"]["units"]}}


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
    q_mean = float(sum(
        sum(1 for m in hr["mu"].values() if m >= mu_min) for hr in half_results
    ) / max(1, len(half_results)))
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

    return {
        "dataset": dataset, "R": R, "B": B,
        "concepts": concept_rows,
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
