"""E5.1/E5.2/E5.3 — the MVL battery (Build Plan P5, milestone M7).

Runs the full certification pipeline on the seven public MVL datasets
(small -> large), evaluates the M-#7 bars per dataset, the M-#9 v4 anchors
(E5.2), and the K3 wall-clock budget (E5.3).  Synthetic-suite results (H-X1-2,
H-X1-3 synthetic half) come from e31.json.  Small-n datasets (n < 2000) are
evaluated under the owner-decision-A regime: fidelity + legibility + honest
labels; recovery/certification bars live on the n >= 8k datasets.

Run: nohup .venv/bin/python experiments/e51_ladder.py > experiments/results/e51.log 2>&1 &
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xplain_x1.certify.certify import certify              # noqa: E402
from xplain_x1.data.registry import MVL_PUBLIC, get_dataset  # noqa: E402
from xplain_x1.util.box import wait_until_free             # noqa: E402
from xplain_x1.util.io import load_json, save_json         # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402

POWER_FLOOR_N = 2000
TTT_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
             (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
V4_ANCHORS = {"wine": 0.766, "adult": 0.892, "mushroom": 0.995}


def dataset_row(name: str, cfg: dict) -> dict:
    t0 = time.time()
    cert = certify(name, cfg)
    ds = get_dataset(name)
    concepts = cert["concepts"]
    core = [c for c in concepts if c["label"] == "CORE"]
    main_cs = [c for c in concepts if c.get("uid_main")]

    total_delta = sum(max(0.0, c.get("delta") or 0.0) for c in main_cs)
    core_cov = (sum(max(0.0, c.get("delta") or 0.0) for c in core) / total_delta
                if total_delta > 1e-9 else 0.0)
    fid = cert["main"]["fidelity"]
    fid_ref = cert["main"]["fid_ref"]
    acc = cert["main"]["val"].get("accuracy")

    row = {
        "dataset": name, "n": ds.n, "d": ds.d,
        "regime": "below_power_floor" if ds.n < POWER_FLOOR_N else "standard",
        "widths": cert["main"]["widths"],
        "fid": fid, "fid_ref": fid_ref,
        "fid_ratio": round(fid / fid_ref, 4) if fid_ref else None,
        "acc": acc,
        "n_concepts": len(main_cs), "n_core": len(core),
        "core_coverage": round(core_cov, 3),
        "median_core_mu": (float(np.median([c["mu"] for c in core]))
                           if core else None),
        "min_core_Pi": (float(min(c["Pi"] for c in core)) if core else None),
        "ev_bound": cert["ev_bound"],
        "n_multiplicitous": sum(1 for c in main_cs if c.get("multiplicitous")),
        "periphery_reasons": sorted({r for c in main_cs
                                     for r in c.get("reasons", [])}),
        "core_supports_named": [c.get("support_names") for c in core][:12],
        "wall_s": round(time.time() - t0, 1),
    }

    # dataset-specific checks
    if name == "tictactoe":
        legible = [tuple(sorted(c["modal_support"])) for c in main_cs
                   if (c.get("mu") or 0) >= 0.8 and len(c["modal_support"]) == 3]
        row["ttt_lines_as_legible_units"] = sorted(
            {s for s in legible if s in {tuple(l) for l in TTT_LINES}})
    if name == "bike":
        idx = {f: i for i, f in enumerate(ds.feature_names)}
        target = {idx.get("hour"), idx.get("temp")}
        row["hour_temp_core"] = any(set(c["modal_support"]) == target
                                    for c in core)
    if name == "mushroom":
        row["odor_in_core"] = any(
            any(s.startswith("odor=") for s in (c.get("support_names") or []))
            for c in core)
    if name == "adult":
        row["honest_shallow"] = len(cert["main"]["widths"]) == 1
    return row


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    wait_until_free(float(cfg["compute"]["load_threshold"]))
    t_start = time.time()
    rows = []
    for name in MVL_PUBLIC:
        row = dataset_row(name, cfg)
        rows.append(row)
        print(f"{name}: fid_ratio {row['fid_ratio']} core {row['n_core']}/"
              f"{row['n_concepts']} cov {row['core_coverage']} "
              f"regime {row['regime']} ({row['wall_s']}s)", flush=True)
        save_json(ROOT / "experiments" / "results" / "e51.json",
                  {"experiment": "E5.1", "git_commit": git_commit(),
                   "partial": name != MVL_PUBLIC[-1], "rows": rows})

    # ---- bars (standard-regime datasets carry the certification bars) ----
    std = [r for r in rows if r["regime"] == "standard"]
    e31 = load_json(ROOT / "experiments" / "results" / "e31.json")
    bars = {
        "H-X1-1_convergence": {
            "bar": "CORE coverage >= 0.7 and median CORE mu >= 0.8 (standard regime)",
            "per_dataset": {r["dataset"]: {"cov": r["core_coverage"],
                                           "mu": r["median_core_mu"]} for r in std},
            "met": bool(std) and all(
                r["core_coverage"] >= 0.7 and (r["median_core_mu"] or 0) >= 0.8
                for r in std),
        },
        "H-X1-2_honest_flatness": {
            "bar": "0 certified depth>=2 on additive/noise (synthetic, e31)",
            "met": all(r["bar_met"] for r in e31["rows"]
                       if r["bar_met"] is not None),
        },
        "H-X1-3_recovery": {
            "bar": "synthetic planted recovered (e31); bike hour x temp CORE; "
                   "ttt lines legible (below floor: reported)",
            "synthetic": all(r["bar_met"] for r in e31["rows"]
                             if r["bar_met"] is not None),
            "bike_hour_temp": next((r.get("hour_temp_core") for r in rows
                                    if r["dataset"] == "bike"), None),
            "ttt_lines": next((r.get("ttt_lines_as_legible_units") for r in rows
                               if r["dataset"] == "tictactoe"), None),
        },
        "H-X1-4_concept_tax": {
            "bar": "median fid_ratio >= 0.98 across MVL public",
            "ratios": {r["dataset"]: r["fid_ratio"] for r in rows},
            "median_ratio": float(np.median([r["fid_ratio"] for r in rows])),
            "met": float(np.median([r["fid_ratio"] for r in rows])) >= 0.98,
        },
        "H-X1-5_stability": {
            "bar": "min CORE Pi >= 0.7 (definitional; reported)",
            "per_dataset": {r["dataset"]: r["min_core_Pi"] for r in std},
            "met": all((r["min_core_Pi"] or 0) >= 0.7 for r in std if r["n_core"]),
        },
        "H-X1-6_certified_honesty": {
            "bar": "reality-tested CORE; non-vacuous E[V]; periphery labelled",
            "ev_bounds": {r["dataset"]: r["ev_bound"] for r in std},
            "met": all(r["ev_bound"] < 1.0 for r in std if r["n_core"]),
        },
    }
    bars["H-X1-3_recovery"]["met"] = bool(
        bars["H-X1-3_recovery"]["synthetic"]
        and bars["H-X1-3_recovery"]["bike_hour_temp"])

    anchors = {}
    for name, v4 in V4_ANCHORS.items():
        r = next((x for x in rows if x["dataset"] == name), None)
        if r:
            anchors[name] = {"x1_fid_ratio": r["fid_ratio"], "v4_fid_ratio": v4,
                             "beats_v4": bool(r["fid_ratio"] >= v4)}
    mush = next((x for x in rows if x["dataset"] == "mushroom"), None)
    if mush:
        anchors["mushroom_multiplicity_labelled"] = mush["n_multiplicitous"] > 0
    adult = next((x for x in rows if x["dataset"] == "adult"), None)
    if adult:
        anchors["adult_honest_shallow"] = adult.get("honest_shallow")

    wall_h = (time.time() - t_start) / 3600
    out = {"experiment": "E5.1+E5.2+E5.3", "git_commit": git_commit(),
           "partial": False, "rows": rows, "bars": bars, "anchors": anchors,
           "wall_hours": round(wall_h, 2), "k3_met": wall_h <= 12.0}
    save_json(ROOT / "experiments" / "results" / "e51.json", out)
    print("bars:", {k: v.get("met") for k, v in bars.items()}, flush=True)
    print("anchors:", anchors, flush=True)
    print(f"E5 battery wall {wall_h:.2f}h (K3 <= 12h: {out['k3_met']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
