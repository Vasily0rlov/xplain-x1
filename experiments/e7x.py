"""E7 — Layer F certification battery (Build Plan P7).

E7.1 (synthetic exactness, pre-registered): planted components recovered as
     CORE (Pi_F >= 0.75) on ADD/COMP2/COMP3; NOISE certifies ZERO components
     (the untradeable at function level); ADD certifies order-1 only.
E7.2 (real data): >= 3 of 4 standard datasets certify >= 1 CORE component with
     total CORE group-share >= 0.5; bike certifies an hour x temp-family
     component (the clause that was ill-posed at route level).
E7.3 (descriptive): route/component concordance reported.

Run: nohup .venv/bin/python experiments/e7x.py > experiments/results/e7x.log 2>&1 &
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xplain_x1.certify.certify import certify              # noqa: E402
from xplain_x1.data.registry import get_dataset            # noqa: E402
from xplain_x1.util.box import wait_until_free             # noqa: E402
from xplain_x1.util.io import save_json                    # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402

SYNTH = ["synthetic:add-noisy-8k", "synthetic:comp2-noisy-8k",
         "synthetic:comp3-noisy-8k", "synthetic:noise-8k"]
REAL = ["mushroom", "drybean", "bike", "adult"]


def run_one(name: str, cfg: dict) -> dict:
    t0 = time.time()
    cert = certify(name, cfg)
    ds = get_dataset(name)
    comps = cert["components"]
    core = [c for c in comps if c["label"] == "CORE"]
    row = {
        "dataset": name,
        "n_components": len(comps), "n_core": len(core),
        "core": [{"names": c["support_names"], "share": c["share_main"],
                  "Pi": c["Pi"], "pi": c["pi"]} for c in
                 sorted(core, key=lambda c: -c["share_main"])[:10]],
        "core_share_sum": round(sum(c["share_main"] for c in core), 3),
        "group_shares": cert["group_shares"],
        "ev_bound_components": cert["ev_bound_components"],
        "fanova_r2": cert["fanova_r2"],
        "core_routes": [r["support_names"] for r in cert["routes"]
                        if r["label"] == "CORE"],
        "reliance_top": sorted(cert["reliance"],
                               key=lambda r: -r["min_reliance"])[:3],
        "wall_s": round(time.time() - t0, 1),
    }
    if name.startswith("synthetic:"):
        planted = [tuple(sorted(c["support"]))
                   for c in ds.ground_truth["concepts"]]
        core_sups = {tuple(sorted(c["support"])) for c in core}
        row["planted"] = [list(p) for p in planted]
        row["planted_core"] = [list(p) for p in planted if p in core_sups]
        row["max_core_order"] = max((len(c["support"]) for c in core), default=0)
    if name == "bike":
        row["hour_temp_component"] = any(
            any("hour" in n for n in c["names"])
            and any(("temp" in n) or ("atemp" in n) for n in c["names"])
            for c in row["core"])
    return row


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    wait_until_free(float(cfg["compute"]["load_threshold"]))
    out = {"experiment": "E7", "git_commit": git_commit(),
           "synthetic": [], "real": []}
    for name in SYNTH:
        row = run_one(name, cfg)
        out["synthetic"].append(row)
        print(f"{name}: core {row['n_core']} planted_core "
              f"{row.get('planted_core')} ({row['wall_s']}s)", flush=True)
        save_json(ROOT / "experiments" / "results" / "e7x.json", out)

    def srow(n):
        return next(r for r in out["synthetic"] if n in r["dataset"])
    e71 = (srow("add-").get("planted_core") == srow("add-").get("planted")
           and srow("add-")["max_core_order"] <= 1
           and srow("comp2")["planted_core"] == srow("comp2")["planted"]
           and srow("comp3")["planted_core"] == srow("comp3")["planted"]
           and srow("noise")["n_core"] == 0)
    out["e71_bar_met"] = bool(e71)
    print(f"E7.1: {'MET' if e71 else 'NOT MET'}", flush=True)

    for name in REAL:
        row = run_one(name, cfg)
        out["real"].append(row)
        print(f"{name}: core {row['n_core']}/{row['n_components']} share_sum "
              f"{row['core_share_sum']} ({row['wall_s']}s)", flush=True)
        save_json(ROOT / "experiments" / "results" / "e7x.json", out)
    certifying = [r for r in out["real"]
                  if r["n_core"] >= 1 and r["core_share_sum"] >= 0.5]
    bike = next((r for r in out["real"] if r["dataset"] == "bike"), None)
    out["e72"] = {"datasets_certifying": [r["dataset"] for r in certifying],
                  "bike_hour_temp": bike and bike.get("hour_temp_component"),
                  "bar_met": bool(len(certifying) >= 3 and bike
                                  and bike.get("hour_temp_component"))}
    save_json(ROOT / "experiments" / "results" / "e7x.json", out)
    print(f"E7.2: {out['e72']}", flush=True)
    return 0 if out["e71_bar_met"] and out["e72"]["bar_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
