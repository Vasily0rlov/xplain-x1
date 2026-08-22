"""E-C.1 — CONFIRMATORY ONE-SHOT (Build Plan post-P5 gate; owner approval
2026-08-23).  Frozen config; untouched seeds (restarts 20-24, CPSS 2100+);
NO reruns; verdicts stand as read (v4 discipline).

Endpoints (frozen, identical bars to the dev batteries):
  primary   E7.1: planted components CORE on ADD/COMP2/COMP3; NOISE certifies
            ZERO components (untradeable);
  primary   E7.2 clause 1: >= 3 of 4 real datasets certify CORE components
            with share-sum >= 0.5;
  report    per-dataset certified component tables, coverage, reliance;
            bike hour x temp purified share (finding follow-up, no bar).
Run: nohup .venv/bin/python experiments/ec1_confirmatory.py > experiments/results/ec1.log 2>&1 &
"""
from __future__ import annotations

import copy
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
from xplain_x1.util.provenance import git_commit, sha256_config  # noqa: E402

SYNTH = ["synthetic:add-noisy-8k", "synthetic:comp2-noisy-8k",
         "synthetic:comp3-noisy-8k", "synthetic:noise-8k"]
REAL = ["mushroom", "drybean", "bike", "adult"]


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["certify"]["seed_base"] = 20
    cfg["certify"]["restarts"] = 5          # seeds 20-24, one shot
    wait_until_free(float(cfg["compute"]["load_threshold"]))
    t0 = time.time()
    out = {"experiment": "E-C.1 confirmatory", "git_commit": git_commit(),
           "config_hash": sha256_config(cfg), "seed_base": 20,
           "synthetic": [], "real": []}

    for name in SYNTH + REAL:
        cert = certify(name, cfg)
        ds = get_dataset(name)
        core = [c for c in cert["components"] if c["label"] == "CORE"]
        row = {"dataset": name, "n_core": len(core),
               "core": [{"names": c["support_names"], "share": c["share_main"],
                         "Pi": c["Pi"], "pi": c["pi"]}
                        for c in sorted(core, key=lambda c: -c["share_main"])],
               "core_share_sum": round(sum(c["share_main"] for c in core), 3),
               "fanova_r2": cert["fanova_r2"],
               "reliance_top": sorted(cert["reliance"],
                                      key=lambda r: -r["min_reliance"])[:3]}
        if name.startswith("synthetic:"):
            planted = [tuple(sorted(c["support"]))
                       for c in ds.ground_truth["concepts"]]
            core_sups = {tuple(sorted(c["support"])) for c in core}
            row["planted"] = [list(p) for p in planted]
            row["planted_core"] = [list(p) for p in planted if p in core_sups]
            row["max_core_order"] = max((len(c["support"]) for c in core),
                                        default=0)
        bucket = "synthetic" if name.startswith("synthetic:") else "real"
        out[bucket].append(row)
        print(f"{name}: core {row['n_core']} share_sum {row['core_share_sum']}"
              + (f" planted_core {row.get('planted_core')}"
                 if bucket == "synthetic" else ""), flush=True)
        save_json(ROOT / "experiments" / "results" / "ec1.json", out)

    def srow(n):
        return next(r for r in out["synthetic"] if n in r["dataset"])
    e71 = (srow("add-")["planted_core"] == srow("add-")["planted"]
           and srow("add-")["max_core_order"] <= 1
           and srow("comp2")["planted_core"] == srow("comp2")["planted"]
           and srow("comp3")["planted_core"] == srow("comp3")["planted"]
           and srow("noise")["n_core"] == 0)
    certifying = [r["dataset"] for r in out["real"]
                  if r["n_core"] >= 1 and r["core_share_sum"] >= 0.5]
    out["verdict"] = {
        "e71_synthetic": bool(e71),
        "real_certifying": certifying,
        "e72_clause1": len(certifying) >= 3,
        "wall_hours": round((time.time() - t0) / 3600, 2),
    }
    save_json(ROOT / "experiments" / "results" / "ec1.json", out)
    print("CONFIRMATORY:", out["verdict"], flush=True)
    return 0 if e71 and len(certifying) >= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
