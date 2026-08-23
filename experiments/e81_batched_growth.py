"""E8.1 — batched width-growth trials (Build Plan P8, pre-registered).

Question: does letting a width trial add grow_batch=8 units at once (so diffuse
per-unit gains aggregate above the delta_grow=0.005 accept bar) close bike's
fidelity gap without breaking the honesty properties?

Pre-registered bar (Build Plan P8/E8.1):
  bike (seeds 0-3): median val fid ratio >= 0.97  AND  median live-unit mu >= 0.9
  guards:  NOISE  -> zero ACCEPTED growth actions
           ADD    -> stays 1 hidden layer AND at ceiling (gap <= delta_stop)
           COMP2  -> planted {x2,x3} support at mu >= 0.8 in >= 3/4 seeds
  adult (seeds 0-1): report-only.

Run: nohup .venv/bin/python experiments/e81_batched_growth.py \
       > experiments/results/e81.log 2>&1 &
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from joblib import Parallel, delayed                       # noqa: E402

from xplain_x1.pipeline import run_pipeline                # noqa: E402
from xplain_x1.util.box import pin_threads, wait_until_free  # noqa: E402
from xplain_x1.util.io import save_json                    # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402

GROW_BATCH = 8
JOBS = [  # (dataset, seed)
    *[("bike", s) for s in range(4)],
    *[("adult", s) for s in range(2)],
    *[("synthetic:noise-8k", s) for s in range(4)],
    *[("synthetic:add-noisy-8k", s) for s in range(4)],
    *[("synthetic:comp2-noisy-8k", s) for s in range(4)],
]


def run_one(name: str, seed: int, cfg: dict, workers: int) -> dict:
    pin_threads(n_workers=workers)
    t0 = time.time()
    r = run_pipeline(name, cfg, seed)
    au = r["final_audit"]
    live = [u for u in au["units"] if u["act_std"] > 1e-6]
    accepted_growth = [a for a in r["actions"]
                      if a["action"] in ("grow_width", "grow_depth")]
    reverted = {a["action"] for a in r["actions"]
                if a["action"].startswith("revert_")}
    # net accepted growth = trials minus reverts (a revert cancels the trial)
    n_width_trials = sum(1 for a in accepted_growth if a["action"] == "grow_width")
    n_width_reverts = sum(1 for a in r["actions"] if a["action"] == "revert_width")
    n_depth_trials = sum(1 for a in accepted_growth if a["action"] == "grow_depth")
    n_depth_reverts = sum(1 for a in r["actions"] if a["action"] == "revert_depth")
    planted = [u for u in live
               if set(u["support_names"]) == {"x2", "x3"} and u["mu"] >= 0.8]
    return {
        "dataset": name, "seed": seed,
        "fid_val": round(r["val"]["fidelity"], 4),
        "fid_test": round(r["test"]["fidelity"], 4),
        "fid_ref": round(r["fid_ref"], 4),
        "ratio": round(r["val"]["fidelity"] / r["fid_ref"], 4) if r["fid_ref"] else None,
        "gap": round(r["fid_ref"] - r["val"]["fidelity"], 4),
        "widths": r["widths"], "n_layers": len(r["widths"]),
        "n_live": len(live),
        "median_mu_live": au["median_mu_live"],
        "width_accepted": n_width_trials - n_width_reverts,
        "depth_accepted": n_depth_trials - n_depth_reverts,
        "comp2_planted_hit": bool(planted),
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["controller"]["grow_batch"] = GROW_BATCH

    print(f"E8.1 batched growth: grow_batch={GROW_BATCH}, {len(JOBS)} runs", flush=True)
    wait_until_free()
    n_par = min(len(JOBS), 18)
    per = max(2, 64 // n_par)
    rows = Parallel(n_jobs=n_par, backend="loky")(
        delayed(run_one)(name, seed, cfg, per) for name, seed in JOBS)

    by = lambda d: [r for r in rows if r["dataset"] == d]
    import statistics as st
    bike = by("bike")
    bike_ratio = st.median(r["ratio"] for r in bike)
    bike_mu = st.median(r["median_mu_live"] for r in bike)
    noise_ok = all(r["width_accepted"] == 0 and r["depth_accepted"] == 0
                   for r in by("synthetic:noise-8k"))
    add_ok = all(r["n_layers"] == 1 and r["gap"] <= cfg["controller"]["delta_stop"]
                 for r in by("synthetic:add-noisy-8k"))
    comp2_hits = sum(r["comp2_planted_hit"] for r in by("synthetic:comp2-noisy-8k"))
    comp2_ok = comp2_hits >= 3
    bar_met = (bike_ratio >= 0.97 and bike_mu >= 0.9
               and noise_ok and add_ok and comp2_ok)

    out = {
        "experiment": "E8.1 batched width growth",
        "git_commit": git_commit(), "grow_batch": GROW_BATCH,
        "rows": rows,
        "bike_median_ratio": round(bike_ratio, 4),
        "bike_median_mu": round(bike_mu, 4),
        "guards": {"noise_zero_growth": noise_ok, "add_flat_at_ceiling": add_ok,
                   "comp2_recovery": f"{comp2_hits}/4"},
        "adult_report": [{k: r[k] for k in ("seed", "ratio", "widths",
                                            "median_mu_live")} for r in by("adult")],
        "bar_met": bar_met,
    }
    save_json(ROOT / "experiments" / "results" / "e81.json", out)
    for r in rows:
        print(f"  {r['dataset']:28s} s{r['seed']}: ratio={r['ratio']} "
              f"widths={r['widths']} live={r['n_live']} mu={r['median_mu_live']:.3f} "
              f"w+={r['width_accepted']} d+={r['depth_accepted']} ({r['wall_s']}s)",
              flush=True)
    print(f"\nbike median ratio={bike_ratio:.4f} (bar >=0.97) "
          f"median mu={bike_mu:.3f} (bar >=0.9)")
    print(f"guards: noise={noise_ok} add={add_ok} comp2={comp2_hits}/4")
    print("E8.1 BAR:", "MET" if bar_met else "NOT MET")


if __name__ == "__main__":
    main()
