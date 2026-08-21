"""E2.1 / E2.2 / E2.3 — growth controller behaviour (Build Plan P2, milestone M3).

E2.1 honest flatness: ADD + NOISE, 4 seeds.  Bar: ends at 1 hidden layer, no
      accepted depth growth; NOISE ends trivial (no accepted growth at all).
E2.2 earned depth:   COMP2 + COMP3 (noisy 8k), 4 seeds.  Bar (pre-registered
      aggregation: >= 3/4 seeds per structure): reaches L=2, fid within
      delta_stop of Fid_ref, and a planted-support unit present with mu >= 0.8.
E2.3 power floor:    COMP2 (noisy 2k), 4 seeds, descriptive record for P3.

Run: .venv/bin/python experiments/e2x.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xplain_x1.data.registry import get_dataset            # noqa: E402
from xplain_x1.pipeline import run_pipeline                # noqa: E402
from xplain_x1.util.box import pin_threads, wait_until_free  # noqa: E402
from xplain_x1.util.io import save_json                    # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402

SEEDS = [0, 1, 2, 3]
FLAT = ["synthetic:add-noisy-8k", "synthetic:noise-8k"]
DEEP = ["synthetic:comp2-noisy-8k", "synthetic:comp3-noisy-8k"]
FLOOR = ["synthetic:comp2-noisy-2k"]
ALL = FLAT + DEEP + FLOOR


def input_support(uid: str, units_by_id: dict) -> set[int]:
    """Recursive input-feature support of a unit via its audit supports."""
    u = units_by_id[uid]
    if u["layer"] == 1:
        return set(u["support"])
    out: set[int] = set()
    for pname in u["support_names"]:
        if pname in units_by_id:
            out |= input_support(pname, units_by_id)
    return out


def one_run(name: str, seed: int, cfg: dict) -> dict:
    pin_threads(n_workers=16)
    t0 = time.time()
    r = run_pipeline(name, cfg, seed)
    audit = r["final_audit"]
    units_by_id = {u["uid"]: u for u in audit["units"]}
    ds = get_dataset(name)
    planted = [set(c["support"]) for c in ds.ground_truth["concepts"]
               if c.get("planted")]
    planted_hits = []
    for u in audit["units"]:
        if u["act_std"] <= 1e-6 or u["mu"] < 0.8:
            continue
        sup = input_support(u["uid"], units_by_id)
        if planted and sup == planted[0] and abs(u["contribution"]) >= 0.005:
            planted_hits.append(u["uid"])
    accepted = [a["action"] for a in r["actions"] if a["action"].startswith("accept")]
    return {"dataset": name, "seed": seed, "widths": r["widths"],
            "n_layers": len(r["widths"]), "rounds": r["rounds"],
            "fid": r["final_audit"]["fidelity"], "fid_ref": r["fid_ref"],
            "gap": r["fid_ref"] - r["final_audit"]["fidelity"],
            "accepted_growth": accepted, "planted_hits": planted_hits,
            "median_mu_live": audit["median_mu_live"],
            "actions": [a["action"] for a in r["actions"]],
            "wall_s": round(time.time() - t0, 1)}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    wait_until_free(float(cfg["compute"]["load_threshold"]))
    jobs = [(n, s) for n in ALL for s in SEEDS]
    runs = Parallel(n_jobs=min(20, len(jobs)), backend="loky")(
        delayed(one_run)(n, s, cfg) for n, s in jobs)
    by = lambda name: [r for r in runs if r["dataset"] == name]  # noqa: E731

    # E2.1 — honest flatness
    flat_ok = {}
    for name in FLAT:
        rs = by(name)
        no_depth = all("accept_depth" not in r["accepted_growth"] for r in rs)
        one_layer = all(r["n_layers"] == 1 for r in rs)
        trivial = (name != "synthetic:noise-8k"
                   or all(not r["accepted_growth"] for r in rs))
        flat_ok[name] = {"one_layer": one_layer, "no_accepted_depth": no_depth,
                         "noise_trivial": trivial,
                         "bar_met": bool(one_layer and no_depth and trivial)}

    # E2.2 — earned depth (>= 3/4 seeds)
    deep_ok = {}
    for name in DEEP:
        rs = by(name)
        good = [r for r in rs
                if r["n_layers"] >= 2 and r["gap"] <= 0.01 + 1e-9
                and r["planted_hits"]]
        deep_ok[name] = {"seeds_passing": len(good), "of": len(rs),
                         "bar_met": len(good) >= 3}

    floor_desc = [{k: r[k] for k in
                   ("seed", "widths", "fid", "gap", "planted_hits", "rounds")}
                  for r in by(FLOOR[0])]

    out = {"experiment": "E2.x", "git_commit": git_commit(), "runs": runs,
           "e21": flat_ok, "e22": deep_ok, "e23_descriptive": floor_desc,
           "e21_bar_met": all(v["bar_met"] for v in flat_ok.values()),
           "e22_bar_met": all(v["bar_met"] for v in deep_ok.values())}
    save_json(ROOT / "experiments" / "results" / "e2x.json", out)

    print(f"E2.1 flatness: {out['e21_bar_met']} {flat_ok}")
    print(f"E2.2 depth:    {out['e22_bar_met']} {deep_ok}")
    for r in floor_desc:
        print(f"E2.3 floor seed {r['seed']}: widths {r['widths']} fid {r['fid']:.3f} "
              f"hits {r['planted_hits']}")
    return 0 if out["e21_bar_met"] and out["e22_bar_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
