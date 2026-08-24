"""E8.3 — is the hour-interaction PARTNER stable across restarts? (diagnostic)

Literature calls the canonical bike interaction hour × workingday; our pipeline
certified hour × weekday (a collinear sibling: workingday is derived from
weekday + holiday).  This checks, per restart and for BOTH growth regimes
(frozen grow_batch=2 vs batched grow_batch=8), which hour-pair carries the
purified interaction mass — i.e. whether the partner choice is stable or flips
with the carving.

Per restart: run the pipeline, purified-decompose the learned function, read the
share of every hour-containing pair.  Report the dominant partner per seed and
the presence/dominance tallies per regime.  Descriptive — no pass bar.

Run: nohup .venv/bin/python experiments/e83_partner_stability.py \
       > experiments/results/e83.log 2>&1 &
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

from xplain_x1.certify.fanova import V_MIN, component_shares  # noqa: E402
from xplain_x1.data.registry import get_dataset            # noqa: E402
from xplain_x1.data.splits import make_splits              # noqa: E402
from xplain_x1.pipeline import run_pipeline                # noqa: E402
from xplain_x1.util.box import pin_threads, wait_until_free  # noqa: E402
from xplain_x1.util.io import save_json                    # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402

HOUR, WEEKDAY, WORKINGDAY = 3, 5, 6
NAME = {WEEKDAY: "weekday", WORKINGDAY: "workingday"}
SEEDS = list(range(8))


def run_one(grow_batch: int, seed: int, workers: int) -> dict:
    pin_threads(n_workers=workers)
    cfg = copy.deepcopy(yaml.safe_load((ROOT / "configs" / "default.yaml").read_text()))
    cfg["controller"]["grow_batch"] = grow_batch
    t0 = time.time()
    r = run_pipeline("bike", cfg, seed)
    ds = get_dataset("bike")
    splits = make_splits(ds, int(cfg["data"]["split_seed"]),
                         int(cfg["data"]["probe_size"]),
                         tuple(cfg["data"]["fractions"]))
    comps = component_shares(r["model"], ds, splits)["components"]
    # every pair component that contains hour
    hour_pairs = {tuple(sorted(k)): round(float(v), 4)
                  for k, v in comps.items()
                  if len(k) == 2 and HOUR in k and v >= V_MIN}
    def share(other):
        return comps.get((HOUR, other), comps.get((other, HOUR), 0.0))
    wk, wd = float(share(WEEKDAY)), float(share(WORKINGDAY))
    partner = None
    if max(wk, wd) >= V_MIN:
        partner = "weekday" if wk >= wd else "workingday"
    named_pairs = {}
    for (a, b), v in hour_pairs.items():
        other = b if a == HOUR else a
        nm = ds.feature_names[other]
        named_pairs[nm] = v
    return {
        "grow_batch": grow_batch, "seed": seed,
        "widths": r["widths"],
        "weekday_share": round(wk, 4), "workingday_share": round(wd, 4),
        "dominant_partner": partner,
        "all_hour_pairs": named_pairs,
        "wall_s": round(time.time() - t0, 1),
    }


def summarise(rows: list[dict], gb: int) -> dict:
    rs = [r for r in rows if r["grow_batch"] == gb]
    from collections import Counter
    dom = Counter(r["dominant_partner"] for r in rs)
    wk_present = sum(1 for r in rs if r["weekday_share"] >= V_MIN)
    wd_present = sum(1 for r in rs if r["workingday_share"] >= V_MIN)
    both = sum(1 for r in rs if r["weekday_share"] >= V_MIN
               and r["workingday_share"] >= V_MIN)
    return {
        "grow_batch": gb, "n": len(rs),
        "dominant_counts": dict(dom),
        "weekday_present": wk_present, "workingday_present": wd_present,
        "both_present": both,
        "median_weekday_share": round(sorted(r["weekday_share"] for r in rs)[len(rs)//2], 4),
        "median_workingday_share": round(sorted(r["workingday_share"] for r in rs)[len(rs)//2], 4),
        "stable_partner": (len(dom) == 1 and None not in dom),
    }


def main() -> None:
    jobs = [(gb, s) for gb in (2, 8) for s in SEEDS]
    print(f"E8.3 partner stability: {len(jobs)} runs (2 regimes x 8 seeds)", flush=True)
    wait_until_free()
    n_par = min(len(jobs), 16)
    per = max(2, 64 // n_par)
    rows = Parallel(n_jobs=n_par, backend="loky")(
        delayed(run_one)(gb, s, per) for gb, s in jobs)
    summ = {gb: summarise(rows, gb) for gb in (2, 8)}
    out = {"experiment": "E8.3 hour-interaction partner stability",
           "git_commit": git_commit(), "rows": rows, "summary": summ}
    save_json(ROOT / "experiments" / "results" / "e83.json", out)

    for gb in (2, 8):
        tag = "FROZEN (grow_batch=2)" if gb == 2 else "BATCHED (grow_batch=8)"
        print(f"\n=== {tag} ===", flush=True)
        for r in [x for x in rows if x["grow_batch"] == gb]:
            print(f"  seed {r['seed']}: widths={r['widths']} "
                  f"weekday={r['weekday_share']} workingday={r['workingday_share']} "
                  f"-> {r['dominant_partner']}  | all hour-pairs: {r['all_hour_pairs']}",
                  flush=True)
        s = summ[gb]
        print(f"  SUMMARY: dominant {s['dominant_counts']} | "
              f"weekday present {s['weekday_present']}/8, workingday {s['workingday_present']}/8, "
              f"both {s['both_present']}/8 | stable partner: {s['stable_partner']}", flush=True)


if __name__ == "__main__":
    main()
