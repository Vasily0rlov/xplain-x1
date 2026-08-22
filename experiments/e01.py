"""E0.1 — harness soundness (Build Plan P0, milestone M1).

Question: does a plain (pressure-free) settle of the unconstrained MLP reach the
reference ceiling?  Bar: median val fidelity within 2% relative of Fid_ref on
zoo + wine.  Dev seeds 0-3.  Run: .venv/bin/python experiments/e01.py
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
from xplain_x1.data.splits import make_splits              # noqa: E402
from xplain_x1.model.mlp import build_model                # noqa: E402
from xplain_x1.train.reference import reference_ceiling    # noqa: E402
from xplain_x1.train.settle import (evaluate, null_statistics,  # noqa: E402
                                    settle)
from xplain_x1.util.box import pin_threads, wait_until_free  # noqa: E402
from xplain_x1.util.io import save_json                    # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402

DATASETS = ["zoo", "wine"]
SEEDS = [0, 1, 2, 3]
BAR_REL = 0.02


def one_run(name: str, seed: int, cfg: dict) -> dict:
    pin_threads(n_workers=len(DATASETS) * len(SEEDS))
    ds = get_dataset(name)
    splits = make_splits(ds, int(cfg["data"]["split_seed"]))
    null_stats = null_statistics(ds, splits)
    t0 = time.time()
    model = build_model(ds.d, list(cfg["reference"]["mlp_widths"]), ds.task,
                        ds.n_classes, seed=seed)
    res = settle(model, ds, splits, cfg, seed=seed)
    return {"dataset": name, "seed": seed,
            "val": evaluate(model, ds, splits, splits.val, null_stats),
            "test": evaluate(model, ds, splits, splits.test, null_stats),
            "epochs": res.epochs_run, "wall_s": round(time.time() - t0, 1)}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    wait_until_free(float(cfg["compute"]["load_threshold"]))

    refs = {}
    for name in DATASETS:
        ds = get_dataset(name)
        splits = make_splits(ds, int(cfg["data"]["split_seed"]))
        refs[name] = reference_ceiling(ds, splits, cfg)

    jobs = [(n, s) for n in DATASETS for s in SEEDS]
    runs = Parallel(n_jobs=len(jobs), backend="loky")(
        delayed(one_run)(n, s, cfg) for n, s in jobs)

    verdict = {}
    for name in DATASETS:
        fids = sorted(r["val"]["fidelity"] for r in runs if r["dataset"] == name)
        median = fids[len(fids) // 2]
        ref = refs[name]["fid_ref_val"]
        ok = median >= ref - BAR_REL * abs(ref)
        verdict[name] = {"median_val_fid": round(median, 4),
                         "fid_ref_val": round(ref, 4),
                         "ref_winner": refs[name]["winner"], "bar_met": bool(ok)}

    out = {"experiment": "E0.1", "bar": f"median val fid >= Fid_ref - {BAR_REL:.0%} rel",
           "git_commit": git_commit(), "runs": runs, "refs": refs, "verdict": verdict}
    save_json(ROOT / "experiments" / "results" / "e01.json", out)
    print("E0.1 verdict:")
    for name, v in verdict.items():
        print(f"  {name:6s} median {v['median_val_fid']:.3f} vs ref "
              f"{v['fid_ref_val']:.3f} ({v['ref_winner']}) -> "
              f"{'MET' if v['bar_met'] else 'NOT MET'}")
    return 0 if all(v["bar_met"] for v in verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
