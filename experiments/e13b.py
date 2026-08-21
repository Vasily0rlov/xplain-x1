"""E1.3b — recalibration with the Hoyer fan-in term (S-#14 alternative invoked after E1.3).

E1.1 found lambda=1e-3 pressures produce NO measurable effect (ef ~8 ON and OFF)
and that COMP2 saturates the mu metric (intrinsic arity <= F_max).  This sweep
calibrates on the discriminating ADD6 fixture (6 relevant inputs > F_max).

Grid: joint scale in {1, 3, 10, 30, 100} x base (act 1e-3, fanin 1e-3), plus OFF
control; 2 dev seeds; anneal_frac 0.1.  Selection rule (pre-registered): the
smallest scale with median ef_live <= 4 AND fid >= fid_OFF - 2% rel.
Run: .venv/bin/python experiments/e13.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xplain_x1.audit.audit import run_audit                # noqa: E402
from xplain_x1.data.registry import get_dataset            # noqa: E402
from xplain_x1.data.splits import make_splits              # noqa: E402
from xplain_x1.model.mlp import build_model                # noqa: E402
from xplain_x1.train.losses import Pressures               # noqa: E402
from xplain_x1.train.settle import settle                  # noqa: E402
from xplain_x1.util.box import pin_threads, wait_until_free  # noqa: E402
from xplain_x1.util.io import save_json                    # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402

DATASET = "synthetic:add6-noisy-8k"
SEEDS = [0, 1]
SCALES = [0, 1, 3, 10, 30, 100]        # 0 = OFF control
BASE_ACT, BASE_FANIN = 1e-3, 1e-3
WIDTHS = [12, 12]


def one_run(seed: int, scale: int, cfg: dict) -> dict:
    pin_threads(n_workers=len(SEEDS) * len(SCALES))
    ds = get_dataset(DATASET)
    splits = make_splits(ds, int(cfg["data"]["split_seed"]))
    run_cfg = {**cfg, "train": {**cfg["train"], "anneal_frac": 0.1}}
    model = build_model(ds.d, WIDTHS, ds.task, ds.n_classes, seed=seed)
    p = (Pressures(BASE_ACT * scale, BASE_FANIN * scale) if scale > 0 else None)
    settle(model, ds, splits, run_cfg, seed=seed, pressures=p)
    a = run_audit(model, ds, splits, cfg)
    return {"seed": seed, "scale": scale, "fidelity": a["fidelity"],
            "median_mu_live": a["median_mu_live"],
            "median_ef_live": a["median_ef_live"],
            "n_live": a["n_live"], "n_units": a["n_units"]}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    wait_until_free(float(cfg["compute"]["load_threshold"]))
    jobs = [(s, sc) for s in SEEDS for sc in SCALES]
    runs = Parallel(n_jobs=len(jobs), backend="loky")(
        delayed(one_run)(s, sc, cfg) for s, sc in jobs)

    fid_off = float(np.median([r["fidelity"] for r in runs if r["scale"] == 0]))
    table, chosen = [], None
    for sc in SCALES:
        rs = [r for r in runs if r["scale"] == sc]
        row = {"scale": sc,
               "fid": float(np.median([r["fidelity"] for r in rs])),
               "mu": float(np.median([r["median_mu_live"] for r in rs])),
               "ef": float(np.median([r["median_ef_live"] for r in rs])),
               "alive": float(np.mean([r["n_live"] / r["n_units"] for r in rs]))}
        table.append(row)
        if (chosen is None and sc > 0 and row["ef"] <= 4.0
                and row["fid"] >= fid_off - 0.02 * abs(fid_off)):
            chosen = sc

    out = {"experiment": "E1.3b", "dataset": DATASET,
           "rule": "smallest scale with ef<=4 and fid within 2% of OFF",
           "git_commit": git_commit(), "runs": runs, "table": table,
           "fid_off": fid_off, "chosen_scale": chosen,
           "chosen_lambdas": ({"lambda_act": BASE_ACT * chosen,
                               "lambda_fanin": BASE_FANIN * chosen}
                              if chosen else None)}
    save_json(ROOT / "experiments" / "results" / "e13b.json", out)
    for row in table:
        mark = " <== chosen" if chosen == row["scale"] else ""
        print(f"scale {row['scale']:>3}: fid {row['fid']:.3f} mu {row['mu']:.3f} "
              f"ef {row['ef']:.1f} alive {row['alive']:.0%}{mark}")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
