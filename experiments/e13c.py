"""E1.3c — recalibration after the settle fix (pre-registered).

E1.3/E1.3b found the pressures ineffective; root cause: settle stopped at the
fidelity plateau and restored a best-fid epoch — both blind to the structural
objective, cutting the shaping phase short.  Settle now plateaus on TOTAL train
loss and keeps final weights (best-val restore only as a safety guard).

Sweep: lambda_fanin (Hoyer) in {0.01, 0.03, 0.1, 0.3, 1.0}, lambda_act fixed at
0.01, anneal_frac 0.1, ADD6 fixture, 2 dev seeds, plus OFF control.
Selection rule (pre-registered): smallest lambda_fanin with median ef_live <= 4
AND fid >= fid_OFF - 2% rel.  Run: .venv/bin/python experiments/e13c.py
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
LAMBDA_FANIN = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]   # 0 = OFF control
LAMBDA_ACT = 0.01
WIDTHS = [12, 12]


def one_run(seed: int, lam: float, cfg: dict) -> dict:
    pin_threads(n_workers=len(SEEDS) * len(LAMBDA_FANIN))
    ds = get_dataset(DATASET)
    splits = make_splits(ds, int(cfg["data"]["split_seed"]))
    run_cfg = {**cfg, "train": {**cfg["train"], "anneal_frac": 0.1}}
    model = build_model(ds.d, WIDTHS, ds.task, ds.n_classes, seed=seed)
    p = Pressures(LAMBDA_ACT, lam) if lam > 0 else None
    res = settle(model, ds, splits, run_cfg, seed=seed, pressures=p)
    a = run_audit(model, ds, splits, cfg)
    mu_l1 = [u["mu"] for u in a["units"] if u["layer"] == 1 and u["act_std"] > 1e-6]
    return {"seed": seed, "lambda_fanin": lam, "fidelity": a["fidelity"],
            "median_mu_live": a["median_mu_live"],
            "median_mu_layer1": float(np.median(mu_l1)) if mu_l1 else 1.0,
            "median_ef_live": a["median_ef_live"],
            "n_live": a["n_live"], "n_units": a["n_units"],
            "epochs": res.epochs_run}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    wait_until_free(float(cfg["compute"]["load_threshold"]))
    jobs = [(s, lam) for s in SEEDS for lam in LAMBDA_FANIN]
    runs = Parallel(n_jobs=len(jobs), backend="loky")(
        delayed(one_run)(s, lam, cfg) for s, lam in jobs)

    fid_off = float(np.median([r["fidelity"] for r in runs if r["lambda_fanin"] == 0]))
    table, chosen = [], None
    for lam in LAMBDA_FANIN:
        rs = [r for r in runs if r["lambda_fanin"] == lam]
        row = {"lambda_fanin": lam,
               "fid": float(np.median([r["fidelity"] for r in rs])),
               "mu": float(np.median([r["median_mu_live"] for r in rs])),
               "mu_l1": float(np.median([r["median_mu_layer1"] for r in rs])),
               "ef": float(np.median([r["median_ef_live"] for r in rs])),
               "alive": float(np.mean([r["n_live"] / r["n_units"] for r in rs])),
               "epochs": float(np.median([r["epochs"] for r in rs]))}
        table.append(row)
        if (chosen is None and lam > 0 and row["ef"] <= 4.0
                and row["fid"] >= fid_off - 0.02 * abs(fid_off)):
            chosen = lam

    out = {"experiment": "E1.3c", "dataset": DATASET, "lambda_act": LAMBDA_ACT,
           "rule": "smallest lambda_fanin with ef<=4 and fid within 2% of OFF",
           "git_commit": git_commit(), "runs": runs, "table": table,
           "fid_off": fid_off, "chosen_lambda_fanin": chosen}
    save_json(ROOT / "experiments" / "results" / "e13c.json", out)
    for row in table:
        mark = " <== chosen" if chosen == row["lambda_fanin"] else ""
        print(f"lam_fanin {row['lambda_fanin']:>5}: fid {row['fid']:.3f} "
              f"mu {row['mu']:.3f} mu_L1 {row['mu_l1']:.3f} ef {row['ef']:.1f} "
              f"alive {row['alive']:.0%} ep {row['epochs']:.0f}{mark}")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
