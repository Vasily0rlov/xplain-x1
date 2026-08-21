"""E1.1 — do the pressures work? (Build Plan P1, milestone M2)

COMP2 (noisy, 8k), fixed 2x12 architecture, pressures ON vs OFF, 4 dev seeds each.
Bar (pre-registered): median live mu (ON) - (OFF) >= +0.10, and the ON fidelity
within 2% relative of OFF.  Run: .venv/bin/python experiments/e11.py
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
from xplain_x1.train.losses import make_pressures          # noqa: E402
from xplain_x1.train.settle import settle                  # noqa: E402
from xplain_x1.util.box import pin_threads, wait_until_free  # noqa: E402
from xplain_x1.util.io import save_json                    # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402

DATASET = "synthetic:comp2-noisy-8k"
SEEDS = [0, 1, 2, 3]
WIDTHS = [12, 12]
BAR_MU_GAIN = 0.10
BAR_FID_REL = 0.02


def one_run(seed: int, pressures_on: bool, cfg: dict) -> dict:
    pin_threads(n_workers=8)
    ds = get_dataset(DATASET)
    splits = make_splits(ds, int(cfg["data"]["split_seed"]))
    model = build_model(ds.d, WIDTHS, ds.task, ds.n_classes, seed=seed)
    p = make_pressures(cfg) if pressures_on else None
    settle(model, ds, splits, cfg, seed=seed, pressures=p)
    audit = run_audit(model, ds, splits, cfg)
    return {"seed": seed, "pressures": pressures_on,
            "fidelity": audit["fidelity"], "median_mu_live": audit["median_mu_live"],
            "median_ef_live": audit["median_ef_live"], "n_live": audit["n_live"]}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    wait_until_free(float(cfg["compute"]["load_threshold"]))
    jobs = [(s, on) for s in SEEDS for on in (True, False)]
    runs = Parallel(n_jobs=len(jobs), backend="loky")(
        delayed(one_run)(s, on, cfg) for s, on in jobs)

    med = lambda key, on: float(np.median(  # noqa: E731
        [r[key] for r in runs if r["pressures"] == on]))
    mu_on, mu_off = med("median_mu_live", True), med("median_mu_live", False)
    fid_on, fid_off = med("fidelity", True), med("fidelity", False)
    mu_ok = (mu_on - mu_off) >= BAR_MU_GAIN
    fid_ok = fid_on >= fid_off - BAR_FID_REL * abs(fid_off)

    out = {"experiment": "E1.1", "dataset": DATASET, "widths": WIDTHS,
           "bar": f"mu gain >= {BAR_MU_GAIN}, fid within {BAR_FID_REL:.0%}",
           "git_commit": git_commit(), "runs": runs,
           "verdict": {"mu_on": round(mu_on, 4), "mu_off": round(mu_off, 4),
                       "mu_gain": round(mu_on - mu_off, 4),
                       "fid_on": round(fid_on, 4), "fid_off": round(fid_off, 4),
                       "bar_met": bool(mu_ok and fid_ok)}}
    save_json(ROOT / "experiments" / "results" / "e11.json", out)
    v = out["verdict"]
    print(f"E1.1: mu {v['mu_off']:.3f} -> {v['mu_on']:.3f} (gain {v['mu_gain']:+.3f}), "
          f"fid {v['fid_off']:.3f} -> {v['fid_on']:.3f} -> "
          f"{'MET' if v['bar_met'] else 'NOT MET'}")
    return 0 if v["bar_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
