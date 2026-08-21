"""E1.2 — pinned-lambda sanity (Build Plan P1).

ADD + COMP2 (noisy, 8k), fixed 2x12, pressures at pinned defaults, 3 settle
rounds with audits between.  Bar: >= 90% units alive at the end; median live mu
trend upward (last > first).  Failing here triggers the S-#14 alternative path
with owner sign-off.  Run: .venv/bin/python experiments/e12.py
"""
from __future__ import annotations

import sys
from pathlib import Path

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

DATASETS = ["synthetic:add-noisy-8k", "synthetic:comp2-noisy-8k"]
ROUNDS = 3
SEED = 0


def one_dataset(name: str, cfg: dict) -> dict:
    pin_threads(n_workers=len(DATASETS))
    ds = get_dataset(name)
    splits = make_splits(ds, int(cfg["data"]["split_seed"]))
    model = build_model(ds.d, [12, 12], ds.task, ds.n_classes, seed=SEED)
    pressures = make_pressures(cfg)
    round_cfg = {**cfg, "train": {**cfg["train"], "max_epochs": 120}}
    audits = []
    for r in range(ROUNDS):
        settle(model, ds, splits, round_cfg, seed=SEED, pressures=pressures)
        a = run_audit(model, ds, splits, cfg)
        audits.append({"round": r, "fidelity": a["fidelity"],
                       "median_mu_live": a["median_mu_live"],
                       "n_live": a["n_live"], "n_units": a["n_units"]})
    alive_frac = audits[-1]["n_live"] / audits[-1]["n_units"]
    mu_up = audits[-1]["median_mu_live"] > audits[0]["median_mu_live"] - 1e-6
    return {"dataset": name, "audits": audits,
            "alive_frac": round(alive_frac, 3), "mu_trend_up": bool(mu_up),
            "bar_met": bool(alive_frac >= 0.9 and mu_up)}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    wait_until_free(float(cfg["compute"]["load_threshold"]))
    results = Parallel(n_jobs=len(DATASETS), backend="loky")(
        delayed(one_dataset)(n, cfg) for n in DATASETS)
    out = {"experiment": "E1.2", "bar": ">=90% units alive; mu trend upward",
           "git_commit": git_commit(), "results": results,
           "bar_met": all(r["bar_met"] for r in results)}
    save_json(ROOT / "experiments" / "results" / "e12.json", out)
    for r in results:
        mus = [a["median_mu_live"] for a in r["audits"]]
        print(f"E1.2 {r['dataset']}: alive {r['alive_frac']:.0%}, "
              f"mu {mus[0]:.3f}->{mus[-1]:.3f} -> "
              f"{'MET' if r['bar_met'] else 'NOT MET'}")
    return 0 if out["bar_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
