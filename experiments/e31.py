"""E3.1 — certified recovery on known ground truth (Build Plan P3, milestone M4).

Full pinned 6-config synthetic suite through the complete certification pipeline
(R=8 restarts + 2B=40 CPSS runs each).  Bars (pre-registered):
  - every planted concept (COMP2 x3 regimes, COMP3) above the power floor is
    recovered as a CORE concept whose modal input-support equals the planted
    support, with Pi >= 2/3;  the noisy-2k regime sits AT the order-2 power
    floor — recovery there is reported, not barred;
  - NOISE certifies ZERO concepts (untradeable, K2);
  - ADD certifies only order-1 supports (no invented interactions).
Run: .venv/bin/python experiments/e31.py
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

SUITE = ["synthetic:add-noisy-8k", "synthetic:comp2-clean-8k",
         "synthetic:comp2-noisy-8k", "synthetic:comp2-noisy-2k",
         "synthetic:comp3-noisy-8k", "synthetic:noise-8k"]
AT_FLOOR = {"synthetic:comp2-noisy-2k"}


def evaluate_dataset(name: str, cfg: dict) -> dict:
    t0 = time.time()
    cert = certify(name, cfg)
    ds = get_dataset(name)
    planted = [frozenset(c["support"]) for c in ds.ground_truth["concepts"]
               if c.get("planted")]
    core = [c for c in cert["concepts"] if c["label"] == "CORE"]
    core_supports = {frozenset(c["modal_support"]) for c in core}

    recovered = [sorted(p) for p in planted if p in core_supports
                 and next(c for c in core
                          if frozenset(c["modal_support"]) == p)["Pi"] >= 2 / 3]
    max_core_order = max((len(c["modal_support"]) for c in core), default=0)
    row = {
        "dataset": name, "n_core": cert["n_core"],
        "n_concepts": len(cert["concepts"]),
        "planted": [sorted(p) for p in planted],
        "recovered": recovered,
        "core_supports": [sorted(s) for s in core_supports],
        "max_core_order": max_core_order,
        "fid": cert["main"]["fidelity"], "fid_ref": cert["main"]["fid_ref"],
        "widths": cert["main"]["widths"], "ev_bound": cert["ev_bound"],
        "restart_fids": cert["restart_fids"],
        "wall_s": round(time.time() - t0, 1),
    }
    if name == "synthetic:noise-8k":
        row["bar_met"] = cert["n_core"] == 0
    elif name == "synthetic:add-noisy-8k":
        row["bar_met"] = max_core_order <= 1
    elif name in AT_FLOOR:
        row["bar_met"] = None            # reported, not barred
    else:
        row["bar_met"] = len(recovered) == len(planted)
    return row


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    wait_until_free(float(cfg["compute"]["load_threshold"]))
    rows = []
    for name in SUITE:
        row = evaluate_dataset(name, cfg)
        rows.append(row)
        print(f"{name}: core {row['n_core']}, planted {row['planted']} -> "
              f"recovered {row['recovered']}, bar {row['bar_met']}, "
              f"{row['wall_s']}s")
        save_json(ROOT / "experiments" / "results" / "e31.json",
                  {"experiment": "E3.1", "git_commit": git_commit(),
                   "partial": name != SUITE[-1], "rows": rows})
    barred = [r for r in rows if r["bar_met"] is not None]
    ok = all(r["bar_met"] for r in barred)
    print(f"E3.1 verdict: {'MET' if ok else 'NOT MET'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
