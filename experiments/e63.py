"""E6.3 — canonicalisation probe (Build Plan P6; owner decision "C + probe").

Question: does a tiny DETERMINISTIC per-feature preference (identical across
runs) tip equal-loss carvings the same way and raise UNIT-level stability on
collinear data?  drybean, lambda_pref in {0 (control), 0.01}; report per-arm:
unit-level CORE count, top-concept Pi distribution, route-level CORE/coverage,
fidelity.  No bar — a probe.
Run: nohup .venv/bin/python experiments/e63.py > experiments/results/e63.log 2>&1 &
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xplain_x1.certify.certify import certify              # noqa: E402
from xplain_x1.util.box import wait_until_free             # noqa: E402
from xplain_x1.util.io import save_json                    # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402


def arm(lam_pref: float, cfg: dict) -> dict:
    cfg = {**cfg, "train": {**cfg["train"], "lambda_pref": lam_pref}}
    cert = certify("drybean", cfg)
    main_cs = sorted((c for c in cert["concepts"] if c.get("uid_main")),
                     key=lambda c: -(c.get("delta") or 0))
    return {
        "lambda_pref": lam_pref,
        "fid": cert["main"]["fidelity"], "fid_ref": cert["main"]["fid_ref"],
        "n_core_units": cert["n_core"],
        "top_unit_Pis": [round(c["Pi"], 3) for c in main_cs[:6]],
        "top_unit_pis": [round(c.get("pi", 0), 3) for c in main_cs[:6]],
        "top_unit_supports": [c.get("support_names") for c in main_cs[:4]],
        "n_core_routes": cert["n_core_routes"],
        "route_supports": [r["support_names"] for r in cert["routes"]
                           if r["label"] == "CORE"],
    }


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    wait_until_free(float(cfg["compute"]["load_threshold"]))
    rows = [arm(0.0, cfg), arm(0.01, cfg)]
    out = {"experiment": "E6.3", "git_commit": git_commit(), "arms": rows}
    save_json(ROOT / "experiments" / "results" / "e63.json", out)
    for r in rows:
        print(f"lam_pref {r['lambda_pref']}: units CORE {r['n_core_units']} "
              f"Pis {r['top_unit_Pis']} | routes CORE {r['n_core_routes']} "
              f"fid {r['fid']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
