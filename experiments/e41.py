"""E4.1 — first real-data end-to-end run: Wine (Build Plan P4, milestone M5).

Full certification pipeline + DAG + certificate on Wine, R=8.
Bar (pre-registered): complete certificate.md + dag.dot; at least one CORE
concept supported on the known Wine discriminative core (flavanoids / proline /
colour intensity / OD280 / alcohol) including flavanoids or proline; Pi of the
core concepts reported against the v4 anchor (Pi = 1.0 on the Wine core, M-#9).
Expert review of nameability happens at P4-4 (owner session).
Run: .venv/bin/python experiments/e41.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xplain_x1.util.io import load_json, save_json         # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402

WINE_CORE = {"flavanoids", "proline", "color_intensity",
             "od280/od315_of_diluted_wines", "alcohol"}
MUST_HIT = {"flavanoids", "proline"}


def main() -> int:
    t0 = time.time()
    r = subprocess.run([str(ROOT / ".venv/bin/xplain-x1"), "certify",
                        "--dataset", "wine"], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1

    run_dir = ROOT / "runs" / "wine" / "certified"
    concepts = load_json(run_dir / "concepts.json")
    cert = load_json(run_dir / "certificate.json")
    dag_ok = (run_dir / "dag.dot").exists() and (run_dir / "certificate.md").exists()

    core = [c for c in concepts if c["label"] == "CORE"]
    def names(c):
        return set(c.get("support_names", []))
    core_on_known = [c for c in core if names(c) & WINE_CORE]
    must_hit = [c for c in core if names(c) & MUST_HIT]
    pis = {c["uid_main"]: c["Pi"] for c in core}

    verdict = {
        "artifacts_complete": bool(dag_ok),
        "n_core": len(core),
        "core_supports": [sorted(names(c)) for c in core],
        "core_on_known_wine_core": len(core_on_known),
        "flavanoids_or_proline_core": len(must_hit) > 0,
        "core_Pi": pis,
        "v4_anchor_Pi": 1.0,
        "fidelity_ratio": cert["performance_and_limits"]["fidelity_ratio"],
        "v4_anchor_fid_ratio": 0.766,
        "bar_met": bool(dag_ok and core and len(must_hit) > 0),
        "wall_s": round(time.time() - t0, 1),
    }
    save_json(ROOT / "experiments" / "results" / "e41.json",
              {"experiment": "E4.1", "git_commit": git_commit(),
               "verdict": verdict, "concepts": concepts})
    print(f"E4.1: core={verdict['n_core']} on-known={verdict['core_on_known_wine_core']} "
          f"flav/proline={verdict['flavanoids_or_proline_core']} "
          f"fid-ratio {verdict['fidelity_ratio']} (v4 0.766) -> "
          f"{'MET' if verdict['bar_met'] else 'NOT MET'} ({verdict['wall_s']}s)")
    return 0 if verdict["bar_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
