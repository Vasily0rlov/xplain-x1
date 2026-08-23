"""E8.2 — bike claims-invariance under batched growth (Build Plan P8, owner (a)).

Full bike certification (R=8 restarts + B=20 CPSS pairs, dev seeds) with
grow_batch=8, compared against the frozen dev reference (e7x):

  CORE components: hour .589 | temp .080 | hour x weekday .051 | year .022
  (all Pi=1.0), coverage 0.743, purified hour x temp NOT certified.

Pre-registered bar:
  all four reference components re-certify CORE at Pi >= 0.875
  AND purified hour x temp stays uncertified
  AND route level keeps an hour-anchored CORE route.
Additions are listed for owner review (not auto-fail).

Run: nohup .venv/bin/python experiments/e82_bike_recert.py \
       > experiments/results/e82.log 2>&1 &
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xplain_x1.certify.certify import certify              # noqa: E402
from xplain_x1.util.box import wait_until_free             # noqa: E402
from xplain_x1.util.io import save_json                    # noqa: E402
from xplain_x1.util.provenance import git_commit           # noqa: E402

REF = {("hour",): 0.589, ("temp",): 0.0804,
       ("hour", "weekday"): 0.0511, ("year",): 0.0222}


def main() -> None:
    cfg = copy.deepcopy(yaml.safe_load((ROOT / "configs" / "default.yaml").read_text()))
    cfg["controller"]["grow_batch"] = 8
    print("E8.2: full bike certification at grow_batch=8", flush=True)
    wait_until_free()
    t0 = time.time()
    cert = certify("bike", cfg)
    wall = time.time() - t0

    comps = cert["components"]
    core = [c for c in comps if c["label"] == "CORE"]
    core_by_names = {tuple(c["support_names"]): c for c in core}

    checks = {}
    for names, ref_share in REF.items():
        c = core_by_names.get(names) or core_by_names.get(tuple(reversed(names)))
        checks[" x ".join(names)] = {
            "recertified": c is not None,
            "Pi": c["Pi"] if c else None, "pi": c["pi"] if c else None,
            "share": round(c["share_main"], 4) if c else None,
            "ref_share": ref_share}
    ref_keys = {frozenset(k) for k in REF}
    additions = [{"names": list(c["support_names"]),
                  "share": round(c["share_main"], 4), "Pi": c["Pi"], "pi": c["pi"]}
                 for c in core if frozenset(c["support_names"]) not in ref_keys]
    ht = next((c for c in comps
               if frozenset(c["support_names"]) == frozenset(("hour", "temp"))), None)
    ht_certified = bool(ht and ht["label"] == "CORE")
    routes_core = [r for r in cert["routes"] if r["label"] == "CORE"]
    hour_route = any("hour" in " ".join(r.get("support_names", []))
                     for r in routes_core)

    bar_met = (all(v["recertified"] and v["Pi"] >= 0.875 for v in checks.values())
               and not ht_certified and hour_route)

    out = {
        "experiment": "E8.2 bike claims-invariance (grow_batch=8)",
        "git_commit": git_commit(), "wall_hours": round(wall / 3600, 2),
        "reference_components": checks,
        "additions_for_owner_review": additions,
        "hour_temp_certified": ht_certified,
        "hour_temp_share": round(ht["share_main"], 4) if ht else 0.0,
        "core_share_sum": round(sum(c["share_main"] for c in core), 4),
        "n_core": len(core),
        "core_routes": [{"support": r.get("support_names"), "Pi": r["Pi"],
                         "pi": r["pi"]} for r in routes_core],
        "hour_route_core": hour_route,
        "restart_widths": [None],   # filled below
        "fid": cert["main"]["fidelity"], "fid_ref": cert["main"]["fid_ref"],
        "ratio": round(cert["main"]["fidelity"] / cert["main"]["fid_ref"], 4),
        "restart_fids": cert["restart_fids"],
        "unit_n_core": cert["n_core"],
        "bar_met": bar_met,
    }
    out["restart_widths"] = cert["main"]["widths"]
    save_json(ROOT / "experiments" / "results" / "e82.json", out)

    print(f"wall {wall/3600:.2f}h · fid {out['fid']:.4f} ratio {out['ratio']}")
    for k, v in checks.items():
        print(f"  ref {k:16s}: recert={v['recertified']} Pi={v['Pi']} "
              f"share={v['share']} (ref {v['ref_share']})")
    print(f"  hour x temp certified: {ht_certified} (share {out['hour_temp_share']})")
    print(f"  additions: {additions}")
    print(f"  coverage: {out['core_share_sum']} (ref 0.743) · core routes "
          f"{len(routes_core)} · hour-anchored: {hour_route}")
    print("E8.2 BAR:", "MET" if bar_met else "NOT MET")


if __name__ == "__main__":
    main()
