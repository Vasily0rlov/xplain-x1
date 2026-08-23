"""Build owner-review dashboards (Build Plan M6).

For each processed dataset: run the full certification pipeline, extract the
concept DAG + certificate, and render one self-contained HTML dashboard.  Also
writes an index.html linking them.  Follows the xplain-v4 dashboard principle
(single standalone HTML files with an interactive drill-down DAG), rebuilt on
x1's three-layer as-built artefacts.

Run (full battery, background):
  nohup .venv/bin/python experiments/build_dashboards.py \
      > experiments/results/dashboards.log 2>&1 &

Single dataset (fast, for iterating on the renderer):
  .venv/bin/python experiments/build_dashboards.py synthetic:comp2-noisy-8k
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
from xplain_x1.extract.certificate import build_certificate  # noqa: E402
from xplain_x1.extract.dag import build_dag                # noqa: E402
from xplain_x1.extract.dashboard import (render_dashboard,  # noqa: E402
                                         render_index)
from xplain_x1.util.box import wait_until_free             # noqa: E402
from xplain_x1.util.io import save_json                    # noqa: E402

OUT = ROOT / "experiments" / "dashboards"

# Review set: standard-regime real datasets (the certified ones) + small-n real
# (honest-labels regime) + two synthetics as ground-truth sanity references.
REVIEW = [
    "synthetic:comp2-noisy-8k", "synthetic:comp3-noisy-8k",
    "mushroom", "drybean", "bike", "adult",
    "wine", "zoo", "tictactoe",
]


def _enrich_dag(dag: dict, concepts: list[dict]) -> dict:
    """Attach support_names + form to each unit node from its concept row."""
    by_uid = {c.get("uid_main"): c for c in concepts if c.get("uid_main")}
    for n in dag["nodes"]:
        if n["kind"] != "unit":
            continue
        c = by_uid.get(n["id"])
        if c:
            n["support_names"] = c.get("support_names", [])
            n["form"] = c.get("form")
    return dag


def _bars(cert_doc: dict, core_share_sum: float) -> list[dict]:
    perf = cert_doc["performance_and_limits"]
    fdec = cert_doc["function_decomposition"]
    stat = cert_doc["statistical_certification"]
    struct = cert_doc["interpretability_structure"]
    ratio = perf.get("fidelity_ratio") or 0
    ev = fdec.get("ev_bound")
    depth = struct.get("honest_depth", {})
    earned = depth.get("earned_depth", depth.get("depth", len(struct.get("widths", []))))
    rows = [
        {"id": "fidelity ≥ ceiling", "pass": ratio >= 0.98,
         "detail": f"ratio {ratio:.3f} vs unconstrained MLP/HGB"},
        {"id": "core coverage ≥ 0.5", "pass": core_share_sum >= 0.5,
         "detail": f"certified components cover {core_share_sum:.2f} of function variance"},
        {"id": "E[V] non-vacuous", "pass": (ev is not None and ev < 1.0),
         "detail": f"E[V] ≤ {ev:.3f} over the structure-level universe"
                   if ev is not None else "—"},
        {"id": "honest depth", "pass": True,
         "detail": f"depth {earned}; widths {'×'.join(map(str, struct.get('widths', [])))}"},
    ]
    return rows


def build_one(name: str, cfg: dict, suffix: str = "") -> dict:
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] certifying {name} ...", flush=True)
    cert = certify(name, cfg)
    ds = get_dataset(name)
    cert_doc = build_certificate(cert, ds, cert["splits"], cfg, seed=0)
    dag = _enrich_dag(build_dag(cert["model"], ds, cert["splits"],
                                cert["concepts"], cfg), cert["concepts"])
    core_share_sum = round(sum(
        (c.get("share_main", 0) or 0) for c in cert["components"]
        if c.get("label") == "CORE"), 4)

    data = {
        "dag": dag,
        "cert": cert_doc,
        "core_share_sum": core_share_sum,
        "bars": _bars(cert_doc, core_share_sum),
    }
    safe = name.replace(":", "_") + suffix
    if suffix:   # variant runs carry their tag into the page identity
        cert_doc["identification"]["dataset"] = name + suffix
    save_json(OUT / f"{safe}.json", data)
    (OUT / f"{safe}.html").write_text(render_dashboard(data), encoding="utf-8")
    perf = cert_doc["performance_and_limits"]
    meta = {
        "dataset": name + suffix, "task": ds.task, "n": ds.n, "d": ds.d,
        "fidelity": perf.get("fidelity_val", 0),
        "ratio": perf.get("fidelity_ratio"),
        "n_core": cert_doc["function_decomposition"].get("n_core", 0),
        "coverage": core_share_sum,
        "regime": cert_doc["statistical_certification"].get("regime", "standard"),
        "file": f"{safe}.html",
    }
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] {name}: fid {meta['fidelity']:.3f} "
          f"ratio {meta['ratio']} core {meta['n_core']} cov {core_share_sum:.2f} "
          f"({dt:.0f}s)", flush=True)
    return meta


def rerender_from_json() -> list[dict]:
    """Regenerate every HTML (and the index) from saved <ds>.json — no compute.

    Used after changing the renderer: the certification results are the source
    of truth, so we never re-certify just to restyle.
    """
    import json
    metas = []
    for jf in sorted(OUT.glob("*.json")):
        if jf.name.startswith("_"):
            continue
        data = json.loads(jf.read_text())
        cert_doc = data["cert"]
        ident = cert_doc["identification"]
        perf = cert_doc["performance_and_limits"]
        (OUT / f"{jf.stem}.html").write_text(render_dashboard(data), encoding="utf-8")
        metas.append({
            "dataset": ident["dataset"], "task": ident["task"],
            "n": ident["n"], "d": ident["d"],
            "fidelity": perf.get("fidelity_val", 0),
            "ratio": perf.get("fidelity_ratio"),
            "n_core": cert_doc["function_decomposition"].get("n_core", 0),
            "coverage": data.get("core_share_sum"),
            "regime": cert_doc["statistical_certification"].get("regime", "standard"),
            "file": f"{jf.stem}.html"})
        print(f"re-rendered {jf.stem}.html", flush=True)
    save_json(OUT / "_index.json", metas)
    (OUT / "index.html").write_text(render_index(metas), encoding="utf-8")
    print(f"index.html written with {len(metas)} dashboards", flush=True)
    return metas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:2] == ["--rerender"]:
        rerender_from_json()
        return
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    args = sys.argv[1:]
    suffix = ""
    if "--suffix" in args:
        i = args.index("--suffix")
        suffix = args[i + 1]
        del args[i:i + 2]
    if "--grow-batch" in args:
        i = args.index("--grow-batch")
        cfg["controller"]["grow_batch"] = int(args[i + 1])
        del args[i:i + 2]
        print(f"grow_batch override: {cfg['controller']['grow_batch']}", flush=True)
    targets = args or REVIEW
    single = bool(args)
    if not single:
        print("Box-capacity gate (CLAUDE.md §3): waiting until free ...", flush=True)
        wait_until_free()
    metas = []
    for name in targets:
        try:
            metas.append(build_one(name, cfg, suffix))
        except Exception as e:  # keep going; one dataset failing must not sink the run
            print(f"!! {name} FAILED: {type(e).__name__}: {e}", flush=True)
    if metas:
        # merge with any previously-built metas so a single-dataset rebuild keeps the index whole
        existing = {}
        idx_json = OUT / "_index.json"
        if idx_json.exists():
            import json
            for m in json.loads(idx_json.read_text()):
                existing[m["dataset"]] = m
        for m in metas:
            existing[m["dataset"]] = m
        allmeta = list(existing.values())
        save_json(idx_json, allmeta)
        (OUT / "index.html").write_text(render_index(allmeta), encoding="utf-8")
        print(f"index.html written with {len(allmeta)} dashboards -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
