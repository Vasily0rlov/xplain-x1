"""Side-by-side Bike comparison: a standard dense NN vs the XPLAIN-x1 method.

Trains a plain 1-hidden-layer MLP (baseline.py), reproduces the x1 grown+audited
model (deterministic seed-0 pipeline), measures BOTH with the same audit
instrument, computes matched performance widgets, and renders one self-contained
comparison dashboard: standard-NN DAG (left) vs x1 DAG (right), with regression
and derived-classification performance widgets underneath.

Run:  .venv/bin/python baselines/standard_nn/compare_bike.py
Out:  baselines/standard_nn/bike_compare.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import predict, train_standard_nn           # noqa: E402
from xplain_x1.audit.audit import run_audit                # noqa: E402
from xplain_x1.data.registry import get_dataset            # noqa: E402
from xplain_x1.data.splits import make_splits              # noqa: E402
from xplain_x1.extract.dag import build_dag                # noqa: E402
from xplain_x1.pipeline import run_pipeline                # noqa: E402

from render_compare import render_comparison               # noqa: E402

DATASET = "bike"
STD_WIDTH = 32
MU_LEGIBLE = 0.8
FANIN_LEGIBLE = 3


def _concepts_from_audit(audit: dict) -> list[dict]:
    """Turn audit units into DAG 'concepts', labelled by per-model legibility
    (monosemantic μ ≥ 0.8 AND effective fan-in ≤ F_max).  No certification here —
    the head-to-head axis is legibility, measured identically for both models."""
    concepts = []
    for u in audit["units"]:
        legible = (u["mu"] >= MU_LEGIBLE and u["effective_fanin"] <= FANIN_LEGIBLE)
        concepts.append({
            "uid_main": u["uid"], "mu": u["mu"], "form": u["form"],
            "support_names": u["support_names"], "contribution": u["contribution"],
            "effective_fanin": u["effective_fanin"],
            "Pi": None, "pi": None, "delta": None,
            "label": "CORE" if legible else "PERIPHERY", "reasons": []})
    return concepts


def _enrich_dag(dag: dict, concepts: list[dict]) -> dict:
    """Attach support_names / form / effective_fanin to each unit node (build_dag
    keeps only a form_text label), so the DAG tooltips read the unit's parents."""
    by_uid = {c["uid_main"]: c for c in concepts}
    for n in dag["nodes"]:
        if n["kind"] != "unit":
            continue
        c = by_uid.get(n["id"])
        if c:
            n["support_names"] = c["support_names"]
            n["form"] = c["form"]
            n["effective_fanin"] = c["effective_fanin"]
    return dag


def _to_counts(pred_scaled: np.ndarray, splits) -> np.ndarray:
    """scaled-y -> log1p(count) -> count."""
    log1p = pred_scaled * splits.y_std + splits.y_mean
    return np.expm1(log1p)


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y_pred - y_true
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"r2": round(1 - ss_res / max(ss_tot, 1e-12), 4),
            "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 2),
            "mae": round(float(np.mean(np.abs(resid))), 2)}


def _roc_block(y_bin: np.ndarray, score: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_bin, score)
    # thin to ~60 points for embedding
    idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
    return {"auc": round(float(roc_auc_score(y_bin, score)), 4),
            "fpr": [round(float(x), 4) for x in fpr[idx]],
            "tpr": [round(float(x), 4) for x in tpr[idx]]}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    ds = get_dataset(DATASET)
    splits = make_splits(ds, int(cfg["data"]["split_seed"]),
                         int(cfg["data"]["probe_size"]),
                         tuple(cfg["data"]["fractions"]))

    print(f"[1/3] training standard dense NN (1 hidden layer, width {STD_WIDTH}) ...",
          flush=True)
    std = train_standard_nn(ds, splits, cfg, width=STD_WIDTH, seed=0)

    print("[2/3] reproducing the XPLAIN-x1 grown+audited model (seed 0) ...",
          flush=True)
    ours = run_pipeline(DATASET, cfg, seed=0)
    ours_audit = run_audit(ours["model"], ds, splits, cfg)
    fid_ref = ours["fid_ref"]

    print("[3/3] measuring, scoring, rendering ...", flush=True)
    std_concepts = _concepts_from_audit(std["audit"])
    ours_concepts = _concepts_from_audit(ours_audit)
    std_dag = _enrich_dag(build_dag(std["model"], ds, splits, std_concepts, cfg),
                          std_concepts)
    ours_dag = _enrich_dag(build_dag(ours["model"], ds, splits, ours_concepts, cfg),
                           ours_concepts)

    # ---- predictions on the held-out TEST set, in count space ---------------
    ytrue = np.expm1(ds.y[splits.test])
    std_pred = _to_counts(predict(std["model"], ds, splits, splits.test), splits)
    ours_pred = _to_counts(predict(ours["model"], ds, splits, splits.test), splits)
    std_pred = np.clip(std_pred, 0, None)
    ours_pred = np.clip(ours_pred, 0, None)

    # derived binary task: "high-demand hour" = count >= test median
    thr = float(np.median(ytrue))
    y_bin = (ytrue >= thr).astype(int)

    rng = np.random.default_rng(0)
    samp = rng.choice(len(ytrue), size=min(700, len(ytrue)), replace=False)

    def model_block(name, subtitle, dag, audit, pred, fid_test):
        live = [u for u in audit["units"] if u["act_std"] > 1e-6]
        legible = [u for u in live
                   if u["mu"] >= MU_LEGIBLE and u["effective_fanin"] <= FANIN_LEGIBLE]
        cm = confusion_matrix(y_bin, (pred >= thr).astype(int)).tolist()
        return {
            "name": name, "subtitle": subtitle,
            "dag": dag,
            "widths": audit["widths"],
            # fidelity = held-out R² in the model's native (log1p) space — the
            # programme's metric, directly comparable to the reference ceiling.
            "fidelity": round(float(fid_test), 4),
            "n_units": len(live),
            "n_legible": len(legible),
            "median_mu": round(float(np.median([u["mu"] for u in live])), 3) if live else None,
            "median_fanin": round(float(np.median([u["effective_fanin"] for u in live])), 2) if live else None,
            "reg": _reg_metrics(ytrue, pred),
            "roc": _roc_block(y_bin, pred),
            "confusion": cm,
            "scatter": [[round(float(ytrue[i]), 1), round(float(pred[i]), 1)] for i in samp],
        }

    data = {
        "dataset": DATASET, "task": ds.task, "n": ds.n, "d": ds.d,
        "target": "count (bike rentals / hour)",
        "fid_ref": round(fid_ref, 4),
        "binary_task": f"high-demand hour (count ≥ median {int(thr)})",
        "pos_rate": round(float(y_bin.mean()), 3),
        "left": model_block(
            "Standard NN", f"plain dense MLP · 1 hidden layer × {STD_WIDTH} · "
            "vanilla MSE · no pressures / growth / certification",
            std_dag, std["audit"], std_pred, std["test"]["fidelity"]),
        "right": model_block(
            "XPLAIN-x1", "pressured + grown + audited · sparse, depth-honest",
            ours_dag, ours_audit, ours_pred, ours["test"]["fidelity"]),
        "test_median": int(thr),
    }

    here = Path(__file__).resolve().parent
    (here / "bike_compare.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    out = here / "bike_compare.html"
    out.write_text(render_comparison(data), encoding="utf-8")
    L, R = data["left"], data["right"]
    print(f"\nStandard NN : R²={L['reg']['r2']} RMSE={L['reg']['rmse']} "
          f"AUC={L['roc']['auc']} | units={L['n_units']} legible={L['n_legible']} "
          f"medμ={L['median_mu']} med-fanin={L['median_fanin']}")
    print(f"XPLAIN-x1   : R²={R['reg']['r2']} RMSE={R['reg']['rmse']} "
          f"AUC={R['roc']['auc']} | units={R['n_units']} legible={R['n_legible']} "
          f"medμ={R['median_mu']} med-fanin={R['median_fanin']}")
    print(f"reference ceiling R² (val) = {fid_ref:.4f}")
    print(f"wrote {out}")


def rerender() -> None:
    here = Path(__file__).resolve().parent
    data = json.loads((here / "bike_compare.json").read_text())
    (here / "bike_compare.html").write_text(render_comparison(data), encoding="utf-8")
    print("re-rendered bike_compare.html from cached bike_compare.json")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--render"]:
        rerender()
    else:
        main()
