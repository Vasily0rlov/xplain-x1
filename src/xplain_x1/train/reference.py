"""Reference ceiling f_ref (S-#7): unconstrained MLP and HistGradientBoosting;
Fid_ref/Acc_ref = the better of the two on val.  Cached per dataset+split."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.mlp import build_model
from ..util.io import load_json, save_json
from .metrics import classification_fidelity, regression_fidelity
from .settle import evaluate, null_statistics, settle

REF_CACHE = Path(__file__).resolve().parents[3] / "data_cache" / "ref"
REF_SEED = 990001  # reference model seed, outside dev (0-7) and confirmatory (20+) pools


def _hgb_eval(ds: Dataset, splits: Splits, idx: np.ndarray, null_stats: dict) -> dict:
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)

    Xtr, ytr = ds.X[splits.train], ds.y[splits.train]
    Xe = ds.X[idx]
    if ds.task == "classification":
        m = HistGradientBoostingClassifier(random_state=REF_SEED)
        m.fit(Xtr, ytr)
        proba_full = m.predict_proba(Xe)
        proba = np.zeros((len(idx), ds.n_classes))
        proba[:, m.classes_] = proba_full          # classes absent from train -> p=0
        fid = classification_fidelity(proba, ds.y[idx], null_stats["class_prior"])
        acc = float((proba.argmax(1) == ds.y[idx]).mean())
    else:
        m = HistGradientBoostingRegressor(random_state=REF_SEED)
        m.fit(Xtr, ytr)
        pred = m.predict(Xe)
        fid = regression_fidelity(pred, ds.y[idx], null_stats["y_train_mean"])
        acc = None
    return {"fidelity": fid, "accuracy": acc}


def reference_ceiling(ds: Dataset, splits: Splits, cfg: dict,
                      use_cache: bool = True) -> dict:
    cache_key = REF_CACHE / f"{ds.name.replace(':', '_')}-{cfg['data']['split_seed']}.json"
    if use_cache and cache_key.exists():
        return load_json(cache_key)

    null_stats = null_statistics(ds, splits)
    ref_cfg = {"train": dict(cfg["train"])}

    mlp = build_model(ds.d, list(cfg["reference"]["mlp_widths"]), ds.task, ds.n_classes,
                      seed=REF_SEED)
    settle(mlp, ds, splits, ref_cfg, seed=REF_SEED, pressures=None)
    mlp_val = evaluate(mlp, ds, splits, splits.val, null_stats)
    mlp_test = evaluate(mlp, ds, splits, splits.test, null_stats)

    hgb_val = _hgb_eval(ds, splits, splits.val, null_stats)
    hgb_test = _hgb_eval(ds, splits, splits.test, null_stats)

    winner = "mlp" if mlp_val["fidelity"] >= hgb_val["fidelity"] else "hgb"
    result = {
        "dataset": ds.name,
        "mlp": {"val": mlp_val, "test": mlp_test},
        "hgb": {"val": hgb_val, "test": hgb_test},
        "winner": winner,
        "fid_ref_val": max(mlp_val["fidelity"], hgb_val["fidelity"]),
        "fid_ref_test": (mlp_test if winner == "mlp" else hgb_test)["fidelity"],
        "acc_ref_val": max(filter(None.__ne__, [mlp_val["accuracy"], hgb_val["accuracy"]]),
                           default=None),
    }
    save_json(cache_key, result)
    return result
