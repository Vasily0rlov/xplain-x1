"""One full pipeline execution: reference ceiling -> grow -> final evals.

Used by experiments, restarts (P3), and CPSS subsample runs (which pass a
train-index override).
"""
from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np

from .controller.growth import GrowthTrace, grow
from .data.registry import get_dataset
from .data.splits import Splits, make_splits
from .train.reference import reference_ceiling
from .train.settle import evaluate, null_statistics


def _resolve_controller(cfg: dict):
    """Controller dispatch (x2 enabling hook): the default is the x1 grow
    controller; an alternative is selected by controller.kind != "grow" plus
    controller.entry (a dotted path to a callable with grow's signature).  With
    the default config (no controller.kind, or "grow") this returns `grow`, so
    behaviour is byte-identical to before the hook.
    """
    ccfg = cfg.get("controller", {})
    if ccfg.get("kind", "grow") == "grow":
        return grow
    module_path, _, fn = ccfg["entry"].rpartition(".")
    return getattr(importlib.import_module(module_path), fn)


def run_pipeline(dataset: str, cfg: dict, seed: int,
                 train_override: np.ndarray | None = None,
                 fid_ref: float | None = None) -> dict:
    ds = get_dataset(dataset)
    splits = make_splits(ds, int(cfg["data"]["split_seed"]),
                         int(cfg["data"]["probe_size"]),
                         tuple(cfg["data"]["fractions"]))
    if train_override is not None:
        splits = replace(splits, train=np.sort(np.asarray(train_override)))

    if fid_ref is None:
        ref = reference_ceiling(ds, splits, cfg,
                                use_cache=train_override is None)
        fid_ref = float(ref["fid_ref_val"])
    trace: GrowthTrace = _resolve_controller(cfg)(ds, splits, cfg, seed, fid_ref)

    null_stats = null_statistics(ds, splits)
    val = evaluate(trace.model, ds, splits, splits.val, null_stats)
    test = evaluate(trace.model, ds, splits, splits.test, null_stats)
    return {
        "dataset": dataset, "seed": seed, "fid_ref": fid_ref,
        "widths": trace.model.widths, "rounds": trace.rounds,
        "actions": trace.actions, "audits": trace.audits,
        "final_audit": trace.audits[-1], "val": val, "test": test,
        "model": trace.model, "splits": splits,
    }
