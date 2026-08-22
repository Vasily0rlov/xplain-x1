"""Per-unit and per-edge contribution (S-#8).

Unit contribution = ablation drop: zero the unit's outgoing edges (no retraining),
measure the val-fidelity drop.  Edge contribution = |w_eff| * std(parent), row-normalised.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.mlp import MaskedMLP
from ..train.settle import evaluate


@torch.no_grad()
def unit_ablation_drops(model: MaskedMLP, ds: Dataset, splits: Splits,
                        null_stats: dict) -> dict[str, float]:
    base = evaluate(model, ds, splits, splits.val, null_stats)["fidelity"]
    drops: dict[str, float] = {}
    for li in range(len(model.layers)):
        out_mask = (model.mask(li + 1) if li + 1 < len(model.layers)
                    else model.mask_head)
        for i, uid in enumerate(model.unit_ids[li]):
            saved = out_mask[:, i].clone()
            out_mask[:, i] = 0.0
            fid = evaluate(model, ds, splits, splits.val, null_stats)["fidelity"]
            out_mask[:, i] = saved
            drops[uid] = float(base - fid)
    return drops


@torch.no_grad()
def edge_contributions(model: MaskedMLP, parent_acts: list[np.ndarray]
                       ) -> list[np.ndarray]:
    """Per layer: [width, n_parents] normalised contribution matrix.

    parent_acts[li] = activations feeding layer li (standardised X for li=0),
    computed on the train split.
    """
    out = []
    for li, lin in enumerate(model.layers):
        w_eff = (lin.weight * model.mask(li)).abs().numpy()
        p_std = parent_acts[li].std(axis=0)
        c = w_eff * p_std[None, :]
        norm = c.sum(axis=1, keepdims=True)
        out.append(c / np.clip(norm, 1e-12, None))
    return out
