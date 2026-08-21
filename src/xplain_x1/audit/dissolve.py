"""Depth-honesty dissolution test (M-#3.4, S-#8).

Distill layers l, l+1 into one layer (width of l+1) by matching the ORIGINAL
MODEL'S OUTPUTS (function-level distillation; matching pre-activations with a
linear map is impossible — the composed map is nonlinear), then fine-tune the
head briefly on the task; report the val fidelity delta.  A layer is EARNED iff
dissolving costs > eps_depth: theatre depth distils away cheaply, earned depth
resists.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.mlp import MaskedMLP
from ..train.settle import evaluate, null_statistics
from ..util.seeding import torch_seed


def dissolved_candidate(model: MaskedMLP, li: int, ds: Dataset, splits: Splits,
                        seed: int, distill_epochs: int = 300,
                        head_epochs: int = 20) -> MaskedMLP:
    """A candidate model with layers li and li+1 merged into one layer."""
    assert li + 1 < len(model.layers), "need two hidden layers to merge"
    torch.manual_seed(torch_seed(seed, "dissolve", li))
    Xtr = torch.from_numpy(splits.standardise(ds.X[splits.train]))
    with torch.no_grad():
        teacher_out = model(Xtr)

    cand = MaskedMLP(model.d_in,
                     [w for j, w in enumerate(model.widths) if j != li],
                     model.d_out, model.task)
    merged = torch.nn.Linear(
        model.d_in if li == 0 else model.widths[li - 1], model.widths[li + 1])
    with torch.no_grad():
        src_layers = [l for j, l in enumerate(model.layers) if j not in (li, li + 1)]
        src_masks = [model.mask(j) for j in range(len(model.layers))
                     if j not in (li, li + 1)]
        src_ids = [ids for j, ids in enumerate(model.unit_ids) if j not in (li, li + 1)]
        new_layers = src_layers[:li] + [merged] + src_layers[li:]
        new_masks = src_masks[:li] + [torch.ones_like(merged.weight)] + src_masks[li:]
        new_ids = src_ids[:li] + [model.unit_ids[li + 1]] + src_ids[li:]
        for j, (lin, mask, ids) in enumerate(zip(new_layers, new_masks, new_ids)):
            cand.layers[j].weight.copy_(lin.weight)
            cand.layers[j].bias.copy_(lin.bias)
            cand.mask(j).copy_(mask)
            cand.unit_ids[j] = list(ids)
        cand.head.weight.copy_(model.head.weight)
        cand.head.bias.copy_(model.head.bias)
        cand.mask_head.copy_(model.mask_head)
    cand._unit_counter = list(model._unit_counter)

    # function-level distillation: candidate matches the teacher's outputs
    opt = torch.optim.AdamW(cand.parameters(), lr=1e-2)
    for _ in range(distill_epochs):
        opt.zero_grad()
        F.mse_loss(cand(Xtr), teacher_out).backward()
        opt.step()

    # brief head-only fine-tune on the task
    ytr = (torch.from_numpy(ds.y[splits.train]) if ds.task == "classification"
           else torch.from_numpy(ds.y[splits.train]).unsqueeze(1))
    loss_fn = F.cross_entropy if ds.task == "classification" else F.mse_loss
    opt = torch.optim.AdamW(cand.head.parameters(), lr=1e-3)
    for _ in range(head_epochs):
        opt.zero_grad()
        loss_fn(cand(Xtr), ytr).backward()
        opt.step()
    return cand


def dissolution_cost(model: MaskedMLP, li: int, ds: Dataset, splits: Splits,
                     seed: int) -> tuple[float, MaskedMLP]:
    """(val fidelity drop from dissolving layer li into li+1, candidate model)."""
    null_stats = null_statistics(ds, splits)
    base = evaluate(model, ds, splits, splits.val, null_stats)["fidelity"]
    cand = dissolved_candidate(model, li, ds, splits, seed)
    fid = evaluate(cand, ds, splits, splits.val, null_stats)["fidelity"]
    return float(base - fid), cand
