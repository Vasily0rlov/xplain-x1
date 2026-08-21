"""Exact gauge canonicalisation (M-#3.3, #4.2): permutation + positive scaling only.

Function-preserving for ReLU by positive homogeneity; asserted to ~1e-5 on a
probe batch after every pass.  Makes signatures, matching, and diffs across
runs well-defined.
"""
from __future__ import annotations

import torch

from .mlp import MaskedMLP


@torch.no_grad()
def gauge_pass(model: MaskedMLP, probe_x: torch.Tensor, atol: float = 1e-5) -> None:
    before = model(probe_x).clone()
    acts = model.hidden(probe_x)

    for li, lin in enumerate(model.layers):
        # 1) scale-normalise: masked incoming row norm -> 1; absorb into outgoing
        w_eff = lin.weight * model.mask(li)
        norms = w_eff.norm(dim=1).clamp_min(1e-12)
        lin.weight.div_(norms.unsqueeze(1))
        lin.bias.div_(norms)
        out_lin = model.layers[li + 1] if li + 1 < len(model.layers) else model.head
        out_lin.weight.mul_(norms.unsqueeze(0))

        # 2) permutation-sort by contribution proxy: outgoing norm x activation std
        out_mask = (model.mask(li + 1) if li + 1 < len(model.layers)
                    else model.mask_head)
        contrib = ((out_lin.weight * out_mask).norm(dim=0)
                   * (acts[li].std(dim=0) / norms).clamp_min(0))
        order = torch.argsort(contrib, descending=True)

        lin.weight.copy_(lin.weight[order])
        lin.bias.copy_(lin.bias[order])
        model.mask(li).copy_(model.mask(li)[order])
        out_lin.weight.copy_(out_lin.weight[:, order])
        out_mask.copy_(out_mask[:, order])
        model.unit_ids[li] = [model.unit_ids[li][i] for i in order.tolist()]

    after = model(probe_x)
    max_diff = (before - after).abs().max().item()
    assert max_diff < atol, f"gauge pass changed outputs by {max_diff:.2e}"
