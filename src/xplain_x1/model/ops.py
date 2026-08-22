"""Structural growth/prune ops (S-#6).  Functional style: each op returns a NEW
MaskedMLP with weights copied and the unit registry carried over — optimiser
state is rebuilt after any accepted structural action.

Invariants: persistent unit ids never reused; masks carried; where an op claims
fidelity preservation (insert_layer on post-ReLU input) it is exact.
"""
from __future__ import annotations

import torch

from ..util.seeding import torch_seed
from .mlp import MaskedMLP


def _clone_into(model: MaskedMLP, widths: list[int]) -> MaskedMLP:
    m = MaskedMLP(model.d_in, widths, model.d_out, model.task)
    m._unit_counter = max(model._unit_counter, m._unit_counter)
    return m


def _copy_layer(dst: MaskedMLP, dj: int, src: MaskedMLP, sj: int) -> None:
    with torch.no_grad():
        dst.layers[dj].weight.copy_(src.layers[sj].weight)
        dst.layers[dj].bias.copy_(src.layers[sj].bias)
        dst.mask(dj).copy_(src.mask(sj))
    dst.unit_ids[dj] = list(src.unit_ids[sj])


def _copy_head(dst: MaskedMLP, src: MaskedMLP) -> None:
    with torch.no_grad():
        dst.head.weight.copy_(src.head.weight)
        dst.head.bias.copy_(src.head.bias)
        dst.mask_head.copy_(src.mask_head)


def add_unit(model: MaskedMLP, li: int, mode: str, seed: int) -> MaskedMLP:
    """mode: 'fresh' or 'split:<uid>' (S-#6: clone + 0.05-scaled noise, halve outgoing)."""
    torch.manual_seed(torch_seed(seed, "add_unit", li, mode))
    widths = model.widths
    widths[li] += 1
    new = _clone_into(model, widths)
    for j in range(len(model.layers)):
        if j not in (li, li + 1):        # li+1's input dim changed: handled below
            _copy_layer(new, j, model, j)
    if li + 1 < len(model.layers):       # head shape changes only when li is last
        _copy_head(new, model)
        new.unit_ids[li + 1] = list(model.unit_ids[li + 1])

    lin_old, lin_new = model.layers[li], new.layers[li]
    out_old = model.layers[li + 1] if li + 1 < len(model.layers) else model.head
    out_new = new.layers[li + 1] if li + 1 < len(new.layers) else new.head
    out_mask_old = (model.mask(li + 1) if li + 1 < len(model.layers)
                    else model.mask_head)
    out_mask_new = (new.mask(li + 1) if li + 1 < len(new.layers) else new.mask_head)
    w = lin_old.weight.shape[0]
    with torch.no_grad():
        lin_new.weight[:w].copy_(lin_old.weight)
        lin_new.bias[:w].copy_(lin_old.bias)
        new.mask(li)[:w].copy_(model.mask(li))
        new.mask(li)[w:] = 1.0
        out_new.weight[:, :w].copy_(out_old.weight)
        if li + 1 < len(model.layers):
            out_new.bias.copy_(out_old.bias)
        else:
            new.head.bias.copy_(model.head.bias)
        out_mask_new[:, :w].copy_(out_mask_old)
        out_mask_new[:, w:] = 1.0

        if mode.startswith("split:"):
            uid = mode.split(":", 1)[1]
            src = model.unit_ids[li].index(uid)
            noise = 0.05 * lin_old.weight[src].std().clamp_min(1e-3)
            lin_new.weight[w].copy_(lin_old.weight[src]
                                    + noise * torch.randn_like(lin_old.weight[src]))
            lin_new.bias[w] = lin_old.bias[src]
            new.mask(li)[w].copy_(model.mask(li)[src])
            out_new.weight[:, src].mul_(0.5)               # halve both clones
            out_new.weight[:, w].copy_(out_new.weight[:, src])
            out_mask_new[:, w].copy_(out_mask_old[:, src])
        else:
            torch.nn.init.normal_(lin_new.weight[w:w + 1], std=0.1)
            lin_new.bias[w] = 0.0
            torch.nn.init.normal_(out_new.weight[:, w:w + 1], std=0.01)

    new.unit_ids[li] = list(model.unit_ids[li]) + [new._new_id(li)]
    return new


def insert_layer(model: MaskedMLP, pos: int) -> MaskedMLP:
    """Insert a near-identity layer at position pos >= 1 (its input is post-ReLU,
    non-negative, so ReLU(I x) = x — exactly fidelity-preserving at insert time)."""
    assert pos >= 1, "insert only after a hidden layer (identity needs ReLU-safe input)"
    d = model.widths[pos - 1]
    widths = model.widths
    widths.insert(pos, d)
    new = _clone_into(model, widths)
    for j in range(len(model.layers)):
        _copy_layer(new, j if j < pos else j + 1, model, j)
    _copy_head(new, model)
    with torch.no_grad():
        new.layers[pos].weight.copy_(torch.eye(d))
        new.layers[pos].bias.zero_()
        new.mask(pos).copy_(torch.ones(d, d))
    new.unit_ids[pos] = [new._new_id(pos) for _ in range(d)]
    return new


def remove_units(model: MaskedMLP, uids: set[str]) -> MaskedMLP:
    """Drop units (row + outgoing column).  Refuses to empty a layer."""
    keep = [[i for i, u in enumerate(ids) if u not in uids]
            for ids in model.unit_ids]
    assert all(k for k in keep), "cannot remove all units of a layer"
    widths = [len(k) for k in keep]
    new = _clone_into(model, widths)
    with torch.no_grad():
        for j in range(len(model.layers)):
            rows = torch.tensor(keep[j])
            cols = (torch.tensor(keep[j - 1]) if j > 0
                    else torch.arange(model.d_in))
            new.layers[j].weight.copy_(model.layers[j].weight[rows][:, cols])
            new.layers[j].bias.copy_(model.layers[j].bias[rows])
            new.mask(j).copy_(model.mask(j)[rows][:, cols])
            new.unit_ids[j] = [model.unit_ids[j][i] for i in keep[j]]
        cols = torch.tensor(keep[-1])
        new.head.weight.copy_(model.head.weight[:, cols])
        new.head.bias.copy_(model.head.bias)
        new.mask_head.copy_(model.mask_head[:, cols])
    return new


def merge_units(model: MaskedMLP, keep_uid: str, drop_uid: str) -> MaskedMLP:
    """Sum outgoing weights into keep, then drop (S-#6): exact when activations
    equal.  MUST NOT mutate the input model — merge is used as a trial by the
    budgeted prune, and an in-place add corrupted the base model on every
    REJECTED merge (bike: dozens of rejected trials drove fid 0.93 -> -14)."""
    li = next(j for j, ids in enumerate(model.unit_ids) if keep_uid in ids)
    assert drop_uid in model.unit_ids[li], "merge partners must share a layer"
    di = model.unit_ids[li].index(drop_uid)
    out_old = model.layers[li + 1] if li + 1 < len(model.layers) else model.head
    drop_col = out_old.weight[:, di].detach().clone()
    new = remove_units(model, {drop_uid})
    ki_new = new.unit_ids[li].index(keep_uid)
    out_new = new.layers[li + 1] if li + 1 < len(new.layers) else new.head
    with torch.no_grad():
        out_new.weight[:, ki_new].add_(drop_col)
    return new


def prune_edges(model: MaskedMLP, edge_contribs: list, eps_edge: float) -> int:
    """Mask edges below the contribution threshold, in place.  Returns #edges cut."""
    cut = 0
    with torch.no_grad():
        for li in range(len(model.layers)):
            keep = torch.from_numpy(edge_contribs[li] >= eps_edge).float()
            newly = int((model.mask(li) * (1 - keep)).sum().item())
            model.mask(li).mul_(keep)
            cut += newly
    return cut
