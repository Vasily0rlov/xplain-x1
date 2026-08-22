"""Unit signatures for cross-run concept identity (M-#3.6, S-#10).

signature(u) = (a) input-attribution vector — E[grad(a_u) * x] over the fixed
probe set, aggregated to the d input features, L2-normalised; and (b) the unit's
activation vector on the probe set.  The probe set is identical across restarts,
so both components are comparable between independently trained models.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.mlp import MaskedMLP


@dataclass
class UnitSignature:
    uid: str
    layer: int
    attribution: np.ndarray      # [d], L2-normalised
    probe_acts: np.ndarray       # [n_probe]


def unit_signatures(model: MaskedMLP, ds: Dataset, splits: Splits
                    ) -> list[UnitSignature]:
    X = torch.from_numpy(splits.standardise(ds.X[splits.probe]))
    X.requires_grad_(True)
    acts = model.hidden(X)
    sigs: list[UnitSignature] = []
    for li, a in enumerate(acts):
        a_np = a.detach().numpy()
        for i, uid in enumerate(model.unit_ids[li]):
            if a_np[:, i].std() <= 1e-6:
                continue                      # dead units carry no signature
            grad = torch.autograd.grad(a[:, i].sum(), X, retain_graph=True)[0]
            attr = (grad * X).detach().mean(dim=0).numpy()
            norm = float(np.linalg.norm(attr))
            if norm < 1e-12:
                continue
            sigs.append(UnitSignature(uid=uid, layer=li + 1,
                                      attribution=attr / norm,
                                      probe_acts=a_np[:, i]))
    return sigs


def input_support(uid: str, units_by_id: dict,
                  _seen: frozenset[str] = frozenset()) -> frozenset[int]:
    """Recursive input-feature support of a unit from its audit supports.
    Cycle-guarded: ids are globally unique by construction, but a malformed
    audit must degrade, not recurse forever."""
    if uid in _seen:
        return frozenset()
    u = units_by_id[uid]
    if u["layer"] == 1:
        return frozenset(u["support"])
    out: set[int] = set()
    for pname in u["support_names"]:
        if pname in units_by_id:
            out |= input_support(pname, units_by_id, _seen | {uid})
    return frozenset(out)
