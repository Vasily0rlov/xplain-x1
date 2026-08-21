"""Convergence pressures (M-#4.1, S-#7; fan-in term revised per E1.3).

L = L_task + lambda_act * mean(|a| / std_ema(a))
           + lambda_fanin * mean_u ( ||W_in[u]*mask||_1 / ||W_in[u]*mask||_2 )

- L1 on std-normalised activations: selective firing; the running std (detached)
  stops the scale gauge from cheating the penalty.
- Fan-in term: HOYER RATIO per incoming row — scale-invariant concentration
  pressure (1 = one-hot row, sqrt(d) = uniform).  The originally pinned row-wise
  group lasso (L2,1) penalises whole-row norms, i.e. *unit* sparsity: E1.3
  measured it killing units at high lambda while leaving fan-in untouched (ef ~8
  at every scale).  That is the S-#14 trigger; the Hoyer alternative is the
  pre-registered replacement — it concentrates rows without shrinking them.
- Deliberately absent: any decorrelation term (M-C3, measured dead end).
Both terms multiply the anneal ramp passed in by the settle loop.
"""
from __future__ import annotations

import torch

from ..model.mlp import MaskedMLP


class Pressures:
    def __init__(self, lambda_act: float, lambda_fanin: float, ema: float = 0.99):
        self.lambda_act = float(lambda_act)
        self.lambda_fanin = float(lambda_fanin)
        self.ema = ema
        # Discovery/cleanup gate (M-C4 ordering: accuracy first, minimality on
        # top): the controller sets scale < 1 while fidelity is below ceiling so
        # fresh units can find weak high-order structure before being squeezed
        # (E2.x finding: full pressure strangles order-3 discovery).
        self.scale = 1.0
        self._std: dict[str, torch.Tensor] = {}   # keyed by unit id: survives growth

    def _update_std(self, model: MaskedMLP, acts: list[torch.Tensor]) -> list[torch.Tensor]:
        stds = []
        for li, a in enumerate(acts):
            batch_std = a.detach().std(dim=0).clamp_min(1e-3)
            ids = model.unit_ids[li]
            cur = torch.stack([
                self._std.get(uid, batch_std[i]) for i, uid in enumerate(ids)])
            new = self.ema * cur + (1 - self.ema) * batch_std
            for i, uid in enumerate(ids):
                self._std[uid] = new[i]
            stds.append(new)
        return stds

    def __call__(self, model: MaskedMLP, acts: list[torch.Tensor],
                 ramp: float) -> torch.Tensor:
        stds = self._update_std(model, acts)
        act_pen = sum((a.abs() / s.unsqueeze(0)).mean() for a, s in zip(acts, stds))
        act_pen = act_pen / max(1, len(acts))
        hoyer_terms = []
        for li, lin in enumerate(model.layers):
            w = lin.weight * model.mask(li)
            l1 = w.abs().sum(dim=1)
            l2 = w.norm(dim=1)
            live = l2 > 1e-8                      # dead rows: no revival pressure
            if live.any():
                hoyer_terms.append((l1[live] / l2[live]).mean())
        fanin_pen = (torch.stack(hoyer_terms).mean() if hoyer_terms
                     else torch.tensor(0.0))
        return (ramp * self.scale
                * (self.lambda_act * act_pen + self.lambda_fanin * fanin_pen))


def make_pressures(cfg: dict) -> Pressures:
    t = cfg["train"]
    return Pressures(t["lambda_act"], t["lambda_fanin"])
