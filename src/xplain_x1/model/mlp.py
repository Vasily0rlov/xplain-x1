"""MaskedMLP: standard Linear->ReLU stack with edge masks + persistent unit ids (S-#6).

Pruning is masking, never deletion — shapes and optimiser state survive while the
effective computation becomes sparse.  Every hidden unit gets a persistent id
("L1U0", ...) at creation that survives growth, pruning, and gauge permutation;
all audit and certification artefacts key on these ids (M-#3.8).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMLP(nn.Module):
    def __init__(self, d_in: int, widths: list[int], d_out: int, task: str):
        super().__init__()
        self.d_in, self.d_out, self.task = d_in, d_out, task
        self._unit_counter = 0     # GLOBAL counter: ids stay unique across
        self.layers = nn.ModuleList()   # growth, pruning, and layer insertion
        self.unit_ids: list[list[str]] = []
        prev = d_in
        for li, w in enumerate(widths):
            lin = nn.Linear(prev, w)
            self.layers.append(lin)
            self.register_buffer(f"mask_{li}", torch.ones(w, prev))
            self.unit_ids.append([self._new_id(li) for _ in range(w)])
            prev = w
        self.head = nn.Linear(prev, d_out)
        self.register_buffer("mask_head", torch.ones(d_out, prev))

    # -- identity ---------------------------------------------------------
    def _new_id(self, layer: int) -> str:
        uid = f"L{layer + 1}U{self._unit_counter}"
        self._unit_counter += 1
        return uid

    @property
    def widths(self) -> list[int]:
        return [lin.out_features for lin in self.layers]

    def mask(self, li: int) -> torch.Tensor:
        return getattr(self, f"mask_{li}")

    # -- forward ----------------------------------------------------------
    def hidden(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Post-activations per hidden layer."""
        acts = []
        h = x
        for li, lin in enumerate(self.layers):
            h = F.relu(F.linear(h, lin.weight * self.mask(li), lin.bias))
            acts.append(h)
        return acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for li, lin in enumerate(self.layers):
            h = F.relu(F.linear(h, lin.weight * self.mask(li), lin.bias))
        return F.linear(h, self.head.weight * self.mask_head, self.head.bias)

    # -- persistence ------------------------------------------------------
    def config(self) -> dict:
        return {"d_in": self.d_in, "widths": self.widths, "d_out": self.d_out,
                "task": self.task, "unit_ids": self.unit_ids,
                "unit_counter": self._unit_counter}

    def save(self, path) -> None:
        torch.save({"state": self.state_dict(), "config": self.config()}, path)

    @classmethod
    def load(cls, path) -> "MaskedMLP":
        blob = torch.load(path, weights_only=False)
        cfg = blob["config"]
        m = cls(cfg["d_in"], cfg["widths"], cfg["d_out"], cfg["task"])
        m.load_state_dict(blob["state"])
        m.unit_ids = cfg["unit_ids"]
        m._unit_counter = cfg["unit_counter"]
        return m


def build_model(d_in: int, widths: list[int], task: str,
                n_classes: int | None, seed: int | None = None) -> MaskedMLP:
    """seed pins the weight init (S-#2: every stochastic step derives from the run seed)."""
    if seed is not None:
        from ..util.seeding import torch_seed

        torch.manual_seed(torch_seed(seed, "init"))
    d_out = (n_classes if task == "classification" else 1)
    return MaskedMLP(d_in, list(widths), d_out, task)
