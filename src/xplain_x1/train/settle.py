"""The settle loop (S-#7): AdamW, plateau stop, best-val restore.

P0 runs it pressure-free; P1 adds the loss terms via the `pressures` hook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.mlp import MaskedMLP
from ..util.seeding import rng, torch_seed
from .metrics import accuracy, classification_fidelity, regression_fidelity


@dataclass
class SettleResult:
    best_val_fid: float
    epochs_run: int
    history: list[dict] = field(default_factory=list)


def _tensors(ds: Dataset, splits: Splits, idx: np.ndarray):
    X = torch.from_numpy(splits.standardise(ds.X[idx]))
    if ds.task == "classification":
        y = torch.from_numpy(ds.y[idx])
    else:
        y = torch.from_numpy(ds.y[idx]).unsqueeze(1)
    return X, y


@torch.no_grad()
def evaluate(model: MaskedMLP, ds: Dataset, splits: Splits, idx: np.ndarray,
             null_stats: dict) -> dict:
    model.eval()
    X, _ = _tensors(ds, splits, idx)
    out = model(X)
    y = ds.y[idx]
    if ds.task == "classification":
        proba = F.softmax(out, dim=1).numpy()
        fid = classification_fidelity(proba, y, null_stats["class_prior"])
        acc = accuracy(proba, y, ds.task)
    else:
        pred = out.squeeze(1).numpy()
        fid = regression_fidelity(pred, y, null_stats["y_train_mean"])
        acc = None
    return {"fidelity": fid, "accuracy": acc}


def null_statistics(ds: Dataset, splits: Splits) -> dict:
    ytr = ds.y[splits.train]
    if ds.task == "classification":
        prior = np.bincount(ytr, minlength=ds.n_classes).astype(np.float64)
        return {"class_prior": prior / prior.sum()}
    return {"y_train_mean": float(ytr.mean())}


def settle(model: MaskedMLP, ds: Dataset, splits: Splits, cfg: dict, seed: int,
           pressures: Callable[[MaskedMLP, list[torch.Tensor], float], torch.Tensor]
           | None = None) -> SettleResult:
    tcfg = cfg["train"]
    torch.manual_seed(torch_seed(seed, "settle"))
    g = rng(seed, "batches")
    null_stats = null_statistics(ds, splits)

    Xtr, ytr = _tensors(ds, splits, splits.train)
    n = len(splits.train)
    batch = min(int(tcfg["batch_size"]), n)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg["weight_decay"]))
    loss_fn = (F.cross_entropy if ds.task == "classification" else F.mse_loss)

    # Stopping is TOTAL-train-loss driven (task + pressures): the structural
    # shaping happens during the fidelity-flat phase, so a fidelity-plateau stop
    # would cut it short and a best-fid restore would discard it (E1.3 finding).
    # Best-val restore remains only as a safety guard against real degradation.
    best_fid, best_state = -np.inf, None
    best_loss, evals_since_best, history = np.inf, 0, []
    plateau_rel = float(tcfg["plateau_rel"])
    plateau_evals = int(tcfg["plateau_evals"])
    max_epochs = int(tcfg["max_epochs"])
    anneal_frac = float(tcfg.get("anneal_frac", 0.25))
    safety_margin = float(tcfg.get("safety_margin", 0.01))
    epochs_run = 0

    for epoch in range(max_epochs):
        model.train()
        epochs_run = epoch + 1
        ramp = (1.0 if pressures is None else
                min(1.0, (epoch + 1) / max(1, int(max_epochs * anneal_frac))))
        perm = torch.from_numpy(g.permutation(n))
        loss_sum, n_batches = 0.0, 0
        for s in range(0, n, batch):
            bi = perm[s:s + batch]
            opt.zero_grad()
            if pressures is None:
                loss = loss_fn(model(Xtr[bi]), ytr[bi])
            else:
                acts = model.hidden(Xtr[bi])
                h = acts[-1]
                out = F.linear(h, model.head.weight * model.mask_head,
                               model.head.bias)
                loss = loss_fn(out, ytr[bi]) + pressures(model, acts, ramp)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n_batches += 1

        train_loss = loss_sum / max(1, n_batches)
        ev = evaluate(model, ds, splits, splits.val, null_stats)
        history.append({"epoch": epoch, "train_loss": train_loss, **ev})
        if ev["fidelity"] > best_fid:
            best_fid = ev["fidelity"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # plateau on total train loss, once the anneal ramp is complete.
        # CUMULATIVE criterion: patience resets when loss has fallen plateau_rel
        # below the reference snapshot (slow-but-steady descent accumulates and
        # keeps training; a per-epoch margin would stop it prematurely).
        if (not np.isfinite(best_loss)
                or train_loss < best_loss - plateau_rel * max(abs(best_loss), 1e-6)):
            best_loss, evals_since_best = train_loss, 0
        elif ramp >= 1.0:
            evals_since_best += 1
            if evals_since_best >= plateau_evals:
                break

    final_fid = history[-1]["fidelity"] if history else -np.inf
    if best_state is not None and final_fid < best_fid - safety_margin:
        model.load_state_dict(best_state)          # genuine degradation: restore
        final_fid = best_fid
    return SettleResult(best_val_fid=float(final_fid), epochs_run=epochs_run,
                        history=history)
