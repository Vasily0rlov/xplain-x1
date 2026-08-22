"""The audit pass (S-#8): run after every settle, on validation data."""
from __future__ import annotations

import numpy as np
import torch

from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.mlp import MaskedMLP
from ..train.settle import evaluate, null_statistics
from .contribution import edge_contributions, unit_ablation_drops
from .monosemanticity import UnitAudit, effective_fanin, unit_mu


@torch.no_grad()
def _parent_activations(model: MaskedMLP, X_std: np.ndarray) -> list[np.ndarray]:
    """parent_acts[li] = activations feeding layer li (X for li=0)."""
    acts = model.hidden(torch.from_numpy(X_std))
    return [X_std] + [a.numpy() for a in acts[:-1]]


def run_audit(model: MaskedMLP, ds: Dataset, splits: Splits, cfg: dict) -> dict:
    acfg = cfg["audit"]
    null_stats = null_statistics(ds, splits)
    Xtr = splits.standardise(ds.X[splits.train])
    Xva = splits.standardise(ds.X[splits.val])

    P_tr = _parent_activations(model, Xtr)
    P_va = _parent_activations(model, Xva)
    with torch.no_grad():
        acts_tr = [a.numpy() for a in model.hidden(torch.from_numpy(Xtr))]
        acts_va = [a.numpy() for a in model.hidden(torch.from_numpy(Xva))]

    drops = unit_ablation_drops(model, ds, splits, null_stats)
    edges = edge_contributions(model, P_tr)

    units: list[UnitAudit] = []
    for li in range(len(model.layers)):
        parent_names = (ds.feature_names if li == 0 else model.unit_ids[li - 1])
        for i, uid in enumerate(model.unit_ids[li]):
            mu, support, form = unit_mu(
                acts_tr[li][:, i], acts_va[li][:, i], P_tr[li], P_va[li],
                edges[li][i], parent_names,
                f_max=int(acfg["f_max"]), top_parents=int(acfg["top_parents"]))
            units.append(UnitAudit(
                uid=uid, layer=li + 1, mu=mu, support=support,
                support_names=[parent_names[j] for j in support], form=form,
                effective_fanin=effective_fanin(edges[li][i], float(acfg["eps_edge"])),
                contribution=drops[uid],
                act_std=float(acts_va[li][:, i].std())))

    val = evaluate(model, ds, splits, splits.val, null_stats)
    live = [u for u in units if u.act_std > 1e-6]
    return {
        "fidelity": val["fidelity"], "accuracy": val["accuracy"],
        "widths": model.widths,
        "units": [u.__dict__ for u in units],
        "n_units": len(units), "n_live": len(live),
        "median_mu_live": float(np.median([u.mu for u in live])) if live else 1.0,
        "median_ef_live": float(np.median([u.effective_fanin for u in live])) if live else 0,
    }


def mu_stalled(audit_history: list[dict], tol: float = 0.01) -> bool:
    """Monosemanticity stall: median live mu improved < tol over the last two audits."""
    if len(audit_history) < 3:
        return False
    m = [a["median_mu_live"] for a in audit_history[-3:]]
    return (m[-1] - m[0]) < tol
