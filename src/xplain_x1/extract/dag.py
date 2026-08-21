"""Concept-DAG extraction (M-#4.5, S-#11): a derived VIEW of the masked matrices.

Nodes: input features (used ones), surviving units (full payload), outputs.
Edges: masked-matrix connections above eps_edge, weight = contribution share.
CORE solid, PERIPHERY grey dashed in the DOT render.
"""
from __future__ import annotations

import numpy as np
import torch

from ..audit.contribution import edge_contributions
from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.mlp import MaskedMLP


def _edges(model: MaskedMLP, ds: Dataset, splits: Splits,
           eps_edge: float) -> list[dict]:
    Xtr = splits.standardise(ds.X[splits.train])
    with torch.no_grad():
        acts = model.hidden(torch.from_numpy(Xtr))
    parent_acts = [Xtr] + [a.numpy() for a in acts[:-1]]
    contribs = edge_contributions(model, parent_acts)
    out = []
    for li, C in enumerate(contribs):
        parents = (ds.feature_names if li == 0 else model.unit_ids[li - 1])
        for i, uid in enumerate(model.unit_ids[li]):
            for j, pname in enumerate(parents):
                if C[i, j] >= eps_edge:
                    out.append({"src": pname, "dst": uid,
                                "share": round(float(C[i, j]), 4)})
    # head edges: |w| * act std, normalised per output
    W = (model.head.weight * model.mask_head).detach().abs().numpy()
    stds = acts[-1].numpy().std(axis=0)
    C = W * stds[None, :]
    C = C / np.clip(C.sum(axis=1, keepdims=True), 1e-12, None)
    outputs = ([f"class_{k}" for k in range(model.d_out)]
               if ds.task == "classification" else ["output"])
    for k, oname in enumerate(outputs):
        for j, uid in enumerate(model.unit_ids[-1]):
            if C[k, j] >= eps_edge:
                out.append({"src": uid, "dst": oname,
                            "share": round(float(C[k, j]), 4)})
    return out


def build_dag(model: MaskedMLP, ds: Dataset, splits: Splits,
              concepts: list[dict], cfg: dict) -> dict:
    eps_edge = float(cfg["audit"]["eps_edge"])
    edges = _edges(model, ds, splits, eps_edge)
    by_uid = {c["uid_main"]: c for c in concepts if c.get("uid_main")}

    used_features = {e["src"] for e in edges if e["src"] in ds.feature_names}
    nodes = [{"id": f, "kind": "input", "label": f}
             for f in ds.feature_names if f in used_features]
    for li, ids in enumerate(model.unit_ids):
        for uid in ids:
            c = by_uid.get(uid)
            form_text = (f"{c['form']}({', '.join(c['support_names'])})"
                         if c and c.get("form") else "?")
            nodes.append({
                "id": uid, "kind": "unit", "layer": li + 1,
                "label": form_text,
                "mu": c.get("mu") if c else None,
                "Pi": c.get("Pi") if c else None,
                "pi": c.get("pi") if c else None,
                "delta": c.get("delta") if c else None,
                "coverage": c.get("contribution") if c else None,
                "tag": c.get("label", "PERIPHERY") if c else "PERIPHERY",
                "reasons": c.get("reasons", []) if c else ["unmatched"],
            })
    outputs = ([f"class_{k}" for k in range(model.d_out)]
               if ds.task == "classification" else ["output"])
    nodes += [{"id": o, "kind": "output", "label": o} for o in outputs]
    return {"dataset": ds.name, "nodes": nodes, "edges": edges}


def to_dot(dag: dict) -> str:
    lines = ["digraph xplain {", '  rankdir=LR;',
             '  node [fontname="Helvetica", fontsize=10];']
    for n in dag["nodes"]:
        nid = n["id"].replace(":", "_")
        if n["kind"] == "input":
            lines.append(f'  "{nid}" [shape=box, style=filled, '
                         f'fillcolor="#eef4ff", label="{n["label"]}"];')
        elif n["kind"] == "output":
            lines.append(f'  "{nid}" [shape=box, style=filled, '
                         f'fillcolor="#fff4e6", label="{n["label"]}"];')
        else:
            core = n.get("tag") == "CORE"
            mu = n.get("mu")
            label = (f'{n["id"]}\\n{n["label"]}'
                     + (f'\\nmu={mu:.2f} Pi={n.get("Pi") or 0:.2f}' if mu else ""))
            style = ('style=filled, fillcolor="#e8f8ee"' if core
                     else 'style="filled,dashed", fillcolor="#f2f2f2", color=grey')
            lines.append(f'  "{nid}" [shape=ellipse, {style}, label="{label}"];')
    for e in dag["edges"]:
        s, d = e["src"].replace(":", "_"), e["dst"].replace(":", "_")
        w = 0.5 + 4 * e["share"]
        lines.append(f'  "{s}" -> "{d}" [penwidth={w:.2f}];')
    lines.append("}")
    return "\n".join(lines)
