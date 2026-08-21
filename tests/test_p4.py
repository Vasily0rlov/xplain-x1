"""P4 tests: DAG builder and certificate structure on a small settled model."""
from __future__ import annotations

from xplain_x1.data.splits import make_splits
from xplain_x1.data.synthetic import make_synthetic
from xplain_x1.extract.certificate import build_certificate, render_markdown
from xplain_x1.extract.dag import build_dag, to_dot
from xplain_x1.model.mlp import build_model
from xplain_x1.train.settle import settle

CFG = {"train": {"lr": 1e-3, "batch_size": 256, "max_epochs": 40,
                 "weight_decay": 1e-4, "plateau_rel": 0.001, "plateau_evals": 10,
                 "anneal_frac": 0.25},
       "audit": {"eps_edge": 0.02, "f_max": 3, "top_parents": 6},
       "controller": {"eps_depth": 0.005},
       "certify": {"tau_match": 0.7, "pi_thr": 0.7, "mu_min": 0.8}}


def _mini_cert():
    ds = make_synthetic("synthetic:comp2-noisy-2k")
    splits = make_splits(ds, split_seed=1000)
    m = build_model(ds.d, [5], ds.task, ds.n_classes, seed=0)
    settle(m, ds, splits, CFG, seed=0)
    concepts = [{
        "cid": "c0", "uid_main": m.unit_ids[0][0], "layer": 1, "mu": 0.95,
        "Pi": 1.0, "pi": 0.9, "delta": 0.02, "ci_low": 0.01, "ci_high": 0.03,
        "form": "smooth", "support_names": ["x2"], "modal_support": [1],
        "support_variants": [[1]], "multiplicitous": False,
        "contribution": 0.02, "members": {0: m.unit_ids[0][0]},
        "label": "CORE", "reasons": []},
        {"cid": "c1", "uid_main": m.unit_ids[0][1], "layer": 1, "mu": 0.5,
         "Pi": 0.25, "pi": 0.1, "delta": 0.0, "ci_low": -0.01, "ci_high": 0.01,
         "form": "tree(d3)", "support_names": ["x2", "x3"], "modal_support": [1, 2],
         "support_variants": [[1, 2], [3]], "multiplicitous": True,
         "contribution": 0.001, "members": {0: m.unit_ids[0][1]},
         "label": "PERIPHERY", "reasons": ["polysemantic", "unstable"]}]
    cert = {"model": m, "splits": splits, "concepts": concepts, "R": 2, "B": 2,
            "n_core": 1, "ev_bound": 0.4, "q_mean": 3.0, "p_universe": 175,
            "main": {"widths": m.widths, "fidelity": 0.8, "fid_ref": 0.85,
                     "val": {"fidelity": 0.8, "accuracy": None},
                     "test": {"fidelity": 0.78, "accuracy": None},
                     "actions": []},
            "restart_fids": [0.8, 0.79]}
    return ds, splits, m, cert


def test_dag_structure_and_dot():
    ds, splits, m, cert = _mini_cert()
    dag = build_dag(m, ds, splits, cert["concepts"], CFG)
    kinds = {n["kind"] for n in dag["nodes"]}
    assert kinds == {"input", "unit", "output"}
    unit_nodes = [n for n in dag["nodes"] if n["kind"] == "unit"]
    assert len(unit_nodes) == 5
    tags = {n["id"]: n["tag"] for n in unit_nodes}
    assert tags[m.unit_ids[0][0]] == "CORE"
    assert tags[m.unit_ids[0][1]] == "PERIPHERY"
    assert all(e["share"] >= CFG["audit"]["eps_edge"] for e in dag["edges"])
    dot = to_dot(dag)
    assert dot.startswith("digraph") and "dashed" in dot and "#e8f8ee" in dot


def test_certificate_sections_and_render():
    ds, splits, m, cert = _mini_cert()
    doc = build_certificate(cert, ds, splits,
                            {**CFG, "data": {"split_seed": 1000}}, seed=0)
    for section in ("identification", "performance_and_limits",
                    "interpretability_structure", "statistical_certification",
                    "concepts", "non_claims"):
        assert section in doc
    assert doc["interpretability_structure"]["n_core"] == 1
    assert doc["interpretability_structure"]["core_coverage_share"] == 1.0
    md = render_markdown(doc)
    assert "Audit Certificate" in md and "CORE concepts" in md
    assert "polysemantic, unstable" in md
