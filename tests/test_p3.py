"""P3 tests: signatures, matching of permuted clones, CPSS arithmetic, reality test."""
from __future__ import annotations

import numpy as np
import torch

from xplain_x1.certify.cpss import ev_bound, structure_universe_size
from xplain_x1.certify.matching import cluster_concepts, match_pair
from xplain_x1.certify.reality import reality_test
from xplain_x1.certify.signatures import input_support, unit_signatures
from xplain_x1.data.splits import make_splits
from xplain_x1.data.synthetic import make_synthetic
from xplain_x1.model.mlp import build_model


def _model_and_data(seed=0):
    ds = make_synthetic("synthetic:comp2-noisy-2k")
    splits = make_splits(ds, split_seed=1000)
    m = build_model(ds.d, [6], ds.task, ds.n_classes, seed=seed)
    return ds, splits, m


def test_signatures_shapes_and_normalisation():
    ds, splits, m = _model_and_data()
    sigs = unit_signatures(m, ds, splits)
    assert 0 < len(sigs) <= 6
    for s in sigs:
        assert s.attribution.shape == (ds.d,)
        assert abs(np.linalg.norm(s.attribution) - 1.0) < 1e-5
        assert len(s.probe_acts) == len(splits.probe)


def test_matching_identifies_permuted_clone():
    ds, splits, m = _model_and_data()
    sigs_a = unit_signatures(m, ds, splits)
    perm = torch.randperm(6)
    m2 = build_model(ds.d, [6], ds.task, ds.n_classes, seed=0)
    with torch.no_grad():
        m2.layers[0].weight.copy_(m.layers[0].weight[perm])
        m2.layers[0].bias.copy_(m.layers[0].bias[perm])
        m2.head.weight.copy_(m.head.weight[:, perm])
        m2.head.bias.copy_(m.head.bias)
    sigs_b = unit_signatures(m2, ds, splits)
    matches = match_pair(sigs_a, sigs_b, tau=0.7)
    assert len(matches) == len(sigs_a)
    for uid_a, uid_b, cos in matches:
        ia = m.unit_ids[0].index(uid_a)
        ib = m2.unit_ids[0].index(uid_b)
        assert perm[ib] == ia or cos > 0.999


def test_cluster_concepts_pi():
    ds, splits, m = _model_and_data(seed=0)
    sigs = unit_signatures(m, ds, splits)
    runs = [sigs, sigs, sigs]                      # 3 identical runs
    sup = {s.uid: frozenset({0, 1}) for s in sigs}
    concepts = cluster_concepts(runs, [sup, sup, sup], tau=0.7)
    full = [c for c in concepts if c.Pi == 1.0]
    assert len(full) == len(sigs)
    assert all(not c.multiplicitous for c in full)


def test_universe_size_and_bound():
    # d=10 inputs, widths [8, 4]: parents 10 then 8
    p = structure_universe_size(10, [8, 4], f_max=3)
    assert p == (10 + 45 + 120) + (8 + 28 + 56)
    import pytest
    assert ev_bound(5.0, p, 0.7) == pytest.approx(25.0 / (0.4 * p))
    assert ev_bound(5.0, p, 0.5) == float("inf")


def test_input_support_recursion():
    units = {
        "L1U0": {"layer": 1, "support": [2, 3], "support_names": ["x3", "x4"]},
        "L1U1": {"layer": 1, "support": [5], "support_names": ["x6"]},
        "L2U0": {"layer": 2, "support": [0, 1], "support_names": ["L1U0", "L1U1"]},
    }
    assert input_support("L2U0", units) == frozenset({2, 3, 5})


def test_reality_ablation_of_useful_unit():
    ds, splits, _ = _model_and_data()
    m = build_model(ds.d, [4], ds.task, ds.n_classes, seed=0)
    from xplain_x1.train.settle import settle
    cfg = {"train": {"lr": 1e-3, "batch_size": 256, "max_epochs": 60,
                     "weight_decay": 1e-4, "plateau_rel": 0.001,
                     "plateau_evals": 15, "anneal_frac": 0.25}}
    settle(m, ds, splits, cfg, seed=0)
    from xplain_x1.audit.contribution import unit_ablation_drops
    from xplain_x1.train.settle import null_statistics
    drops = unit_ablation_drops(m, ds, splits, null_statistics(ds, splits))
    top_uid = max(drops, key=drops.get)
    rt = reality_test(m, top_uid, ds, splits, n_bootstrap=200)
    assert rt["delta"] > 0 and rt["ci_low"] < rt["delta"] < rt["ci_high"]
