"""P1 tests: pressures, contribution, monosemanticity metric, dissolution."""
from __future__ import annotations

import numpy as np
import torch

from xplain_x1.audit.audit import run_audit
from xplain_x1.audit.contribution import edge_contributions, unit_ablation_drops
from xplain_x1.audit.dissolve import dissolution_cost
from xplain_x1.audit.monosemanticity import unit_mu
from xplain_x1.data.splits import make_splits
from xplain_x1.data.synthetic import make_synthetic
from xplain_x1.model.mlp import build_model
from xplain_x1.train.losses import Pressures
from xplain_x1.train.settle import null_statistics, settle

CFG = {"train": {"lr": 1e-3, "batch_size": 256, "max_epochs": 40,
                 "weight_decay": 1e-4, "plateau_rel": 0.001, "plateau_evals": 10,
                 "anneal_frac": 0.25, "lambda_act": 1e-3, "lambda_fanin": 1e-3},
       "audit": {"eps_edge": 0.02, "f_max": 3, "top_parents": 6}}


def test_pressures_positive_and_ramped():
    m = build_model(5, [6], "regression", None, seed=0)
    p = Pressures(1e-3, 1e-3)
    x = torch.randn(32, 5)
    acts = m.hidden(x)
    full = p(m, acts, ramp=1.0)
    half = p(m, acts, ramp=0.5)
    assert full.item() > 0 and abs(half.item() - 0.5 * full.item()) / full.item() < 0.05


def test_pressures_std_keyed_by_unit_id():
    m = build_model(5, [6], "regression", None, seed=0)
    p = Pressures(1e-3, 1e-3)
    p(m, m.hidden(torch.randn(32, 5)), 1.0)
    assert set(p._std) == set(m.unit_ids[0])


def _handbuilt_net():
    """4 inputs -> 3 units: u0 = relu(x0) (big head weight), u1 = relu(x1+x2+x3) (small),
    u2 dead."""
    m = build_model(4, [3], "regression", None, seed=0)
    with torch.no_grad():
        m.layers[0].weight.zero_()
        m.layers[0].bias.zero_()
        m.layers[0].weight[0, 0] = 1.0
        m.layers[0].weight[1, 1] = 1.0
        m.layers[0].weight[1, 2] = 1.0
        m.layers[0].weight[1, 3] = 1.0
        m.layers[0].weight[2, :] = 0.0
        m.head.weight.zero_()
        m.head.bias.zero_()
        m.head.weight[0, 0] = 2.0
        m.head.weight[0, 1] = 0.1
    return m


def test_contribution_orders_units():
    ds = make_synthetic("synthetic:add-noisy-8k")
    ds.X = ds.X[:, :4]
    ds.feature_names = ds.feature_names[:4]
    ds.continuous_mask = ds.continuous_mask[:4]
    # make y depend on relu(x0) so ablation of u0 hurts
    ds.y = np.maximum(ds.X[:, 0], 0) * 2.0
    splits = make_splits(ds, split_seed=1000)
    m = _handbuilt_net()
    # the model must live in the SCALED-target space (regression targets are
    # standardised): out_scaled = (2*relu(x0) - y_mean) / y_std
    import torch as _t
    with _t.no_grad():
        m.head.weight.mul_(1.0 / splits.y_std)
        m.head.bias.fill_(-splits.y_mean / splits.y_std)
    drops = unit_ablation_drops(m, ds, splits, null_statistics(ds, splits))
    assert drops["L1U0"] > 0.5
    assert drops["L1U0"] > drops["L1U1"]
    # u1 adds a term absent from y: ablating it legitimately *helps* (small negative)
    assert abs(drops["L1U1"]) < 0.05
    assert abs(drops["L1U2"]) < 1e-9

    Xtr = splits.standardise(ds.X[splits.train])
    edges = edge_contributions(m, [Xtr])
    assert edges[0][0].argmax() == 0
    assert edges[0][2].sum() == 0.0


def test_mu_pure_vs_mixed_unit():
    g = np.random.default_rng(0)
    P_tr, P_va = g.standard_normal((2000, 6)), g.standard_normal((800, 6))
    contrib = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
    pure_tr, pure_va = np.tanh(P_tr[:, 0]), np.tanh(P_va[:, 0])
    mu, S, form = unit_mu(pure_tr, pure_va, P_tr, P_va, contrib,
                          [f"p{i}" for i in range(6)])
    assert mu > 0.95 and S == [0]
    # a genuine 5-way mix cannot be captured by any <=3-parent surrogate
    contrib5 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    mix_tr = np.sin(3 * P_tr[:, :5]).prod(axis=1)
    mix_va = np.sin(3 * P_va[:, :5]).prod(axis=1)
    mu2, _, _ = unit_mu(mix_tr, mix_va, P_tr, P_va, contrib5,
                        [f"p{i}" for i in range(6)])
    assert mu2 < 0.5


def test_dead_unit_mu_is_one():
    P = np.random.default_rng(0).standard_normal((100, 3))
    a = np.zeros(100)
    mu, S, form = unit_mu(a, a, P, P, np.array([1.0, 0, 0]), ["a", "b", "c"])
    assert mu == 1.0 and form == "constant"


def test_dissolution_of_redundant_layer_is_cheap():
    # needs a real training budget: at ~40 epochs even a FROM-SCRATCH flat model
    # cannot fit ADD's three nonlinearities, so the teacher-vs-flat gap would be
    # a budget artifact, not earned depth
    cfg = {"train": {**CFG["train"], "max_epochs": 150, "plateau_evals": 15}}
    ds = make_synthetic("synthetic:add-noisy-8k")
    splits = make_splits(ds, split_seed=1000)
    m = build_model(ds.d, [12, 12], ds.task, ds.n_classes, seed=0)
    settle(m, ds, splits, cfg, seed=0)   # additive task: depth 2 is theatre
    cost, cand = dissolution_cost(m, 0, ds, splits, seed=0)
    assert cost < 0.02, f"dissolving an unearned layer cost {cost:.4f}"
    assert cand.widths == [12]


def test_run_audit_shapes():
    ds = make_synthetic("synthetic:comp2-noisy-2k")
    splits = make_splits(ds, split_seed=1000)
    m = build_model(ds.d, [8], ds.task, ds.n_classes, seed=0)
    settle(m, ds, splits, CFG, seed=0)
    audit = run_audit(m, ds, splits, CFG)
    assert audit["n_units"] == 8
    assert 0 <= audit["median_mu_live"] <= 1
    uids = {u["uid"] for u in audit["units"]}
    assert uids == set(m.unit_ids[0])
