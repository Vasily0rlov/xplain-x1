"""P6 tests: feature grouping, route building, joint reality ablation."""
from __future__ import annotations

import numpy as np

from xplain_x1.certify.groups import (discover_groups, group_index, group_names,
                                      group_support)
from xplain_x1.certify.routes import (build_routes, route_q_mean,
                                      route_selection, route_universe_size)
from xplain_x1.data.dataset import Dataset
from xplain_x1.data.splits import make_splits
from xplain_x1.data.synthetic import make_synthetic


def test_iid_features_give_singleton_groups():
    ds = make_synthetic("synthetic:comp2-noisy-8k")
    sp = make_splits(ds, 1000)
    groups = discover_groups(ds, sp)
    assert groups == [[j] for j in range(ds.d)]      # E6.1 reduction property


def test_collinear_features_cluster():
    g = np.random.default_rng(0)
    base = g.standard_normal((3000, 3))
    X = np.column_stack([
        base[:, 0], base[:, 0] * 2 + 0.05 * g.standard_normal(3000),  # pair
        base[:, 1], -base[:, 1] + 0.05 * g.standard_normal(3000),     # pair
        base[:, 2]])                                                   # alone
    ds = Dataset(name="t", X=X, y=base[:, 2], task="regression",
                 feature_names=[f"f{i}" for i in range(5)])
    sp = make_splits(ds, 1000)
    groups = discover_groups(ds, sp)
    assert [0, 1] in groups and [2, 3] in groups and [4] in groups
    gi = group_index(groups, 5)
    assert gi[0] == gi[1] and gi[2] == gi[3] and gi[4] not in (gi[0], gi[2])
    assert group_support({0, 3}, gi) == frozenset({gi[0], gi[2]})
    names = group_names(groups, ds.feature_names)
    assert names[gi[4]] == "f4" and names[gi[0]].startswith("G")


def test_routes_unify_carvings_across_runs():
    # run 0 carves {A}, run 1 carves {A'} (same group), run 2 both -> one route
    A, B = frozenset({0}), frozenset({1})
    run_gs = {0: {A: 0.95}, 1: {A: 0.9}, 2: {A: 0.92, B: 0.85}, 3: {A: 0.99}}
    routes = build_routes(run_gs, {A: ["L1U0"]}, run_gs[0], n_runs=4, mu_min=0.8)
    ra = next(r for r in routes if r.certified_support == A)
    rb = next(r for r in routes if r.certified_support == B)
    assert ra.Pi == 1.0 and ra.members_main == ["L1U0"]
    assert rb.Pi == 0.25 and rb.members_main == []


def test_routes_chain_collapse_and_multiplicity():
    # {0} and {0,1} in complementary runs -> ONE route certified {0} (jitter);
    # {2} vs {3} present in different runs stay separate routes
    g0, g01, g2, g3 = (frozenset({0}), frozenset({0, 1}),
                       frozenset({2}), frozenset({3}))
    run_gs = {0: {g0: 0.9, g2: 0.9}, 1: {g01: 0.9, g2: 0.9},
              2: {g0: 0.9, g3: 0.9}, 3: {g01: 0.9, g3: 0.9}}
    routes = build_routes(run_gs, {g0: ["L1U0"], g2: ["L1U1"]}, run_gs[0],
                          n_runs=4, mu_min=0.8)
    r0 = next(r for r in routes if r.certified_support == g0)
    assert r0.Pi == 1.0                                # jitter unified
    assert sorted(map(sorted, r0.variants)) == [[0], [0, 1]]
    assert not any(r.certified_support == g01 for r in routes)
    assert {tuple(sorted(r.certified_support)) for r in routes} >= {(2,), (3,)}


def test_routes_modal_beats_underdetection():
    # 7 runs detect the full product {0,1,2}; 1 run under-detects {0}.
    # Certified support must be the MODAL {0,1,2}, not the minimal {0};
    # the common core {0} is reported separately.
    full, part = frozenset({0, 1, 2}), frozenset({0})
    run_gs = {r: {full: 0.95} for r in range(7)}
    run_gs[7] = {part: 0.9}
    routes = build_routes(run_gs, {full: ["L1U0"]}, run_gs[0], 8, 0.8)
    r = routes[0]
    assert r.certified_support == full
    assert r.common_core == part
    assert r.Pi == 1.0                    # presence unified across the chain


def test_route_selection_and_universe():
    A = frozenset({0})
    routes = build_routes({0: {A: 0.9}}, {A: ["u"]}, {A: 0.9}, 1, 0.8)
    half_gs = [{A: 0.9}, {A: 0.7}, {A: 0.85}, {}]
    pi = route_selection(half_gs, routes, mu_min=0.8)
    assert pi[routes[0].rid] == 0.5
    assert route_universe_size(4, 3) == 4 + 6 + 4
    assert route_q_mean(half_gs, 0.8) == 0.5


def test_joint_reality_ablation():
    import yaml

    from xplain_x1.audit.contribution import unit_ablation_drops
    from xplain_x1.certify.reality import reality_test
    from xplain_x1.model.mlp import build_model
    from xplain_x1.train.settle import null_statistics, settle

    ds = make_synthetic("synthetic:comp2-noisy-2k")
    sp = make_splits(ds, 1000)
    m = build_model(ds.d, [6], ds.task, ds.n_classes, seed=0)
    cfg = {"train": {"lr": 1e-3, "batch_size": 256, "max_epochs": 60,
                     "weight_decay": 1e-4, "plateau_rel": 0.001,
                     "plateau_evals": 15, "anneal_frac": 0.25}}
    settle(m, ds, sp, cfg, seed=0)
    drops = unit_ablation_drops(m, ds, sp, null_statistics(ds, sp))
    top2 = sorted(drops, key=drops.get, reverse=True)[:2]
    joint = reality_test(m, top2, ds, sp, n_bootstrap=100)
    single = reality_test(m, top2[0], ds, sp, n_bootstrap=100)
    assert joint["delta"] >= single["delta"] - 1e-6   # joint at least as big
