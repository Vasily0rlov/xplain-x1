"""P0 test suite: seeding, splits, synthetics, masked MLP, gauge, determinism."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from xplain_x1.data.registry import get_dataset
from xplain_x1.data.splits import cpss_pairs, make_splits
from xplain_x1.data.synthetic import CONFIGS, make_synthetic
from xplain_x1.model.gauge import gauge_pass
from xplain_x1.model.mlp import build_model
from xplain_x1.train.settle import evaluate, null_statistics, settle
from xplain_x1.util.seeding import rng, torch_seed

CFG = {"train": {"lr": 1e-3, "batch_size": 256, "max_epochs": 30,
                 "weight_decay": 1e-4, "plateau_rel": 0.001, "plateau_evals": 10,
                 "anneal_frac": 0.25}}


# -- seeding ---------------------------------------------------------------

def test_seeding_deterministic_and_independent():
    a1 = rng(7, "x").standard_normal(5)
    a2 = rng(7, "x").standard_normal(5)
    b = rng(7, "y").standard_normal(5)
    assert np.allclose(a1, a2)
    assert not np.allclose(a1, b)
    assert torch_seed(7, "x") == torch_seed(7, "x") != torch_seed(8, "x")


# -- splits ----------------------------------------------------------------

def test_splits_deterministic_disjoint_and_standardised():
    ds = make_synthetic("synthetic:comp2-noisy-8k")
    s1 = make_splits(ds, split_seed=1000)
    s2 = make_splits(ds, split_seed=1000)
    assert np.array_equal(s1.train, s2.train) and np.array_equal(s1.probe, s2.probe)
    all_idx = np.concatenate([s1.train, s1.val, s1.test])
    assert len(np.unique(all_idx)) == ds.n
    Xtr = s1.standardise(ds.X[s1.train])
    assert np.abs(Xtr.mean(0)).max() < 1e-4
    assert set(s1.probe).issubset(set(np.concatenate([s1.train, s1.val])))


def test_cpss_pairs_complementary():
    ds = make_synthetic("synthetic:comp2-noisy-2k")
    s = make_splits(ds, split_seed=1000)
    pairs = cpss_pairs(s.train, n_pairs=5, split_seed=1000)
    assert len(pairs) == 5
    for a, b in pairs:
        assert len(np.intersect1d(a, b)) == 0
        assert abs(len(a) - len(b)) <= 1
        assert set(a) | set(b) <= set(s.train)
    # deterministic
    pairs2 = cpss_pairs(s.train, n_pairs=5, split_seed=1000)
    assert np.array_equal(pairs[0][0], pairs2[0][0])


# -- synthetics ------------------------------------------------------------

def test_synthetic_configs_ground_truth_and_determinism():
    assert len(CONFIGS) == 6
    for name in CONFIGS:
        ds1, ds2 = make_synthetic(name), make_synthetic(name)
        assert np.array_equal(ds1.X, ds2.X) and np.array_equal(ds1.y, ds2.y)
        assert ds1.ground_truth is not None
        assert "honest_depth" in ds1.ground_truth
    comp2 = make_synthetic("synthetic:comp2-noisy-8k")
    planted = [c for c in comp2.ground_truth["concepts"] if c.get("planted")]
    assert planted and planted[0]["support"] == [1, 2]
    assert 0.5 < comp2.ground_truth["r2_ceiling"] < 1.0


# -- masked MLP ------------------------------------------------------------

def test_mask_semantics_and_unit_ids():
    m = build_model(4, [6, 5], "regression", None)
    assert m.unit_ids[0] == [f"L1U{i}" for i in range(6)]
    x = torch.randn(16, 4)
    y0 = m(x)
    m.mask_0[2, :] = 0.0      # sever unit L1U2's inputs: bias may leak, output changes
    y1 = m(x)
    assert not torch.allclose(y0, y1)
    with torch.no_grad():
        m.layers[0].weight[2, :] = 99.0   # masked weights must have no effect
    assert torch.allclose(m(x), y1)


def test_model_save_load_roundtrip(tmp_path):
    m = build_model(4, [6], "classification", 3)
    m.mask_0[1, 2] = 0.0
    p = tmp_path / "m.pt"
    m.save(p)
    m2 = type(m).load(p)
    x = torch.randn(8, 4)
    assert torch.allclose(m(x), m2(x))
    assert m2.unit_ids == m.unit_ids


# -- gauge -----------------------------------------------------------------

def test_gauge_preserves_function():
    torch.manual_seed(0)
    m = build_model(6, [8, 8], "regression", None)
    with torch.no_grad():   # de-normalise weights so the pass has real work to do
        m.layers[0].weight.mul_(3.7)
        m.layers[1].weight.mul_(0.2)
    x = torch.randn(64, 6)
    before = m(x)
    gauge_pass(m, x)        # internal assert: max diff < 1e-5
    after = m(x)
    assert (before - after).abs().max().item() < 1e-5
    norms = (m.layers[0].weight * m.mask_0).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert sorted(m.unit_ids[0]) == [f"L1U{i}" for i in range(8)]  # ids permuted, kept


# -- determinism (P0-10) ---------------------------------------------------

def _settle_once() -> dict:
    ds = make_synthetic("synthetic:add-noisy-8k")
    splits = make_splits(ds, split_seed=1000)
    m = build_model(ds.d, [8], ds.task, ds.n_classes, seed=3)
    settle(m, ds, splits, CFG, seed=3)
    return {k: v.clone() for k, v in m.state_dict().items()}


def test_settle_deterministic():
    torch.set_num_threads(4)
    s1, s2 = _settle_once(), _settle_once()
    for k in s1:
        assert torch.equal(s1[k], s2[k]), f"nondeterministic param {k}"


# -- baseline sanity -------------------------------------------------------

def test_settle_learns_signal_not_noise():
    ds = make_synthetic("synthetic:add-noisy-8k")
    splits = make_splits(ds, split_seed=1000)
    m = build_model(ds.d, [16], ds.task, ds.n_classes)
    settle(m, ds, splits, CFG, seed=0)
    fid = evaluate(m, ds, splits, splits.val, null_statistics(ds, splits))["fidelity"]
    assert fid > 0.5 * ds.ground_truth["r2_ceiling"]

    noise = make_synthetic("synthetic:noise-8k")
    nsplits = make_splits(noise, split_seed=1000)
    mn = build_model(noise.d, [16], noise.task, noise.n_classes)
    settle(mn, noise, nsplits, CFG, seed=0)
    nfid = evaluate(mn, noise, nsplits, nsplits.val,
                    null_statistics(noise, nsplits))["fidelity"]
    assert nfid < 0.05
