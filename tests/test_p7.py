"""P7 tests: fANOVA decomposition — purity property, ground-truth recovery,
honest nulls, model-level smoke."""
from __future__ import annotations

import numpy as np

from xplain_x1.certify.fanova import (V_MIN, bin_features, component_shares,
                                      decompose, screen_triples)
from xplain_x1.data.splits import make_splits
from xplain_x1.data.synthetic import make_synthetic


def _setup(name):
    ds = make_synthetic(name)
    sp = make_splits(ds, 1000)
    Xb, nb = bin_features(ds, sp)
    return ds, sp, Xb, nb


def test_purified_slices_zero_mean():
    ds, sp, Xb, nb = _setup("synthetic:comp2-noisy-8k")
    dec = decompose(ds.y.astype(float), Xb, nb, sp.train, sp.val, K=2)
    pair = dec["_tensors"][(1, 2)]
    tr = sp.train
    cid1 = Xb[tr, 1]
    cid2 = Xb[tr, 2]
    W = np.zeros_like(pair)
    for a, b in zip(cid1, cid2):
        W[a, b] += 1
    # every weighted row/col mean ~ 0 (the exact weighted-fANOVA condition)
    rows = (pair * W).sum(1) / np.clip(W.sum(1), 1e-12, None)
    cols = (pair * W).sum(0) / np.clip(W.sum(0), 1e-12, None)
    assert np.abs(rows).max() < 1e-6 and np.abs(cols).max() < 1e-6


def test_comp2_recovery():
    ds, sp, Xb, nb = _setup("synthetic:comp2-noisy-8k")
    dec = decompose(ds.y.astype(float), Xb, nb, sp.train, sp.val, K=2)
    comp = dec["components"]
    assert comp.get((1, 2), 0) > 0.3 and comp.get((0,), 0) > 0.2
    spurious = {u: s for u, s in comp.items() if u not in ((0,), (1, 2))}
    assert all(s < 0.05 for s in spurious.values())


def test_comp3_triple_via_screen():
    ds, sp, Xb, nb = _setup("synthetic:comp3-noisy-8k")
    fv = ds.y.astype(float)
    triples = screen_triples(fv, Xb, nb, sp.train, sp.val)
    assert (0, 1, 2) in triples
    dec = decompose(fv, Xb, nb, sp.train, sp.val, K=2, triples=triples)
    assert dec["components"].get((0, 1, 2), 0) > 0.2
    assert dec["components"].get((3,), 0) > 0.3


def test_noise_nearly_silent():
    ds, sp, Xb, nb = _setup("synthetic:noise-8k")
    dec = decompose(ds.y.astype(float), Xb, nb, sp.train, sp.val, K=2)
    assert all(s < 0.03 for s in dec["components"].values())


def test_model_component_shares_smoke():
    from xplain_x1.model.mlp import build_model
    from xplain_x1.train.settle import settle

    ds, sp, _, _ = _setup("synthetic:comp2-noisy-8k")
    m = build_model(ds.d, [8], ds.task, ds.n_classes, seed=0)
    cfg = {"train": {"lr": 1e-3, "batch_size": 256, "max_epochs": 60,
                     "weight_decay": 1e-4, "plateau_rel": 0.001,
                     "plateau_evals": 15, "anneal_frac": 0.25}}
    settle(m, ds, sp, cfg, seed=0)
    out = component_shares(m, ds, sp)
    assert out["components"].get((1, 2), 0) > 0.2   # the model's f carries it
    assert out["recon_r2"] > 0.5