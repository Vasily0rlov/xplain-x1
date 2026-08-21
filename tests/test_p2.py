"""P2 tests: structural ops preserve what they claim; controller smoke."""
from __future__ import annotations

import numpy as np
import torch

from xplain_x1.data.splits import make_splits
from xplain_x1.data.synthetic import make_synthetic
from xplain_x1.model.mlp import build_model
from xplain_x1.model.ops import (add_unit, insert_layer, merge_units, prune_edges,
                                 remove_units)


def _rand_model(seed=0, widths=(6, 5)):
    return build_model(4, list(widths), "regression", None, seed=seed)


def test_add_unit_fresh_extends_registry_and_barely_moves_function():
    m = _rand_model()
    x = torch.randn(64, 4)
    y0 = m(x)
    m2 = add_unit(m, 0, "fresh", seed=1)
    assert m2.widths == [7, 5]
    all_old = set(m.unit_ids[0]) | set(m.unit_ids[1])
    new_uid = m2.unit_ids[0][6]
    assert m2.unit_ids[0][:6] == m.unit_ids[0]
    assert new_uid.startswith("L1U") and new_uid not in all_old
    assert (m2(x) - y0).abs().max() < 0.5      # tiny outgoing init
    m3 = add_unit(m2, 0, "fresh", seed=2)
    assert m3.unit_ids[0][7] not in set(m2.unit_ids[0]) | set(m2.unit_ids[1])


def test_add_unit_split_halves_outgoing_and_preserves_function_approx():
    m = _rand_model()
    x = torch.randn(64, 4)
    y0 = m(x)
    uid = m.unit_ids[0][2]
    m2 = add_unit(m, 0, f"split:{uid}", seed=3)
    assert m2.widths == [7, 5]
    # split: clone + 0.05-scale noise, halved outgoing -> function ~preserved
    assert (m2(x) - y0).abs().max() < 0.15


def test_insert_layer_identity_is_exact():
    m = _rand_model(widths=(6, 5))
    x = torch.randn(128, 4)
    y0 = m(x)
    m2 = insert_layer(m, 1)
    assert m2.widths == [6, 6, 5]
    assert torch.allclose(m2(x), y0, atol=1e-6)
    # inserted ids are fresh and GLOBALLY unique (no collision with shifted layer)
    inserted = set(m2.unit_ids[1])
    assert len(inserted) == 6
    assert not inserted & (set(m.unit_ids[0]) | set(m.unit_ids[1]))
    # original layer-2 ids preserved (now layer 3)
    assert m2.unit_ids[2] == m.unit_ids[1]


def test_merge_identical_units_is_exact():
    m = _rand_model()
    with torch.no_grad():   # make unit 1 an exact clone of unit 0
        m.layers[0].weight[1] = m.layers[0].weight[0]
        m.layers[0].bias[1] = m.layers[0].bias[0]
    x = torch.randn(64, 4)
    y0 = m(x)
    m2 = merge_units(m, m.unit_ids[0][0], m.unit_ids[0][1])
    assert m2.widths == [5, 5]
    assert torch.allclose(m2(x), y0, atol=1e-6)
    assert m.unit_ids[0][1] not in m2.unit_ids[0]


def test_remove_units_registry_integrity():
    m = _rand_model()
    drop = {m.unit_ids[0][1], m.unit_ids[1][0]}
    m2 = remove_units(m, drop)
    assert m2.widths == [5, 4]
    assert not drop & set(m2.unit_ids[0]) | drop & set(m2.unit_ids[1])
    x = torch.randn(8, 4)
    m2(x)                                       # forward is consistent


def test_prune_edges_masks_below_threshold():
    m = _rand_model()
    contribs = [np.full((6, 4), 0.5), np.full((5, 6), 0.5)]
    contribs[0][0, 0] = 0.001                   # one edge below threshold
    cut = prune_edges(m, contribs, eps_edge=0.02)
    assert cut == 1 and m.mask_0[0, 0] == 0.0 and m.mask_0.sum() == 23


def test_grow_smoke_terminates():
    from xplain_x1.controller.growth import grow

    ds = make_synthetic("synthetic:comp2-noisy-2k")
    splits = make_splits(ds, split_seed=1000)
    cfg = {
        "model": {"init_layers": 1, "init_width": 6, "max_layers": 3,
                  "max_total_units": 24},
        "train": {"lr": 1e-3, "batch_size": 256, "max_epochs": 25,
                  "weight_decay": 1e-4, "plateau_rel": 0.001, "plateau_evals": 5,
                  "anneal_frac": 0.2, "lambda_act": 1e-3, "lambda_fanin": 1e-2},
        "audit": {"eps_edge": 0.02, "f_max": 3, "top_parents": 6},
        "controller": {"eps_prune": 0.001, "eps_depth": 0.005, "delta_grow": 0.005,
                       "delta_stop": 0.01, "max_rounds": 3},
        "certify": {"tau_match": 0.7},
    }
    trace = grow(ds, splits, cfg, seed=0, fid_ref=0.9)
    assert trace.rounds <= 3
    assert trace.audits and trace.model.widths
    assert sum(trace.model.widths) <= 24 and len(trace.model.widths) <= 3
