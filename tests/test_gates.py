"""CI gates (S-#13, Build Plan P3-6): fast certification smoke.

Run with: pytest -m gates  (deselected by default — ~2-3 min).
Gate 1: COMP2-clean mini-battery (R=2, B=2) recovers the planted support as a
        stable concept.
Gate 2: NOISE certifies ZERO concepts (the K2 untradeable).
"""
from __future__ import annotations

import copy

import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mini_cfg() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["certify"]["restarts"] = 2
    cfg["certify"]["cpss_pairs"] = 2
    cfg["certify"]["bootstrap"] = 200
    cfg["controller"]["max_rounds"] = 6
    cfg["compute"]["n_workers"] = 8
    return cfg


@pytest.mark.gates
def test_gate_comp2_recovery():
    from xplain_x1.certify.certify import certify

    cert = certify("synthetic:comp2-clean-8k", _mini_cfg())
    core_supports = {tuple(c["modal_support"]) for c in cert["concepts"]
                     if c["label"] == "CORE"}
    stable_supports = {tuple(c["modal_support"]) for c in cert["concepts"]
                       if c["Pi"] >= 0.5 and c.get("mu", 0) and c["mu"] >= 0.8}
    assert (1, 2) in core_supports | stable_supports, (
        f"planted (1,2) not found; core={core_supports}, stable={stable_supports}")


@pytest.mark.gates
def test_gate_noise_zero_certification():
    from xplain_x1.certify.certify import certify

    cert = certify("synthetic:noise-8k", _mini_cfg())
    assert cert["n_core"] == 0, (
        f"K2 violated: {cert['n_core']} concepts certified on noise")
