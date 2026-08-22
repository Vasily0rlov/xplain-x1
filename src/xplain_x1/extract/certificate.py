"""Audit certificate (M-#4.5, S-#11): machine-readable + human-readable.

Sections keyed to SR 11-7 model-risk-validation and EU AI Act Article 9
headings.  The certificate asserts NOTHING about the periphery except that it
exists, where it is, and why it failed the bars.
"""
from __future__ import annotations

from ..audit.dissolve import dissolution_cost
from ..data.dataset import Dataset
from ..data.splits import Splits
from ..model.mlp import MaskedMLP
from ..util.provenance import git_commit, sha256_config


def honest_depth_statement(model: MaskedMLP, ds: Dataset, splits: Splits,
                           eps_depth: float, seed: int) -> dict:
    layers = []
    for li in range(len(model.layers) - 1):
        cost, _ = dissolution_cost(model, li, ds, splits, seed)
        layers.append({"layer": li + 1, "dissolution_cost": round(float(cost), 5),
                       "earned": bool(cost > eps_depth)})
    return {"depth": len(model.layers), "eps_depth": eps_depth,
            "pairwise_dissolutions": layers,
            "all_earned": all(l["earned"] for l in layers) if layers else True}


def build_certificate(cert: dict, ds: Dataset, splits: Splits, cfg: dict,
                      seed: int = 0) -> dict:
    model: MaskedMLP = cert["model"]
    ccfg = cfg["certify"]
    core = [c for c in cert["concepts"] if c["label"] == "CORE"]
    periphery = [c for c in cert["concepts"] if c["label"] != "CORE"]
    depth = honest_depth_statement(model, ds, splits,
                                   float(cfg["controller"]["eps_depth"]), seed)
    fid = cert["main"]["fidelity"]
    fid_ref = cert["main"]["fid_ref"]
    acc = cert["main"]["test"].get("accuracy")
    total_delta = sum(c.get("delta", 0.0) or 0.0 for c in cert["concepts"])
    for c in cert["concepts"]:
        d = c.get("delta")
        c["coverage_share"] = (round(d / total_delta, 4)
                               if d and total_delta > 1e-9 else 0.0)
    core_coverage = sum(c["coverage_share"] for c in core)

    return {
        # SR 11-7 / EU AI Act Art 9 keyed sections
        "identification": {
            "system": "XPLAIN-x1", "dataset": ds.name,
            "task": ds.task, "n": ds.n, "d": ds.d,
            "git_commit": git_commit(), "config_hash": sha256_config(cfg),
            "data_hash": ds.data_hash(),
            "seeds": {"main": seed, "restarts": cert["R"],
                      "cpss_runs": 2 * cert["B"]},
        },
        "performance_and_limits": {          # SR 11-7: soundness / Art 9(2)(a)
            "fidelity_val": fid, "fidelity_ceiling": fid_ref,
            "fidelity_ratio": round(fid / fid_ref, 4) if fid_ref else None,
            "accuracy_test": acc,
            "concept_tax_note": "vs best(unconstrained MLP, HGB) on val",
            "restart_fidelities": cert["restart_fids"],
        },
        "interpretability_structure": {      # Art 9(2)(d): transparency measures
            "widths": cert["main"]["widths"],
            "honest_depth": depth,
            "n_concepts": len(cert["concepts"]),
            "n_core": len(core), "n_periphery": len(periphery),
            "core_coverage_share": round(core_coverage, 4),
            "periphery_reasons": sorted({r for c in periphery
                                         for r in c.get("reasons", [])}),
        },
        "statistical_certification": {       # SR 11-7: outcomes analysis
            "regime": ("below stability power floor (n < 2000): Pi/pi reported "
                       "per concept; unit-level stability is not certifiable at "
                       "this sample size (owner decision 2026-08-22, option A)"
                       if ds.n < 2000 else "standard"),
            "restarts_R": cert["R"], "cpss_pairs_B": cert["B"],
            "tau_match": float(ccfg["tau_match"]),
            "pi_thr": float(ccfg["pi_thr"]), "mu_min": float(ccfg["mu_min"]),
            "ev_bound": cert["ev_bound"], "q_mean": cert["q_mean"],
            "p_universe": cert["p_universe"],
            "assumptions": [
                "CPSS exchangeability (Meinshausen-Buhlmann) over an adaptive "
                "learned pipeline is a modelling idealisation; the reality test "
                "provides an independent assumption-light check.",
                "The universe is structure-level: supports of arity <= F_max "
                "per layer interface of the delivered topology.",
            ],
        },
        "concepts": cert["concepts"],        # full rows incl. periphery labels
        "non_claims": [
            "CORE concepts are stable, real structures - not proven causal "
            "mechanisms of the world (M-C7).",
            "The periphery is labelled, not certified.",
            "Unit semantics require expert review (the soft target).",
        ],
    }


def render_markdown(cert_doc: dict) -> str:
    ident = cert_doc["identification"]
    perf = cert_doc["performance_and_limits"]
    struct = cert_doc["interpretability_structure"]
    stat = cert_doc["statistical_certification"]
    lines = [
        f"# XPLAIN-x1 Audit Certificate — {ident['dataset']}",
        "",
        f"commit `{(ident['git_commit'] or '?')[:12]}` · config "
        f"`{ident['config_hash'][:12]}` · data `{ident['data_hash']}` · "
        f"R={ident['seeds']['restarts']} restarts, "
        f"{ident['seeds']['cpss_runs']} CPSS runs",
        "",
        "## Performance",
        f"- fidelity {perf['fidelity_val']:.3f} vs ceiling "
        f"{perf['fidelity_ceiling']:.3f} (ratio {perf['fidelity_ratio']})",
        f"- test accuracy: {perf['accuracy_test']}",
        "",
        "## Structure",
        f"- widths {struct['widths']}, depth {struct['honest_depth']['depth']} "
        f"(all earned: {struct['honest_depth']['all_earned']})",
        f"- concepts: {struct['n_core']} CORE / {struct['n_periphery']} "
        f"PERIPHERY; CORE coverage share {struct['core_coverage_share']}",
        f"- periphery reasons: {', '.join(struct['periphery_reasons']) or '—'}",
        "",
        "## Statistical certification",
        f"- E[V] <= {stat['ev_bound']:.3g} at pi_thr {stat['pi_thr']} "
        f"(q_mean {stat['q_mean']:.1f}, universe {stat['p_universe']})",
        f"- assumptions: {' '.join(stat['assumptions'])}",
        "",
        "## CORE concepts",
        "| unit | layer | form | mu | Pi | pi | delta [95% CI] | coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cert_doc["concepts"]:
        if c["label"] != "CORE":
            continue
        lines.append(
            f"| {c['uid_main']} | {c['layer']} | "
            f"{c['form']}({', '.join(c['support_names'])}) | {c['mu']:.2f} | "
            f"{c['Pi']:.2f} | {c['pi']:.2f} | {c['delta']:.4f} "
            f"[{c['ci_low']:.4f}, {c['ci_high']:.4f}] | {c['coverage_share']} |")
    lines += ["", "## Periphery (labelled, not certified)",
              "| unit | layer | mu | Pi | pi | reasons |", "|---|---|---|---|---|---|"]
    for c in cert_doc["concepts"]:
        if c["label"] == "CORE" or c.get("uid_main") is None:
            continue
        lines.append(
            f"| {c['uid_main']} | {c.get('layer', '?')} | "
            f"{c.get('mu', float('nan')):.2f} | {c['Pi']:.2f} | "
            f"{c.get('pi', 0):.2f} | {', '.join(c['reasons'])} |")
    lines += ["", "## Non-claims"]
    lines += [f"- {nc}" for nc in cert_doc["non_claims"]]
    return "\n".join(lines) + "\n"
