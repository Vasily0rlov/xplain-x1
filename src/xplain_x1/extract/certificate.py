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


def _protected_reliance(cert: dict, ds: Dataset) -> dict:
    """Fair-lending / non-discrimination section (SR 11-7 fairness, EU AI Act
    Art 10).  For each declared protected attribute (`ds.meta['protected']`),
    report whether ANY certified function component relies on it.  The headline
    is an FDR-backed *non-reliance* statement when none certify — a claim the
    black-box + post-hoc-attribution stack cannot make with statistical
    guarantees."""
    protected = ds.meta.get("protected", [])
    if not protected:
        return {"declared": [], "note": "no protected attributes declared for "
                "this dataset"}
    comps = cert.get("components", [])
    rows = []
    any_certified = False
    for attr in protected:
        low = attr.lower()
        hits = [c for c in comps
                if any(low in s.lower() for s in c.get("support_names", []))]
        certified = [c for c in hits if c.get("label") == "CORE"]
        max_share = round(max([c.get("share_main", 0.0) for c in hits], default=0.0), 4)
        if certified:
            any_certified = True
        rows.append({
            "attribute": attr,
            "appears_in_decomposition": bool(hits),
            "certified_component": bool(certified),
            "max_share": max_share,
            "status": ("RELIES (certified component)" if certified
                       else "labelled periphery only" if hits
                       else "absent from decomposition")})
    verdict = ("The model's certified decision structure does NOT rely on any "
               "declared protected attribute: none appears in a certified "
               "(CORE) function component."
               if not any_certified else
               "WARNING: at least one protected attribute appears in a certified "
               "component — review required.")
    proxy = _proxy_reliance(cert, ds, protected)
    return {"declared": protected, "rows": rows,
            "relies_on_protected": any_certified, "verdict": verdict,
            "proxy": proxy,
            "basis": "certified Layer-F components under the declared measure; "
                     "FDR-bounded (E[V]).  Absence here is a certified "
                     "non-reliance statement, not a post-hoc approximation."}


def _proxy_reliance(cert: dict, ds: Dataset, protected: list) -> dict:
    """Indirect-reliance screen: could a CERTIFIED driver act as a PROXY for a
    protected attribute?  For each certified-component driver feature, report the
    strongest rank correlation (|Spearman ρ| on the data) to any protected-
    attribute column.  Direct non-reliance does not by itself rule out proxy
    encoding (the standard fair-lending follow-up); this quantifies it."""
    from scipy.stats import spearmanr
    fn = ds.feature_names
    idx = {n: i for i, n in enumerate(fn)}
    prot_cols = {p: [i for i, n in enumerate(fn) if p.lower() in n.lower()]
                 for p in protected}
    core = [c for c in cert.get("components", []) if c.get("label") == "CORE"]
    drivers = sorted({s for c in core for s in c.get("support_names", [])})
    NOTABLE, STRONG = 0.3, 0.5
    rows, overall = [], 0.0
    for drv in drivers:
        di = idx.get(drv)
        if di is None:
            continue
        best_r, best_p = 0.0, None
        for p, cols in prot_cols.items():
            for ci in cols:
                r = spearmanr(ds.X[:, di], ds.X[:, ci]).correlation
                r = abs(float(r)) if r == r else 0.0     # NaN guard
                if r > best_r:
                    best_r, best_p = r, p
        overall = max(overall, best_r)
        rows.append({"driver": drv, "max_rho": round(best_r, 3),
                     "nearest_protected": best_p,
                     "flag": ("strong" if best_r >= STRONG
                              else "notable" if best_r >= NOTABLE else "weak")})
    flag = ("strong" if overall >= STRONG else "notable" if overall >= NOTABLE
            else "none")
    verdict = (f"No certified driver is a strong or notable proxy for a protected "
               f"attribute (max |ρ| = {overall:.3f}, below the 0.30 screen)."
               if flag == "none" else
               f"At least one certified driver correlates with a protected "
               f"attribute at |ρ| = {overall:.3f} ({flag}) — proxy review required.")
    return {"drivers": rows, "max_rho": round(overall, 3), "flag": flag,
            "verdict": verdict,
            "method": "|Spearman ρ| of each certified driver vs each protected-"
                      "attribute column on the full dataset; screen thresholds "
                      "notable 0.30 / strong 0.50."}


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
        "routes": {                          # P6: the Rashomon-invariant layer
            "feature_groups": cert.get("groups", []),
            "rows": cert.get("routes", []),
            "n_core_routes": cert.get("n_core_routes", 0),
            "ev_bound_routes": cert.get("ev_bound_routes"),
            "note": ("Routes are certified at the collinearity-group level "
                     "(the level stable across retrainings); unit members are "
                     "one carving from an explicit equivalence class."),
        },
        "function_decomposition": {          # P7 Layer F: the certified claims
            "note": ("Purified weighted-fANOVA components of the learned "
                     "function under the DECLARED empirical training measure "
                     "(unique given the measure; carving-invariant).  Shares "
                     "are out-of-sample covariances on validation."),
            "components": cert.get("components", []),
            "n_core": cert.get("n_core_components", 0),
            "group_shares": cert.get("group_shares", {}),
            "ev_bound": cert.get("ev_bound_components"),
            "recon_r2": cert.get("fanova_r2"),
        },
        "portfolio_reliance": cert.get("reliance", []),   # P7 Layer R
        "protected_attribute_reliance": _protected_reliance(cert, ds),
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
    fd = cert_doc.get("function_decomposition", {})
    if fd.get("components"):
        lines += ["", "## Certified function components (Layer F — unique under the declared measure)",
                  f"reconstruction R² {fd.get('recon_r2')} · E[V] <= "
                  f"{fd.get('ev_bound'):.3g}" if fd.get("ev_bound") is not None else "",
                  "| component | groups | share | Pi | pi | label |",
                  "|---|---|---|---|---|---|"]
        for c in sorted(fd["components"], key=lambda c: -c["share_main"]):
            lines.append(
                f"| {' × '.join(c['support_names'])} | "
                f"{', '.join(c['group_names'])} | {c['share_main']:.3f} | "
                f"{c['Pi']:.2f} | {c['pi']:.2f} | {c['label']}"
                + (f" ({', '.join(c['reasons'])})" if c["reasons"] else "") + " |")
    par = cert_doc.get("protected_attribute_reliance", {})
    if par.get("declared"):
        lines += ["", "## Protected-attribute non-reliance (fair lending — "
                  "SR 11-7 fairness / EU AI Act Art 10)",
                  f"**{par['verdict']}**", "",
                  "| protected attribute | in decomposition? | certified? | max share | status |",
                  "|---|---|---|---|---|"]
        for r in par["rows"]:
            lines.append(
                f"| {r['attribute']} | {'yes' if r['appears_in_decomposition'] else 'no'} | "
                f"{'YES' if r['certified_component'] else 'no'} | {r['max_share']:.4f} | "
                f"{r['status']} |")
        lines += ["", f"*Basis: {par['basis']}*"]
        px = par.get("proxy", {})
        if px.get("drivers"):
            lines += ["", "### Proxy screen (indirect reliance)",
                      f"**{px['verdict']}**", "",
                      "| certified driver | nearest protected attr | max \\|ρ\\| | flag |",
                      "|---|---|---|---|"]
            for r in sorted(px["drivers"], key=lambda r: -r["max_rho"]):
                lines.append(f"| {r['driver']} | {r['nearest_protected']} | "
                             f"{r['max_rho']:.3f} | {r['flag']} |")
            lines += ["", f"*Method: {px['method']}*"]
    rel = cert_doc.get("portfolio_reliance", [])
    if rel:
        lines += ["", "## Portfolio reliance (Layer R — every restart relies on)",
                  "| group | min reliance | max |", "|---|---|---|"]
        for r in sorted(rel, key=lambda r: -r["min_reliance"])[:10]:
            lines.append(f"| {r['group']} | {r['min_reliance']} | {r['max_reliance']} |")
    rts = cert_doc.get("routes", {})
    if rts.get("rows"):
        lines += ["", "## Certified routes (group level — stable across retrainings)",
                  f"feature groups: {len(rts['feature_groups'])} · "
                  f"E[V] <= {rts['ev_bound_routes']:.3g} at route universe",
                  "| route | groups | Pi | pi | delta [95% CI] | members (this model) | variants |",
                  "|---|---|---|---|---|---|---|"]
        for r in rts["rows"]:
            if r["label"] != "CORE":
                continue
            lines.append(
                f"| {r['rid']} | {', '.join(r['support_names'])} | "
                f"{r['Pi']:.2f} | {r['pi']:.2f} | {r['delta']:.4f} "
                f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | "
                f"{', '.join(r['members_main'])} | {len(r['variants'])} |")
        lines += ["", "### Route periphery (labelled)",
                  "| route | groups | Pi | pi | reasons |", "|---|---|---|---|---|"]
        for r in rts["rows"]:
            if r["label"] == "CORE":
                continue
            lines.append(f"| {r['rid']} | {', '.join(r['support_names'])} | "
                         f"{r['Pi']:.2f} | {r.get('pi', 0):.2f} | "
                         f"{', '.join(r['reasons'])} |")
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
