# Multiplicity, DAG inspection, and certification — literature scan and options

**Question (owner, 2026-08-22):** we want a way forward that (1) recognises
multiplicity, (2) still produces an inspectable DAG, and (3) certifies the
result.  Is the current proposal (route-level certification + companion-split
refinement) still the best way forward?

## 1. What the field knows — five convergent strands

**(a) Rashomon-set interpretability (Rudin school).**  Don't pick one model and
pretend uniqueness; characterise the set of near-optimal models and report
what's invariant.  Model Class Reliance gives upper/lower bounds on how much
*any* good model relies on a variable; Variable Importance Clouds visualise the
distribution over the set.  Set-level statements ("every good model relies on
SIZE by ≥ x") are exactly regulator-grade claims and are quantified over a
model portfolio — our restart set is a Monte-Carlo sample of it.
[VIC (Dong & Rudin)](https://arxiv.org/pdf/1901.03209) ·
[MCR (Fisher, Rudin, Dominici)](https://deepai.org/publication/model-class-reliance-variable-importance-measures-for-any-machine-learning-model-class-from-the-rashomon-perspective) ·
[10 Grand Challenges](https://arxiv.org/pdf/2103.11251) ·
[Shapley VIC](https://www.sciencedirect.com/science/article/pii/S2666389922000253)

**(b) An impossibility theorem for what we were trying to do.**  2026:
**no feature ranking can be simultaneously faithful, stable, and complete under
collinearity** — for collinear pairs the ranking "reduces to a coin flip".
Recommended escapes: report ties/sets (grouping), or ensemble-average with
explicit tie reporting.  Directly licences our group/route move and warns that
SHAP-style per-feature audits are provably unreliable on collinear data.
[Attribution Impossibility](https://arxiv.org/pdf/2605.21492)

**(c) Mechanistic interpretability hit the identical wall in 2026.**
"**Unstable Features, Reproducible Subspaces**": SAE features across seeds are
individually non-reproducible (30–40% overlap) but **concentrate in
reproducible lower-rank subspaces** — "seed variation reflects basis ambiguity
within shared activation regions".  Their remedies: per-feature stability
metrics, subspace-level analysis, pooling across seeds; a companion line does
Procrustes-conditioned joint training to align frames across seeds.  Our
unit-level Π ≤ 0.5 with certifiable group routes is the same phenomenon and the
same resolution, found independently.
[Unstable Features, Reproducible Subspaces](https://arxiv.org/pdf/2606.12138) ·
[Procrustes cross-seed SAEs](https://arxiv.org/html/2607.08499) ·
[Archetypal-SAE stability critique](https://arxiv.org/pdf/2606.02061)

**(d) Causal discovery: how to certify a DAG under equivalence.**  The mature
precedent for "certified DAG despite multiplicity": report the **equivalence
class** (CPDAG/MEC — one partially-directed graph standing for all DAGs the
data cannot distinguish) plus **bootstrap edge frequencies** (edges scored by
the fraction of resamples in which they appear; ambiguous orientations
labelled).  DAG aggregation (DAGBag etc.) minimises structural distance over an
ensemble.  Maps one-to-one onto our restarts-portfolio + per-route Π.
[Stable Causal Discovery via DAG Aggregation](https://arxiv.org/html/2605.18633v1) ·
[Confidence sets for causal orderings](https://arxiv.org/html/2305.14506)

**(e) Functional ANOVA: a UNIQUE decomposition of the function itself.**  The
keystone strand.  Our runs agree on the *function* (fidelity ratios match);
they disagree on the internal factorisation.  The fANOVA literature addresses
exactly this: effects can be freely moved between mains and interactions
without changing predictions — "permitting contradictory models to represent
the same function" — and **purification** projects any fitted function onto the
canonical decomposition where each component carries only variance
unattributable to any smaller support ("pure interaction effects").  Hooker's
**generalized (weighted) fANOVA** extends uniqueness to **dependent/correlated
inputs** (weighting by the joint density), and 2026 work gives **closed forms**
(continuous via Riesz bases; categorical via discrete Fourier), removing the
old computational barrier.  A purified component is a *statistical estimand of
f under a declared measure* — bootstrap CIs and selection-frequency machinery
apply directly.
[Purification (Lengerich, Tan, Chang, Hooker, Caruana)](https://arxiv.org/abs/1911.04974) ·
[Generalized fANOVA in closed form](https://arxiv.org/abs/2605.18422) ·
[Exact fANOVA for categorical inputs](https://arxiv.org/html/2603.02673v2) ·
[Hooker 2007 (weighted fANOVA)](https://www.tandfonline.com/doi/abs/10.1198/106186007X237892)

**Programme resonance:** v2/v3 already used "functional-ANOVA purification"
(`data/DATASETS.md` cites it for Titanic/Adult), and POSITIONING's "invariant
core up to gauge" is precisely "unique at function level, gauge at unit level".

## 2. Options against the three requirements

| option | recognises multiplicity | inspectable DAG | certifiable | assessment |
|---|---|---|---|---|
| **1. Routes + companion-split** (current) | ✓ (equivalence classes) | ✓ (2-layer DAG) | ✓ but granularity ad hoc (bike: anchor-merge; split rule = another heuristic) | working, incomplete |
| **2. Weighted-fANOVA component certification** | ✓ by construction (components are carving-invariant) | ✓ (annotate DAG nodes with the components they carry) | **✓ unique estimand + CIs + FDR over a clean support universe** | the keystone; moderate build |
| 3. Rashomon MCR/VIC ranges | ✓ (set-level) | ✗ alone (no structure) | ✓ (bounds) | certificate rows, not a structure |
| 4. Consensus/portfolio DAG (CPDAG-style) | ✓ | ✓ (edge frequencies) | partial (no per-claim null) | subsumed by 1 + Π |
| 5. Procrustes cross-seed alignment | ✓ (analysis-side, legitimate — unlike training-side canonicalisation, post-hoc alignment doesn't corrupt Π) | ✓ | ≈ subspace identity = our routes | equivalent to 1 at best |

## 3. Analysis: why option 2 is the missing keystone

The E6 arc showed structural (carving-level) certification works where an
anchor exists but leaves two problems: **granularity** (bike's hour-route
swallowed hour×temp and hour×workingday — companion-split would patch this
with one more heuristic) and **adult** (load-bearing structure never stabilises
at any carving level we defined).  Both dissolve at function level:

- `hour×temp` and `hour×workingday` are **distinct pure components of f** —
  separate estimands with separate effect sizes; no merge ambiguity exists,
  because purification is unique.  The granularity question isn't patched, it
  *disappears*.
- Adult's question becomes empirical and well-posed: is f's component structure
  stable across restarts even though carvings aren't?  (Expected yes — the
  functions agree; that is what fid-ratio agreement means.)
- The impossibility theorem (b) is respected: we never output a per-feature
  ranking under collinearity; we output components under a *declared measure*
  (the training distribution), with grouping (routes) on top.
- Certification becomes cleaner than anything we have: components are
  estimands of a fixed function ⇒ bootstrap CIs; component *selection* across
  restarts/CPSS uses the same Π/π machinery on a support universe that is the
  same object the E[V] bound already counts.

**The three layers compose (recommended architecture):**

1. **Layer F (function, certified, unique):** purified weighted-fANOVA
   components of each restart's f; certify components stable across the
   portfolio (Π/π/CI/E[V]).  The *certified claims* live here.
2. **Layer S (structure, inspectable):** the delivered model's DAG exactly as
   today — legible units, route layer, equivalence-class labels.  Each
   unit/route annotated with the certified components it carries (computable:
   ablate unit → which components of f change).
3. **Layer R (set-level rows in the certificate):** MCR-style range statements
   over the restart portfolio ("every restart relies on SIZE ≥ x").

Multiplicity is then not merely "recognised" — it is *explained*: the DAG shows
THIS model's carving; the certificate proves WHICH function-level structure all
carvings implement; the equivalence classes bridge the two.

## 4. Is the current proposal still best?

**As the sole next step — no.**  Companion-split is a reasonable structural
heuristic, but it patches at carving level what fANOVA resolves at function
level, and it would leave adult unexplained.  **As a component — yes**: Layer S
keeps the current machinery (companion-split becomes optional polish, likely
unnecessary once Layer F answers the granularity question).

**Recommended path:** implement Layer F (purified component extraction on the
model's f — the model is piecewise-linear, so Lengerich-style purification on a
discretised grid or the 2026 closed forms apply; components up to arity
`F_max`), validate on synthetics (planted components are known exactly), then
re-run the real four; add Layer R rows to the certificate.  Then freeze.

## 5. Papers to supply if deeper access is needed (full texts)

Fetched at abstract/summary level so far; full texts would help implementation
detail, in priority order:

1. https://arxiv.org/pdf/2605.18422 — Generalized fANOVA in closed form (implementation basis)
2. https://arxiv.org/pdf/1911.04974 — Purification algorithm (piecewise functions)
3. https://arxiv.org/pdf/2606.12138 — Unstable features / reproducible subspaces (metrics)
4. https://arxiv.org/pdf/2605.21492 — Attribution impossibility (formal statements)
5. https://www.tandfonline.com/doi/abs/10.1198/106186007X237892 — Hooker 2007 weighted fANOVA (likely paywalled)
6. https://arxiv.org/pdf/1901.03209 — Variable Importance Clouds
7. https://arxiv.org/html/2603.02673v2 — Exact fANOVA for categorical inputs
8. https://arxiv.org/html/2605.18633v1 — Stable causal discovery via DAG aggregation
