# XPLAIN-x1 — Method Specification

**Status:** draft for review · **Audience:** researchers · **Basis:** `docs/POSITIONING.md`
(2026-08) · **Scope:** the beachhead — supervised learning on tabular/structured data with
monosemantic columns (`docs/DATASETS.md`), realised on standard deep MLPs. The Transformer/SLM
track is out of scope here (POSITIONING §7, stage 3).

> **Provenance caveat** (inherited from POSITIONING): citations were gathered from abstracts
> and search summaries; verify exact claims against the papers before external use.

This document is a *clean-slate* specification: it derives the method from the positioning
argument alone and defines its own terminology. Prior programme iterations (v1–v4) are
referenced **only as external result anchors** in §9, never as design inputs.

---

## 1. Problem statement and goal

Post-hoc explanation of a black box is approximate, local, and — for regulated decisions —
not faithful enough to audit (Rudin 2019, [arXiv:1811.10154](https://arxiv.org/abs/1811.10154)).
The goal is a model whose **own decision logic is inspectable**: a standard deep network that
has *converged*, under added training pressures, into a state where

1. most units are **monosemantic** — each is one coherent, low-arity thing;
2. the network is **minimal** — fewest units, edges, and layers that preserve fidelity;
3. its depth is **honest** — every layer of claimed hierarchy is earned by the data;
4. its concepts are **certified** — stability and false-discovery evidence re-earn the
   correctness that inspectability alone does not provide; and
5. the **polysemantic periphery** that remains is explicitly labelled, not hidden.

The deliverable per trained model is (a) the model, (b) an extracted **concept DAG** an expert
can traverse, and (c) an **audit certificate** with statistical guarantees, mappable to
model-risk-management and EU AI Act Article 9 validation requirements.

## 2. Foundational commitments (from the positioning argument)

Each commitment below is load-bearing; the method in §4 is their joint consequence.

- **C1 — Target monosemanticity, not disentanglement.** Disentanglement (unit ≈ ground-truth
  factor) is relational, needs an external reference, and is unidentifiable in general.
  Monosemanticity (unit ≈ *one coherent* thing) is intrinsic, identifiability-immune, and
  directly measurable. We optimise and certify the latter only.
- **C2 — An inductive bias is mandatory.** Prediction loss alone carries no gradient toward
  the clean basin: clean and entangled minima are equal-loss, and a plain net recovers the
  information as a *subspace* while entangling it at the *unit* level. Convergence pressure
  must be added to training.
- **C3 — The pressure is sparsity/selectivity plus depth — not statistical independence.**
  Decorrelation penalties are measured dead ends: uncorrelated units are still linear mixes.
- **C4 — Minimality only rides on top of the monosemanticity pressure.** Superposition *is*
  the minimal-width encoding; compression alone drives *more* entanglement. Minimality
  (MDL/Occam) is applied as a secondary pressure, never the primary one. The ordering is
  normative for the method.
- **C5 — Depth ≥ 2 is non-negotiable for composition.** A single-hidden-layer network cannot
  represent a disentangled product (a linear readout adds factor-units, it cannot multiply
  them). Conversely, depth that composition does not warrant is theatre and must be dissolved
  (**depth-honesty**).
- **C6 — Uniqueness is level-dependent; the periphery is real.** A domain has an invariant
  core and a multiplicitous periphery (collinearity, weak signal, fit ceiling). The periphery
  has no unique clean answer to find; a small, *labelled* polysemantic residual is the honest
  target, not a compromise.
- **C7 — Monosemanticity buys inspectability, not correctness.** A clean unit can be a
  dataset shortcut. Correctness is re-earned separately, by stability across
  restarts/resamples and by false-discovery control.
- **C8 — The bias is expected to be ~accuracy-neutral.** Measured on synthetic ground, the
  structured solution fits as well as or better than the entangled one; the pressures select
  among *equal-accuracy* basins. This is a monitored hypothesis with a budget (§7), not an
  assumption.

## 3. Formal setting and definitions

### 3.1 Setting

Supervised task with data `(x, y)`, `x ∈ R^d`, where each input column is **monosemantic**
(one column = one human concept; see `docs/DATASETS.md` for the data requirements — no raw
pixels, no PCA-rotated features, high-cardinality nominals decomposed into binary
predicates). Model class: standard MLP `f_θ` with `L` hidden layers of widths `m_1..m_L`,
ReLU activations, linear readout; classification via softmax/logistic, regression via
identity. **Fidelity** is held-out pseudo-R²: `Fid = 1 − LL_model / LL_null` (regression:
ordinary R²). A **reference model** `f_ref` (unconstrained MLP and/or gradient-boosted trees,
whichever is better) fixes the task's attainable ceiling `Fid_ref` and accuracy `Acc_ref`.

### 3.2 Units, supports, and local legibility

For hidden unit `u` in layer `ℓ`, its **parents** are the layer-`ℓ−1` units (layer 1: the
input features) with incoming edges. Define:

- **Effective fan-in** `ef(u)`: the number of parents whose contribution to `u`'s
  pre-activation variance exceeds a threshold `ε_edge` (contribution = |w| × parent
  activation std, normalised per unit).
- **Monosemanticity** `μ(u) ∈ [0,1]`: the held-out R² of the best surrogate `g_S` predicting `u`'s
  activation from a parent subset `S`, maximised over supports `|S| ≤ F_max` (default
  `F_max = 3`), where `g_S` is drawn from a fixed simple class (1-D: monotone spline; 2–3-D:
  shallow tree or degree-2 polynomial). `μ(u)` measures whether the unit is *effectively a
  low-arity function of few parents* — its **local legibility**. The maximising `S` is the
  unit's **support** `S(u)` and the fitted `g` its **surrogate form**.

In plain terms: a monosemantic unit **responds to the same few variables in the same simple
way** — *not* necessarily "activates on the same (range of) inputs". The feature-detector
reading is deliberately not the definition (a graded, always-active unit can be maximally
legible; a narrow detector over many features is not — see `docs/ALTERNATIVE-OPTIONS.md` A1).

A unit is **locally monosemantic** at level `μ_min` if `μ(u) ≥ μ_min`. Global legibility is
recursive: a decision route is legible iff every unit along it is locally monosemantic —
this is what the DAG certification (§5) establishes route by route. Explainability in this
programme concretely means: **decisions traceable through a minimal number of hops along
unambiguous routes** (POSITIONING §3).

### 3.3 Gauge: what is free and what must be trained

Two distinct freedoms must not be conflated:

- **Architecture-side gauge (exact).** The function-preserving reparameterisations of a ReLU
  MLP layer are generated by unit **permutations** and per-unit **positive rescalings** (with
  inverse scaling of outgoing weights). General invertible mixes of post-activations are *not*
  absorbable (`σ(AWx) ≠ Aσ(Wx)`).
- **Analysis-side gauge (broad).** Any invertible linear transform of a layer's
  post-activations preserves the *information* it carries; probes and dictionary methods
  exploit this freedom, which is why "the concepts are linearly present" and "the units are
  clean" are different facts.

**Consequence (the precise form of the positioning's gauge argument):** apparent entanglement
that is mere basis-mixing *is* removable at zero accuracy cost — but not by rotating a trained
model, because the architecture cannot represent the rotation. It is removed by **steering
training into an equal-loss basin whose native unit basis is the sparse one**. The training
pressures of §4 are exactly that steering; the exact gauge (permutation/scaling) is fixed
post-hoc as mere canonicalisation. What steering cannot remove is **true superposition**
(more concepts than units) — the release valve for that is capacity growth (§4.3).

### 3.4 Depth-honesty

Layer `ℓ` is **earned** iff dissolving it — merging it with an adjacent layer via a brief
distillation into a single layer — costs more than `ε_depth` in held-out fidelity. The
model's **honest depth** is the depth at which every remaining layer is earned. The
compositional order of the data determines the depth actually needed (order 1 ⇒ 1 hidden
layer suffices; order 2–3 ⇒ 2; see the order→depth mapping in `docs/DATASETS.md`); the
method must *discover* this constructively, not assume it (§4.3), and must certify shallow
when the data's interactions add no held-out value.

### 3.5 Minimality

The minimality objective is MDL-flavoured: prefer the model minimising
`description length of the extracted DAG (units, edges, surrogate forms) + residual`.
Operationally it is enforced by the growth/prune controller (§4.3) under the C4 ordering:
every accepted structural action must preserve fidelity within budget, and among
fidelity-preserving states the smaller one wins. Minimality is never traded against monosemanticity.

### 3.6 Stability, identity, and certification objects

- **Concept identity across runs.** The certified object is not "unit 17" but a concept
  `c = (S, form)` — a support plus surrogate form. Units from independent runs are matched
  by **signature**: cosine similarity of their input-attribution profiles (for layer-1
  units, the normalised incoming weight vector; for deeper units, the attribution of their
  activation to input features) *and* activation correlation on a shared probe set. Both
  must exceed `τ_match` for two units to instantiate the same concept.
- **Stability** `Π(c)`: the fraction of independent restarts in which concept `c` re-emerges
  (some unit matches above `τ_match`).
- **Selection frequency and false-discovery control.** Complementary-pairs stability
  selection (CPSS; Meinshausen–Bühlmann,
  [stability selection with error control](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4464883/)):
  the pipeline is rerun on `B` complementary subsample pairs; a concept's selection
  frequency `π(c)` is its fraction of appearances. With selection threshold `π_thr`, the
  expected number of false selections is bounded by `E[V] ≤ q² / ((2·π_thr − 1)·p)`, where
  `q` is the average number of concepts selected per subsample and `p` the size of the
  **candidate universe**. The universe is defined at the *structure level* — all supports of
  arity ≤ `F_max` over the available parents — not at the raw-feature level (a raw-feature
  universe makes the bound vacuous; the structure-level universe is the one the method can
  actually output). The exchangeability assumption behind the bound is reported in the
  certificate, not silently assumed away.
- **Incremental-fidelity (reality) test.** A concept is **real** only if ablating its unit
  (zeroing outgoing edges, no retraining) costs `Δ(c) ≥ δ_min` held-out fidelity with a
  bootstrap confidence interval excluding zero. This is the operational null: a false
  discovery is a concept whose removal costs nothing.
- **Uniqueness levels.** Matched concepts are clustered across runs. A DAG position filled
  by *different* supports in different runs (mutually exclusive alternatives) is
  **multiplicitous** — the Rashomon signature of the periphery (C6). Uniqueness is asserted
  only where the same support re-emerges.

### 3.7 Core and periphery

A concept is **CORE** iff `μ ≥ μ_min`, `Π ≥ Π_min`, `π ≥ π_thr`, and the reality test
passes. Everything else that survives in the final model is **PERIPHERY**, labelled with its
failure reasons (polysemantic / unstable / multiplicitous / below-Δ). The certificate never claims
the periphery; the DAG never hides it.

## 4. The method

Four phases: **converge → canonicalise → certify → extract**. Phases 1–2 produce the model;
3–4 produce the audit artefacts.

### 4.1 Phase A — Convergence training (the pressures)

Total loss: `L = L_task + λ_act·L_selectivity + λ_fanin·L_fanin (+ standard weight decay)`.

- `L_selectivity`: L1 on post-activations, normalised per layer (each unit's activations
  scaled by their std before the penalty, so scale gauge cannot cheat it). Drives units to
  fire selectively — the sparse-gauge steering of §3.3.
- `L_fanin`: group lasso (L2,1) over each unit's incoming weight vector groups — the direct
  pressure toward low effective fan-in and hence low-arity local functions; weight-level
  sparsity is the pressure the literature shows yields interpretable circuits.
- **Deliberately absent:** any decorrelation/independence term (C3). Duplicate-concept
  redundancy is handled structurally at prune time (merge units with near-identical
  signatures), not by a loss.

Pressures are annealed in (task loss settles first, pressures ramp to full strength) and the
model is periodically **audited**: monosemanticity, effective fan-in, per-unit contribution, and
fidelity are measured on validation data (protocols of §3.2).

### 4.2 Phase A′ — Gauge canonicalisation (exact, zero-cost)

After each settle: normalise per-unit incoming weight norms to 1 (absorbing scale into
outgoing weights), canonicalise signs where the activation permits, and sort units by
contribution. Purely cosmetic by §3.3 — bitwise fidelity-preserving — but it makes
signatures, matching, and diffs across runs well-defined.

### 4.3 Phase B — Constructive growth and pruning (minimal, depth-honest)

The controller runs a settle → audit → act loop from a deliberately small start
(**1 hidden layer, narrow**), so that any depth or width in the final model was *demanded*
by evidence — depth-honesty and minimality by construction, honouring the C4 ordering:

- **Grow width** when task fidelity plateaus below the reference ceiling *and* audit shows
  monosemanticity stalling (units saturated/polysemantic under pressure) — the signature of
  true superposition; add or split units (split the least-monosemantic unit, perturbed).
- **Grow depth** when width growth fails to close the fidelity gap — the signature of
  unexpressed composition (C5); insert one layer.
- **Prune** continuously: edges below `ε_edge` contribution, dead/negligible units; **merge**
  near-duplicate units (signature similarity above `τ_match`).
- **Dissolve depth** whenever the §3.4 dissolution test shows a layer unearned.
- **Accept/reject**: growth is accepted only if it improves held-out fidelity by ≥ `δ_grow`;
  pruning/dissolution is accepted only if it costs ≤ `ε_prune`. Terminate when no action is
  accepted — the model has settled into its minimal, depth-honest state.

Accuracy is monitored against `f_ref` throughout under the C8 budget.

### 4.4 Phase C — Certification

1. **Restarts:** run Phases A–B `R` times from independent seeds; match concepts (§3.6);
   compute `Π(c)`.
2. **CPSS:** run the pipeline on `B` complementary subsample pairs; compute `π(c)` and the
   `E[V]` bound over the structure-level universe.
3. **Reality tests:** ablation `Δ(c)` with bootstrap CIs, on held-out data.
4. **Uniqueness analysis:** cluster matches; mark multiplicitous positions.
5. **Label:** CORE / PERIPHERY per §3.7.

### 4.5 Phase D — Extraction and the audit certificate

- **Concept DAG:** nodes = input features, surviving units (with support, surrogate form,
  plain-language template, monosemanticity, stability, frequency, coverage share, CORE/PERIPHERY
  label), and outputs; edges = connections above `ε_edge`. Coverage share = the concept's
  ablation-Δ as a fraction of total. The expert traverses this DAG to assess legibility —
  the expert's assessment is the validation of the interpretability *soft target*
  (POSITIONING §3 note); the method certifies stability and reality, not semantics.
- **Audit certificate:** a machine-readable + human-readable artefact containing: per-concept
  rows (support, form, `μ`, `Π`, `π`, `Δ` with CI, label); model rows (fidelity and accuracy
  vs `f_ref`, concept tax, honest-depth statement with dissolution evidence, periphery
  fraction and its labels, `E[V]` bound and its assumptions); and full provenance (data
  hash, splits, seeds, config hash, code version). Its sections map onto SR 11-7-style model
  risk validation and [EU AI Act Article 9](https://arxiv.org/html/2512.13907v3) headings.
  The certificate asserts *nothing* about the periphery except that it exists, where it is,
  and why it failed the bars.

## 5. What the method claims — and does not

**Claims (certified):** the CORE concepts are stable re-emerging structures with bounded
expected false discoveries, each carrying measurable held-out contribution; the model's
depth is honest at `ε_depth`; the decision routes through CORE are locally legible at
`μ_min` per hop.

**Non-claims:** that CORE concepts are *true mechanisms* of the world (C7 — stability and
reality tests raise the bar, they do not prove causality); that the periphery is
interpretable; that unit semantics match domain semantics without expert review; that the
method applies beyond data meeting the `DATASETS.md` requirements.

## 6. Relation to prior art

The neighbourhood is occupied (full map: POSITIONING §5): post-hoc monosemanticity via
sparse dictionaries ([Anthropic 2023](https://transformer-circuits.pub/2023/monosemantic-features));
MDL-selected decompositions ([MDL-SAEs, 2024](https://arxiv.org/abs/2410.11179)); sparsity
as the identifiability-restoring bias ([Lachapelle et al., 2023](https://arxiv.org/abs/2211.14666));
by-design interpretability ([Rudin 2019](https://arxiv.org/abs/1811.10154);
[CRATE, 2023](https://arxiv.org/abs/2311.13110); [XNNTab, 2025](https://arxiv.org/abs/2512.13442));
feature circuits across layers ([Marks et al., 2025](https://arxiv.org/html/2502.03032v3));
sparsity-enhanced standard architectures
([weight-sparse transformers, 2025](https://arxiv.org/html/2511.13653v1)).

XPLAIN-x1 does not claim any single pillar as novel. Its position is the *package* none of
the above ships (POSITIONING §7): **by-design + constructively-grown-to-minimal +
depth-honest + statistically certified, obtained by enhancing a standard MLP, for bounded
regulated decisions, delivering a false-discovery-bounded audit certificate.** The moat is
certification + audit-artefact packaging + honest depth on small-n expert-auditable data —
not the architecture.

## 7. Hypotheses, bars, and kill criteria

Pre-registered before the implementation runs; evaluated on the `DATASETS.md` ladder
(interpretability core: Zoo, Tic-Tac-Toe, Mushroom; continuous sanity: Iris, Wine, Seeds;
mixed interactions: Titanic, Adult; scale: Covertype subset, Bike Sharing, Dry Bean) plus
the synthetic control suite (Solution Spec §5) whose ground-truth structure is known.

| id | hypothesis | bar |
|---|---|---|
| H-X1-1 convergence | pressures produce a largely-monosemantic model where a plain MLP does not | CORE coverage share ≥ 0.7 of ablation mass on ladder datasets; median CORE monosemanticity ≥ 0.8; plain-MLP control stays far below |
| H-X1-2 honest flatness | no invented structure | 0 certified depth-≥2 concepts on additive and pure-noise controls, all restarts |
| H-X1-3 compositional recovery | true compositions found and certified | planted order-2/3 concepts in synthetics recovered as CORE (matched support) at ≥ 2/3 rate above the power floor; Tic-Tac-Toe lines and Bike `hour×temp` certified depth-2 |
| H-X1-4 concept tax | interpretability ~free (C8) | held-out accuracy within 2% relative of `f_ref` (median across ladder); expectation ≈ 0 |
| H-X1-5 stability | the core re-emerges | `Π ≥ 0.7` for CORE across `R` restarts |
| H-X1-6 certified honesty | the labels can be trusted | every certified concept passes the reality test; `E[V]` bound reported non-vacuously at the structure-level universe; periphery labelled, never certified |

**Kill criteria:** K1 — median concept tax > 10%: the accuracy-neutrality premise fails; the
method as positioned is dead. K2 — any certified expansion on noise/additive controls
(untradeable): the honesty layer is broken. K3 — full-ladder certification battery
wall-clock > 12 h on the reference 64-thread box: the method is not practically auditable.

## 8. Threats to validity

- **The gauge argument's scope** is stated precisely in §3.3; claims of "free" removal of
  entanglement apply only to basin selection during training and exact canonicalisation —
  not to post-hoc rotation. Reviewers should hold the method to that precise form.
- **Matching defines identity.** `Π` and `π` are functions of the signature/matching choices
  (`τ_match`); certificates report them, and the sensitivity of `Π` to `τ_match` is part of
  the validation battery.
- **The `E[V]` bound's exchangeability assumption** is a modelling idealisation over a
  learned, adaptive pipeline; the bound is reported with its assumptions, and the reality
  test provides an independent, assumption-light check.
- **Monosemanticity `μ` is surrogate-relative.** `μ` depends on the surrogate class; a
  fixed, simple, pre-registered class keeps it honest but means "monosemantic" =
  "describable in that class".
- **Accuracy-neutrality (C8) is evidence-based on synthetic ground**, not a theorem; H-X1-4
  and K1 keep it falsifiable.
- **Small-n power floors** bound what can honestly be certified (`DATASETS.md`): absence of
  certified depth on under-powered data is the correct output, not a failure — and
  conversely must not be sold as evidence of additivity.

## 9. External benchmark anchors (XPLAIN v4 results)

The immediately preceding programme iteration (**XPLAIN v4**, a different formalism;
`xplain-v4` repo, docs 09–10, 2026-08-17) published dev + confirmatory results that serve as
*outcome* anchors — x1 is benchmarked against these numbers on shared datasets, not against
v4's design:

| anchor | v4 result | x1 bar |
|---|---|---|
| Fidelity ratio vs ceiling | Wine 0.766 · Adult 0.892 · Mushroom 0.995 | meet or exceed per dataset |
| Honest flatness | 15/15 flat on additive/noise, confirmatory | equal: 0 violations (H-X1-2) |
| Concept tax | negative on every instance measured | ≤ 2% median; expect ≤ 0 (H-X1-4) |
| Cross-run invariance of core | Π = 1.0 on Wine core (8/8) | Π ≥ 0.7 CORE, aspire to 1.0 on Wine (H-X1-5) |
| Expert alignment (Wine) | recovered exactly the previously-certified core | recover an equivalent Wine core as CORE |
| Level-dependent uniqueness (Mushroom) | deep carving stable-within-portfolio, not unique — labelled | reproduce: Mushroom alternatives labelled multiplicitous, not certified unique |
| FDR bound usability | raw-feature-universe bound vacuous | non-vacuous at structure-level universe (H-X1-6) |
| Battery cost | 27-min confirmatory battery (30 instances) | K3 budget: full ladder ≤ 12 h on the 64-thread box |

v4's honestly-stated frontier — under-recovery of hard-geometry flat classes on fresh seeds —
is inherited as a known open risk for any recovery-type hypothesis (H-X1-3): dev-phase
recovery rates must be confirmed on untouched seeds before being claimed.

---

*Companion document: `docs/SOLUTION-SPECIFICATION.md` — the implementable technical
specification realising this method.*
