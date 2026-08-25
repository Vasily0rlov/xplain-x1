# XPLAIN-x1 — Findings

Dev-phase log of experiment verdicts and instrument corrections.  Final
hypothesis verdicts land here at P5-3.  Every as-registered verdict is recorded
before any re-score; re-scores await owner ratification.

## P0 (2026-08-21)

- **E0.1 MET** — plain settle reaches the reference ceiling: zoo median val fid
  0.915 vs `Fid_ref` 0.913; wine 0.965 vs 0.940 (both ref winners: MLP).
  🏁 **M1**.
- Fixed en route: model weight init was not derived from the run seed
  (determinism test caught it).

## P1 (2026-08-21)

### Experiment trail (instrument corrections recorded honestly)

- **E1.1 as-registered: NOT MET** (μ gain +0.004 on COMP2).  Diagnosis: COMP2
  has 3 relevant inputs = intrinsic arity ≤ F_max, so *any* useful unit scores
  high μ — OFF baseline 0.91; the fixture cannot discriminate.  Additionally
  effective fan-in ~8 in BOTH arms: the pinned pressures did nothing.
- **E1.3** (λ sweep ×{1..100} on new ADD6 instrument fixture): group-lasso
  (L2,1) fan-in term shown to be a *unit-sparsity* penalty — at high λ it kills
  units (alive 85%) while fan-in stays ~8.  The formula contradicted its stated
  intent.  → replaced with the **Hoyer ratio** per incoming row (the S-§14
  pre-registered alternative for exactly this trigger): scale-invariant pure
  concentration.
- **E1.3b** (Hoyer sweep): better (ef 6.0 at λ=0.1·scale100) but still short.
  Root cause found in **settle**: stopping and best-restore were
  fidelity-driven, but the structural shaping happens in the fidelity-flat
  phase — training stopped exactly when concentration began, then restored a
  quasi-random best-fid epoch.  → settle now plateaus on **total train loss**
  (cumulative criterion) and keeps final weights (best-val restore only as a
  safety guard, margin 0.01).  An `inf`-arithmetic bug in the first version of
  that criterion (NaN comparison → stop at exactly `plateau_evals` epochs) was
  caught by the E0.1 regression re-run and fixed; E0.1 re-verified MET.
- **E1.3c** (final calibration, fixed settle): clean monotone response.
  **Chosen: λ_act 1e-2, λ_fanin 0.03** — μ 0.991, ef 3.0, fid 0.961 vs OFF
  0.958 (no tax), 90% alive.  λ=1.0 reaches μ=1.000/ef=2.0 at −3.5% fid.
- **E1.1b** (ADD6, calibrated λ): μ 0.906→0.987, **ef 8.2→3.8**, fid 0.959
  both arms.  As-registered gain bar (+0.10) NOT MET — *ceiling effect*: the
  properly-converged OFF baseline sits at 0.906, so max attainable gain is
  0.094.  **Re-scored criterion** (impurity halved at ≤2% fid cost): impurity
  0.094→0.013 (−86%) — **MET**.  🏁 **M2** (per re-score).
- **E1.2 as-registered: NOT MET** (alive 67%/79% vs ≥90%).  The bar was
  designed to catch pressure killing *needed* units; measured: fidelity RISES
  (0.83→0.92) while redundant units die and μ≈0.98 — healthy minimality.
  **Re-scored bar** (fid within 2% of OFF ∧ μ non-decreasing ∧ ≥25% alive):
  **MET** on both fixtures.

### Verdict pending owner ratification

The P1 mechanism verdict on current evidence: **pressures work and are
accuracy-neutral** (M-C8 consistent).  Three instrument corrections
(fixture, loss term, settle semantics) and two bar re-scores are recorded
above and await owner sign-off; none was a silent retune.

## P2 (2026-08-21)

- **E2.1 MET as registered** — honest flatness 8/8: ADD stays 1 hidden layer,
  NOISE grows nothing, zero accepted depth anywhere.  🏁 **M3 (part 1)**.
- **E2.2 MET after two documented corrections** — final: COMP2 3/4, COMP3 4/4
  seeds at ceiling with planted-support units at μ ≥ 0.8.  🏁 **M3 (part 2)**.
  1. *Depth-convention error in the bar:* "reaches L=2" mistranslated
     DATASETS' order→depth mapping — order 2–3 needs **one composition layer**,
     which is **1 hidden layer** here (hidden units take 2–3 parents; readout
     sums).  The basin-census "flat cannot multiply" claim concerns a linear
     readout over *factor units*, not a hidden layer of composition units.
     Bar re-based on outcome: at ceiling ∧ planted-support unit present.
  2. *Separation↔discovery tension in λ_fanin:* 0.03 left the `f1(x1)` main
     effect bleeding into product units (COMP2, supports {x1,x2,x3}); 0.1
     separated cleanly but strangled order-3 discovery (COMP3 collapsed to 2
     units, growth always reverted).  Fix: **discovery-gated pressure**
     (scale 0.3 below ceiling, full strength near ceiling) — a direct encoding
     of M-C4 "minimality rides on top of accuracy"; λ_fanin 0.1 adopted
     (within the E1.3c pre-registered constraints; tie-break moved from
     "smallest" to "separates on compositional evidence").
  3. *Constitutional fix:* `delta_stop` 0.01 → 0.02, aligning the controller's
     stop bar with the pressure fid-neutrality budget (E1.3c / H-X1-4) — the
     stop bar cannot be tighter than the distortion the pressure is allowed.
- **E2.3 recorded** (COMP2 at the n=2k power floor): planted-support units
  present in 4/4 seeds (fid 0.87–0.89); whether they *certify* is P3's
  question, as designed.
- Bugs caught by the battery en route: stale-audit unit reference after prune;
  **duplicate unit ids on layer insertion** (per-layer counters collide when
  layers shift) → globally-unique id counter + cycle-guarded support recursion.

## P3 (2026-08-22)

- **E3.1 MET** (after one definition sharpening) — full 6-config suite through
  the complete certification pipeline (R=8 + 40 CPSS runs each):
  ADD 3 CORE all order-1 (no invented interactions); COMP2 clean/noisy planted
  `[x2,x3]` **recovered as CORE**; COMP2 at the n=2k power floor recovers too
  (reported); COMP3 planted order-3 `[x1,x2,x3]` recovered; **NOISE 0 certified
  concepts** (K2 untradeable holds).  🏁 **M4**.
  - *Definition sharpening (first run failed COMP2):* product concepts scored
    Π=1.00, π=0.97, μ=0.98, Δ≈2.2 and fell ONLY to the `multiplicitous` flag —
    support variants flickered between `[1,2]` and `[0,1,2]` (a weak x1 rider
    in some restarts).  Nested variants are boundary jitter of ONE concept,
    not Rashomon alternatives: when variants form an inclusion chain the
    certified support is now the **minimal common variant** (the run-invariant
    core — POSITIONING's own concept); `multiplicitous` is reserved for
    incomparable variants (the Mushroom case it was designed for).
- **E3.2 MET** — CORE set completely insensitive to the matching threshold:
  6 stable concepts at every τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9} on COMP2-noisy.
- **E3.3 MET** (after a bookkeeping correction) — the MB universe is SUPPORTS,
  so q must count distinct supports per subsample, not μ-passing units
  (the product spans 4 units = one support).  Corrected: q=2.95 →
  **E[V] ≤ 0.124** at the structure-level universe (p=175) vs 2.18 at the
  raw-feature universe — non-vacuous, and the structure-level advantage
  (v4 finding) reproduced.
- CI gates wired (`pytest -m gates`): COMP2 recovery mini-battery + NOISE
  zero-certification.

## P4 (2026-08-22) — E4.1 escalated to owner

Artefact machinery complete (DAG + certificate + provenance; golden tests).
**E4.1 (Wine end-to-end) NOT MET — this is a results finding, not a code
failure:**

- The model is *better than the reference ceiling* (fid 0.956 vs ref 0.940,
  ratio **1.017** — v4's anchor was 0.766) and every live unit is μ-legible
  (0.90–0.98), sitting on the known Wine core (flavanoids, od280,
  colour intensity, alcohol) with large real ablation effects (Δ up to 0.34,
  CI > 0).
- But **0 concepts certify**: across restarts the 13 collinear chemistry
  features get carved into *different* legible 3-feature mixes (Π 0.38–0.88 —
  Rashomon under collinearity), and CPSS on 53-row half-samples has no
  statistical power (π ≤ 0.6) at n=178.  A cleanup-pressure probe (λ_fanin
  0.3) lifted the flavanoids complex to Π=0.88 but π stayed at 0.55 and the
  variants remain incomparable.  The honest-labelling machinery is working
  exactly as designed; Wine at this n sits below the method's unit-level
  stability floor.

**Options for the owner decision:**

- **A — accept honest periphery on small-n (recommended):** re-scope E4.1's
  bar: on datasets below a stability power floor the deliverable is
  fidelity + legibility + honest labels (which Wine passes emphatically);
  certified-recovery bars live on the n ≥ 8k MVL datasets (mushroom, adult,
  bike, drybean) where CPSS has power.  Proceed to P5 unchanged.
- **B — small-n statistical design:** replace 50/50 CPSS with an m-out-of-n
  subsampling variant below a data-size threshold (changes the E[V] bound's
  form; needs care to stay non-vacuous and honest).
- **C — coarser concept identity:** match/certify at route/support level
  (all-pairs union-find + support-overlap), consolidating collinear variants
  into one concept.  Larger redesign of M-#3.6; risks blurring genuinely
  distinct concepts.

**Owner decision (2026-08-22): Option A.** *(see P5 below for how the evidence
resolved the power-vs-multiplicity question)*  Wine recorded as "below stability
power floor / collinear-multiplicitous" with full Π/π reporting; certificates
now carry an explicit regime label at small n.  P5 runs unchanged; whether the
failure mode is statistical power (should vanish at n ≥ 8k) or genuine
multiplicity (would persist and motivate option C's coarser certification
level) is decided by the P5 evidence.  E4.1 re-scored under this decision:
artefacts complete ✓, fidelity ratio 1.017 > v4's 0.766 ✓, full legibility ✓,
honest labelling ✓ — **MET (A)**; the recovery clause moves to the n ≥ 8k
datasets.  Instrument-correction ratification: **"ratify with review later"** —
owner will review the FINDINGS trail before the P5 freeze.

## P5 (2026-08-22) — MVL battery: the dev-phase verdict

Definitive battery at frozen commit `085bea8`; full record in
`experiments/results/e51.json`.  Wall-clock 0.55 h for the whole MVL (K3 ≤ 12h:
comfortable).  En route the battery caught and forced fixes for: unscaled
regression targets (bike untrainable), unbudgeted structural mutations
(prune_edges/merge wrecking near-separable models — all mutations now
trial-and-revert under ε_prune), a settle fidelity ratchet (entry-state now
protected), a float32 gauge tolerance, and a joint-vs-marginal ablation fallacy
in unit removal (now sequential under budget).

### Results per dataset (fid ratio = pipeline fidelity / reference ceiling)

| dataset | n | fid ratio | CORE | notes |
|---|---|---|---|---|
| zoo | 101 | **1.058** | 0/7 | below power floor (regime label) |
| tictactoe | 958 | **1.028** | 0/8 | below floor; lines superposed, not unit-aligned |
| wine | 178 | **1.044** | 0/6 | below floor; v4 anchor 0.766 crushed |
| mushroom | 8 124 | **0.995** | 0/2 | = v4 anchor 0.995; multiplicity labelled (v4-consistent) |
| drybean | 13 611 | 0.984 | 0/8 | all 8 concepts multiplicitous |
| bike | 17 379 | 0.945 | 0/22 | hour×temp present but not stable-unique |
| adult | 48 842 | 0.877 | 0/5 | v4 anchor 0.892 missed by 1.7%; honest-shallow ✓ |

### Bars (M-#7): H2 ✓ · H4 ✓ (median ratio 0.995) · H5 ✓ · H6 ✓ · **H1 ✗ · H3 ✗**

### The central finding — carving multiplicity is real, not statistical

The owner-decision-A question (power vs multiplicity) is now answered by
evidence: **multiplicity**.  On every standard-regime real dataset the units
are individually legible (μ 0.94–0.98) with large real effects (Δ up to 0.48),
but across 8 restarts each run carves the correlated features into *different*
equally-good 3-feature mixes: Π ≤ 0.5 — and the τ-sensitivity sweep shows **0
stable concepts at every matching threshold down to 0.5**, so this is not a
matching-strictness artefact and not (at n = 8k–49k) a CPSS power problem.
Synthetics certified perfectly because their generative process has a unique
sparsest carving; real tabular data's correlated features admit many.  This is
POSITIONING §2's "uniqueness is level-dependent" thesis biting at full
strength, and it reproduces v4's honestly-stated frontier (Mushroom: "stable
within portfolio, NOT unique").

**Implication:** the certifiable invariant on real data lives at a coarser
level than exact unit supports — shared subspaces / feature-group routes —
and/or the training needs a canonicalisation tie-break so equal-loss carvings
collapse to one.  That is ALTERNATIVE-OPTIONS-grade method design (option C
territory), a principal decision for the owner.

### Anchors (E5.2) and cost (E5.3)

Wine 1.044 vs v4 0.766 ✓ · Mushroom 0.995 = 0.995 ✓ with multiplicity
labelled ✓ · Adult 0.877 vs 0.892 ✗ (−1.7%) with honest-shallow ✓ ·
Battery 0.55 h ✓.

## P6 (2026-08-22) — route-level certification: first certified real-data routes

Owner decision C implemented: routes = concepts keyed by collinearity-group
support (train-data Spearman clustering ≥ 0.8), with the invariant-core chain
rule, MODAL-variant certification (a minimal-variant rule mis-certified COMP3's
product as `{x1}` under an under-detection chain — caught by E6.1), and
anchor-based Rashomon multiplicity (variants sharing no common group are rival
explanations; companion variation around a shared anchor is context — the
earlier flag over-fired and blocked drybean's Π=1.00/π=1.00/Δ=0.93 size route).

- **E6.1 (reduction on synthetics): MET, 5/5** under the final ground-truth
  scoring — no invented routes, every planted concept recovered as a route
  modal, nothing unit-level-certified lost.  Two earlier scorings were
  themselves corrected (support-set equality double-counted shared supports;
  a cross-code-version e31 comparison was noise); the route layer twice proved
  MORE correct than the unit-level reference it was compared against (folding a
  redundant partial variant; certifying COMP3's `{x4}` that unit matching
  fumbled).
- **E6.2 (MVL at route level): NOT MET as pre-registered, with substantial
  wins.**  Certified real-data routes, a programme first:
  - **drybean: 2 CORE routes at coverage 1.0** — the size-family route
    (Π=1.00, π=1.00, μ=0.97, joint Δ=0.93) and the elongation×shape composite
    (Π=0.88, π=0.93, μ=0.98, Δ=0.77).  Groups: 16 descriptors → 6 families.
  - **bike: 1 CORE route at coverage 0.97** — hour-anchored (Π=1.00), with
    modal companion `workingday`; all 26 variants contain `hour`.
  - **adult: 1 CORE route, coverage 0.014** — only a minor route certifies;
    the load-bearing routes remain unstable.
  - **mushroom: 0** — parallel alternative rules (odor vs backups) surface as
    separate routes with complementary presence; the Π bar prices them
    honestly.  Consistent with the v4 anchor ("stable within portfolio, NOT
    unique").
- **Open design question (the bar's failing clause): route granularity under a
  dominant anchor.**  Bike's chain-merge folds every hour-interaction into one
  hour-anchored route, so `hour×temp` and `hour×workingday` — both genuine —
  are not separated as distinct certified compositions.  Candidate refinement:
  split a merged route when distinct companion sub-supports are individually
  stable across runs.  Adult's and mushroom's remaining failures are the honest
  cases (instability/alternatives), not instrument artefacts, on present
  evidence.

### E6.3 — canonicalisation probe: NULL RESULT (and a principled reason)

Deterministic per-feature preference (λ_pref 0.01 as input-edge L1, identical
across runs) on drybean: **zero effect** — unit-level Π distribution
(0.25–0.5) and fidelity (0.882) identical to control.  Carvings are fixed
early by init/SGD trajectory; a static linear preference has no grip on the
within-group rotations that distinguish equal-loss carvings.  Deeper: even a
*working* canonicalisation would be epistemically suspect — if runs agree
because a shared bias made them agree, Π measures the bias, not discovered
stability.  The probe's null result strengthens the route-level design as the
correct answer rather than a fallback.  Probe retired; mechanism stays in the
codebase config-gated at 0 (default off).

## P7 (2026-08-22/23) — function-level certification: the programme's answer

Layer F implemented straight (bin → backfit → exact mass-moving purification
under the declared empirical measure; OOS-scored shares), certified with the
existing Π/π/E[V] machinery; Layer R (portfolio reliance) added.  Instrument
corrections measured against ground truth en route: OOS covariance shares
(train-side hit r²=1.0 on pure noise), residualised triple screen (a dominant
main drowned pure order-3), one 8-bin tier (16³ starves cell counts),
task-variance re-basing (a useless model's internal wiggle decomposed "stably"
because restarts share data — NOISE x5), bike `log1p` count target, and
closed-form head recalibration (gauge-legal) for regression.

**A load-bearing bug found by Layer F's fidelity gate:** `merge_units` mutated
the base model in-place, so every REJECTED trial merge in the budgeted prune
corrupted it cumulatively (bike: fid 0.93 → −14 across prune steps; affected
all datasets with rejected merges since the budgeted-merge change).  Fixed +
non-mutation regression test; prune steps now cost exactly 0.000.

### E7.1 (synthetic exactness): **MET** — ADD `{x1},{x2},{x3}` only; COMP2
`{x1},{x2·x3}`; COMP3 `{x1·x2·x3},{x4}`; NOISE **zero** components (the
untradeable holds at function level).  All planted components CORE.

### E7.2 (real data): 3/4 certify (clause MET); bike hour×temp clause
**NOT MET as-registered — and that is a finding, not a failure**:

| dataset | certified components (share, Π, π) |
|---|---|
| **mushroom** | odor=n .44 · gill-spacing .17 · odor=f .09 · stalk-surface .05 · bruises .05 (+2) — 7 CORE, coverage 0.865 |
| **drybean** | 20 CORE across the 7 class logits |
| **bike** | hour .59 · temp .08 · hour×weekday .05 · year .02 — all Π=1.00, π≥0.975; coverage 0.743 |
| **adult** | age .08 · married .075 · capital-gain .06 · education-num .05 · hours .02 — all Π=1.0; all MAINS (honest-shallow confirmed at function level) |

The **purified** hour×temp interaction is genuinely < 1% of target variance:
the v3-era "hour×temp certifies (cov 0.82)" was an UNPURIFIED composition-unit
claim whose mass belongs to the hour and temp mains — exactly the
"contradictory interpretations" phenomenon the purification literature
documents, now demonstrated on our own programme's historical anchor.  The
certified bike structure (hour + temp + hour×weekday + year) is unanimous
across the portfolio and domain-canonical.

Layer R examples: "every bike restart relies on hour ≥ 1.07"; "every adult
restart relies on capital-gain ≥ 0.17, marital-group ≥ 0.16".

### The three-layer answer to the multiplicity question (owner's framing)

Multiplicity recognised (routes + labelled equivalence classes), DAG
inspectable (this model's legible units, route layer, component annotations),
result certified (unique purified components under a declared measure +
portfolio reliance ranges).  On every real dataset the certified function
structure is stable at Π≈1.0 even where unit carvings never exceeded Π≈0.5.

## Ratification (owner, 2026-08-23)

The owner ratifies: the full P0–P7 instrument-correction and bar re-score
trail as recorded above (deferred from 2026-08-22), the E4.1 decision-A
re-score, the E6 route-level scoring corrections, and the P7 re-score of the
bike hour×temp clause as a purification finding.  The programme proceeds to
config freeze and the confirmatory one-shot.

## E-C.1 — CONFIRMATORY ONE-SHOT (2026-08-23, seeds 20-24, frozen `freeze-x1-v1`)

One shot, no reruns; verdicts stand as read.  **PRIMARY ENDPOINTS MET.**

- **Synthetic exactness confirmed on fresh seeds**: ADD `{x1},{x2},{x3}`;
  COMP2 `{x1},{x2·x3}`; COMP3 `{x1·x2·x3},{x4}`; **NOISE zero certified
  components** — the untradeable holds on untouched seeds.
- **Real data confirmed**: mushroom 4 CORE (coverage 0.761), drybean 17 CORE,
  bike 4 CORE (coverage 0.747 — hour/temp/hour×weekday/year again, Π=1.0);
  adult again certifies exactly its 5 textbook mains (age, married,
  capital-gain, education-num, hours) at Π=1.0 with coverage 0.198 — below the
  0.5 coverage bar on fresh seeds as in dev, reported as the method's measured
  frontier on diffuse-structure data.
- Battery wall 0.48 h.  Fresh-seed shrinkage vs dev is modest and one-sided
  (mushroom 7→4 CORE members within the same rule family; bike identical).

**Programme verdict, one paragraph.**  On untouched seeds the honesty layer
holds without exception (zero certification on noise, honest-shallow adult),
planted structure is recovered exactly through the models' own functions, and
three of four real datasets carry certified, unanimous (Π=1.0),
domain-canonical function components with majority coverage.  The deliverable
stands: a legible model whose DAG is inspectable carving-by-carving, an
explicit equivalence-class account of multiplicity, and certified claims
anchored at the level where uniqueness exists — the purified function
decomposition under a declared measure.

## POSITIONING ASSESSMENT (2026-08-23)

Critical scorecard of the completed beachhead build against
`docs/00-POSITIONING.md` (owner-requested).  POSITIONING carries matching
inline **AMENDMENT (2026-08-23)** blocks recording where the refined position
differs from the original.

### Hypothesis drivers (POSITIONING §3)

| driver | verdict | evidence |
|---|---|---|
| Monosemantic | **Achieved** | pressures ~free (impurity −86% at neutral fid, E1.1b); every real dataset's live units legible, μ 0.94–0.98 |
| Minimal | **Achieved** | growth/prune converges small; redundant units die while fidelity *rises*; C4 ordering encoded as discovery-gating |
| Depth-honest | **Achieved, twice-confirmed** | honest flatness 8/8; earned depth on COMP2/3; adult honest-shallow at unit and function level, dev + confirmatory |
| Stable | **Not at unit level — achieved one level up** | unit carvings multiplicitous (Π ≤ 0.5 at every τ, E5.1); function components unanimous (Π ≈ 1.0) everywhere |
| A-priori explainable + certified | **Achieved, re-based** | full Π/π/E[V] machinery, non-vacuous bound (E[V] ≤ 0.124), perfect honesty record; certified object moved units → purified components |
| Polysemantic periphery | **Achieved — but not as predicted** | the real-data residual is *monosemantic-but-multiplicitous*, not polysemantic; refinement amended into POSITIONING §3 |

### The four differentiators (POSITIONING §4)

(a) **Statistical certification — delivered; the strongest asset** (one-shot
confirmatory discipline held, M9).  (b) **Honest depth — delivered**, incl.
dissolution and negative validation (adult refuses depth).  (c) **Constructive
growth to minimal — delivered** at beachhead scale (caps L≤4, Σm≤96).
(d) **Small-n expert-auditable — half-delivered**: above-ceiling fidelity +
legibility + honest regime labels at n < 2k, but no certificates (CPSS
powerless at n=178; the m-out-of-n design was never built).

### Accuracy-neutrality ("no concept tax")

Confirmed 5/7 (wine 1.044, zoo 1.058, tictactoe 1.028, mushroom 0.995, drybean
0.984 vs ceiling).  **Bike 0.945 and adult 0.877 are a real tax** (adult also
−1.7% under the v4 anchor).  "~Free" is true on the beachhead core, overstated
at the diffuse-structure frontier.

### The defensible corner (POSITIONING §7)

Occupied, and arguably strengthened: the multiplicity crisis produced a
positioning weapon (the field's "contradictory interpretations" failure
demonstrated on our own historical anchor — bike hour×temp; canonicalisation
shown null *and* epistemically suspect).  Two exposures the story must answer
(now amended into POSITIONING §7): **coverage honesty** (certified components
cover ~0.74–0.87 of function variance on strong datasets, 0.198 on adult —
"certified where certifiable, labelled elsewhere") and **the EBM question**
(the NN meets/beats reference ceilings where GAMs can't; the decomposition is
exact on our own model's f; the DAG gives mechanism-level inspection).

### Outstanding, ranked

1. **Expert validation of the soft target never happened** — M6 (owner DAG
   nameability review) still open; POSITIONING §3 makes it the test of the
   interpretability claim.  Cheapest highest-value item.
2. **Small-n certification** (m-out-of-n CPSS variant) — undelivered
   differentiator.
3. **Adult-class (diffuse-structure) data**: fidelity tax + 0.198 coverage is
   the measured frontier; no designed next step.
4. **Stability across model families** — restarts + resamples done; a second
   architecture/family never tested (POSITIONING §2 lists it).
5. **Docs drift** — resolved this date: specs/plan harmonised to the
   three-layer as-built state (M-§10, S-§16 addenda).
6. **External validation of the certificate format** — SR 11-7 / EU-AI-Act
   keying is our reading; no model-risk practitioner has reviewed it.
7. **Mushroom at route level** (0 routes; parallel alternative rules) honestly
   priced but unexplained to a domain expert; function level covers it.
8. **Stages 2–3 of sequencing** (regulated-decision product; SLM/Transformer)
   untouched, correctly per beachhead-first.

### One-line verdict

The programme proved the hypothesis with one honest amendment: a minimal,
monosemantic, depth-honest, ~accuracy-neutral network can be grown and
certified — but on real data the certifiable invariant lives one level above
the units (POSITIONING's own "uniqueness is level-dependent" thesis, confirmed
harder than anticipated).  The honesty layer never broke once, including on
quarantined seeds.  What stands between this and the §7 moat being real is not
more method — it is the expert nameability review, small-n power, and a
practitioner's eyes on the certificate.

## P8 (2026-08-23) — batched width growth: the bike gap localised and closed

The standard-NN baseline comparison (`baselines/standard_nn/`, owner-requested)
traced bike's fidelity shortfall to one mechanism, then E8.1 fixed it.

- **Diagnosis** (seed-0 trace + plain capacity curve): the width-growth accept
  bar δ_grow=0.005 is a *per-step* test; bike's residual past ~0.90 arrives as
  ~0.003-sized diffuse increments that never individually clear it though they
  sum to ~0.05 (plain curve w6 0.8725 → w16 0.9384 → w32 0.9424 test fid).
  Crucially, **at matched capacity the tax is negative**: ours-at-6 0.8909 vs
  plain-at-6 0.8725 — pressures cost nothing; the whole gap was capacity.
  Depth gating worked correctly throughout (bike honestly 1 composition layer;
  the depth trial was run and refused at +0.001).
- **E8.1 (grow_batch=8, dev seeds 0–3): MET as pre-registered.**  Bike median
  ratio 0.945 → **0.9877**, median live-unit μ **0.992**; guards 3/3 (NOISE
  zero accepted growth, ADD flat-at-ceiling, COMP2 recovery 4/4).
- **Honest notes:** adult unmoved (0.876/0.885, zero accepted growth even
  batched — a different frontier than bike's); bike models larger (10–15 units
  vs 6; seed 0 accepted a second hidden layer that survived dissolution —
  operationally earned, flagged for owner review); COMP2 seed 2 kept 13 units
  at ceiling (batching trades some minimality for fidelity); ~20 min/seed.
- **P8-X owner decision (2026-08-24): SHIP BOTH AS A CONFIG SWITCH.**  Default
  stays `grow_batch: 2` (frozen, `freeze-x1-v1`-certified — the legible,
  depth-honest interpretability deliverable); `grow_batch: 8` is documented as
  the opt-in **accuracy mode** (S-§14, S-§16.1).  Rationale: the certified Layer-F
  structure is near-identical between the two (E8.2), so batch buys *accuracy*
  (bike count-space R² 0.69→0.91, log1p 0.945→0.988) not more *certified
  interpretable structure* — and the added complexity (6→28 units, +unwarranted
  2nd layer) lives entirely in the uncertified unit layer.  Default unchanged ⇒
  **no re-freeze**; every existing certified/confirmatory claim stands.  Batch is
  inert when already near ceiling (morpher, Spambase both unchanged at [7]/[8]).

### Morpher (Russian declension) — generality probe + expert analysis, 2026-08-24

First non-English, linguistic, categorical-predicate dataset (10,195 words, 38
morphological value-predicates, 3-class declension target сущ/нескл/прил).
Loader replicates the v4 encoding; Cyrillic feature names kept for expert reading.

- **Fidelity 0.878, ratio 0.994** vs ceiling; **honest-shallow [7], 1 layer**
  (declension cues act largely independently — order-1, correctly).
- **3 CORE function components, one per declension paradigm — all textbook-correct
  Russian grammar:** `ЧастьРечи=прил` (part-of-speech = adjective → adjectival
  declension, share 0.115, Π=1.0) · `НаСогласную` (ends in consonant → masc noun
  declension, Π=0.875) · `Суффикс=ово` (the -ово/-ино toponym suffix →
  indeclinable, Π=0.875).  The method recovered real grammar an expert confirms.
- **Coverage only 0.15**, and the honest reason is a NEW periphery mechanism:
  **low-support multiplicity** (distinct from bike's collinearity multiplicity).
  The dominant *raw* shares sit on rare suffix predicates (`Суффикс=цы` 0.72,
  `Суффикс=ца` 0.61) that are near-deterministic on the few words carrying them
  but **unstable (Π 0.25–0.38) and infrequent (π 0.13–0.35)** — genuine sparse
  lexical sub-regularities the honesty layer correctly refuses to certify.
  (Those >0.5 numbers are per-class-logit max-merged raw shares, not fractions of
  total variance — the multiclass share caveat.)
- **0 CORE units, 0 CORE routes** — as everywhere on real data, the certifiable
  invariant is the function component, not the unit/route.

**Generality verdict:** the method transfers cleanly to a new domain and feature
type — honest-shallow, near-ceiling fit, certified components = genuine grammar,
honesty layer intact.  It also surfaced a domain-specific stressor: extreme
categorical **feature sparsity** pushes most mass into an honestly-labelled
periphery, so the certificate covers the coarse grammatical backbone and declines
the sparse-suffix long tail.  A different periphery mechanism (low support, not
collinearity), same honest handling.

### Spambase — high-d generality probe (d=57), 2026-08-24

Second generality dataset (owner-chosen): 4,601 emails × 57 continuous features,
binary spam.  Purpose: stress the interaction screen at C(57,2)=1,596 candidate
pairs (the d-scaling axis flagged in the POSITIONING assessment) with known
structure, on a third domain + continuous features (complements morpher).

- **The screen held at d=57 — no blow-up** (~144 s full certification, well
  within budget).  Frozen and batched are near-identical (both [8], fid 0.778,
  13 CORE, coverage 0.48) — like morpher, batching adds nothing because the
  model is near ceiling with 8 units.
- **13 CORE components, ALL main effects — honest-shallow, zero certified
  interactions.**  The literature's headline capital-run × `!` interaction does
  **not** certify: the purified pairwise mass is negligible.  Another instance of
  the bike-hour×temp lesson — a widely-cited interaction that, purified, is
  mostly additive main effects.  On present evidence Spambase is additive.
- **The certified drivers are a domain fingerprint, not the generic spam
  lexicon.**  Top: `george` .138, `cap_total` .087, `hp` .041, `lab` .034,
  `telnet` .028, `hp`/`lab`/`telnet`/`650`/`85`/`meeting` — these are the
  HP-Labs collection's ham markers (the non-spam mail was one person's work
  email at Hewlett-Packard).  Generic spam tokens (`char_$`, `char_!`, `remove`,
  `your`, `our`) certify too but with smaller shares.  The method faithfully
  recovered that this corpus separates on *sender-context* tokens more than on
  universal spam cues — a correct, expert-checkable read of THIS dataset (and a
  known caveat about Spambase's generalisability).
- **Coverage 0.48, recon R² 0.67**: 57 weak token signals, no single dominant
  driver — the method certifies the robust ~13 and leaves the diffuse tail as
  labelled periphery.

**Generality verdict (morpher + Spambase together):** the method transferred
across three domains (linguistic / spam / the MVL) and both feature types
(categorical-predicate + continuous), stayed honest-shallow where the data is
additive, and recovered domain-correct structure an expert can validate.  The
d=57 screen did not strain — the practical dimensionality ceiling is higher than
the MVL alone showed.  Two datasets, two orthogonal stressors (sparsity, then
dimensionality), honesty layer intact throughout.

### Regulated credit datasets — the first genuinely regulated deliverables, 2026-08-24

Owner decision (roadmap #5): produce the audit deliverable on a genuinely
regulated decision domain.  Two credit-risk datasets (branch
`feat/regulated-credit`): **taiwan_credit** (30,000 real bank clients, default
next month — EU AI Act Annex III creditworthiness / SR 11-7 core scope;
full-power regime) and **german_credit** (credit-g, n=1,000 — the canonical
benchmark; small-n regime by design).  Protected attributes (SEX, MARRIAGE,
AGE) deliberately kept visible as explicit predicates for fair-lending review.

**Taiwan (flagship, full certificate):** fid 0.167 vs ceiling 0.195 (ratio
0.861), test accuracy 0.823, minimal [4]-unit model, E[V] ≤ 0.04.
- **`PAY_0` (most recent repayment status) certifies as THE dominant driver
  (Π=1.0, π=1.0)** — exactly the literature's known answer; `LIMIT_BAL` second
  (Π=1.0).  Older repayment lags (PAY_3) land in the labelled periphery.
- **Fair-lending finding: protected attributes certify NOTHING.**  SEX and AGE
  are absent from the decomposition entirely; MARRIAGE=single appears at share
  0.000 periphery.  The certificate can state affirmatively: the model's
  certified decision structure relies on repayment behaviour and credit limit,
  not on sex, age, or marital status — an auditable non-reliance statement the
  black-box + SHAP stack cannot make with FDR backing.
- Low absolute pseudo-R² (~0.17–0.20 even for HGB) is the domain's known
  irreducible noise; the certificate reports it honestly (accuracy 0.82).
  Ratio 0.861 is below the 0.98 neutrality bar — diffuse-frontier territory
  (like adult), recorded as such.

**German (companion, small-n regime label as designed):** fid 0.194 vs ceiling
0.197 (ratio **0.986**), [8] units.  3 CORE: `checking_status=no checking`
(Π=1.0, π=1.0) · `credit_history=critical/other existing` (Π=1.0) · `duration`
(Π=0.875) — the classic Statlog triad, correct.  Savings/purpose/amount land in
the honest periphery (unstable at n=1,000 half-samples).  No protected-attribute
component certifies (age periphery-only).

**Verdict:** the method produces domain-correct, expert-validatable certified
structure on real credit data at both regimes, and — the practitioner headline —
supports certified NON-reliance statements on protected attributes.  These two
dashboards + certificates are the artifacts for external model-risk review
(roadmap #6).

Literature's canonical bike interaction is hour × **workingday**; our pipeline
certifies hour × **weekday**.  Diagnostic: per restart (dev seeds 0–7, BOTH
regimes), measure the purified interaction share on each hour-pair.

**Result — the partner is perfectly stable, and it is NOT the collinear coin-flip
expected: `weekday` wins 8/8 in BOTH regimes; `hour × workingday` carries zero
purified mass (0/8, both regimes); no other hour-pair appears at all.**
Frozen weekday shares 0.022–0.086; batched 0.063–0.085 — same effect, same
partner, wider models.  So the weekday-vs-workingday difference is NOT
carving-multiplicity: the method consistently prefers `weekday` as the
interaction partner.

**Why (assessment, not yet a proven mechanism):** `workingday` is a *coarsening*
of `weekday` (weekend/holiday → 0).  The commute double-peak differs across the
five weekdays and between Sat/Sun, so `hour × weekday` (7 levels) fits the true
daily-shape modulation strictly better than the binary `hour × workingday` — the
purification then assigns the interaction mass to the finer, more explanatory
term and leaves the binary with ~0 residual.  The literature names the *binary*
because EBM/GA²M demos privilege the 2-level split for display; our method,
given both, measurably prefers the higher-resolution partner.  This is a case
where the certified structure is *more* refined than the textbook, not a
mismatch to fix.  (Not adjudicated against ground truth — bike has none.)

### E8.2 — bike claims-invariance under batched growth: **MET** (2026-08-23)

Full bike certification (R=8 + 40 CPSS) at grow_batch=8, 0.12 h wall:

| component | frozen ref (e7x dev) | at grow_batch=8 |
|---|---|---|
| hour | .589, Π=1.0 | **.5871, Π=1.0** |
| temp | .0804, Π=1.0 | **.0656, Π=1.0** |
| hour×weekday | .0511, Π=1.0 | **.0714, Π=1.0** |
| year | .0222, Π=1.0 | **.0241, Π=1.0** |

Purified hour×temp **stays uncertified** (share 0.0) — the purification finding
survives a 2× capacity change.  Coverage 0.743 → **0.784**; fid ratio 0.9925;
hour-anchored CORE route retained (Π=1.0, π=1.0).  **The certified function
structure is capacity-invariant** — the strongest evidence yet that Layer F
captures the data's structure, not the model size's artefacts.

**Two additions for owner review** (small, domain-canonical mains the 6-unit
model truncated): `season` (.0228, Π=.875, π=.875) and `humidity` (.0134,
Π=1.0, π=.85) — both textbook ridership drivers.  Main restart is [14,14]
(second layer survived dissolution); unit-level n_core=1 (multiplicity remains,
as expected — the claims live at the function level).  Restart fids 0.90–0.94.
