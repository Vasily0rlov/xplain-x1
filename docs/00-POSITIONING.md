# SAGE-Net / XPLAIN — Positioning

A strategic positioning note distilled from a design + literature-review thread
(2026-08). It states the problem, the reasoning that led to the hypothesis, the
hypothesis itself, the recommended approach, the prior art with citations, the disproven / ruled-out approaches, and the defensible corner + moat.

> **Honesty / provenance caveat.** Citations were gathered from paper abstracts
> and search summaries (direct arXiv fetch was egress-blocked in the working
> environment); verify exact claims against the papers before external use. The
> novelty assessment is deliberately conservative — where the thread found an idea
> to be *known*, it says so.

---

## 1. Problem statement

Modern deep networks are hard to inspect because **capacity is fixed**: when the
data holds more useful concepts than the model has neurons, the model resorts to
**superposition / polysemanticity / entanglement** — efficient for prediction,
catastrophic for inspection. Mainstream explainable-AI (SHAP, LIME, sparse
autoencoders) explains a black box *after the fact*: approximate, local, and — for
high-stakes and regulated decisions — not faithful enough for an auditor or
regulator to rely on ([Rudin 2019](https://arxiv.org/abs/1811.10154)).

The goal is a model whose **own decision logic is inspectable** — concepts are
mostly monosemantic, their composition is legible, and the depth of reasoning is honest — rather than a black box wrapped in a post-hoc explanation.

## 2. How we got here — the conceptual moves that shaped the hypothesis

These are the load-bearing insights from the thread; each one narrows *what is
achievable* and therefore *what to aim for*.

- **Disentanglement vs mono-semanticity are different targets, and the second is
  the tractable one.** *Disentanglement* is relational — each unit ≈ one
  *ground-truth factor* — and so needs an external reference, is blocked by the
  identifiability impossibility (which of the equal-loss factorisations is "true"?),
  and is ill-posed on real data. *Mono-semanticity* is intrinsic — each unit is one
  *coherent* thing, whatever it is — so it needs no ground truth, is
  identifiability-immune, and is directly measurable. **We target
  mono-semanticity, not disentanglement.**
- **Gauge freedom splits "entanglement" into two kinds.** A hidden layer is
  identified only up to invertible linear mixing (the readout absorbs any mix), so
  "how monosemantic a unit is" is *basis-dependent*. Much apparent entanglement is
  pure **gauge** (the concepts are linearly present, just rotated) — removable
  cheaply and accuracy-neutrally by choosing the right basis. What is *not* gauge is
  **true superposition** (more concepts than neurons). Pushing toward
  mono-semanticity is exactly *choosing the sparse/selective gauge*; it resolves the
  first kind, and needs more capacity (width/depth) for the second.
- **Minimality canonicalises the gauge — but only on top of a mono-semanticity
  pressure.** Among all monosemantic bases, the *sparsest / fewest-unit* one is
  canonical (the sparse-coding identifiability result), it re-derives "one concept,
  one unit," and it matches how a domain expert compresses a domain to a minimal
  viable representation. **Critical ordering caveat:** minimality *alone* drives
  *more* entanglement — superposition *is* the minimal-width encoding — so
  compression and mono-semanticity pull in opposite directions on width. They are
  two pressures in balance (an MDL trade-off), and the balance lands near the true
  concept count **with a small polysemantic residual** (see §3).
- **Compositional structure needs depth — a flat net cannot be disentangled at all.**
  Measured directly (the basin-census experiment): a one-hidden-layer net *cannot
  represent a disentangled product* — a linear readout can add two factor-units but
  not multiply them, so a flat "correct-support" model gets r² ≈ 0. Disentangled
  compositional concepts live at depth ≥ 2. Hence **depth-honesty**: use exactly as
  much depth as the domain's compositional order warrants, and dissolve depth that
  is not earned.
- **The bias is accuracy-neutral, not accuracy-trading (empirically).** In the
  basin census, the factored/structured solution fit as well or better than the
  entangled one (r² 0.999 ≥ 0.97), and a plain net recovered the *subspace* (the
  information, R² ≈ 0.82) while failing at the *unit level* (superposition). So the
  structural bias selects among *equal-accuracy* basins — it buys interpretability
  at ~no accuracy cost. (A naive **decorrelation** penalty was accuracy-neutral but
  *insufficient* — uncorrelated units are still linear mixes; the effective pressure
  is sparsity/selectivity + depth, not statistical independence.)
- **Uniqueness is level-dependent, which is *why* a polysemantic periphery is
  acceptable.** A domain has a unique **invariant core** (which quantities matter,
  their supports and shapes — up to gauge) and a **multiplicitous periphery** whose
  size grows with collinearity, weak signal, and fit ceiling. The periphery has no
  unique disentangled answer to find, so leaving it as a small polysemantic residual
  is the *honest* target, not a compromise.
- **Mono-semanticity buys inspectability, not correctness.** A monosemantic unit is
  cleanly "one thing," but that thing may be a dataset shortcut, not a true
  mechanism. **Stability** (re-emergence across restarts / resamples / model
  families) and statistical **certification** (false-discovery bounds) are what
  re-earn correctness on top of inspectability.

**The punchline.** You cannot grow a *disentangled* network organically —
disentanglement is unidentifiable without an inductive bias — but you *can* grow a
largely *monosemantic* one, because **mono-semanticity is a gauge choice you are
allowed to make, whereas disentanglement is a truth you would have to discover** —
and making that choice costs ~no accuracy.

## 3. The hypothesis

**Take a *traditional deep neural network* and enhance the architecture and the
training method so that the model converges *toward* a minimal, monosemantic,
depth-honest, stable and therefore a-priori-explainable network — accepting a polysemantic periphery
for corner cases.**

Read the three drivers as **directions of convergence / soft pressures in
balance**, not as hard constraints:

- **Monosemantic (target direction, not a hard rule).** Push units toward being
  individually coherent; **tolerate a residual set of polysemantic neurons** that
  absorb corner cases and the intrinsically-multiplicitous periphery. "Largely
  monosemantic" is the practical — and, per the uniqueness result, the *correct* —
  bar, not "every neuron pure."
- **Minimal (objective within that pressure).** Prefer the fewest coherent units
  and the least honest depth that preserve fidelity — MDL / Occam / the expert's
  minimal viable representation. It only aids interpretability *because* it rides on
  the mono-semanticity pressure; alone it would compress into superposition.
- **Depth-honest (compositional).** Grow / retain exactly the depth the domain's
  compositional order warrants; dissolve depth-theatre. This is inherently a *deep*
  network property.
- **A-priori (by-design) explainable (by the virtue of monosemanticity and compactness) + certified.** The certification layer (stability, false-discovery bounds, uniqueness)
  supplies the correctness guarantee that inspectability alone does not.

**Note on interpretability vs explainability**. Explainability is our main target, in the form of the ability to trace model's decisions through a minimal number of hops and unambiguous decision routes. Part of my hypothesis is that our model will converge to something more interpretable by the virtue of chosen training pressures - which is our soft target, to be validated by the expert. Similarly, we don't mandate neurons in the NN to represent domain concepts straight away (not at the initial stage anyway), but rather settle on monosemanticity where possible.

**Representation formalism.** The preferred path is to **start from a
standard deep architecture (MLP) and let enhanced
training drive it toward the target regime** — via objective terms (sparsity /
selectivity, separation, purity), architectural pressure (bottlenecks / fan-in
limits where appropriate), constructive growth-and-pruning toward minimal size and
honest depth, and a certification pass. The bet is that the
*method*, applied to an ordinary deep NN, can do the work. Once the method is proven, we'll consider applying it to the Transformer architecture to see if we can achieve an explainable Transformer model. We'll want to be able to extract a DAG representation from the trained model, based on the largely 
-monosemantic neurons and their connections, so that an expert can explore it and thereby assess the level of explainability/interpretability of the resulting model.

## 4. Recommended approach

- **Enhance, don't replace.** Begin with a conventional deep NN method for the task; add
  the convergence pressures above to the training loop plus a certification pass on
  the result. The deliverable is a standard-architecture network that has *settled*
  into a largely-monosemantic, minimal, depth-honest state, with its polysemantic
  periphery explicitly labelled (not hidden).
- **Lean on the four differentiators the field is weakest on:** (a) statistical
  **certification** (false-discovery bounds, uniqueness); (b) **honest compositional
  depth** (a gated "don't claim hierarchy you don't need"); (c) **constructive
  growth to minimal** size; (d) **small-n, expert-auditable** data.
- **Architecture reach.** The principles are architecture-agnostic. For structured
  / tabular / scientific data the concept-model realisation is a proven ground. For
  text-heavy tasks a **small, purpose-trained Transformer** applies in principle:
  the residual-stream/superposition story is native, feed-forward sublayers map to
  monosemantic units, and attention's *dynamic routing* is the one genuine seam —
  handled either by **static weight-sparsification** (shown to yield interpretable
  circuits) or by a **conditional-routing** extension (the native model of
  data-dependent edges).

## 5. Prior art (the neighbourhood — mostly occupied)

Every pillar has a home in the 2019–2026 literature; the field has independently
converged on the same reasoning. Grouped:

**Post-hoc monosemanticity (sparse dictionaries).**
- [Towards Monosemanticity (Anthropic, 2023)](https://transformer-circuits.pub/2023/monosemantic-features)
  and [Scaling Monosemanticity (Templeton et al., 2024)](https://transformer-circuits.pub/2023/monosemantic-features)
  — SAEs decompose activations into monosemantic features (post-hoc on a frozen model).

**Minimality / MDL as the principle that avoids over-completeness.**
- [MDL-SAEs — *Interpretability as Compression* (Ayonrinde et al., NeurIPS 2024)](https://arxiv.org/abs/2410.11179):
  naive SAEs prefer "extremely wide and sparse" dictionaries; description length
  selects the minimal, non-split decomposition — the closest analogue to "minimal
  monosemantic".
- [Sparse but Wrong: Incorrect L0 → Incorrect Features (2025)](https://arxiv.org/html/2508.16560v4)
  (there is a floor; the count matters); feature-splitting remedies
  ([subspace-aware SAEs](https://arxiv.org/html/2606.06333v1)).

**Sparsity/minimality gives identifiability (canonicalises the gauge).**
- [Lachapelle et al., *Synergies between Disentanglement and Sparsity* (ICML 2023)](https://arxiv.org/abs/2211.14666)
  and the mechanism-sparsity identifiability line — sparsity breaks the rotational
  ambiguity; the sparse/minimal basis is canonical.

**By-design / inherently interpretable models (the "a-priori" camp).**
- Founding manifesto: [Rudin — *Stop Explaining Black Box Models …* (2019)](https://arxiv.org/abs/1811.10154).
- Compression-derived white-box nets: [CRATE / *White-Box Transformers via Sparse
  Rate Reduction* (Yu, Ma et al., NeurIPS 2023)](https://arxiv.org/abs/2311.13110).
- Intrinsically-interpretable **monosemantic tabular** nets:
  [XNNTab (2025)](https://arxiv.org/abs/2512.13442) (MLP + embedded SAE,
  "mechanistic, not post-hoc"); [TabCBM](https://openreview.net/forum?id=TIsrnWpjQ0);
  the additive NAM / EBM family (already programme baselines).

**Deep / multi-level monosemanticity (mostly post-hoc, on LLMs).**
- [Sparse feature circuits (Marks et al., ICLR 2025)](https://arxiv.org/html/2502.03032v3)
  — interpretable causal *graphs of features across layers* (closest cousin to a
  concept DAG, but post-hoc); crosscoders (Lindsey et al., 2024);
  [transcoders (Dunefsky et al., 2024)](https://www.researchgate.net/publication/397201244_Transcoders_find_interpretable_LLM_feature_circuits);
  matryoshka / meta-SAEs (multi-level hierarchy).

**Enhancing a *standard* architecture toward interpretability (closest to our
"enhance, don't replace" stance).**
- [Weight-sparse transformers have interpretable circuits (OpenAI, 2025)](https://arxiv.org/html/2511.13653v1);
  [Intrinsically Interpretable Attention via Sparse Post-Training (2025)](https://arxiv.org/pdf/2512.05865);
  [Interpreting Attention Layer Outputs with SAEs (2024)](https://arxiv.org/html/2406.17759v1).
  (Attention as dynamic routing is the seam; routing itself can be superposed —
  [MoE routing via SAEs](https://library.sparai.org/reports/interpreting-mixture-of-experts-routing-through-sparse-autoencoders-z8khhk/).)

**Constructive / depth-adaptive growth (for accuracy, not concepts).**
- AutoGrow, ANNA / [depth-adaptive DNNs](https://www.mdpi.com/2076-3417/13/1/398),
  layer-wise growth — grow depth adaptively, but for fit/efficiency, without a
  monosemantic-concept objective.

**Certification / trust (the field's weak spot).**
- [Stability selection with false-discovery control](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4464883/)
  (Meinshausen–Bühlmann; CPSS) — statistical guarantees the SAE/circuit literature
  largely lacks.

**Conclusion of the scan.** "Minimal monosemantic representations are the
interpretable/canonical ones," "build interpretable-by-design instead of explaining
black boxes," "deep monosemantic feature graphs," and "enhance a standard net with
sparsity to get interpretable circuits" are each an active frontier, not open
ground. A paper claiming any one as the contribution would read as rediscovery.
Note that our preferred **"enhance a standard deep NN toward convergence"** framing
sits *closest* to the weight-sparse-transformer / CRATE lineage — so the
differentiation must lean hardest on the items in §7.

## 6. Disproven / ruled-out hypotheses and approaches

Recorded so the dead ends are not re-walked. Basis tags: **[measured]** = the
basin-census experiment (`xplain-v3` PR #2); **[in-principle]** =
identifiability / representability argument; **[literature]** = falsified as a
*novelty / positioning* claim by the prior art in §5; **[programme]** = established
by the v1–v4 findings and reaffirmed here.

- **Disentanglement grows organically from prediction loss alone** —
  **disproven** [measured + in-principle]. A plain net from random init entangles
  (unit-level P(clean) ≈ 0 on compositional data) while still recovering the
  *subspace* (information present, R² ≈ 0.82); clean and entangled minima are
  equal-loss, so the objective carries no gradient toward the clean basin. ⇒ an
  inductive bias is mandatory; target mono-semanticity, not disentanglement.
- **A flat (single-hidden-layer) network can be disentangled on compositional
  data** — **disproven** [measured]. With the *correct* supports but no depth the
  model cannot represent a product (r² ≈ 0: a linear readout adds its factor-units,
  it cannot multiply them). ⇒ depth is non-negotiable for compositional concepts.
- **A decorrelation / statistical-independence penalty yields mono-semantic
  units** — **disproven** [measured]. Accuracy-neutral but insufficient — P(clean)
  stays 0; uncorrelated units are still linear *mixes* of factors. ⇒ the effective
  pressure is sparsity / selectivity + depth, not statistical independence.
- **Random / data-blind support seeding settles into structure** — **disproven**
  [measured]. Random fan-in supports cannot even fit the target (r² ≈ 0.1). ⇒
  basin / support selection is load-bearing (a screen, or growth toward it).
- **Structure/seeding trades accuracy for interpretability (a "concept tax")** —
  **refuted** [measured + programme]. The structured solution fits ≥ the entangled
  one (r² 0.999 ≥ 0.97); v4's concept tax was *negative*. ⇒ interpretability is
  ~free on this ground; do not frame or sell it as a trade-off.
- **Minimality alone drives interpretability** — **inverted** [in-principle].
  Superposition *is* the minimal-width encoding, so minimality left unchecked
  pushes toward *more* entanglement. ⇒ minimality helps only *atop* a
  mono-semanticity pressure; order the objectives accordingly.
- **A monosemantic unit is a true mechanism (mono-semanticity ⇒ correctness)** —
  **rejected** [in-principle + programme]. A cleanly-monosemantic unit can still be
  a dataset shortcut. ⇒ correctness is a separate guarantee, re-earned by stability
  + statistical certification, not implied by inspectability.
- **"A-priori interpretability" is itself the novel / defensible angle** —
  **falsified** [literature]. By-design interpretability is an established camp
  (Rudin 2019; CRATE; XNNTab; NAM / EBM / CBM), and the closest works are already
  a-priori. ⇒ a-priori is table stakes, not the pitch.
- **"Deep / multi-level monosemantic" is open research territory** — **falsified**
  [literature]. It is one of the most active areas (crosscoders, transcoders,
  sparse feature circuits, matryoshka SAEs, CRATE). ⇒ differentiate on
  certification + depth-honesty + small-n + regulated, not on "deep monosemantic".
- **Exact / full disentanglement (every factor recovered) as the deliverable
  target** — **retired** [programme + in-principle]. Unidentifiable in general and
  an over-fitting-prone benchmark; the periphery is intrinsically multiplicitous.
  ⇒ target "largely monosemantic" with an honestly-labelled polysemantic periphery.

## 7. The defensible corner and the moat

**Not defensible** (each is prior art in §5): "minimal monosemantic nets are
interpretable" (MDL-SAE, CRATE, Lachapelle); "a-priori interpretable" (Rudin,
XNNTab, EBM/CBM); "deep monosemantic" (feature circuits, crosscoders, CRATE);
"enhance a net with sparsity for interpretable circuits" (weight-sparse
transformers); "interpretable SLM" as a generic label.

**The defensible corner** is the thin *intersection* none of the above occupies as
a single package:

> A **by-design, constructively-driven, depth-honest, statistically-certified,
> largely-monosemantic** model — obtained by *enhancing a standard deep NN* rather
> than hand-building a bespoke formalism — **for bounded, *regulated*
> decisions**, whose deliverable is a **false-discovery-bounded audit certificate**
> mapped to model-risk-management / [EU AI Act Article 9](https://arxiv.org/html/2512.13907v3)
> validation requirements.

**The moat** is not the architecture — it is **certification + audit-artifact
packaging + by-design faithfulness + honest depth**, applied on the data type where
the programme has evidence. Regulated demand is becoming statutory (EU AI Act
high-risk, SR 11-7-style model risk, medical-device/GDPR duties), and the current
market answer — a black-box LLM plus post-hoc SHAP / audit trails — is a
*compliance patch*, not compliance (exactly Rudin's point). Neither the
post-hoc-LLM vendors nor the mechanistic-interpretability labs ship a model that is
faithful *by construction* **and** carries statistical evidence that its concepts
are real. That certificate is what a model-risk validator or regulator actually
needs, and it is the programme's genuine edge. The accepted polysemantic periphery
is handled honestly here too: it is *labelled* (low-stability / multiplicitous),
not passed off as certified.

**Sequencing (the strategy, and the trap to avoid).** Form factor follows the task;
order is load-bearing:

1. **Beachhead — tabular / structured regulated decisions** (credit, insurance,
   clinical risk, AML, sensor/scientific data) based on deep NN. Evidence exists (the programme's
   results + the basin census + the certification instruments); competition is thin; regulators are receptive.
2. **Then — regulated decision products** around that certified model.
3. **Frontier — purpose-trained small Transformers (SLMs)** for text-heavy
   regulated tasks. The approach applies in principle (§4), but this is the
   **harder, later, unproven** step: the programme has no language evidence, and
   certification on free text (what is the null? what is a false feature?) is
   unsolved. Earn your way here after the beachhead proves the certificate's value.