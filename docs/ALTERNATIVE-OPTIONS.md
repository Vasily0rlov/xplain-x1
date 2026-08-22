# XPLAIN-x1 — Alternative Options

Options deliberately **not** taken in the current specs, recorded for future assessment.
Each entry states the option, why it was not chosen now, and what evidence would justify
revisiting it. Companion to `docs/01-METHOD-SPECIFICATION.md` (M-) and
`docs/02-SOLUTION-SPECIFICATION.md` (S-); mechanism-level substitutions already live in S-§14 —
this document is for *approach-level* alternatives.

---

## A1. Feature-detector monosemanticity (selectivity-based definition)

**Status:** not adopted · candidate for future assessment · recorded 2026-08-21.

### The option

Define a monosemantic unit as a **feature detector**: a unit that *activates only on one
coherent type/region of inputs* — "fires iff odor = foul", "fires iff temp ∈ [20, 30]" —
rather than the adopted definition (M-§3.2: monosemanticity `μ` = the unit is one simple **low-arity
function** of ≤ `F_max` parents, reproducible by a fixed simple surrogate class).

A detector-based variant of the method would change:

- **Metric.** Replace (or augment) monosemanticity `μ` with a **selectivity index**: concentration of
  the unit's activation distribution — e.g. activation frequency (lifetime sparsity), a
  class-/region-conditional selectivity score, or precision/recall of "unit active" against
  a candidate input predicate.
- **Pressure.** Strengthen the activation-sparsity term (S-§7) from instrument to primary
  objective — pushing units toward rare, gated, on/off firing; possibly add a
  bimodality/gating regulariser.
- **Certification objects.** A concept becomes "the predicate the unit fires on" (with the
  predicate itself extracted and certified) rather than "support + surrogate form".
- **DAG semantics.** Nodes read as *named events* ("foul-odor present") rather than
  *quantities/functions* ("≈ 0.8·hour·temp").

### Summary in the interpretability context

The two definitions optimise different halves of "interpretability" (POSITIONING §3 note
splits these deliberately):

| | adopted (low-arity function) | detector (selective activation) |
|---|---|---|
| optimises | **explainability-as-traceability**: every hop verifiable, routes auditable end-to-end | **interpretability-as-nameability**: units are pointable, nameable, category-like |
| unit reads as | a quantity / simple function | an event / predicate |
| auditor can | check each computation step | recognise what the unit "is about" |
| natural fit | continuous & graded structure (Bike `hour×temp`, Wine chemistry) | Boolean/predicate data (Zoo, Mushroom, Tic-Tac-Toe) |
| always-on graded unit (`≈ petal_length`) | monosemantic (correctly) | scores as non-selective — penalised despite being maximally legible |
| narrow detector over many features | polysemantic (correctly — not traceable) | scores as monosemantic — **nameable but not checkable** |

On predicate-style data the two definitions largely coincide (a monosemantic low-arity function of
binary inputs *is* a detector); they diverge on continuous data, where the current specs
knowingly chose traceability.

### Caveats (why it was not adopted now)

1. **The audit-trap caveat (decisive for the beachhead).** A detector can hide arbitrary
   complexity inside *how* it detects: a unit firing on one narrow region jointly defined
   over ten features is perfectly selective yet has no small-support explanation. Certifying
   it as "monosemantic" produces a nice label over an unverifiable computation — the post-hoc
   trap (Rudin) reproduced inside the model. For regulated audit, the validator must check
   reasoning, not admire labels; traceability is the harder, more valuable guarantee.
2. **It penalises legible graded units.** Continuous quantities that are maximally
   transparent (one parent, monotone form, always active) score badly on selectivity. On
   continuous-heavy ladder datasets this would push the model toward discretising smooth
   structure — inventing thresholds the data does not contain, in tension with honesty
   (M-C5/H-X1-2 spirit).
3. **Selectivity is distribution-relative.** "Fires rarely" depends on the evaluation
   distribution; under covariate shift a selective unit can become promiscuous with no
   change in mechanism. Monosemanticity `μ` (held-out functional describability) is the more
   shift-robust certificate object.
4. **Certification machinery would need rework.** The CPSS universe (M-§3.6) is defined
   over supports; a detector variant needs a predicate universe (region hypotheses), which
   is larger and makes the false-discovery bound harder to keep non-vacuous.
5. **Partial redundancy.** The adopted method already gets detector-like behaviour where the
   data warrants it, via the L1 selectivity pressure (S-§7) and tree surrogates that express
   thresholds/regions. The marginal gain is nameability on continuous data — exactly where
   the caveats bite hardest.

### What would justify revisiting

- **Expert-review evidence** from Phase P4–P5 DAG reviews that monosemantic-but-graded nodes are
  systematically hard for domain experts to name/validate (the soft target failing while the
  hard target passes).
- A **hybrid reading layer** proving cheap: keep `μ` as the certified definition, add a
  *reported-only* selectivity index per node so the DAG can render detector-style nodes as
  named events where both hold. (Lowest-cost first step; no change to certification.)
- The **SLM/text frontier** (POSITIONING §7, stage 3): in language models the
  feature-detector view is the native formalism of the surrounding literature
  (SAE features ≈ detectors); a detector-based definition may be the right bridge there.

### Assessment protocol (when picked up)

Run the ladder + synthetic suite (S-§5, S-§13) under three conditions — adopted definition,
detector definition, hybrid reporting — and compare: (a) expert nameability scores on DAG
review; (b) H-X1-2 honest-flatness violations (does detectorisation invent thresholds?);
(c) recovery matched-rates on planted concepts; (d) certificate strength (non-vacuous
`E[V]`, CI widths); (e) concept tax. Adopt the detector or hybrid variant only if (a)
improves without degrading (b)–(e).


## A2. Explicit abstraction levels (is-a hierarchy above composition)

**Status:** not adopted · candidate for a future iteration · recorded 2026-08-22
(owner request, following the depth-vs-abstraction terminology discussion).

### The option

Give the representation an explicit ABSTRACTION axis, orthogonal to depth
(which is strictly compositional order in the current method):

- **Recursive grouping** — groups of groups: the P6 collinearity groups
  (feature → determinable, e.g. Area/Perimeter/… → SIZE) applied again at
  group level, yielding a ladder of determinables (SIZE + SHAPE → MORPHOLOGY).
- **Route taxonomies** — routes clustered by shared anchors/sub-supports into
  more abstract route families, certified at each level with the same
  machinery (Π/π/E[V] per level).
- **Typed DAG edges** — distinguish `comprised-of` (composition, today's only
  edge meaning) from `is-a` (abstraction), so a unit like `milk ∧ hair` can be
  linked upward as an instance of class-predicate "mammal" where the data
  warrants it.

### Where abstraction already lives in the current method (implicit)

1. **Type vs token** — a concept/route is the class; the delivered model's
   unit is the instance.  All certification statistics are type-level claims
   validated over token instantiations.
2. **Feature → group (P6)** — one explicit determinable step, adopted because
   certification demanded it: uniqueness lives at the more abstract level.
3. **Abstraction through depth on predicate data** — a learned conjunction of
   Boolean predicates IS a class predicate (FCA-style formal concept), so
   composition and abstraction coincide there; on continuous data they diverge
   and depth stays purely compositional.

### Caveats

1. **Only certify levels the data forces.**  P6's lesson generalises: an
   abstraction level earns its place when stability/uniqueness exists there
   and not below.  Imposing a taxonomy a priori would be representational
   ideology — the opposite of the programme's certify-what-is-stable stance.
2. **Level discovery needs its own stability validation** (do the same
   higher groups re-emerge across runs/resamples?) and its own FDR universe
   per level; bounds must stay non-vacuous as universes shrink.
3. **Expert semantics risk**: auto-named abstract nodes (G0{Area, …}) are
   less nameable than their members; naming remains the expert's soft-target
   review.
4. **DAG complexity**: two edge types and multi-level nodes raise the
   reading burden; the two-layer view (routes + members) should stay the
   default, with deeper levels expand-on-demand.

### Revisit triggers

- A dataset with KNOWN taxonomy structure certifying flat routes that
  visibly share structure — the designed probe is **20 Newsgroups**
  (two-level label taxonomy; does the DAG recover comp.*/rec.* as
  intermediate abstractions?), already surveyed in `data/DATASETS.md`.
- P6 route tables on wider datasets showing many routes with common
  anchors/sub-supports (the raw material of a route taxonomy).
- Expert review reporting that group-level nodes remain too concrete to
  audit at domain level.

### Assessment protocol (when picked up)

Extend the E6 battery with a taxonomy-bearing dataset (20 Newsgroups or a
synthetic with planted 2-level structure): (a) does recursive grouping
recover the planted/known hierarchy (levels stable across runs)?  (b) do
per-level E[V] bounds stay non-vacuous?  (c) does the expert find the
multi-level DAG more navigable (soft target)?  Adopt per level only where
(a)+(b) hold.

---

*New approach-level options go here as further entries (A3, A4, …) with the same structure:
option → interpretability summary → caveats → revisit triggers → assessment protocol.*
