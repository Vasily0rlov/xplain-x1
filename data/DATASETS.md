# Candidate Datasets for XPLAIN / SAGE-Net Evaluation

Consolidated survey of small-to-large public benchmark datasets, evaluated against XPLAIN-relevant criteria: **monosemanticity** (one column = one human concept), **additive vs interaction-driven** label structure, human-observable size, and CPU-feasibility.

### Reading the two method columns (added 2026-08-11)

- **Top order → depth** — the highest feature-interaction order in the label's *true*
  structure, and the DAG **depth** SAGE-Net needs to express it. With arity `F_max=3` a single
  composition unit absorbs up to a 3-way, so: order 1 ⇒ **d1** (additive / main effects);
  order 2–3 ⇒ **d2** (one composition layer); order 4–9 ⇒ **d3**; higher ⇒ deeper. `spatial`
  = effectively unbounded *local* structure (images); `tree` = recursive AST structure;
  `obscured` = rotated/anonymised features, so intrinsic order is undefined. A claimed order
  is the data's structure, not a promise the method certifies it — the power floor (n) and the
  gates decide (e.g. Adult's marital×sex×hours is order 2–3 but certifies honest-shallow d1).
- **Method fit** — ✅ suits SAGE-Net as-is · ⚠️ only with caveats (re-encoding, n below the
  power floor for the target order, very high dimensionality, or a non-standard task) · ❌
  unsuitable for the compositional-**factor** method. The driver is the *SAGE-Net data
  requirements* section at the foot of this doc — in one line: the method needs **monosemantic
  columns** (continuous or low-cardinality ordinal) and **n above the power floor** for the
  target order. Raw pixels (no spatial/convolutional inductive bias — it is not a vision model),
  PCA-rotated features (anti-monosemantic), and generative tasks (no target to certify) are ❌.

---

## Tiny / Small (<10k rows)

### Tabular

| Dataset | Rows | Features | Classes | Discriminative signal | Ceiling (approx) | Top order → depth | Method fit | XPLAIN relevance |
|---|---|---|---|---|---|---|---|---|
| **Iris** | 150 | 4 continuous | 3 | petal_len, petal_wid | ~97% | 1 → d1 | ✅ continuous, monosemantic | Baseline sanity check. Near-additive; setosa separable on one feature. Low interaction depth — good for verifying the architecture doesn't invent structure that isn't there |
| **Zoo** | 101 | 15 binary + 1 int (legs) | 7 | milk, feathers, backbone, legs | ~96–100% | 2 → d2 | ✅ Boolean predicates (ideal) | **Strongest monosemanticity fit.** Each column is a discrete human concept; label is a near-deterministic Boolean composition. Ideal for testing whether learned nodes align to interpretable predicates |
| **SPECT Heart** | 267 | 22 binary | 2 (binary) | diffuse, no single dominant | ~80–84% | 1–2 → d1–2 | ⚠️ binary (C3-ok) but anonymised regions | Purest binary structure (every column and target ∈ {0,1}), but features are anonymised image regions — tests separation-driven training *without* semantic priors to lean on |
| **Congressional Voting** | 435 | 16 binary (+missing) | 2 | physician-fee-freeze | ~95% | 1 → d1 | ✅ binary, monosemantic | Semantically loaded binary features with strong correlation structure. Good stress test for purification — one feature dominates while 15 others are collinear with it |
| **Seeds** | 210 | 7 continuous | 3 | groove length, kernel length, compactness | ~93–96% | 2 → d2 | ✅ continuous (certifies; first native multiclass) | **First native-multiclass certificate** (D-43, 2026-08-13). Screen: the real order-2 rung is **kernel_length × groove_length** (+0.29 rel-logloss lift), not the compactness ratio first hypothesised. OvR: **Kama** certifies depth-2 `area·groove_length` (cov 0.68); Rosa additively separable; Canadian shallow — the rung is **class-specific**. Native 3-class softmax certifies `groove_length` (cov 0.90) + compactness + area as depth-1 shared routes (pseudo-R² 0.61) |
| **Wine** | 178 | 13 continuous | 3 | flavanoids, proline, colour intensity | ~98% | 1 → d1 | ✅ continuous (certifies additive) | Chemically near-orthogonal discriminating axes. Clean case for testing whether learned factors recover the true low-dimensional structure (LDA finds it in 2D) |
| **Palmer Penguins** | 344 | 4 cont. + 3 cat. | 3 | bill_len × bill_dep | ~98% | 2 → d2 | ✅ mixed low-card (⚠️ missingness) | Mixed types + genuine missingness. `island` is a partial-information categorical — useful for testing how the DAG handles a feature that constrains but doesn't determine |
| **Heart Disease (Cleveland)** | 303 | 13 mixed | 2 (or 5) | thal, ca, cp | ~82–85% | 2 → d2 | ⚠️ mixed types, noisy | Realistic mixed-type medical data; moderate noise ceiling |
| **Breast Cancer Wisconsin (Diagnostic)** | 569 | 30 continuous | 2 | worst-radius, worst-concave-points | ~97–98% | 1 → d1 | ✅ continuous (redundancy test) | High feature count relative to rows, with heavy redundancy (mean/SE/worst triplets of the same 10 measures). Excellent purification test |
| **Breast Cancer (Original)** | 699 | 9 ordinal (1–10) | 2 | uniformity of cell size | ~96% | 1–2 → d1–2 | ✅ ordinal, meaningful order | Ordinal-discrete, human-readable scale. Sits between the binary and continuous regimes |
| **Pima Diabetes** | 768 | 8 continuous | 2 | glucose, BMI | ~77% | 1–2 → d1–2 | ✅ continuous (honest-uncertainty test) | Deliberate low-ceiling counterpoint — features only partially discriminate. Tests that the architecture reports honest uncertainty rather than overfitting structure |
| **Titanic** | 891 | ~10 mixed | 2 | sex, pclass, age | ~82–84% | 3 → d2 | ⚠️ order-3 but n<floor | Known interaction: sex × class is strongly non-additive ("women and children first" applied unevenly by class). **Good compositional depth probe** |
| **Tic-Tac-Toe Endgame** | 958 | 9 ternary (x/o/b) | 2 | none singly — 8 winning lines | 100% | 3 → d2 | ⚠️ recovers some lines; n=958 below floor | **Best compositional test in this bucket.** Label is a pure disjunction of eight 3-way conjunctions. Zero main effects, all interaction. If SAGE-Net recovers the eight lines as discrete nodes, that's a direct monosemanticity win |
| **Wine Quality (Red)** | 1,599 | 11 continuous | ordinal 3–8 | alcohol, volatile acidity | ~65% (exact) | 1–2 → d1–2 | ✅ continuous, ordinal target | Ordinal regression with class imbalance; low ceiling, realistic |
| **Spambase** | 4,601 | 57 continuous (word/char freq + capital-run stats) | 2 | "free", "!", "$", "remove", capital-run-length | ~94–95% | 1–2 → d1–2 | ✅ continuous, high-d (screen-pruning stress) | **High-dimensional *binary* — exercises screen pruning with no readout change** (added 2026-08-13; item #3 axis A). C(57,2)=1,596 candidate order-2 pairs, so the interaction screen must pre-rank rather than test-all. Mostly additive token main effects (`free`, `!`, `$`) plus genuine second-order lift (e.g. capital-run-length × `!`); the honest question is whether pruning + C2 parsimony keep it shallow rather than inventing depth |
| **Mushroom** | 8,124 | 22 categorical | 2 | **odor** alone → ~98% | ~100% | 2 → d2 | ✅ low-card ordinal (certified d2 backup) | Single-feature shortcut. Valuable adversarially: does the architecture collapse onto odor, or recover the redundant backup rules (spore-print-colour, gill-size)? |

### Images

| Dataset | Rows | Features | Classes | Discriminative signal | Ceiling (approx) | Top order → depth | Method fit | XPLAIN relevance |
|---|---|---|---|---|---|---|---|---|
| **Olivetti Faces** | 400 | 64×64 grey | 40 | eigenface subspace | ~95% | spatial → d3+ | ❌ raw pixels, not a vision model | Tiny, many classes, few examples each — tests separation under extreme class-to-sample ratio |
| **Semeion Digits** | 1,593 | 16×16 binary | 10 | stroke topology | ~93% | spatial → d3+ | ❌ binary pixels still non-monosemantic | **Binary pixels.** Rare combination of image structure with strictly Boolean inputs |
| **sklearn Digits** | 1,797 | 8×8 grey (0–16) | 10 | central/loop pixels | ~99% | spatial → d3+ | ❌ pixels not monosemantic (even at 8×8) | 64 features — small enough to inspect every input weight by hand |
| **Fashion-MNIST (test split)** | 10,000 | 28×28 grey | 10 | silhouette, texture | ~90% | spatial → d3+ | ❌ raw pixels, no spatial prior | Harder than MNIST; shoe/shirt confusions are semantically meaningful |
| **CIFAR-10 (test split)** | 10,000 | 32×32 RGB | 10 | colour + texture | ~70% w/o CNN | spatial → d3+ | ❌ raw pixels, no spatial prior | Realistically hard on CPU; include only if you want a difficulty anchor |

> **Can SAGE-Net process image data in any shape or form?** Not as raw pixels: an individual
> pixel is not a human concept, and the method has no convolutional/spatial inductive bias, so
> C3 has no monosemantic unit to stabilise. The only route is to first re-represent an image as
> a handful of **engineered monosemantic concept features** (e.g. segment counts, has-loop,
> stroke-width, region occupancy) and run the method on *those* — i.e. it can interpret a
> concept-level tabular view of images, but it is not itself a vision model.

### Text / Code

| Dataset | Rows | Features | Classes | Discriminative signal | Ceiling (approx) | Top order → depth | Method fit | XPLAIN relevance |
|---|---|---|---|---|---|---|---|---|
| **HumanEval** | 164 | NL prompt + tests | — (gen) | — | — | n/a (generative) | ❌ no target to certify | Too small for training; useful as a qualitative probe only |
| **MBPP** | 974 | NL + code + tests | — (gen) | — | — | n/a (generative) | ❌ no target to certify | "Basic Python problems" — construct-level inspection |
| **CoNaLa** | 2,879 | intent → snippet | — (gen) | — | — | n/a (generative) | ❌ no target to certify | One-liners; smallest useful code corpus |
| **BBC News** | 2,225 | TF-IDF | 5 topics | topic-marker terms | ~97% | 1 → d1 | ✅ monosemantic tokens (additive) | **Highly monosemantic after TF-IDF** — individual tokens map to topics. Good text analogue of Zoo |
| **Reuters R8** | ~7,600 | TF-IDF | 8 topics | topic terms | ~95% | 1 → d1 | ✅ monosemantic tokens (imbalanced) | Class-imbalanced variant of the above |
| **SMS Spam** | 5,574 | bag-of-words | 2 | "free", "claim", "£", digits | ~98% | 1 → d1 | ✅ few-token main effects | Near-perfect discrimination from a handful of tokens; heavily imbalanced (~13% spam) |
| **APPS** | 10,000 | problem text | 3 difficulty tiers | — | — | 1–2 → d1–2 | ⚠️ text→features; diffuse label | Difficulty labels let you isolate introductory-level constructs |

---

## Medium (10k–100k rows)

| Dataset | Rows | Features | Classes | Discriminative signal | Ceiling (approx) | Top order → depth | Method fit | XPLAIN relevance |
|---|---|---|---|---|---|---|---|---|
| **20 Newsgroups** | ~18,800 | TF-IDF (~50k vocab) | 20 | topic terms; header leakage | ~90% | 2 → d2 | ✅ hierarchical (⚠️ 50k-dim) | **Hierarchical label structure** (comp.\*, rec.\*, talk.\*) — a natural test for compositional depth discovery: does the DAG recover the two-level taxonomy? |
| **Twitter US Airline Sentiment** | ~14,600 | text + metadata | 3 | negation, airline mentions | ~80% | 2 → d2 | ⚠️ short noisy text | Short noisy text; negation handling is a genuine interaction |
| **Bike Sharing** (hourly) | ~17,400 | 8 cont.+ordinal | reg (count) | hour, temp, **hour×temp** | ~0.93 R² (GBM) | 2 → d2 | ✅ **certifies hour×temp depth-2** | **First real dataset with a certifiable order-2 rung** (realworld-interactions.md, 2026-08-12): `hour × temp` certifies as a depth-2 composition in **6/6 restarts** (coverage 0.82, a main route), val_r2 ~0.48. Interaction-lift screen winner; the honest depth-machinery testbed. §4b heredity found no order-3 to extend it (negative) |
| **Dry Bean** | 13,611 | 16 continuous (morphological) | 7 | ShapeFactor1, Compactness, Perimeter, roundness | ~92–93% | 2 → d2 | ✅ continuous; **certifies (native 7-class, d>15)** | **Item #3 flagship — d>15 AND native multiclass, both axes (D-43, 2026-08-13).** Native 7-class softmax certifies **pseudo-R² 0.88** with a robust **depth-2 `Compactness × ShapeFactor1`** composition (cov 0.26, all 6 restarts) atop 6 shape-descriptor depth-1 routes — the first native-multiclass certified interaction. Needed the multiclass tree_screen nominator's tree budget capped (`300/K` rounds) to tame the SHAP-interaction cost. ~23.5 min / 6 restarts |
| **Morpher — Russian declension, single words** (local: `data/morpher/morpher-sample-single-words-only.csv`) | 10,195 | 18 raw (UTF-16) → ~38 monosemantic binary predicates under the v4 encoding recipe | 3 (сущ 67% / нескл 21% / прил 12%) | ЧастьРечи (POS), Суффикс, РодЧисло, НаСогласную | ~96% (v4 teacher, OvR сущ 0.964) | 1–2 → d1–2 | ✅ low-cardinality categoricals → binary predicates (proven in v4) | **Local dataset, used successfully in xplain-v4.** Morphological features of Russian words (gender/number, animacy, POS, stress, syllable count, capitalisation, suffix, final-consonant, is-name) predicting declension type — near-deterministic *linguistic rules*, auditable by any Russian speaker: a POS main effect (прил ⇒ adjectival declension) plus genuine low-order morphological conjunctions (e.g. gender × final-consonant for indeclinables). Encoding: v4 `morpher_encode.py` recipe — drop the word-form id and the sentence-context punctuation columns (empty in this file). Companion `morpher-sample-sentences.csv` (rows in sentence order, "." separator rows) adds *context* — out of scope for the exchangeable-rows beachhead; a candidate future context-features variant (see `data/morpher/description.md`) |
| **Bank Marketing** | ~45,200 | 16 mixed | 2 | duration (leaky), poutcome, month | ~90% AUC | 1–2 → d1–2 | ✅ mixed (leak-surfacing diagnostic) | Contains a well-known **leakage feature** (`duration`). Diagnostic for whether interpretability surfaces the leak rather than hiding it |
| **Adult / Census Income** | ~48,800 | 14 mixed | 2 | relationship, education-num, capital-gain | ~87% | 2–3 claimed / d1 effective | ✅ mixed; certifies honest-shallow | Standard fairness benchmark; strong categorical interactions (marital × sex × hours). Good ANOVA decomposition target |
| **Connect-4** | ~67,600 | 42 ternary (board cells) | 3 (win/loss/draw) | none singly — high-order line patterns | ~87% | ≥8 → d3+ (native) | ❌ native (honest-limit) · ⚠️ re-tasked to lines (≤4) | **Very-high-order compositional test** (used 2026-08 as a binary-family probe). Order-probe shows the effective interaction order is >8 (unsaturated at depth 8): a genuine **honest-limit** case — SAGE-Net certifies 0 because the structure is above the finite-data power floor, not a failure. Use single-column ordinal encoding (one-hot is C3-unstable) |
| **IMDB Reviews** | 50,000 | text | 2 | sentiment lexicon | ~90% (BoW) | 1–2 → d1–2 | ✅ BoW main effects + negation | Balanced, long-form; lexical main effects plus negation interactions |
| **Covertype (subset)** | ~58,000 | 54 (10 cont. + 44 binary) | 7 | Elevation (dominant) | ~85% (GBM) | 1–2 → **d1 certified** | ✅ mixed; **certifies shallow** (distributed-interaction + one-hot limit) | **Baseline run 2026-08-15 (D-44, 50k subset, native 7-class):** certifies `Elevation` (cov 0.92) + one wilderness indicator, **pseudo-R² 0.38, NO depth-2** in any of 6 restarts. The screen shows real `Elevation×distance` interaction (+0.06 lift) but it is **distributed** (Elevation modulates several distances moderately) — no single stable rung, so C1/C3 keep it shallow (contrast Dry Bean's *concentrated* `Compactness×ShapeFactor1`, which certifies). The 40 one-hot soils confirm **D-32 at scale** (nominated in triples, none survive C3). Nominator tractable post-`300/K` fix (176 s at d=54). ~40 min / 6 restarts at 50k |
| **MNIST** | 70,000 | 28×28 grey | 10 | stroke geometry | ~99.7% | spatial → d3+ | ❌ raw pixels, not a vision model | The reference point — if XPLAIN loses much accuracy here, that's the interpretability tax quantified |
| **Fashion-MNIST (full)** | 70,000 | 28×28 grey | 10 | silhouette | ~93% | spatial → d3+ | ❌ raw pixels, not a vision model | Same shape, harder, more semantically structured confusions |
| **SVHN (cropped)** | ~73,000 | 32×32 RGB | 10 | digit shape + distractors | ~95% | spatial → d3+ | ❌ raw pixels, not a vision model | Real-world clutter; distractor digits at frame edges |

---

## Large (>100k rows)

These mostly exceed "human-observable," so treat them as sources to subsample from rather than training targets — especially given XPLAIN's CPU-only constraint.

| Dataset | Size | Features | Labels | Top order → depth | Method fit | XPLAIN relevance |
|---|---|---|---|---|---|---|
| **AG News (full)** | 120,000 | text | 4 topics | 1 → d1 | ✅ monosemantic tokens (subsample) | Very high ceiling (~92%); subsample to 10–20k for a clean, easy text benchmark |
| **Py150 / JS150** | 150,000 files | parsed ASTs | every node typed | 2 → d2 | ✅ **certifies `is_Assign × pos_0`** (binary predicates + D-42 NaN-guard) | **Highest value in this tier.** Construct labels come free from the tree. *The `parent` nominal must be decomposed into a few monosemantic binary predicates (raw one-hot is interchangeable-unstable D-32; single-column ordinal is arbitrary-code-unstable D-39).* With that, NameStore-vs-Load **certifies the depth-2 `parent=Assign × pos=0` rule (val_r2 0.89, cert 7)** — the first certified interaction on a high-cardinality categorical dataset — once the settle's near-separable NaN-divergence is guarded (D-42) |
| **Credit Card Fraud** | 284,807 | 30 numeric (28 are PCA components) | binary | obscured (PCA) | ❌ anti-monosemantic by construction | Extreme imbalance (~0.17%). Features are already PCA-rotated, so they're *anti*-monosemantic by construction — an interesting negative control |
| **Covertype (full)** | 581,012 | 54 | 7 | 1–2 → d1–2 | ✅ mixed cont/binary monosemantic | Full version of the medium entry; use only if you want the scaling curve |
| **SVHN (with extra)** | ~630,000 | 32×32 RGB | 10 | spatial → d3+ | ❌ raw pixels, not a vision model | Scale ceiling for the image track |
| **CodeSearchNet** | ~2M | function + docstring | 6 languages | 1–2 → d1–2 | ⚠️ needs monosemantic feature extraction from code | Function-granularity pairs; useful for a language-ID or docstring-alignment task |
| **CodeNet** | ~14M | source files | problem ID + accept/reject | tree → d2+ | ⚠️ oversized; subset + feature extraction | Vastly oversized; subset to a few thousand for a clean multi-class code task |

---

## Suggested Sequencing for XPLAIN

Proposed validation ladder alongside the 12-dataset synthetic suite:

1. **Zoo → Tic-Tac-Toe → Mushroom** as the interpretability core. Zoo tests monosemantic recovery on Boolean concepts; Tic-Tac-Toe tests pure compositional depth with zero main effects; Mushroom tests whether the architecture resists a single-feature shortcut. All three are CPU-trivial and fully inspectable.
2. **Iris / Wine / Seeds** as continuous-feature sanity checks — the known 2D discriminative subspaces should be recovered.
3. **Titanic + Adult** for genuine mixed-type interactions, where functional-ANOVA purification has something real to separate.
4. **Covertype subset** as the first test at scale with mixed continuous/binary inputs.
5. **Py150-derived construct classification** as the "real data, perfect labels" capstone.

### Known Gap

Nothing in this list has *known ground-truth* compositional structure except Tic-Tac-Toe. That is exactly what the 12-dataset synthetic suite is for — these public sets validate that the architecture behaves sensibly on data not designed by us, but they cannot confirm it recovered the true DAG, because nobody knows what the true DAG is.

---

## SAGE-Net data requirements (what suits the method — 2026-08-10)

The binding constraint is **C3 reproducibility** (a certified unit must re-emerge across bootstrap
resamples). Full rationale + the heredity-testbed implications in **Upgrade-Plan §4b.2**; in brief:

- **Monosemantic columns** — one column = one human concept. PCA/rotated features (Credit-Card-Fraud)
  are anti-monosemantic — a negative control, nothing interpretable to certify.
- **Continuous, or low-cardinality ordinal with a *meaningful order*** (Wine chemistry; TTT {-1,0,1};
  ordinal ratings). Interactions must add real held-out value (≥ δ_rel) over additive, else the method
  honestly certifies shallow (Adult, Wine).
- **n above the power floor** for the target order (~2k for order-2; steeply more for higher orders —
  Connect-4's >8-way *strategic* label is unreachable even at n≈67k; TTT n=958 is below the floor).
- **AVOID high-cardinality *nominal* categoricals** (D-32 + D-39, Py150): *one-hot* is C3-unstable
  (interchangeable indicators) AND *single-column ordinal* is C3-unstable (arbitrary code →
  non-reproducible fit). Fix = decompose into a few **monosemantic binary predicates**, whose
  conjunctions certify as value-free 2-ways.
- **Not raw pixels, and not a generative target.** Images have no monosemantic columns and the method
  has no spatial inductive bias (interpret a *concept-level* tabular view instead); generative code/NL
  tasks have no label to certify. These are the ❌ rows above.
- **Best heredity-lever testbed candidate:** Connect-4 **re-tasked to local line detection** (order ≤4)
  — a genuine 2→3→4 ladder, clean ternary→ordinal encoding, n≈67k above the floor. Native Connect-4
  (game outcome) does NOT work — it is the honest-limit case (Upgrade-Plan §4b.2).
