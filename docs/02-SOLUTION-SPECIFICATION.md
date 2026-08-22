# XPLAIN-x1 — Solution Specification

**Status:** draft for review · **Audience:** the implementing Claude Code model ·
**Companion:** `docs/01-METHOD-SPECIFICATION.md` (cited as **M-§n**; rationale lives there, this
document is the *how*). Scope: beachhead only — tabular/structured data, MLP realisation.

Every design decision here is an **opinionated default**: implement exactly what is pinned;
alternatives are listed in §14 and must not be silently substituted. All hyperparameters
live in the config schema (§12) — nothing numeric is hard-coded outside it.

---

## 1. Deliverable

A Python package `xplain_x1` + CLI that, given a dataset name and config, runs the full
pipeline **converge → canonicalise → certify → extract** (M-§4) and emits a run directory:

```
runs/<dataset>/<run-id>/
  model.pt              # final model (+ unit registry)
  audit.json            # per-unit metrics per audit round (time series)
  concepts.json         # matched concepts with μ, Π, π, Δ, label
  dag.json  dag.dot     # extracted concept DAG (M-§4.5)
  certificate.json      # machine-readable audit certificate
  certificate.md        # human-readable render
  provenance.json       # data hash, splits, seeds, config hash, git commit
```

## 2. Environment and compute

- Python ≥ 3.11; PyTorch (CPU build sufficient), numpy, pandas, scikit-learn, scipy, pyyaml,
  joblib. Package managed with `uv` (`pyproject.toml`); tests with `pytest`.
- **Device-agnostic:** all tensor code takes `device` from config (`cpu` default). The
  current box has **no GPU** (64-thread Xeon Gold 6246R, 125 GB RAM); if a GPU appears,
  `device: cuda` must work without code changes. Models are small — CPU is the primary
  target; GPU is only ever an option, never a requirement.
- **Parallelism model (pinned):** parallelise at the *process* level across pipeline
  repetitions (restarts, CPSS subsample runs, datasets) via `joblib` `loky`; inside each
  worker set `torch.set_num_threads(max(1, 64 // n_workers))` and
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to match — never oversubscribe. Default
  `n_workers = 16` for batteries (fits R=8 restarts × CPSS comfortably in queue), raise per
  config on big batteries.
- **Box etiquette (per project `CLAUDE.md`):** every battery entry point first calls
  `xplain_x1.util.box.wait_until_free(load_threshold=8.0, check_interval_s=60)` — poll
  1-min loadavg; start only when below threshold (sister projects may be running). Once
  started, use the box fully.
- **Determinism:** every stochastic step seeded from a single run seed via
  `numpy.random.SeedSequence` spawning; `torch.manual_seed` per worker; seeds recorded in
  `provenance.json`. Two runs with the same config + seed must produce identical
  certificates.

## 3. Repository layout

```
src/xplain_x1/
  data/        # registry.py, loaders.py, synthetic.py, encode.py, splits.py
  model/       # mlp.py (masked MLP + unit registry), gauge.py, ops.py (grow/prune/merge)
  train/       # losses.py, settle.py (train loop), reference.py (f_ref)
  audit/       # monosemanticity.py, fanin.py, contribution.py, dissolve.py, plateau.py
  controller/  # growth.py (settle→audit→act loop, M-§4.3)
  certify/     # restarts.py, matching.py, cpss.py, reality.py, labels.py
  extract/     # dag.py, certificate.py
  util/        # box.py, seeding.py, provenance.py, io.py
  cli.py
configs/default.yaml
experiments/   # battery scripts (ladder.py, synthetic_suite.py, v4_anchors.py)
tests/
docs/
```

## 4. Data module

- **Registry:** one entry per dataset with loader, task type, encoding recipe, and metadata
  (`n`, `d`, expected order→depth from `data/DATASETS.md`). Two tiers:
  - **Minimum viable ladder (MVL — the build target).** `synthetic:*` (§5) plus seven
    public datasets, each the sole carrier of a distinct evidential role, in build order:
    `zoo` (monosemantic recovery of nameable Boolean predicates; native multiclass),
    `tictactoe` (pure composition, zero main effects; the only public set with known
    ground-truth structure; below the power floor — honest partial recovery expected),
    `mushroom` (single-feature-shortcut resistance; multiplicity labelling; v4 anchor),
    `wine` (continuous features certifying additive; core-recovery + expert-alignment v4
    anchor; 3-class), `adult` (mixed types at moderate scale; honest-shallow certification;
    v4 anchor), `bike` (the only regression task; the certified real-data order-2
    interaction `hour×temp` named in the H-X1-3 bar), `drybean` (native-multiclass
    certified interaction at scale — the strongest multiclass evidence: 7-class,
    `Compactness×ShapeFactor1` certified depth-2 in the prior programme). All three M-§9
    v4 anchors and every M-§7 bar are carriable by this set.
  - **Extended tier (post-MVL, optional).** `iris`, `seeds`, `titanic`, `covertype50k` —
    additional redundancy and scale evidence; run only after the MVL bars are met.

  Public sets fetched via `sklearn.datasets`/OpenML with local cache under `data_cache/`;
  SHA-256 of the raw file recorded in provenance.
- **Encoding rules** (from `data/DATASETS.md` requirements, mandatory): continuous → standardise;
  low-cardinality ordinal → integer with meaningful order, standardised; binary → {0,1};
  **high-cardinality nominals → a small set of monosemantic binary predicates** (recipe
  stored per dataset in the registry; never one-hot, never arbitrary ordinal codes);
  missingness → explicit indicator column + median impute. Every encoded column carries a
  human-readable name — these names flow through to DAG nodes.
- **Splits:** stratified train/val/test = 60/20/20 by default, split seed independent of
  model seeds; plus a fixed **probe set** (2 048 rows sampled from train+val, or the whole
  set if smaller) used for signatures and matching (M-§3.6) — identical across restarts.
- **CPSS subsampler:** `B` complementary pairs — each pair is a random half/half partition of
  the training set; both halves are run (2·B pipeline executions) (M-§3.6).

## 5. Synthetic control suite (ground truth known)

Generator `data/synthetic.py`, **6 pinned configs**; all with `d = 10` standard-normal
features (irrelevant features included). Structures:

- **ADD:** additive `y = f1(x1) + f2(x2) + f3(x3)` (monotone nonlinear f's) — honest depth 1.
- **COMP2:** `y = f1(x1) + x2·x3` — one planted order-2 concept, honest depth 2.
- **COMP3:** `y = x1·x2·x3 + f4(x4)` — one planted order-3 concept, honest depth 2 (single
  composition unit at `F_max = 3`).
- **NOISE:** `y` independent of `x` — honest output: nothing certified.

Configs: COMP2 — the decisive recovery carrier — runs all three regimes: (clean, n=8k),
(noisy σ=0.3, n=8k), (noisy, n=2k, near the order-2 power floor); ADD and COMP3 run
(noisy, n=8k) only; NOISE runs (n=8k) only.
Ground-truth scoring: a planted concept counts **recovered** iff a CORE concept's support
equals the planted support; the suite reports matched-rate, false-support rate, and honest
flatness (H-X1-2/-3 in M-§7).

## 6. Model module

`MaskedMLP`: standard `Linear→ReLU` stack + linear head, with

- a boolean **edge mask** per weight matrix (pruning = mask, so optimizer state survives);
- a **unit registry**: every unit gets a persistent id `L{layer}U{counter}` at creation,
  surviving growth/prune/permutation — all audit/certify artefacts key on unit ids;
- **growth ops** (`model/ops.py`): `add_unit(layer, init='fresh'|'split:<uid>')` (split =
  clone the least-monosemantic unit's weights + Gaussian perturbation ×0.05, halve both clones'
  outgoing weights), `insert_layer(pos)` (initialised near-identity), `remove_unit`,
  `merge_units(a, b)` (sum outgoing weights, drop b), `dissolve_layers(ℓ, ℓ+1)` (replace by
  one distilled layer, §8);
- **gauge ops** (`model/gauge.py`, M-§4.2): scale-normalise incoming weight rows to unit L2
  norm (absorb into outgoing), permutation-sort units by contribution. Assert bitwise-level
  fidelity preservation (max output diff < 1e-5) after every gauge pass.

## 7. Training module

- Loss (M-§4.1): `CE/MSE + λ_act·mean(|a|/std(a)) + λ_fanin·Σ_u ||W_in[u]||_2 + wd·||θ||²`,
  where `std(a)` is a running per-unit activation std (detached).
- Optimiser AdamW, lr `1e-3`, batch 256 (full-batch if n < 2 048), weight decay `1e-4`.
- **Settle:** train until val fidelity plateau (no improvement ≥ `0.1%` relative over 20
  evals, eval every epoch), max 500 epochs, restore best-val weights.
- **Pressure annealing:** λ's ramp linearly 0 → full over the first 25% of the first settle;
  full strength thereafter (including after growth actions).
- `f_ref` (`train/reference.py`): unconstrained MLP (2×64, no pressures, same budget) **and**
  `HistGradientBoosting`; `Fid_ref`/`Acc_ref` = the better of the two on val. Computed once
  per dataset+split, cached.

## 8. Audit module

Run after every settle, on validation data; results appended to `audit.json`.

- **Contribution** of unit `u`: ablation drop — zero `u`'s outgoing edges, measure val
  fidelity drop (no retraining). Edge contribution: `|w| · std(parent)` normalised per unit.
- **Effective fan-in** `ef(u)`: count of parents with edge contribution ≥ `ε_edge = 0.02`
  of the unit's total.
- **Monosemanticity** `μ(u)` (M-§3.2): candidate supports = all subsets of the top-6 parents by edge
  contribution with `|S| ≤ F_max = 3`; surrogates: `|S|=1` → `sklearn` isotonic ∨ 5-knot
  cubic spline (best of); `|S|∈{2,3}` → depth-3 `DecisionTreeRegressor` ∨ degree-2
  polynomial ridge (best of). Fit on train activations, score R² on val; `μ(u)` = best val
  R², `S(u)` = argmax support, `form(u)` = winning surrogate (pickled + templated to text).
- **Plateau detector** (`audit/plateau.py`): fidelity plateau (settle criterion met) and
  monosemanticity stall (median layer `μ` improved < 0.01 over last two audits).
- **Dissolution test** (`audit/dissolve.py`, M-§3.4): distill layers `ℓ, ℓ+1` into one layer
  (width = current `m_{ℓ+1}`) trained ≤ 50 epochs to match the composed map (MSE on
  pre-activations) then fine-tune head 10 epochs; report val fidelity delta.

## 9. Growth controller (M-§4.3)

```
model = MaskedMLP(L=1, m=[8])
loop (max 12 rounds):
    settle(); gauge_pass(); audit()
    prune: mask edges with contribution < ε_edge
           remove units with ablation drop < ε_prune·Fid AND μ-support empty
           merge unit pairs with signature cosine > τ_match and act-corr > 0.95
    if any layer dissolution costs ≤ ε_depth: dissolve, continue
    gap = Fid_ref − Fid
    if gap ≤ δ_stop: break                       # converged at ceiling
    if mono_stalled and gap > δ_grow:
        if width_grown_last_round and gap unchanged (< δ_grow improvement):
            insert_layer(after least-monosemantic layer)  # depth is the remaining move
        else:
            add 2 units to the least-monosemantic layer
            (split its least-monosemantic unit + 1 fresh)
    else: continue settling (pressures still working)
    accept growth iff val fidelity gain ≥ δ_grow after next settle, else revert (keep a
    model snapshot per round)
final: settle → gauge → audit → prune once more
```

Pinned: `ε_prune = 0.001` (relative fid), `ε_depth = 0.005`, `δ_grow = 0.005`,
`δ_stop = 0.01`, caps `L ≤ 4`, `Σm ≤ 96`, rounds ≤ 12. Revert-on-no-gain makes growth
monotone in val fidelity; caps + round limit guarantee termination.

## 10. Certification module (M-§4.4)

- **Restarts:** `R = 8` (dev) / `16` (confirmatory batteries) full pipeline runs,
  independent seeds, in parallel (§2). Seeds 0–7 are dev; **seeds 20+ are reserved for
  confirmatory one-shot runs** and must never be used during development or tuning
  (M-§9 inherited risk).
- **Signatures & matching** (`certify/matching.py`, M-§3.6): signature of `u` = concat of
  (a) input-attribution vector — expected gradient × input over the probe set, aggregated to
  the `d` input features, L2-normalised — and (b) activation vector on the probe set.
  Match units across two runs greedily by descending cosine(a); accept iff cosine(a) ≥
  `τ_match = 0.7` **and** |Pearson(b)| ≥ 0.7. A **concept** = the transitive cluster of
  matched units + the modal support `S`; `Π(c)` = fraction of restarts represented.
- **CPSS** (`certify/cpss.py`): `B = 20` complementary pairs (40 pipeline runs, parallel);
  a concept is *selected* in a run iff some unit matches it with `μ ≥ μ_min`. `π(c)` =
  selection fraction; `E[V] ≤ q²/((2π_thr−1)·p)` with `π_thr = 0.7`, `q` = mean selected
  count, `p` = Σ_a C(d_parents, a) for a ≤ F_max computed per layer interface (structure-
  level universe, M-§3.6). Report bound + assumptions verbatim in the certificate.
- **Reality test** (`certify/reality.py`): `Δ(c)` = ablation fidelity drop of `c`'s unit in
  the final model; 1 000-resample bootstrap of the test set → 95% CI; pass iff
  `Δ ≥ δ_min = 0.005` and CI low > 0.
- **Labels** (`certify/labels.py`, M-§3.7): CORE iff `μ ≥ 0.8 ∧ Π ≥ 0.7 ∧ π ≥ 0.7 ∧`
  reality pass; else PERIPHERY with reason codes `{polysemantic, unstable, infrequent,
  no_effect, multiplicitous}`. Multiplicitous = concept cluster whose matched units carry
  ≥ 2 distinct modal supports across runs.

## 11. Extraction module (M-§4.5)

- `extract/dag.py`: nodes = input features + final-model units + outputs; node payload
  `{id, layer, support (named), form_text, μ, Π, π, Δ±CI, coverage_share, label, reasons}`;
  edges above `ε_edge` with weight = contribution share. `form_text` from surrogate
  templates, e.g. `"petal_length > 2.6 AND petal_width low"` (tree), `"≈ 0.8·hour·temp"`
  (polynomial). Emit `dag.json` + Graphviz `dag.dot` (CORE solid, PERIPHERY grey dashed).
- `extract/certificate.py`: `certificate.json` per the M-§4.5 content list, sections keyed
  to SR 11-7 / EU-AI-Act-Art-9 headings; `certificate.md` render with the DAG summary, the
  bars table (M-§7) evaluated, and the honest-depth statement (dissolution evidence per
  retained layer).

## 12. Config schema (`configs/default.yaml`)

All pinned values above, grouped: `data` (dataset, split seed, probe size), `model`
(L0=1, m0=8, caps), `train` (lr 1e-3, batch 256, max epochs 500, plateau 0.001/20,
λ_act 1e-3, λ_fanin 1e-3, wd 1e-4, anneal 0.25), `audit` (ε_edge 0.02, F_max 3, top-6
parents), `controller` (ε_prune 0.001, ε_depth 0.005, δ_grow 0.005, δ_stop 0.01,
rounds 12), `certify` (R 8, B 20, τ_match 0.7, π_thr 0.7, μ_min 0.8, Π_min 0.7,
δ_min 0.005), `compute` (device cpu, n_workers 16, load_threshold 8.0). CLI:
`xplain-x1 run --dataset wine [--config path] [--override key=val …]`, plus
`xplain-x1 battery --suite synthetic|mvl|extended|v4-anchors`.

## 13. Testing and acceptance

- **Unit tests:** gauge pass preserves outputs; growth/prune/merge/dissolve ops preserve
  registry integrity and (where claimed) fidelity; `μ` of a hand-built monosemantic unit
  ≈ 1 and of a hand-built mixed unit ≪ 1; matching identifies permuted clones; CPSS bound arithmetic;
  determinism (same seed ⇒ identical `concepts.json`).
- **Integration gates (CI, fast):** `synthetic:COMP2(clean, 8k)` — planted concept recovered
  as CORE with correct support in a 2-restart mini-battery; `synthetic:NOISE` — zero
  certified concepts (the K2 untradeable, M-§7).
- **Acceptance = the M-§7 bars + M-§9 anchor table on the MVL tier**, produced by
  `experiments/ladder.py` (MVL by default, `--extended` for the second tier) and
  `experiments/v4_anchors.py`. Dev on seeds 0–7; confirmatory one-shot on seeds 20+ after
  config freeze.

## 14. Flagged alternatives (do not substitute silently)

| pinned default | alternative (when to revisit) |
|---|---|
| L1 activation + group-lasso fan-in | proximal/hard-threshold group lasso; Hoyer sparsity (if L1 kills units or monosemanticity stalls globally) |
| start L=1, grow | start at data's claimed order→depth (if growth proves slow/unstable) |
| ablation-Δ contribution | path-integrated gradients (if ablation too noisy on correlated units) |
| greedy signature matching | Hungarian assignment (if greedy produces unstable clusters) |
| half/half CPSS pairs | r-concave CPSS variant with tighter bound |
| no decorrelation loss (M-C3) | none — measured dead end, do not add |
| post-hoc rotation of trained model | none — architecturally unsound (M-§3.3), do not add |

## 15. Build order

| phase | contents | exit criterion |
|---|---|---|
| P0 | package skeleton, data module + registry (synthetics + zoo/wine), MaskedMLP + registry + gauge, settle loop, f_ref | plain settle reaches `Fid_ref` ± 2% on zoo/wine; determinism test green |
| P1 | pressures + audit module | on `COMP2`: pressures raise median μ vs plain control; audit.json time series produced |
| P2 | growth controller | `ADD` stays L=1; `COMP2/3` grow to L=2 and match ceiling; `NOISE` stays trivial |
| P3 | certification (restarts, matching, CPSS, reality, labels) | CI gates green (§13); synthetic matched-rates reported |
| P4 | DAG + certificate | wine run yields complete `certificate.md` + `dag.dot` |
| P5 | MVL + v4-anchor batteries (extended tier optional, after) | M-§7 bars table + M-§9 anchor table filled on MVL, dev seeds |

Each phase lands as its own PR on a feature branch; merges to `main` only on owner approval
(project `CLAUDE.md`). Battery runs obey the box-etiquette gate (§2) and then use all 64
threads.
