# XPLAIN-x1 — Models & Run Results Catalog

*Single index of every trained model and run-result artifact produced by this
programme, attributed to the build-plan phase / milestone that produced it.
Maintained so any result can be found and referred back to in future.*

**Basis:** [`docs/03-BUILD-PLAN.md`](docs/03-BUILD-PLAN.md) (phases P0–P8, milestones
M1–M9) · [`docs/FINDINGS.md`](docs/FINDINGS.md) (verdicts). Cataloged 2026-08-26 at
`main`. Frozen config tag: **`freeze-x1-v1`**.

## How to read this catalog

- **Persisted weights vs recorded structure.** Only one run on the box persisted
  actual model *weights* (`runs/wine/certified/model.pt` — the M5 end-to-end
  deliverable). Every other run recorded its **certified structure** (DAG +
  certificate + fANOVA components) as JSON/HTML rather than weights; those live in
  `experiments/results/` and `experiments/dashboards/` and are the reviewable
  objects the programme actually certifies. This is by design — the certified
  claim is about *stable structure across restarts*, not a single weight tensor.
- **Reproducibility.** Each artifact is reproducible from its provenance triple
  (`config_hash`, `data_hash`, `git_commit`) plus the frozen config. The wine run
  carries a full `provenance.json`; experiment JSONs record their seeds/config.
- **Milestones** (🏁) mark programme-level reviewable deliverables; see the map in
  the build plan.

---

## 1. Saved model checkpoints (persisted weights)

| run | files | phase · milestone | what it is | provenance |
|---|---|---|---|---|
| `runs/wine/certified/` | `model.pt`, `certificate.json`, `certificate.md`, `concepts.json`, `dag.json`, `dag.dot`, `provenance.json` | **P4 · E4.1 · 🏁 M5** | First real-data end-to-end deliverable: trained MaskedMLP (widths [7], depth 1, all-earned) + concept DAG + audit certificate. fidelity 0.956 vs ceiling 0.940 (ratio 1.017, beats v4 anchor 0.766); 0 CORE / 37 PERIPHERY at unit level (carving multiplicity — later re-earned at route/function level in P6/P7). | commit `822caf1399dd` · config `ad6a837b2b50` · data `fdd1a162030a…` · R=8, 40 CPSS |

> **Note.** This is the *only* run that wrote a full weight+artifact directory. The
> route-level (P6), function-level (P7), regulated-credit, and batched-growth (P8)
> runs recorded their certified structure into `experiments/dashboards/` and
> `experiments/results/` instead of persisting weights. If we later want persisted
> weights for the frozen bike / taiwan / drybean / mushroom / adult models, they
> can be regenerated deterministically from the frozen config + seeds and added
> here under `runs/<dataset>/`.

---

## 2. Experiment run results — `experiments/results/*.json`

Pre-registered experiments (`E<phase>.<n>`), each with its result JSON and (where
present) a run log. Verdicts and instrument-correction trail are in `docs/FINDINGS.md`.

| result file | experiment | phase · milestone | one-line result |
|---|---|---|---|
| `e01.json` | E0.1 | P0 · 🏁 **M1** | Harness sound — plain settle runs at the reference ceiling on zoo/wine. |
| `e11.json`, `e11b.json` | E1.1 / E1.1b | P1 · 🏁 **M2** | Pressures work — impurity 0.094→0.013 (−86%) at ≤2% fidelity cost (re-scored on the ADD6 instrument fixture). |
| `e12.json` | E1.2 | P1 | Pinned-λ sanity — healthy minimality, not pressure pathology (MET under corrected bar). |
| `e13.json`, `e13b.json`, `e13c.json` | E1.3 chain | P1 | Pressure-strength calibration → Hoyer fan-in term; final λ_act 1e-2, λ_fanin 0.03. |
| `e2x.json` | E2.1 / E2.2 / E2.3 | P2 · 🏁 **M3** | Honest structure — flat where flat (8/8), earned depth on COMP2/COMP3, power-floor characterised. |
| `e31.json`, `e31_diag.json` | E3.1 | P3 · 🏁 **M4** | First certified concepts — recovery on all compositional configs; 0 certified on NOISE (untradeable). |
| `e32_e33.json` | E3.2 / E3.3 | P3 | Matching stable τ∈[0.5,0.9]; FDR bound non-vacuous (E[V]≤0.124 structure-level vs 2.18 raw). |
| `e41.json` | E4.1 | P4 · 🏁 **M5** | Wine end-to-end (see saved run above) — v4-anchor beaten on fidelity ratio. |
| `e51.json`, `e51_diag_drybean.json` | E5.1 / E5.2 / E5.3 | P5 · 🏁 **M7** | MVL dev battery — H2/H4/H5/H6 ✓; **H1/H3 ✗** at unit level (carving multiplicity confirmed real, not power); cost 0.55h. |
| `e6x.json`, `e6x_pass2.json`, `e62_diag.json` | E6.1 / E6.2 | P6 · 🏁 **M7b** | First certified real-data **routes** (feature-group reliance); granularity shortfall recorded, escalated to function level. |
| `e63.json` | E6.3 | P6 | Canonicalisation probe (deterministic tie-break) — unit-level Π change quantified. |
| `e7x.json` | E7.1 / E7.2 / E7.3 | P7 · 🏁 **M7c** | Certified **function components** (purified fANOVA, Layer F) on real data at Π≈1.0; bike hour×temp shown to be a <1% artifact of the unpurified v3-era screen. |
| `e81.json` | E8.1 | P8 | Batched width growth (`grow_batch=8`) — bike fidelity ratio 0.945→**0.9877**, μ 0.992, guards 3/3. |
| `e82.json` | E8.2 | P8 | Claims-invariance under batched growth — the 4 frozen bike components re-certify CORE (Π=1.0); hour×temp stays uncertified. |
| `e83.json` | E8.3 | P8 | Interaction-partner diagnostic — partner is stably `weekday` 8/8 in both regimes (not multiplicitous). |
| `ec1.json` | E-C.1 | post-P5 · 🏁 **M9** | **Confirmatory one-shot** on untouched seeds 20–24 (frozen config, no reruns) — the programme's headline verdict, stands as read. |

Companion `*.log` files record each run's console trace. The P8 experiment logs
(`e81.log`, `e82.log`, `e83.log`) are committed alongside the earlier e-series logs;
transient dashboard-*build* logs (`dash_*.log`, `dashboards.log`) are left out as
regenerable tooling output — the dashboards they produced are committed (§4).

---

## 3. Certified-model dashboards — `experiments/dashboards/`

The **M6 owner-review** deliverable: for each processed dataset, a self-contained
interactive HTML dashboard (drill-down concept DAG: Layer S routes → Layer F
function components → certificate) plus its machine-readable `<dataset>.json`
(certified DAG + certificate + measured-carrier membership). Open
`experiments/dashboards/index.html` for the gallery.

**Certified datasets (default `grow_batch=2`, frozen-certified):**
`adult`, `bike`, `drybean`, `german_credit`, `morpher`, `mushroom`, `spambase`,
`synthetic_comp2-noisy-8k`, `synthetic_comp3-noisy-8k`, `taiwan_credit`,
`tictactoe`, `wine`, `zoo` — each as `.html` + `.json`.

**Accuracy-mode variants (`grow_batch=8`, P8 opt-in):**
`bike@grow-batch-8`, `morpher@grow-batch-8`, `spambase@grow-batch-8` — the
higher-accuracy / higher-complexity models (S-§16.1 trade-off).

**Meta-models (certified structure as a relational schema):**
`bike_ermodel.html`, `taiwan_credit_ermodel.html`.

**Domain relational models (the real-world entities behind the flat table):**
`bike_domain_ermodel.html`, `taiwan_credit_domain_ermodel.html` — each with a
"certified findings → domain model" mapping.

Producers: `experiments/build_dashboards.py`, `experiments/build_ermodels.py`,
`experiments/build_domain_ermodels.py`.

---

## 4. Baseline comparison — `baselines/standard_nn/`

`bike_compare.json` + `bike_compare.html` — side-by-side standard feed-forward NN
(1 hidden layer, plain backprop) vs the x1 certified model on bike, with AU-ROC /
confusion / performance widgets. **This comparison is what triggered P8**: it
traced bike's fidelity shortfall to per-step under-growth on diffuse residuals.
Code: `baseline.py`, `compare_bike.py`, `render_compare.py`.

---

## 5. Regulated-credit certificates — `docs/regulated-review/`

The genuinely-regulated deliverable (roadmap item #5) + the practitioner package (#6):

- `taiwan_credit_certificate.md` — fully-powered run (n=30,000); PAY_0 certifies
  (Π=1.0, literature-correct dominant driver); **protected-attribute non-reliance**
  (sex/age/marriage in no certified component) + clean proxy screen (max |ρ| 0.186).
- `german_credit_certificate.md` — small-n regime (n=1,000), honest degraded certification.
- `REVIEW-BRIEF.md` — SR 11-7 / EU AI Act mapping + validator questions + limitations.

Interactive versions: `experiments/dashboards/{taiwan_credit,german_credit}.html` (§3).

---

## 6. Frozen config & provenance

- **Config freeze:** tag **`freeze-x1-v1`** (P5-4 / 🏁 M8) — the pinned config all
  standard-regime and confirmatory claims were produced under. Default
  `grow_batch: 2`; `grow_batch: 8` is the documented opt-in accuracy mode (P8-X).
- **Reproducibility:** the wine run's `provenance.json` carries the full triple
  (`config_hash`, `data_hash`, `git_commit`, seeds). Experiment JSONs record their
  own seeds/config. Reference empirical-measure samples used at certification
  (`data_cache/ref/<dataset>-1000.json`) are deterministic from the split seed and
  are **not** committed (regenerable; see below).

## 7. Deliberately not committed

| path | why |
|---|---|
| `data_cache/` | Regenerable raw-data cache (OpenML downloads + seeded reference samples) — inputs, not models/results. Datasets are documented in [`data/DATASETS.md`](data/DATASETS.md). |
| `experiments/results/dash_*.log`, `dashboards.log` | Transient dashboard-build console output; the dashboards themselves are committed (§3). |
| `.venv/`, `__pycache__/`, `papers/`, `uv.lock` | Environment / build artifacts (see `.gitignore`). |

> Sibling projects (`/opt/xplain-x2`, `xplain-v3/4/5`) have their own model
> artifacts (e.g. `xplain-x2/experiments/results/a4_bike_collapsed_model.pt`) and
> are **out of scope** for this repo's catalog.
