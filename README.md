# xplain-x1

**XPLAIN-x1** — convergence-trained, statistically certified, largely-monosemantic
MLPs for regulated tabular decisions.  A standard deep net is trained under
sparsity/selectivity pressures into a minimal, depth-honest, mostly-monosemantic
state; certification (restarts, complementary-pairs stability selection,
reality tests) separates the trustworthy CORE from the labelled polysemantic
periphery; the deliverables are a concept DAG and an audit certificate.

## Documents

| doc | what |
|---|---|
| [docs/00-POSITIONING.md](docs/00-POSITIONING.md) | problem, hypothesis, prior art, defensible corner |
| [docs/01-METHOD-SPECIFICATION.md](docs/01-METHOD-SPECIFICATION.md) | formal method (researcher-facing) |
| [docs/02-SOLUTION-SPECIFICATION.md](docs/02-SOLUTION-SPECIFICATION.md) | implementable technical spec, pinned defaults |
| [docs/03-BUILD-PLAN.md](docs/03-BUILD-PLAN.md) | trackable delivery plan, milestones M1–M9 |
| [docs/DATASETS.md](docs/DATASETS.md) | dataset survey + method data requirements |
| [docs/ALTERNATIVE-OPTIONS.md](docs/ALTERNATIVE-OPTIONS.md) | approach-level options not taken |
| [docs/FINDINGS.md](docs/FINDINGS.md) | experiment verdicts + instrument corrections |

## Quickstart

```bash
uv sync                                   # py3.11 venv + CPU torch
.venv/bin/python -m pytest tests/ -q      # test suite
.venv/bin/xplain-x1 list                  # registered datasets
.venv/bin/xplain-x1 run --dataset wine    # baseline settle + reference ceiling
.venv/bin/xplain-x1 certify --dataset wine  # full pipeline -> DAG + certificate
```

Experiments live in `experiments/` (`e01.py`, `e11b.py`, `e13c.py`, `e2x.py`,
`e31.py`, `e41.py`, …); each is pre-registered in the build plan, writes
`experiments/results/<id>.json`, and prints a one-line verdict.  Batteries wait
for the box to be free (`compute.load_threshold`), then use all cores.
