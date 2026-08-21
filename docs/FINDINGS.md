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
