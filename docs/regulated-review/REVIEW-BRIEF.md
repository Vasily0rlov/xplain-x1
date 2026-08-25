# Model-Risk Validation Brief — XPLAIN-x1 Credit Certificates

**Purpose.** This package is prepared for review by a model-risk / model-validation
or compliance practitioner (SR 11-7 model risk management; EU AI Act high-risk
obligations). It asks a domain expert to judge whether the XPLAIN-x1 **audit
certificate** answers the questions a validator actually needs answered — and
where it falls short. It is *not* a claim that the models below are fit for
deployment; the datasets are public benchmarks used to exercise the certificate
format on a genuinely regulated decision type (consumer creditworthiness).

**Package contents.**
- `taiwan_credit_certificate.md` — the machine-and-human-readable certificate for
  a fully-powered run (Taiwan credit-card default, n=30,000).
- `german_credit_certificate.md` — a small-n run (German/Statlog credit, n=1,000)
  showing the honest degraded-certification regime.
- The interactive dashboards (`experiments/dashboards/taiwan_credit.html`,
  `german_credit.html`) — the same content with a drill-down concept DAG.

---

## 1. What the certificate is (in one paragraph)

XPLAIN-x1 trains a small, standard neural network under sparsity pressures into a
minimal, inspectable state, then **certifies the structure of its learned
function** — not a post-hoc approximation of a black box. The certified objects
are *purified additive components* of the model's own function under the declared
empirical measure (Layer F), each carrying a stability score across independent
retrainings (Π), a subsample-selection frequency (π, complementary-pairs
stability selection), and a **false-discovery bound E[V]**. A component is CORE
(certified) only if it clears all three plus a real-effect ablation test;
everything else is labelled periphery, never certified.

## 2. Section-by-section mapping to the frameworks

| certificate section | SR 11-7 | EU AI Act |
|---|---|---|
| Identification (data hash, seeds, config hash, git commit) | conceptual soundness; reproducibility | Art 11 technical documentation; Art 12 record-keeping |
| Performance and limits (fidelity vs ceiling, accuracy, restart spread) | outcomes analysis; benchmarking | Art 15 accuracy |
| Statistical certification (Π, π, E[V], universe, assumptions) | ongoing monitoring; effective challenge | Art 9 risk management; Art 15 robustness |
| Certified function components (Layer F) | model theory / variable roles | Art 13 transparency |
| **Protected-attribute non-reliance** | fair-lending / disparate-impact review | Art 10 data governance / bias |
| Portfolio reliance (Layer R) | sensitivity analysis | Art 15 robustness |
| Non-claims | model limitations; use constraints | Art 13 limitations disclosure |

## 3. The headline the certificate can make that a black-box + SHAP stack cannot

For each **declared protected attribute** the certificate states, with an
FDR-bounded guarantee, whether **any certified component of the model's decision
function relies on it**. On Taiwan this is an affirmative **non-reliance**
statement: sex, age, and marital status appear in *no* certified component (sex
is absent from the decomposition entirely; age and marital status sit in the
labelled, uncertified periphery). This is a *certified* structural statement, not
a post-hoc attribution average that can flip with the explainer's baseline.

## 4. Questions for the reviewer

Please assess each and note where the certificate is sufficient / insufficient:

1. **Faithfulness.** Do you accept that certifying the *model's own function*
   (rather than a surrogate) is the right object for reliance, and that the
   purified-component decomposition is a defensible variable-role account?
2. **Non-reliance.** Is the protected-attribute non-reliance statement, as framed
   (no *certified* component relies on the attribute; periphery appearances
   disclosed), sufficient for a fair-lending file? What would you additionally
   require — e.g. proxy/redundant-encoding analysis, outcome-level disparate
   impact, intersectional checks?
3. **The FDR bound.** Is E[V] (expected number of false certified components)
   over the declared structure-level universe a statistic your framework can use,
   and is the exchangeability assumption (stated in the certificate) acceptable
   for an adaptive training pipeline?
4. **The honest frontier.** Taiwan's fidelity ratio (0.861) and low absolute
   pseudo-R² are reported plainly; German carries an explicit small-n regime
   label (weaker certification at n=1,000). Does this honest degradation *help*
   or *hurt* the certificate's credibility with a validator?
5. **Gaps.** What does a model-risk validator need that this certificate does not
   provide (e.g. stability under data drift, challenger-model comparison,
   reason-code generation for adverse-action notices, calibration)?

## 5. Honest limitations (stated up front)

- These are **public benchmark datasets**, not a lender's production data; the
  purpose is to exercise the certificate format, not to validate a deployable
  model. Absolute accuracy is benchmark-limited.
- Non-reliance is asserted at the level of **certified components**; it does not
  by itself rule out reliance via an *unmeasured proxy* (a legitimate reviewer
  request — see Q2). Protected attributes were kept as visible predicates
  precisely so this analysis is possible.
- The certificate certifies *stable, real structure*, **not causal mechanism**
  and **not fitness for a specific decision** (see the certificate's Non-claims).
- No reason-code / adverse-action-notice generation is included yet (a known gap
  for a US credit deployment).

*Prepared by the XPLAIN-x1 programme, 2026-08-25. Certificates and dashboards are
reproducible from the committed run artifacts (config + data + git hashes in each
certificate's Identification section).*
