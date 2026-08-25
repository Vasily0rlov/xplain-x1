# XPLAIN-x1 Audit Certificate — german_credit

commit `e91e26368d35` · config `528920056f3f` · data `5a69e884239d5055-44b4ee009af0848b` · R=8 restarts, 40 CPSS runs

## Performance
- fidelity 0.194 vs ceiling 0.197 (ratio 0.9862)
- test accuracy: 0.71

## Structure
- widths [8], depth 1 (all earned: True)
- concepts: 0 CORE / 59 PERIPHERY; CORE coverage share 0
- periphery reasons: absent_in_main, infrequent, multiplicitous, no_effect, polysemantic, unstable

## Statistical certification
- E[V] <= 1.56e-05 at pi_thr 0.7 (q_mean 0.3, universe 19649)
- assumptions: CPSS exchangeability (Meinshausen-Buhlmann) over an adaptive learned pipeline is a modelling idealisation; the reality test provides an independent assumption-light check. The universe is structure-level: supports of arity <= F_max per layer interface of the delivered topology.

## CORE concepts
| unit | layer | form | mu | Pi | pi | delta [95% CI] | coverage |
|---|---|---|---|---|---|---|---|

## Certified function components (Layer F — unique under the declared measure)
reconstruction R² 0.86 · E[V] <= 0.00505
| component | groups | share | Pi | pi | label |
|---|---|---|---|---|---|
| checking_status=no checking | checking_status=no checking | 0.049 | 1.00 | 1.00 | CORE |
| credit_history=critical/other existing credit | credit_history=critical/other existing credit | 0.023 | 1.00 | 0.82 | CORE |
| savings_status=no known savings | savings_status=no known savings | 0.021 | 0.50 | 0.28 | PERIPHERY (unstable, infrequent) |
| purpose=used car | purpose=used car | 0.020 | 0.62 | 0.05 | PERIPHERY (unstable, infrequent) |
| duration | duration | 0.019 | 0.88 | 0.95 | CORE |
| credit_amount | credit_amount | 0.015 | 0.50 | 0.38 | PERIPHERY (unstable, infrequent) |
| checking_status=<0 | checking_status=<0 | 0.012 | 0.88 | 0.65 | PERIPHERY (infrequent) |
| age | age | 0.012 | 0.75 | 0.55 | PERIPHERY (infrequent) |
| purpose=new car | purpose=new car | 0.000 | 0.62 | 0.17 | PERIPHERY (absent_in_main, unstable, infrequent) |
| savings_status=<100 | savings_status=<100 | 0.000 | 0.25 | 0.47 | PERIPHERY (absent_in_main, unstable, infrequent) |
| housing=own | housing=own | 0.000 | 0.25 | 0.47 | PERIPHERY (absent_in_main, unstable, infrequent) |

## Protected-attribute non-reliance (fair lending — SR 11-7 fairness / EU AI Act Art 10)
**The model's certified decision structure does NOT rely on any declared protected attribute: none appears in a certified (CORE) function component.**

| protected attribute | in decomposition? | certified? | max share | status |
|---|---|---|---|---|
| personal_status | no | no | 0.0000 | absent from decomposition |
| age | yes | no | 0.0120 | labelled periphery only |
| foreign | no | no | 0.0000 | absent from decomposition |

*Basis: certified Layer-F components under the declared measure; FDR-bounded (E[V]).  Absence here is a certified non-reliance statement, not a post-hoc approximation.*

### Proxy screen (indirect reliance)
**No certified driver is a strong or notable proxy for a protected attribute (max |ρ| = 0.183, below the 0.30 screen).**

| certified driver | nearest protected attr | max \|ρ\| | flag |
|---|---|---|---|
| credit_history=critical/other existing credit | age | 0.183 | weak |
| duration | foreign | 0.171 | weak |
| checking_status=no checking | age | 0.095 | weak |

*Method: |Spearman ρ| of each certified driver vs each protected-attribute column on the full dataset; screen thresholds notable 0.30 / strong 0.50.*

## Portfolio reliance (Layer R — every restart relies on)
| group | min reliance | max |
|---|---|---|
| checking_status=no checking | 0.0612 | 0.1367 |
| credit_history=critical/other existing credit | 0.0342 | 0.1122 |
| checking_status=<0 | 0.0256 | 0.0662 |
| age | 0.0185 | 0.0416 |
| purpose=used car | 0.0154 | 0.0896 |
| duration | 0.011 | 0.1339 |
| other_parties=guarantor | 0.0069 | 0.0446 |
| employment=4<=X<7 | 0.0031 | 0.0183 |
| savings_status=no known savings | 0.0023 | 0.0195 |
| property_magnitude=real estate | 0.002 | 0.0132 |

## Certified routes (group level — stable across retrainings)
feature groups: 48 · E[V] <= 1.66e-05 at route universe
| route | groups | Pi | pi | delta [95% CI] | members (this model) | variants |
|---|---|---|---|---|---|---|

### Route periphery (labelled)
| route | groups | Pi | pi | reasons |
|---|---|---|---|---|
| r0 | checking_status=no checking, credit_history=critical/other existing credit, employment=4<=X<7 | 0.12 | 0.00 | unstable, infrequent, no_effect |
| r1 | checking_status=no checking, purpose=new car, existing_credits | 0.12 | 0.00 | absent_in_main |
| r2 | duration, credit_amount, age | 0.12 | 0.00 | absent_in_main |
| r3 | credit_history=critical/other existing credit, purpose=used car, credit_amount | 0.12 | 0.00 | polysemantic, unstable, infrequent, no_effect |
| r4 | credit_history=delayed previously, credit_amount, existing_credits | 0.12 | 0.00 | absent_in_main |
| r5 | purpose=radio/tv, credit_amount, employment=>=7 | 0.12 | 0.00 | unstable, infrequent, no_effect |
| r6 | purpose=new car, installment_commitment, age | 0.12 | 0.00 | absent_in_main |
| r7 | purpose=used car, age, existing_credits | 0.12 | 0.00 | unstable, infrequent, no_effect |
| r8 | credit_amount, savings_status=no known savings, other_parties=guarantor | 0.12 | 0.00 | absent_in_main |
| r9 | credit_amount, employment=<1, age | 0.12 | 0.00 | absent_in_main |
| r10 | personal_status=male mar/wid, property_magnitude=real estate, property_magnitude=life insurance | 0.12 | 0.00 | absent_in_main |

## Periphery (labelled, not certified)
| unit | layer | mu | Pi | pi | reasons |
|---|---|---|---|---|---|
| L1U7 | 1 | 0.51 | 0.38 | 0.03 | polysemantic, unstable, infrequent, no_effect, multiplicitous |
| L1U1 | 1 | 0.74 | 0.12 | 0.00 | polysemantic, unstable, infrequent, no_effect |
| L1U2 | 1 | 0.73 | 0.12 | 0.00 | polysemantic, unstable, infrequent, no_effect |
| L1U3 | 1 | 0.92 | 0.38 | 0.10 | unstable, infrequent, no_effect, multiplicitous |
| L1U4 | 1 | 0.74 | 0.12 | 0.00 | polysemantic, unstable, infrequent, no_effect |
| L1U5 | 1 | 0.94 | 0.12 | 0.00 | unstable, infrequent, no_effect |
| L1U6 | 1 | 0.84 | 0.12 | 0.00 | unstable, infrequent, no_effect |
| L1U9 | 1 | 0.55 | 0.12 | 0.00 | polysemantic, unstable, infrequent, no_effect |

## Non-claims
- CORE concepts are stable, real structures - not proven causal mechanisms of the world (M-C7).
- The periphery is labelled, not certified.
- Unit semantics require expert review (the soft target).
