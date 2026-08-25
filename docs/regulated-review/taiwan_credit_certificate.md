# XPLAIN-x1 Audit Certificate — taiwan_credit

commit `e91e26368d35` · config `528920056f3f` · data `9fec8ba454fae460-246c692533334b26` · R=8 restarts, 40 CPSS runs

## Performance
- fidelity 0.167 vs ceiling 0.195 (ratio 0.8605)
- test accuracy: 0.8231666666666667

## Structure
- widths [4], depth 1 (all earned: True)
- concepts: 0 CORE / 47 PERIPHERY; CORE coverage share 0
- periphery reasons: absent_in_main, infrequent, multiplicitous, polysemantic, unstable

## Statistical certification
- E[V] <= 0.0397 at pi_thr 0.7 (q_mean 6.1, universe 2324)
- assumptions: CPSS exchangeability (Meinshausen-Buhlmann) over an adaptive learned pipeline is a modelling idealisation; the reality test provides an independent assumption-light check. The universe is structure-level: supports of arity <= F_max per layer interface of the delivered topology.

## CORE concepts
| unit | layer | form | mu | Pi | pi | delta [95% CI] | coverage |
|---|---|---|---|---|---|---|---|

## Certified function components (Layer F — unique under the declared measure)
reconstruction R² 0.8936 · E[V] <= 0.0235
| component | groups | share | Pi | pi | label |
|---|---|---|---|---|---|
| PAY_0 | PAY_0 | 0.078 | 1.00 | 1.00 | CORE |
| LIMIT_BAL | LIMIT_BAL | 0.018 | 1.00 | 0.85 | CORE |
| PAY_3 | G7{PAY_3, PAY_4} | 0.011 | 0.75 | 0.57 | PERIPHERY (infrequent) |
| MARRIAGE=single | G3{MARRIAGE=married, MARRIAGE=single} | 0.000 | 0.38 | 0.25 | PERIPHERY (absent_in_main, unstable, infrequent) |
| PAY_4 | G7{PAY_3, PAY_4} | 0.000 | 0.12 | 0.25 | PERIPHERY (absent_in_main, unstable, infrequent) |
| PAY_5 | G8{PAY_5, PAY_6} | 0.000 | 0.12 | 0.12 | PERIPHERY (absent_in_main, unstable, infrequent) |
| PAY_6 | G8{PAY_5, PAY_6} | 0.000 | 0.38 | 0.35 | PERIPHERY (absent_in_main, unstable, infrequent) |
| PAY_AMT1 | PAY_AMT1 | 0.000 | 0.38 | 0.35 | PERIPHERY (absent_in_main, unstable, infrequent) |
| PAY_AMT2 | PAY_AMT2 | 0.000 | 0.12 | 0.33 | PERIPHERY (absent_in_main, unstable, infrequent) |
| BILL_AMT3 | G9{BILL_AMT1, BILL_AMT2, BILL_AMT3, +1} | 0.000 | 0.12 | 0.03 | PERIPHERY (absent_in_main, unstable, infrequent) |

## Protected-attribute non-reliance (fair lending — SR 11-7 fairness / EU AI Act Art 10)
**The model's certified decision structure does NOT rely on any declared protected attribute: none appears in a certified (CORE) function component.**

| protected attribute | in decomposition? | certified? | max share | status |
|---|---|---|---|---|
| SEX | no | no | 0.0000 | absent from decomposition |
| AGE | yes | no | 0.0000 | labelled periphery only |
| MARRIAGE | yes | no | 0.0000 | labelled periphery only |

*Basis: certified Layer-F components under the declared measure; FDR-bounded (E[V]).  Absence here is a certified non-reliance statement, not a post-hoc approximation.*

### Proxy screen (indirect reliance)
**No certified driver is a strong or notable proxy for a protected attribute (max |ρ| = 0.186, below the 0.30 screen).**

| certified driver | nearest protected attr | max \|ρ\| | flag |
|---|---|---|---|
| LIMIT_BAL | AGE | 0.186 | weak |
| PAY_0 | AGE | 0.064 | weak |

*Method: |Spearman ρ| of each certified driver vs each protected-attribute column on the full dataset; screen thresholds notable 0.30 / strong 0.50.*

## Portfolio reliance (Layer R — every restart relies on)
| group | min reliance | max |
|---|---|---|
| PAY_0 | 0.1495 | 0.2165 |
| G7{PAY_3, PAY_4} | 0.013 | 0.0343 |
| LIMIT_BAL | 0.0126 | 0.0252 |
| G9{BILL_AMT1, BILL_AMT2, BILL_AMT3, +1} | 0.0065 | 0.0782 |
| G8{PAY_5, PAY_6} | 0.0046 | 0.0232 |
| PAY_2 | 0.0026 | 0.0619 |
| PAY_AMT1 | 0.0003 | 0.0183 |
| G3{MARRIAGE=married, MARRIAGE=single} | 0.0001 | 0.0065 |
| SEX=female | 0.0 | 0.002 |
| AGE | 0.0 | 0.0017 |

## Certified routes (group level — stable across retrainings)
feature groups: 17 · E[V] <= 0.11 at route universe
| route | groups | Pi | pi | delta [95% CI] | members (this model) | variants |
|---|---|---|---|---|---|---|
| r0 | PAY_0 | 1.00 | 0.97 | 1.2541 [1.1801, 1.3343] | L2U10, L2U11, L2U12 | 26 |

### Route periphery (labelled)
| route | groups | Pi | pi | reasons |
|---|---|---|---|---|
| r1 | PAY_2 | 0.25 | 0.00 | absent_in_main |
| r2 | G7{PAY_3, PAY_4}, PAY_AMT1, PAY_AMT2 | 0.50 | 0.12 | absent_in_main |
| r3 | G8{PAY_5, PAY_6} | 0.50 | 0.28 | absent_in_main |
| r4 | G9{BILL_AMT1, BILL_AMT2, BILL_AMT3, +1}, PAY_AMT1 | 0.12 | 0.00 | absent_in_main |
| r5 | G3{MARRIAGE=married, MARRIAGE=single}, AGE, G9{BILL_AMT1, BILL_AMT2, BILL_AMT3, +1} | 0.12 | 0.00 | absent_in_main |

## Periphery (labelled, not certified)
| unit | layer | mu | Pi | pi | reasons |
|---|---|---|---|---|---|
| L2U10 | 1 | 0.93 | 0.88 | 0.88 | multiplicitous |
| L2U12 | 1 | 0.96 | 0.50 | 0.38 | unstable, infrequent, multiplicitous |
| L2U11 | 1 | 0.97 | 0.50 | 0.35 | unstable, infrequent, multiplicitous |
| L2U13 | 1 | 0.70 | 0.12 | 0.03 | polysemantic, unstable, infrequent |

## Non-claims
- CORE concepts are stable, real structures - not proven causal mechanisms of the world (M-C7).
- The periphery is labelled, not certified.
- Unit semantics require expert review (the soft target).
