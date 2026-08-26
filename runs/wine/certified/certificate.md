# XPLAIN-x1 Audit Certificate — wine

commit `822caf1399dd` · config `ad6a837b2b50` · data `fdd1a162030a8e5b-28570293bbf6f313` · R=8 restarts, 40 CPSS runs

## Performance
- fidelity 0.956 vs ceiling 0.940 (ratio 1.0166)
- test accuracy: 0.9444444444444444

## Structure
- widths [7], depth 1 (all earned: True)
- concepts: 0 CORE / 37 PERIPHERY; CORE coverage share 0
- periphery reasons: absent_in_main, infrequent, multiplicitous, no_effect, unstable

## Statistical certification
- E[V] <= 0.249 at pi_thr 0.7 (q_mean 6.1, universe 377)
- assumptions: CPSS exchangeability (Meinshausen-Buhlmann) over an adaptive learned pipeline is a modelling idealisation; the reality test provides an independent assumption-light check. The universe is structure-level: supports of arity <= F_max per layer interface of the delivered topology.

## CORE concepts
| unit | layer | form | mu | Pi | pi | delta [95% CI] | coverage |
|---|---|---|---|---|---|---|---|

## Periphery (labelled, not certified)
| unit | layer | mu | Pi | pi | reasons |
|---|---|---|---|---|---|
| L1U4 | 1 | 0.94 | 0.38 | 0.40 | unstable, infrequent, multiplicitous |
| L1U0 | 1 | 0.90 | 0.88 | 0.62 | infrequent, no_effect, multiplicitous |
| L1U3 | 1 | 0.98 | 0.50 | 0.60 | unstable, infrequent, multiplicitous |
| L1U6 | 1 | 0.92 | 0.38 | 0.42 | unstable, infrequent, no_effect, multiplicitous |
| L1U1 | 1 | 0.93 | 0.62 | 0.30 | unstable, infrequent, no_effect, multiplicitous |
| L1U8 | 1 | 0.95 | 0.12 | 0.17 | unstable, infrequent, no_effect |
| L1U9 | 1 | 0.94 | 0.25 | 0.03 | unstable, infrequent, no_effect |

## Non-claims
- CORE concepts are stable, real structures - not proven causal mechanisms of the world (M-C7).
- The periphery is labelled, not certified.
- Unit semantics require expert review (the soft target).
