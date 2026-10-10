# Phase 1: Experiment health and A/B test

## Design

The estimand is the intent-to-treat effect of randomized `treatment` assignment. Conversion is the primary outcome and visit is secondary. Realized `exposure` is not used as the randomized treatment indicator.

## Experiment health

- Total users: 13,979,592
- Observed treatment share: 85.00%
- Largest absolute standardized mean difference across f0–f11: 0.0488
- No sample-ratio-mismatch p-value is reported because the intended allocation probability is unavailable.

## A/B test results

| Outcome    | Control rate   | Treatment rate   | Absolute lift   | 95% CI            | Relative lift   | p-value   | 80% MDE   |
|:-----------|:---------------|:-----------------|:----------------|:------------------|:----------------|:----------|:----------|
| Conversion | 0.194%         | 0.309%           | 0.115 pp        | [0.108, 0.122] pp | 59.4%           | 7.31e-179 | 0.009 pp  |
| Visit      | 3.820%         | 4.854%           | 1.034 pp        | [1.006, 1.063] pp | 27.1%           | <1e-927   | 0.040 pp  |

The confidence level is 95%. MDE uses a local normal approximation with 80% target power. Statistical significance should be interpreted together with absolute lift, relative lift, and business value.

## Generated artifacts

- `reports/tables/experiment_arms.csv`
- `reports/tables/ab_test_results.csv`
- `reports/tables/covariate_balance.csv`
- `figures/ab_test_effects.png`
- `figures/covariate_balance.png`
