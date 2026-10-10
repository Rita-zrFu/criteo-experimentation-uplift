# Phase 2B: T-Learner uplift and targeting policy

Two LightGBM response models were trained separately for randomized treatment and control users. A deterministic hash of `row_id` created 70% train, 15% validation, and 15% untouched test partitions. Model selection and early stopping used validation data; all uplift and policy results below use only the test partition.

Training is capped at 2,000,000 rows per arm and validation at 500,000 rows per arm using deterministic hash sampling. No outcome-based sampling or class weighting is used, preserving probability scale.

## Split counts

| data_split   |   treatment |   users |   conversions |
|:-------------|------------:|--------:|--------------:|
| test         |           0 |  314435 |           588 |
| test         |           1 | 1782643 |          5502 |
| train        |           0 | 1468361 |          2811 |
| train        |           1 | 8316836 |         25759 |
| validation   |           0 |  314141 |           664 |
| validation   |           1 | 1783176 |          5450 |

## Factual response-model diagnostics

| arm       |   users |   conversion_rate |   roc_auc |   log_loss |
|:----------|--------:|------------------:|----------:|-----------:|
| control   |  314435 |        0.00187002 |  0.959851 | 0.00814658 |
| treatment | 1782643 |        0.00308643 |  0.957815 | 0.0123345  |

ROC AUC and log loss diagnose the two response models; they are not treated as uplift metrics.

## Uplift evaluation

- AUUC: 0.00095595
- Qini coefficient: 0.00034775
- Test treatment propensity: 85.01%

Qini/AUUC use an inverse-propensity transformed outcome and are computed only on the untouched test set. Uplift-decile effects are estimated directly from randomized treatment-control differences within each predicted-score decile.

## Targeting policies

| policy         | coverage   |   incremental_conversions_per_100k |   ci_low_per_100k |   ci_high_per_100k | assumed_net_value_per_100k   |
|:---------------|:-----------|-----------------------------------:|------------------:|-------------------:|:-----------------------------|
| Treat none     | 0.0%       |                               0    |              0    |               0    | 0.00                         |
| Top 20% uplift | 20.0%      |                              85.26 |             69.04 |             100.75 | 7,526.44                     |
| Top 30% uplift | 30.0%      |                              92.14 |             76.33 |             107.21 | 7,713.99                     |
| Treat all      | 100.0%     |                             121.64 |            103.41 |             139.11 | 7,164.08                     |

Confidence intervals are stratified bootstrap intervals conditional on each fixed targeting rule. The illustrative net-value calculation assumes a conversion is worth $100.00 and each treatment costs $0.05; it is a sensitivity scenario, not an observed financial result.

## Leading features by average gain importance

| feature   |   control_gain |   treatment_gain |   average_gain |
|:----------|---------------:|-----------------:|---------------:|
| f11       |     0.423375   |        0.353195  |      0.388285  |
| f4        |     0.198487   |        0.190236  |      0.194362  |
| f8        |     0.109646   |        0.192159  |      0.150903  |
| f2        |     0.102316   |        0.0783651 |      0.0903404 |
| f9        |     0.0437375  |        0.0870197 |      0.0653786 |
| f10       |     0.0831668  |        0.0426132 |      0.06289   |
| f6        |     0.0140544  |        0.0170704 |      0.0155624 |
| f3        |     0.00633698 |        0.0191736 |      0.0127553 |

## Outputs

- `reports/tables/uplift_deciles.csv`
- `reports/tables/qini_curve.csv`
- `reports/tables/policy_comparison.csv`
- `reports/tables/model_diagnostics.csv`
- `reports/tables/feature_importance.csv`
- `figures/uplift_by_decile.png`
- `figures/qini_curve.png`
