# Double Machine Learning extension

This analysis applies a cross-fitted interactive-regression Double Machine Learning estimator to conversion. The treatment nuisance model is logistic regression with standardized covariates; separate treatment and control outcome nuisances use LightGBM. Each observation is scored by nuisance models trained without its fold, using the orthogonal augmented inverse-propensity score.

Because treatment was randomized, DML is used for covariate-adjusted estimation, orthogonalization, and a robustness comparison—not to repair confounding or claim stronger causal identification.

## Results

| estimator                      | estimate   | standard_error   | ci_low    | ci_high   |
|:-------------------------------|:-----------|:-----------------|:----------|:----------|
| Randomized difference in means | 0.1152 pp  | 0.0034 pp        | 0.1085 pp | 0.1219 pp |
| Cross-fitted DML               | 0.0996 pp  | 0.0034 pp        | 0.0930 pp | 0.1062 pp |

The DML estimate differs from the raw randomized estimate by -0.0156 percentage points. Its standard error is 2.0% smaller than the unadjusted standard error. The confidence intervals do not overlap; this is a material covariate-adjustment sensitivity result rather than a near-identical replication of the raw estimate.

## Specification

- Outcome: `conversion`
- Treatment: randomized `treatment`
- Covariates: `f0`–`f11`
- Cross-fitting folds: 3
- Outcome nuisances: separate treatment/control LightGBM classifiers, fixed parameters, no class weighting
- Propensity nuisance: standardized logistic regression
- Maximum outcome-nuisance training rows per arm and fold: 1,000,000
- Maximum propensity training rows per fold: 500,000
- Score sample: all 13,979,592 users
- Uncertainty: heteroskedasticity-robust influence-function standard error

## Fold diagnostics

|   fold |   score_rows |   mean_propensity |   min_propensity |   max_propensity |   mean_control_prediction |   mean_treatment_prediction |
|-------:|-------------:|------------------:|-----------------:|-----------------:|--------------------------:|----------------------------:|
|      1 |  4.65765e+06 |          0.849846 |         0.802182 |         0.922764 |                0.00207715 |                  0.00309035 |
|      2 |  4.65969e+06 |          0.849556 |         0.820932 |         0.929016 |                0.00200558 |                  0.0030266  |
|      3 |  4.66226e+06 |          0.84997  |         0.7712   |         0.908384 |                0.00204792 |                  0.00298758 |
