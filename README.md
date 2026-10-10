# Large-Scale Product Experimentation & Uplift Modeling

This portfolio project studies a randomized advertising experiment with nearly 14 million users. It combines a reproducible DuckDB experimentation pipeline, intent-to-treat inference, exploratory treatment-effect heterogeneity, and an out-of-sample LightGBM T-Learner translated into targeting decisions.

> Project status: **Core analysis complete — data pipeline, A/B testing, exploratory heterogeneity, T-Learner, and test-set policy evaluation.**

## Business question

Does assignment to the advertising treatment increase visits and conversions, and is the estimated lift large enough to matter in practice?

The primary outcome is `conversion`; `visit` is a secondary outcome. The causal analysis uses randomized `treatment` assignment, not realized `exposure`. Exposure requires winning an ad auction and therefore should not be treated as if it were independently randomized.

## Dataset

The project uses the corrected **Criteo Uplift Prediction Dataset v2.1**:

- 13,979,592 users
- 12 anonymized features (`f0`–`f11`)
- randomized treatment assignment
- binary visit and conversion outcomes
- approximately 297 MB compressed

The raw data is excluded from Git. Review the [official Criteo dataset page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) and its CC BY-NC-SA 4.0 terms before downloading.

## Phase 1 scope

- Stream the compressed CSV from the official source and verify its expected columns
- Convert the 14M-row CSV to compressed Parquet with DuckDB
- Report row counts, missingness, treatment allocation, and covariate balance
- Estimate absolute lift, relative lift, risk ratio, 95% confidence intervals, and two-proportion z-tests
- Report Cohen's h and an approximate 80%-power minimum detectable effect (MDE)
- Save reproducible tables and figures for the final README

The treatment allocation ratio is reported but no sample-ratio-mismatch test is performed because the intended randomization probability is not published with the dataset.

## Phase 1 results

The pipeline successfully validated all 13,979,592 rows with no missing values or invalid binary indicators. All 12 pre-treatment features have absolute standardized mean differences below 0.05.

| Outcome | Control | Treatment | Absolute lift (95% CI) | Relative lift |
|---|---:|---:|---:|---:|
| Conversion | 0.194% | 0.309% | +0.115 pp [0.108, 0.122] | +59.4% |
| Visit | 3.820% | 4.854% | +1.034 pp [1.006, 1.063] | +27.1% |

Within this benchmark sample, assignment to treatment produced approximately 115 additional conversions and 1,034 additional visits per 100,000 users. These are precisely estimated intent-to-treat effects, not effects of realized exposure.

![Treatment and control outcome rates](figures/ab_test_effects.png)

![Pre-treatment covariate balance](figures/covariate_balance.png)

Because Criteo anonymized and prepared this dataset as a public uplift benchmark, these results support method comparison but should not be interpreted as production advertising ROI without treatment-cost and conversion-value information.

## Phase 2: Heterogeneity and uplift targeting

All 12 anonymized features were screened using quantile-based exploratory treatment effects. Tied cutpoints were collapsed rather than forcing artificial groups. Feature-level Cochran Q tests were adjusted with Benjamini–Hochberg separately for conversion and visit; 10 features had estimable and statistically detectable conversion-effect heterogeneity. Because features are anonymized, these patterns are not presented as customer personas.

![Exploratory treatment-effect heterogeneity](figures/exploratory_heterogeneity.png)

For conversion uplift, a LightGBM T-Learner was trained on a deterministic 70% train split, tuned with a 15% validation split, and evaluated once on the untouched 15% test split (2,097,078 users). Training used up to two million observations per randomized arm without class weighting or outcome-based sampling.

Test-set results:

- AUUC: **0.000956**
- Qini coefficient: **0.000348**
- Top 20% targeting retained **70.1%** of treat-all incremental conversions while treating **80% fewer users**
- Top 30% targeting retained **75.7%** of treat-all incremental conversions while treating **70% fewer users**

| Policy | Coverage | Incremental conversions per 100K (95% CI) | Illustrative net value per 100K |
|---|---:|---:|---:|
| Treat none | 0% | 0.0 | $0 |
| Top 20% uplift | 20% | 85.3 [69.0, 100.8] | $7,526 |
| Top 30% uplift | 30% | 92.1 [76.3, 107.2] | $7,714 |
| Treat all | 100% | 121.6 [103.4, 139.1] | $7,164 |

Net value assumes a conversion value of $100 and treatment cost of $0.05 per user. These values are a transparent sensitivity scenario, not observed financial data. Under that scenario, top-30% targeting has the highest estimated net value.

![Out-of-sample uplift deciles](figures/uplift_by_decile.png)

![Out-of-sample Qini curve](figures/qini_curve.png)

The uplift ranking is useful but not perfectly monotonic: the lowest predicted decile still shows positive observed lift. This is retained as an honest model limitation and motivates monitoring, calibration, and alternative learners before production use.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── sql/
│   └── metrics.sql
├── src/
│   ├── data.py
│   ├── experiment.py
│   ├── heterogeneity.py
│   ├── evaluation.py
│   └── uplift.py
├── models/
│   ├── t_learner_arm_0.txt
│   └── t_learner_arm_1.txt
├── notebooks/
│   └── README.md
├── figures/
└── reports/
    └── tables/
```

## Reproduce the analysis

Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Download the official corrected dataset (about 300 MB compressed).
python src/data.py download

# Validate it and create data/processed/criteo_uplift.parquet.
python src/data.py prepare

# Run experiment checks and A/B tests.
python src/experiment.py

# Run exploratory quantile-based heterogeneity analysis.
python src/heterogeneity.py

# Train the T-Learner and evaluate uplift/policies on the untouched test set.
python src/uplift.py
```

If the dataset has already been downloaded elsewhere:

```bash
python src/data.py prepare --input /absolute/path/to/criteo-uplift-v2.1.csv.gz
python src/experiment.py --data data/processed/criteo_uplift.parquet
```

Generated outputs:

- `data/processed/data_quality.json`
- `reports/tables/ab_test_results.csv`
- `reports/tables/experiment_arms.csv`
- `reports/tables/covariate_balance.csv`
- `reports/phase1_experiment_report.md`
- `figures/ab_test_effects.png`
- `figures/covariate_balance.png`
- `reports/phase2_heterogeneity_report.md`
- `reports/phase2_uplift_report.md`
- `reports/tables/heterogeneity_bins.csv`
- `reports/tables/uplift_deciles.csv`
- `reports/tables/policy_comparison.csv`
- `figures/exploratory_heterogeneity.png`
- `figures/uplift_by_decile.png`
- `figures/qini_curve.png`

## Statistical notes

- Difference-in-proportions confidence intervals use an unpooled standard error.
- The null-hypothesis z-test uses the pooled proportion.
- Relative effects are accompanied by log-scale risk-ratio confidence intervals.
- MDE is a local normal approximation around the observed control rate, not post-hoc “observed power.”
- With millions of observations, practical magnitude and confidence intervals matter more than p-values alone.
- Uplift deciles, Qini/AUUC, and policy comparisons are calculated only on the untouched test set.
- Policy confidence intervals use a stratified bootstrap conditional on each fixed targeting rule.
- `exposure` is not substituted for randomized `treatment` in causal estimates.

## Citation

```bibtex
@inproceedings{Diemert2018,
  author = {{Diemert Eustache, Betlei Artem} and Renaudin, Christophe and Massih-Reza, Amini},
  title = {A Large Scale Benchmark for Uplift Modeling},
  booktitle = {Proceedings of the AdKDD and TargetAd Workshop},
  year = {2018}
}
```
