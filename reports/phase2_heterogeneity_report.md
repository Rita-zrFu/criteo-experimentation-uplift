# Phase 2A: Exploratory treatment-effect heterogeneity

All 12 anonymized features were analyzed using global 20th/40th/60th/80th percentile cutpoints. Adjacent bins are collapsed when tied feature values share a cutpoint. Within each resulting bin, the intent-to-treat effect is the treatment-minus-control difference in outcome rates. Feature-level heterogeneity uses a fixed-effect Cochran Q test across estimable bin effects, followed by Benjamini–Hochberg adjustment across estimable features separately for each outcome.

Three non-duplicate conversion-effect profiles are visualized for exploration only. They were selected after inspecting the results and must not be interpreted as pre-specified confirmatory findings or named customer segments. Features without enough distinct quantile bins remain in the output as non-testable (`NaN`).

## Feature-level summary

| outcome    | feature   |   effect_range_pp |   heterogeneity_p_value |   bh_adjusted_p_value |
|:-----------|:----------|------------------:|------------------------:|----------------------:|
| conversion | f2        |          0.841641 |            2.42676e-188 |          2.42676e-187 |
| conversion | f8        |          0.490432 |            1.20269e-187 |          6.01345e-187 |
| conversion | f9        |          0.46313  |            8.09376e-178 |          2.69792e-177 |
| conversion | f3        |          0.443926 |            1.43122e-171 |          3.57804e-171 |
| conversion | f0        |          0.53515  |            2.25947e-165 |          4.51894e-165 |
| conversion | f6        |          0.371878 |            8.51194e-155 |          1.41866e-154 |
| conversion | f4        |          1.51073  |            1.44118e-102 |          1.80148e-102 |
| conversion | f10       |          1.51073  |            1.44118e-102 |          1.80148e-102 |
| conversion | f7        |          0.610459 |            3.39409e-57  |          3.77121e-57  |
| conversion | f1        |          0.787832 |            5.22988e-10  |          5.22988e-10  |
| conversion | f5        |          0        |          nan            |        nan            |
| conversion | f11       |          0        |          nan            |        nan            |
| visit      | f0        |          5.01325  |            0            |          0            |
| visit      | f2        |          4.9606   |            0            |          0            |
| visit      | f3        |          4.18346  |            0            |          0            |
| visit      | f6        |          3.68876  |            0            |          0            |
| visit      | f7        |          5.17595  |            0            |          0            |
| visit      | f8        |          3.65042  |            0            |          0            |
| visit      | f9        |          3.48045  |            0            |          0            |
| visit      | f1        |          8.51408  |            2.90361e-193 |          3.62952e-193 |
| visit      | f4        |          3.66334  |            2.21548e-108 |          2.21548e-108 |
| visit      | f10       |          3.66334  |            2.21548e-108 |          2.21548e-108 |
| visit      | f5        |          0        |          nan            |        nan            |
| visit      | f11       |          0        |          nan            |        nan            |

## Outputs

- `reports/tables/heterogeneity_bins.csv`
- `reports/tables/heterogeneity_tests.csv`
- `figures/exploratory_heterogeneity.png`
