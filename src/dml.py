"""Cross-fitted Double Machine Learning for the randomized conversion effect."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [f"f{i}" for i in range(12)]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "criteo_uplift.parquet"
DEFAULT_RESULTS = PROJECT_ROOT / "reports" / "tables" / "dml_results.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "dml_report.md"


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def create_view(
    connection: duckdb.DuckDBPyConnection,
    data_path: Path,
    folds: int,
    seed: int,
) -> None:
    columns = [
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{sql_path(data_path)}')"
        ).fetchall()
    ]
    if "row_id" not in columns:
        raise ValueError("Processed data has no row_id; rerun `python src/data.py prepare --force`")
    connection.execute(
        f"""
        CREATE TEMP VIEW dml_data AS
        SELECT
            row_id,
            {', '.join(FEATURES)},
            treatment,
            conversion,
            hash(row_id + {seed}) % {folds} AS dml_fold
        FROM read_parquet('{sql_path(data_path)}')
        """
    )


def load_nuisance_training(
    connection: duckdb.DuckDBPyConnection,
    held_out_fold: int,
    max_rows: int,
    seed: int,
    arm: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arm_filter = "" if arm is None else f"AND treatment = {arm}"
    available = connection.execute(
        f"SELECT COUNT(*) FROM dml_data WHERE dml_fold != ? {arm_filter}", [held_out_fold]
    ).fetchone()[0]
    threshold = min(1_000_000, math.ceil(max_rows / available * 1_000_000))
    if available <= max_rows:
        sample_clause = ""
    else:
        sample_clause = (
            f"AND hash(row_id + {seed + held_out_fold}) % 1000000 < {threshold} "
            f"ORDER BY hash(row_id + {seed + held_out_fold}) LIMIT {max_rows}"
        )
    frame = connection.execute(
        f"""
        SELECT {', '.join(FEATURES)}, treatment, conversion
        FROM dml_data
        WHERE dml_fold != ? {arm_filter} {sample_clause}
        """,
        [held_out_fold],
    ).fetchdf()
    outcome = frame.pop("conversion").to_numpy(dtype=np.uint8)
    treatment = frame.pop("treatment").to_numpy(dtype=np.uint8)
    covariates = frame.to_numpy(dtype=np.float32)
    return covariates, treatment, outcome


def load_score_fold(
    connection: duckdb.DuckDBPyConnection,
    fold: int,
    max_rows: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    limit_clause = (
        ""
        if max_rows is None
        else f"ORDER BY hash(row_id + 104729) LIMIT {max_rows}"
    )
    frame = connection.execute(
        f"""
        SELECT {', '.join(FEATURES)}, treatment, conversion
        FROM dml_data
        WHERE dml_fold = ?
        {limit_clause}
        """,
        [fold],
    ).fetchdf()
    outcome = frame.pop("conversion").to_numpy(dtype=np.uint8)
    treatment = frame.pop("treatment").to_numpy(dtype=np.uint8)
    covariates = frame.to_numpy(dtype=np.float32)
    return covariates, treatment, outcome


def fit_propensity_model(
    covariates: np.ndarray,
    treatment: np.ndarray,
) -> object:
    propensity_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, tol=1e-7),
    )
    propensity_model.fit(covariates, treatment)
    return propensity_model


def fit_outcome_model(
    covariates: np.ndarray,
    outcome: np.ndarray,
    seed: int,
) -> lgb.LGBMClassifier:
    outcome_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=500,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    outcome_model.fit(covariates, outcome, feature_name=FEATURES)
    return outcome_model


def estimate_dml(
    data_path: Path,
    folds: int,
    max_outcome_rows_per_arm: int,
    max_propensity_rows: int,
    max_score_rows_per_fold: int | None,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    if folds < 2:
        raise ValueError("DML requires at least two folds")

    connection = duckdb.connect()
    create_view(connection, data_path, folds, seed)

    total_n = 0
    score_sum = score_squared_sum = 0.0
    control_n = treatment_n = control_events = treatment_events = 0
    fold_diagnostics: list[dict[str, float]] = []

    try:
        for fold in range(folds):
            X_propensity, d_propensity, _ = load_nuisance_training(
                connection,
                fold,
                max_propensity_rows,
                seed + 100,
            )
            propensity_model = fit_propensity_model(X_propensity, d_propensity)
            propensity_training_rate = float(d_propensity.mean())
            del X_propensity, d_propensity

            outcome_models = {}
            outcome_training_rows = {}
            outcome_training_rates = {}
            for arm in (0, 1):
                X_outcome, _, y_outcome = load_nuisance_training(
                    connection,
                    fold,
                    max_outcome_rows_per_arm,
                    seed + 200 + arm,
                    arm=arm,
                )
                outcome_models[arm] = fit_outcome_model(
                    X_outcome,
                    y_outcome,
                    seed + 10 * fold + arm,
                )
                outcome_training_rows[arm] = len(y_outcome)
                outcome_training_rates[arm] = float(y_outcome.mean())
                del X_outcome, y_outcome

            print(
                f"Fold {fold + 1}/{folds}: propensity train={max_propensity_rows:,} "
                f"(treatment={propensity_training_rate:.2%}); "
                f"outcome train control={outcome_training_rows[0]:,} "
                f"(rate={outcome_training_rates[0]:.4%}), "
                f"treatment={outcome_training_rows[1]:,} "
                f"(rate={outcome_training_rates[1]:.4%})"
            )

            X_score, d_score, y_score = load_score_fold(
                connection, fold, max_score_rows_per_fold
            )
            propensity = propensity_model.predict_proba(X_score)[:, 1]
            propensity = np.clip(propensity, 0.01, 0.99)
            outcome_control = outcome_models[0].booster_.predict(X_score)
            outcome_treatment = outcome_models[1].booster_.predict(X_score)
            treatment_float = d_score.astype(np.float64)
            outcome_float = y_score.astype(np.float64)
            orthogonal_score = (
                outcome_treatment
                - outcome_control
                + treatment_float * (outcome_float - outcome_treatment) / propensity
                - (1 - treatment_float) * (outcome_float - outcome_control) / (1 - propensity)
            )

            total_n += len(y_score)
            score_sum += float(np.sum(orthogonal_score, dtype=np.float64))
            score_squared_sum += float(
                np.sum(orthogonal_score * orthogonal_score, dtype=np.float64)
            )

            treated = d_score == 1
            treatment_n += int(treated.sum())
            control_n += int((~treated).sum())
            treatment_events += int(y_score[treated].sum())
            control_events += int(y_score[~treated].sum())
            fold_diagnostics.append(
                {
                    "fold": fold + 1,
                    "score_rows": len(y_score),
                    "mean_propensity": float(propensity.mean()),
                    "min_propensity": float(propensity.min()),
                    "max_propensity": float(propensity.max()),
                    "mean_control_prediction": float(outcome_control.mean()),
                    "mean_treatment_prediction": float(outcome_treatment.mean()),
                }
            )
            del (
                X_score,
                d_score,
                y_score,
                propensity,
                outcome_control,
                outcome_treatment,
                orthogonal_score,
                outcome_models,
            )
    finally:
        connection.close()

    dml_estimate = score_sum / total_n
    score_variance = (
        score_squared_sum - total_n * dml_estimate**2
    ) / (total_n - 1)
    dml_se = math.sqrt(score_variance / total_n)

    control_rate = control_events / control_n
    treatment_rate = treatment_events / treatment_n
    raw_estimate = treatment_rate - control_rate
    raw_se = math.sqrt(
        treatment_rate * (1 - treatment_rate) / treatment_n
        + control_rate * (1 - control_rate) / control_n
    )

    records = []
    for estimator, estimate, standard_error in (
        ("Randomized difference in means", raw_estimate, raw_se),
        ("Cross-fitted DML", dml_estimate, dml_se),
    ):
        z_statistic = estimate / standard_error
        records.append(
            {
                "outcome": "conversion",
                "estimator": estimator,
                "observations": total_n,
                "estimate": estimate,
                "standard_error": standard_error,
                "ci_low": estimate - 1.96 * standard_error,
                "ci_high": estimate + 1.96 * standard_error,
                "z_statistic": z_statistic,
                "p_value": 2 * norm.sf(abs(z_statistic)),
            }
        )
    return pd.DataFrame(records), fold_diagnostics


def write_report(
    results: pd.DataFrame,
    fold_diagnostics: list[dict[str, float]],
    output: Path,
    folds: int,
    max_outcome_rows_per_arm: int,
    max_propensity_rows: int,
) -> None:
    display = results[["estimator", "estimate", "standard_error", "ci_low", "ci_high"]].copy()
    for column in ("estimate", "standard_error", "ci_low", "ci_high"):
        display[column] = display[column].map(lambda value: f"{value * 100:.4f} pp")
    raw = results.loc[results["estimator"] == "Randomized difference in means"].iloc[0]
    dml = results.loc[results["estimator"] == "Cross-fitted DML"].iloc[0]
    estimate_difference = dml["estimate"] - raw["estimate"]
    se_change = dml["standard_error"] / raw["standard_error"] - 1
    intervals_overlap = max(raw["ci_low"], dml["ci_low"]) <= min(
        raw["ci_high"], dml["ci_high"]
    )
    comparison = (
        "The confidence intervals overlap, so the adjusted result is statistically compatible "
        "with the original estimate at this level of precision."
        if intervals_overlap
        else "The confidence intervals do not overlap; this is a material covariate-adjustment "
        "sensitivity result rather than a near-identical replication of the raw estimate."
    )
    precision_text = (
        f"{abs(se_change):.1%} smaller"
        if se_change < 0
        else f"{se_change:.1%} larger"
    )
    diagnostics = pd.DataFrame(fold_diagnostics)

    report = f"""# Double Machine Learning extension

This analysis applies a cross-fitted interactive-regression Double Machine Learning estimator to conversion. The treatment nuisance model is logistic regression with standardized covariates; separate treatment and control outcome nuisances use LightGBM. Each observation is scored by nuisance models trained without its fold, using the orthogonal augmented inverse-propensity score.

Because treatment was randomized, DML is used for covariate-adjusted estimation, orthogonalization, and a robustness comparison—not to repair confounding or claim stronger causal identification.

## Results

{display.to_markdown(index=False)}

The DML estimate differs from the raw randomized estimate by {estimate_difference * 100:.4f} percentage points. Its standard error is {precision_text} than the unadjusted standard error. {comparison}

## Specification

- Outcome: `conversion`
- Treatment: randomized `treatment`
- Covariates: `f0`–`f11`
- Cross-fitting folds: {folds}
- Outcome nuisances: separate treatment/control LightGBM classifiers, fixed parameters, no class weighting
- Propensity nuisance: standardized logistic regression
- Maximum outcome-nuisance training rows per arm and fold: {max_outcome_rows_per_arm:,}
- Maximum propensity training rows per fold: {max_propensity_rows:,}
- Score sample: all {int(results.iloc[0]['observations']):,} users
- Uncertainty: heteroskedasticity-robust influence-function standard error

## Fold diagnostics

{diagnostics.to_markdown(index=False, floatfmt='.6g')}
"""
    output.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-outcome-rows-per-arm", type=int, default=1_000_000)
    parser.add_argument("--max-propensity-rows", type=int, default=500_000)
    parser.add_argument("--max-score-rows-per-fold", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        results, diagnostics = estimate_dml(
            args.data,
            args.folds,
            args.max_outcome_rows_per_arm,
            args.max_propensity_rows,
            args.max_score_rows_per_fold,
            args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results, index=False)
    write_report(
        results,
        diagnostics,
        args.report,
        args.folds,
        args.max_outcome_rows_per_arm,
        args.max_propensity_rows,
    )
    print(results.to_string(index=False))
    print(f"Wrote DML results to {args.results}")
    print(f"Wrote DML report to {args.report}")


if __name__ == "__main__":
    main()
