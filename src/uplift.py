"""Train a LightGBM T-Learner and evaluate targeting on an untouched test set."""

from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import log_loss, roc_auc_score

from evaluation import evaluate_policies, qini_curve, uplift_by_decile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURES = [f"f{i}" for i in range(12)]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "criteo_uplift.parquet"
DEFAULT_TABLES = PROJECT_ROOT / "reports" / "tables"
DEFAULT_FIGURES = PROJECT_ROOT / "figures"
DEFAULT_MODELS = PROJECT_ROOT / "models"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "phase2_uplift_report.md"
SPLIT_CASE = "CASE WHEN hash(row_id) % 100 < 70 THEN 'train' WHEN hash(row_id) % 100 < 85 THEN 'validation' ELSE 'test' END"


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def create_view(connection: duckdb.DuckDBPyConnection, data_path: Path) -> None:
    columns = [row[0] for row in connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{sql_path(data_path)}')"
    ).fetchall()]
    if "row_id" not in columns:
        raise ValueError("Processed data has no row_id; rerun `python src/data.py prepare --force`")
    connection.execute(
        f"CREATE TEMP VIEW model_data AS "
        f"SELECT *, {SPLIT_CASE} AS data_split FROM read_parquet('{sql_path(data_path)}')"
    )


def split_counts(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(
        """
        SELECT data_split, treatment, COUNT(*) AS users, SUM(conversion) AS conversions
        FROM model_data
        GROUP BY data_split, treatment
        ORDER BY data_split, treatment
        """
    ).fetchdf()


def load_arm_sample(
    connection: duckdb.DuckDBPyConnection,
    split: str,
    arm: int,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    available = connection.execute(
        "SELECT COUNT(*) FROM model_data WHERE data_split = ? AND treatment = ?",
        [split, arm],
    ).fetchone()[0]
    threshold = min(1_000_000, math.ceil(max_rows / available * 1_000_000))
    if available <= max_rows:
        sample_clause = ""
    else:
        sample_clause = (
            f"AND hash(row_id + {seed}) % 1000000 < {threshold} "
            f"ORDER BY hash(row_id + {seed}) LIMIT {max_rows}"
        )
    feature_sql = ", ".join(FEATURES)
    frame = connection.execute(
        f"""
        SELECT {feature_sql}, conversion
        FROM model_data
        WHERE data_split = ? AND treatment = ? {sample_clause}
        """,
        [split, arm],
    ).fetchdf()
    y = frame.pop("conversion").to_numpy(dtype=np.uint8)
    X = frame.to_numpy(dtype=np.float32)
    return X, y


def load_test(connection: duckdb.DuckDBPyConnection) -> tuple[np.ndarray, ...]:
    columns = ", ".join(["row_id", *FEATURES, "treatment", "conversion"])
    frame = connection.execute(
        f"SELECT {columns} FROM model_data WHERE data_split = 'test' ORDER BY row_id"
    ).fetchdf()
    row_id = frame.pop("row_id").to_numpy(dtype=np.int64)
    treatment = frame.pop("treatment").to_numpy(dtype=np.uint8)
    outcome = frame.pop("conversion").to_numpy(dtype=np.uint8)
    X = frame.to_numpy(dtype=np.float32)
    return row_id, X, treatment, outcome


def fit_arm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
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
    model.fit(
        X_train,
        y_train,
        feature_name=FEATURES,
        eval_set=[(X_validation, y_validation)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return model


def model_diagnostics(
    treatment: np.ndarray,
    outcome: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for arm, prediction in [(0, p0), (1, p1)]:
        mask = treatment == arm
        rows.append(
            {
                "arm": "control" if arm == 0 else "treatment",
                "users": int(mask.sum()),
                "conversion_rate": float(outcome[mask].mean()),
                "roc_auc": float(roc_auc_score(outcome[mask], prediction[mask])),
                "log_loss": float(log_loss(outcome[mask], prediction[mask])),
            }
        )
    return pd.DataFrame(rows)


def save_decile_plot(deciles: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5))
    axis.bar(deciles["uplift_decile"], deciles["effect"] * 100, color="#2A9D8F")
    axis.errorbar(
        deciles["uplift_decile"],
        deciles["effect"] * 100,
        yerr=np.vstack(
            [
                (deciles["effect"] - deciles["ci_low"]) * 100,
                (deciles["ci_high"] - deciles["effect"]) * 100,
            ]
        ),
        fmt="none",
        color="#333333",
        capsize=3,
    )
    axis.axhline(0, color="#777777", linewidth=1)
    axis.set_xlabel("Predicted uplift decile (1 = highest)")
    axis.set_ylabel("Observed conversion lift (percentage points)")
    axis.set_title("Out-of-sample uplift by predicted decile")
    axis.set_xticks(range(1, 11))
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_qini_plot(curve: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5))
    axis.plot(
        curve["targeted_fraction"] * 100,
        curve["incremental_conversion_rate"] * 100_000,
        label="T-Learner ranking",
        color="#E76F51",
        linewidth=2,
    )
    axis.plot(
        curve["targeted_fraction"] * 100,
        curve["random_targeting_rate"] * 100_000,
        label="Random targeting",
        color="#7A8CA5",
        linestyle="--",
    )
    axis.set_xlabel("Population targeted (%)")
    axis.set_ylabel("Estimated incremental conversions per 100K users")
    axis.set_title("Out-of-sample cumulative uplift curve")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    counts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    deciles: pd.DataFrame,
    policies: pd.DataFrame,
    metrics: dict[str, float],
    feature_importance: pd.DataFrame,
    output: Path,
    conversion_value: float,
    treatment_cost: float,
) -> None:
    policy_display = policies[
        [
            "policy",
            "coverage",
            "incremental_conversions_per_100k",
            "ci_low_per_100k",
            "ci_high_per_100k",
            "assumed_net_value_per_100k",
        ]
    ].copy()
    policy_display["coverage"] = policy_display["coverage"].map(lambda x: f"{x:.1%}")
    for column in policy_display.columns[2:]:
        policy_display[column] = policy_display[column].map(lambda x: f"{x:,.2f}")
    report = f"""# Phase 2B: T-Learner uplift and targeting policy

Two LightGBM response models were trained separately for randomized treatment and control users. A deterministic hash of `row_id` created 70% train, 15% validation, and 15% untouched test partitions. Model selection and early stopping used validation data; all uplift and policy results below use only the test partition.

Training is capped at 2,000,000 rows per arm and validation at 500,000 rows per arm using deterministic hash sampling. No outcome-based sampling or class weighting is used, preserving probability scale.

## Split counts

{counts.to_markdown(index=False)}

## Factual response-model diagnostics

{diagnostics.to_markdown(index=False, floatfmt='.6g')}

ROC AUC and log loss diagnose the two response models; they are not treated as uplift metrics.

## Uplift evaluation

- AUUC: {metrics['auuc']:.8f}
- Qini coefficient: {metrics['qini_coefficient']:.8f}
- Test treatment propensity: {metrics['treatment_propensity']:.2%}

Qini/AUUC use an inverse-propensity transformed outcome and are computed only on the untouched test set. Uplift-decile effects are estimated directly from randomized treatment-control differences within each predicted-score decile.

## Targeting policies

{policy_display.to_markdown(index=False)}

Confidence intervals are stratified bootstrap intervals conditional on each fixed targeting rule. The illustrative net-value calculation assumes a conversion is worth ${conversion_value:,.2f} and each treatment costs ${treatment_cost:,.2f}; it is a sensitivity scenario, not an observed financial result.

## Leading features by average gain importance

{feature_importance.head(8).to_markdown(index=False, floatfmt='.6g')}

## Outputs

- `reports/tables/uplift_deciles.csv`
- `reports/tables/qini_curve.csv`
- `reports/tables/policy_comparison.csv`
- `reports/tables/model_diagnostics.csv`
- `reports/tables/feature_importance.csv`
- `figures/uplift_by_decile.png`
- `figures/qini_curve.png`
"""
    output.write_text(report, encoding="utf-8")


def run_uplift(
    data_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    models_dir: Path,
    report_path: Path,
    max_train_per_arm: int,
    max_validation_per_arm: int,
    conversion_value: float,
    treatment_cost: float,
    seed: int,
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    for directory in (tables_dir, figures_dir, models_dir, report_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    create_view(connection, data_path)
    counts = split_counts(connection)
    training = {}
    validation = {}
    for arm in (0, 1):
        training[arm] = load_arm_sample(connection, "train", arm, max_train_per_arm, seed + arm)
        validation[arm] = load_arm_sample(
            connection, "validation", arm, max_validation_per_arm, seed + 10 + arm
        )
        print(
            f"Arm {arm}: train={len(training[arm][1]):,} "
            f"(rate={training[arm][1].mean():.4%}), validation={len(validation[arm][1]):,}"
        )

    models = {}
    for arm in (0, 1):
        print(f"Training {'control' if arm == 0 else 'treatment'} model...")
        models[arm] = fit_arm_model(
            *training[arm],
            *validation[arm],
            seed + arm,
        )
        models[arm].booster_.save_model(str(models_dir / f"t_learner_arm_{arm}.txt"))
        del training[arm], validation[arm]

    row_id, X_test, treatment, outcome = load_test(connection)
    connection.close()
    print(f"Predicting uplift for {len(outcome):,} untouched test users...")
    p0 = models[0].booster_.predict(X_test, num_iteration=models[0].best_iteration_)
    p1 = models[1].booster_.predict(X_test, num_iteration=models[1].best_iteration_)
    uplift_score = p1 - p0

    diagnostics = model_diagnostics(treatment, outcome, p0, p1)
    deciles = uplift_by_decile(treatment, outcome, uplift_score)
    curve, metrics = qini_curve(treatment, outcome, uplift_score)
    policies = evaluate_policies(
        treatment,
        outcome,
        uplift_score,
        conversion_value,
        treatment_cost,
        seed=seed,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "control_gain": models[0].booster_.feature_importance(importance_type="gain"),
            "treatment_gain": models[1].booster_.feature_importance(importance_type="gain"),
        }
    )
    for column in ("control_gain", "treatment_gain"):
        total = feature_importance[column].sum()
        feature_importance[column] /= total
    feature_importance["average_gain"] = feature_importance[
        ["control_gain", "treatment_gain"]
    ].mean(axis=1)
    feature_importance = feature_importance.sort_values("average_gain", ascending=False)

    counts.to_csv(tables_dir / "data_split_counts.csv", index=False)
    diagnostics.to_csv(tables_dir / "model_diagnostics.csv", index=False)
    deciles.to_csv(tables_dir / "uplift_deciles.csv", index=False)
    curve.to_csv(tables_dir / "qini_curve.csv", index=False)
    policies.to_csv(tables_dir / "policy_comparison.csv", index=False)
    feature_importance.to_csv(tables_dir / "feature_importance.csv", index=False)
    (tables_dir / "uplift_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_decile_plot(deciles, figures_dir / "uplift_by_decile.png")
    save_qini_plot(curve, figures_dir / "qini_curve.png")
    write_report(
        counts,
        diagnostics,
        deciles,
        policies,
        metrics,
        feature_importance,
        report_path,
        conversion_value,
        treatment_cost,
    )
    print(policies.to_string(index=False))
    print(f"Wrote uplift report to {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-train-per-arm", type=int, default=2_000_000)
    parser.add_argument("--max-validation-per-arm", type=int, default=500_000)
    parser.add_argument("--conversion-value", type=float, default=100.0)
    parser.add_argument("--treatment-cost", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_uplift(
            args.data,
            args.tables_dir,
            args.figures_dir,
            args.models_dir,
            args.report,
            args.max_train_per_arm,
            args.max_validation_per_arm,
            args.conversion_value,
            args.treatment_cost,
            args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
