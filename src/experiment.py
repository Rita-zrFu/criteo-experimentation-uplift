"""Run experiment health checks and intent-to-treat A/B tests."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import duckdb
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "criteo_uplift.parquet"
DEFAULT_TABLES = PROJECT_ROOT / "reports" / "tables"
DEFAULT_FIGURES = PROJECT_ROOT / "figures"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "phase1_experiment_report.md"
FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ("conversion", "visit")


def create_data_view(connection: duckdb.DuckDBPyConnection, data_path: Path) -> None:
    safe_path = data_path.resolve().as_posix().replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW experiment_data AS "
        f"SELECT * FROM read_parquet('{safe_path}')"
    )


def load_arm_metrics(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = (PROJECT_ROOT / "sql" / "metrics.sql").read_text(encoding="utf-8")
    arms = connection.execute(sql).fetchdf()
    observed = set(arms["treatment"].astype(int))
    if observed != {0, 1}:
        raise ValueError(f"Expected treatment values 0 and 1; observed {sorted(observed)}")
    return arms


def estimate_effect(
    control_events: int,
    control_n: int,
    treatment_events: int,
    treatment_n: int,
    alpha: float,
    power: float,
) -> dict[str, float]:
    """Estimate a binary intent-to-treat effect and large-sample uncertainty."""
    p0 = control_events / control_n
    p1 = treatment_events / treatment_n
    difference = p1 - p0
    z_critical = norm.ppf(1 - alpha / 2)
    standard_error = math.sqrt(p0 * (1 - p0) / control_n + p1 * (1 - p1) / treatment_n)
    ci_low = difference - z_critical * standard_error
    ci_high = difference + z_critical * standard_error

    pooled = (control_events + treatment_events) / (control_n + treatment_n)
    null_se = math.sqrt(pooled * (1 - pooled) * (1 / control_n + 1 / treatment_n))
    z_stat = difference / null_se if null_se else math.nan
    p_value = 2 * norm.sf(abs(z_stat)) if math.isfinite(z_stat) else math.nan
    log10_p_value = (
        (math.log(2) + norm.logsf(abs(z_stat))) / math.log(10)
        if math.isfinite(z_stat)
        else math.nan
    )

    risk_ratio = p1 / p0 if p0 else math.nan
    relative_lift = difference / p0 if p0 else math.nan
    if control_events and treatment_events:
        log_rr_se = math.sqrt(
            1 / treatment_events - 1 / treatment_n + 1 / control_events - 1 / control_n
        )
        rr_ci_low = math.exp(math.log(risk_ratio) - z_critical * log_rr_se)
        rr_ci_high = math.exp(math.log(risk_ratio) + z_critical * log_rr_se)
    else:
        rr_ci_low = rr_ci_high = math.nan

    cohens_h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p0)))
    # Local normal approximation around the control rate. This is a sensitivity
    # threshold, not post-hoc observed power.
    mde = (z_critical + norm.ppf(power)) * math.sqrt(
        p0 * (1 - p0) * (1 / control_n + 1 / treatment_n)
    )
    return {
        "control_n": control_n,
        "treatment_n": treatment_n,
        "control_events": control_events,
        "treatment_events": treatment_events,
        "control_rate": p0,
        "treatment_rate": p1,
        "absolute_lift": difference,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "relative_lift": relative_lift,
        "risk_ratio": risk_ratio,
        "risk_ratio_ci_low": rr_ci_low,
        "risk_ratio_ci_high": rr_ci_high,
        "z_statistic": z_stat,
        "p_value": p_value,
        "log10_p_value": log10_p_value,
        "cohens_h": cohens_h,
        "mde_absolute": mde,
        "mde_relative_to_control": mde / p0 if p0 else math.nan,
    }


def build_ab_results(arms: pd.DataFrame, alpha: float, power: float) -> pd.DataFrame:
    indexed = arms.set_index("treatment")
    records = []
    for outcome in OUTCOMES:
        record = estimate_effect(
            control_events=int(indexed.loc[0, f"{outcome}s"]),
            control_n=int(indexed.loc[0, "users"]),
            treatment_events=int(indexed.loc[1, f"{outcome}s"]),
            treatment_n=int(indexed.loc[1, "users"]),
            alpha=alpha,
            power=power,
        )
        records.append({"outcome": outcome, **record})
    return pd.DataFrame(records)


def covariate_balance(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    expressions = []
    for feature in FEATURES:
        expressions.extend(
            [f"AVG({feature}) AS {feature}_mean", f"STDDEV_SAMP({feature}) AS {feature}_sd"]
        )
    query = f"SELECT treatment, {', '.join(expressions)} FROM experiment_data GROUP BY treatment"
    stats = connection.execute(query).fetchdf().set_index("treatment")

    rows = []
    for feature in FEATURES:
        mean_control = float(stats.loc[0, f"{feature}_mean"])
        mean_treatment = float(stats.loc[1, f"{feature}_mean"])
        sd_control = float(stats.loc[0, f"{feature}_sd"])
        sd_treatment = float(stats.loc[1, f"{feature}_sd"])
        pooled_sd = math.sqrt((sd_control**2 + sd_treatment**2) / 2)
        smd = (mean_treatment - mean_control) / pooled_sd if pooled_sd else 0.0
        rows.append(
            {
                "feature": feature,
                "control_mean": mean_control,
                "treatment_mean": mean_treatment,
                "control_sd": sd_control,
                "treatment_sd": sd_treatment,
                "standardized_mean_difference": smd,
                "absolute_smd": abs(smd),
            }
        )
    return pd.DataFrame(rows).sort_values("absolute_smd", ascending=False)


def save_effect_plot(results: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    colors = ["#7A8CA5", "#E76F51"]
    labels = ["Control", "Treatment"]
    for axis, (_, row) in zip(axes, results.iterrows()):
        rates = np.array([row["control_rate"], row["treatment_rate"]]) * 100
        counts = np.array([row["control_n"], row["treatment_n"]])
        standard_errors = np.sqrt((rates / 100) * (1 - rates / 100) / counts) * 100
        bars = axis.bar(labels, rates, color=colors, width=0.62)
        axis.errorbar(
            labels,
            rates,
            yerr=1.96 * standard_errors,
            fmt="none",
            ecolor="#333333",
            capsize=4,
            linewidth=1.2,
        )
        axis.set_title(row["outcome"].title())
        axis.set_ylabel("Outcome rate (%)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(y=0.2)
        for bar, value in zip(bars, rates):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Observed outcomes by randomized assignment (95% CI)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_balance_plot(balance: pd.DataFrame, output: Path) -> None:
    ordered = balance.sort_values("absolute_smd")
    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(ordered["standardized_mean_difference"], ordered["feature"], color="#2A9D8F")
    axis.axvline(0, color="#444444", linewidth=1)
    axis.axvline(-0.1, color="#BBBBBB", linestyle="--", linewidth=1)
    axis.axvline(0.1, color="#BBBBBB", linestyle="--", linewidth=1)
    axis.set_xlabel("Standardized mean difference (treatment − control)")
    axis.set_ylabel("")
    axis.set_title("Pre-treatment covariate balance")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    arms: pd.DataFrame,
    results: pd.DataFrame,
    balance: pd.DataFrame,
    output: Path,
    alpha: float,
    power: float,
) -> None:
    treatment_share = arms.loc[arms["treatment"] == 1, "users"].iloc[0] / arms["users"].sum()
    max_smd = balance["absolute_smd"].max()
    report_table = pd.DataFrame(
        {
            "Outcome": results["outcome"].str.title(),
            "Control rate": results["control_rate"].map(lambda value: f"{value:.3%}"),
            "Treatment rate": results["treatment_rate"].map(lambda value: f"{value:.3%}"),
            "Absolute lift": results["absolute_lift"].map(lambda value: f"{value * 100:.3f} pp"),
            "95% CI": results.apply(
                lambda row: f"[{row['ci_low'] * 100:.3f}, {row['ci_high'] * 100:.3f}] pp",
                axis=1,
            ),
            "Relative lift": results["relative_lift"].map(lambda value: f"{value:.1%}"),
            "p-value": results.apply(
                lambda row: (
                    f"{row['p_value']:.2e}"
                    if row["p_value"] >= 1e-300
                    else f"<1e{math.floor(row['log10_p_value'])}"
                ),
                axis=1,
            ),
            "80% MDE": results["mde_absolute"].map(lambda value: f"{value * 100:.3f} pp"),
        }
    )

    report = f"""# Phase 1: Experiment health and A/B test

## Design

The estimand is the intent-to-treat effect of randomized `treatment` assignment. Conversion is the primary outcome and visit is secondary. Realized `exposure` is not used as the randomized treatment indicator.

## Experiment health

- Total users: {int(arms['users'].sum()):,}
- Observed treatment share: {treatment_share:.2%}
- Largest absolute standardized mean difference across f0–f11: {max_smd:.4f}
- No sample-ratio-mismatch p-value is reported because the intended allocation probability is unavailable.

## A/B test results

{report_table.to_markdown(index=False)}

The confidence level is {1-alpha:.0%}. MDE uses a local normal approximation with {power:.0%} target power. Statistical significance should be interpreted together with absolute lift, relative lift, and business value.

## Generated artifacts

- `reports/tables/experiment_arms.csv`
- `reports/tables/ab_test_results.csv`
- `reports/tables/covariate_balance.csv`
- `figures/ab_test_effects.png`
- `figures/covariate_balance.png`
"""
    output.write_text(report, encoding="utf-8")


def run_analysis(
    data_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    report_path: Path,
    alpha: float,
    power: float,
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {data_path}. Run `python src/data.py prepare` first."
        )
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    try:
        create_data_view(connection, data_path)
        arms = load_arm_metrics(connection)
        results = build_ab_results(arms, alpha, power)
        balance = covariate_balance(connection)
    finally:
        connection.close()

    arms.to_csv(tables_dir / "experiment_arms.csv", index=False)
    results.to_csv(tables_dir / "ab_test_results.csv", index=False)
    balance.to_csv(tables_dir / "covariate_balance.csv", index=False)
    save_effect_plot(results, figures_dir / "ab_test_effects.png")
    save_balance_plot(balance, figures_dir / "covariate_balance.png")
    write_report(arms, results, balance, report_path, alpha, power)
    print(results[["outcome", "absolute_lift", "ci_low", "ci_high", "p_value"]].to_string(index=False))
    print(f"Wrote Phase 1 report to {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 < args.alpha < 1 or not 0 < args.power < 1:
        raise SystemExit("--alpha and --power must both be between 0 and 1")
    try:
        run_analysis(
            args.data,
            args.tables_dir,
            args.figures_dir,
            args.report,
            args.alpha,
            args.power,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
