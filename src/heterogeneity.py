"""Exploratory quantile-based treatment-effect heterogeneity analysis."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ("conversion", "visit")
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "criteo_uplift.parquet"
DEFAULT_TABLES = PROJECT_ROOT / "reports" / "tables"
DEFAULT_FIGURE = PROJECT_ROOT / "figures" / "exploratory_heterogeneity.png"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "phase2_heterogeneity_report.md"


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def quantile_cutpoints(connection: duckdb.DuckDBPyConnection) -> dict[str, list[float]]:
    expressions = ", ".join(
        f"approx_quantile({feature}, [0.2, 0.4, 0.6, 0.8]) AS {feature}" for feature in FEATURES
    )
    row = connection.execute(f"SELECT {expressions} FROM experiment_data").fetchone()
    return {feature: [float(value) for value in values] for feature, values in zip(FEATURES, row)}


def feature_bins(
    connection: duckdb.DuckDBPyConnection,
    feature: str,
    cutpoints: list[float],
) -> pd.DataFrame:
    q20, q40, q60, q80 = cutpoints
    query = f"""
        WITH binned AS (
            SELECT
                treatment,
                conversion,
                visit,
                CASE
                    WHEN {feature} <= ? THEN 1
                    WHEN {feature} <= ? THEN 2
                    WHEN {feature} <= ? THEN 3
                    WHEN {feature} <= ? THEN 4
                    ELSE 5
                END AS quantile_bin
            FROM experiment_data
        )
        SELECT
            quantile_bin,
            treatment,
            COUNT(*) AS users,
            SUM(conversion) AS conversion_events,
            SUM(visit) AS visit_events
        FROM binned
        GROUP BY quantile_bin, treatment
        ORDER BY quantile_bin, treatment
    """
    return connection.execute(query, [q20, q40, q60, q80]).fetchdf()


def bin_effects(aggregates: pd.DataFrame, feature: str, outcome: str) -> pd.DataFrame:
    rows = []
    for quantile_bin, group in aggregates.groupby("quantile_bin"):
        indexed = group.set_index("treatment")
        if set(indexed.index.astype(int)) != {0, 1}:
            continue
        n0, n1 = int(indexed.loc[0, "users"]), int(indexed.loc[1, "users"])
        y0 = int(indexed.loc[0, f"{outcome}_events"])
        y1 = int(indexed.loc[1, f"{outcome}_events"])
        p0, p1 = y0 / n0, y1 / n1
        effect = p1 - p0
        variance = p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1
        se = math.sqrt(variance)
        rows.append(
            {
                "feature": feature,
                "outcome": outcome,
                "quantile_bin": int(quantile_bin),
                "control_n": n0,
                "treatment_n": n1,
                "control_rate": p0,
                "treatment_rate": p1,
                "effect": effect,
                "standard_error": se,
                "ci_low": effect - 1.96 * se,
                "ci_high": effect + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def heterogeneity_test(effects: pd.DataFrame) -> dict[str, float]:
    valid = effects.loc[effects["standard_error"] > 0].copy()
    weights = 1 / valid["standard_error"].pow(2)
    pooled = np.average(valid["effect"], weights=weights)
    statistic = float(np.sum(weights * (valid["effect"] - pooled) ** 2))
    degrees_freedom = len(valid) - 1
    p_value = float(chi2.sf(statistic, degrees_freedom))
    return {
        "weighted_effect": pooled,
        "heterogeneity_statistic": statistic,
        "degrees_freedom": degrees_freedom,
        "heterogeneity_p_value": p_value,
        "effect_range": float(valid["effect"].max() - valid["effect"].min()),
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    finite = np.isfinite(values)
    adjusted = np.full_like(values, np.nan)
    finite_values = values[finite]
    if not len(finite_values):
        return pd.Series(adjusted, index=p_values.index)
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    adjusted_ranked = ranked * len(finite_values) / np.arange(1, len(finite_values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    finite_adjusted = np.empty_like(adjusted_ranked)
    finite_adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    adjusted[finite] = finite_adjusted
    return pd.Series(adjusted, index=p_values.index)


def run_heterogeneity(
    data_path: Path,
    tables_dir: Path,
    figure_path: Path,
    report_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute(
        f"CREATE TEMP VIEW experiment_data AS SELECT * FROM read_parquet('{sql_path(data_path)}')"
    )
    try:
        cutpoints = quantile_cutpoints(connection)
        effect_frames = []
        summaries = []
        for feature in FEATURES:
            aggregates = feature_bins(connection, feature, cutpoints[feature])
            for outcome in OUTCOMES:
                effects = bin_effects(aggregates, feature, outcome)
                effect_frames.append(effects)
                summaries.append({"feature": feature, "outcome": outcome, **heterogeneity_test(effects)})
    finally:
        connection.close()

    effects = pd.concat(effect_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    summary["bh_adjusted_p_value"] = summary.groupby("outcome", group_keys=False)[
        "heterogeneity_p_value"
    ].transform(benjamini_hochberg)
    effects.to_csv(tables_dir / "heterogeneity_bins.csv", index=False)
    summary.to_csv(tables_dir / "heterogeneity_tests.csv", index=False)
    save_heterogeneity_plot(effects, summary, figure_path)
    write_report(effects, summary, report_path)
    return effects, summary


def save_heterogeneity_plot(effects: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    ranked = summary.loc[
        (summary["outcome"] == "conversion") & summary["effect_range"].gt(0)
    ].sort_values("effect_range", ascending=False)
    selected = []
    signatures = []
    for feature in ranked["feature"]:
        signature = (
            effects.loc[
                (effects["feature"] == feature) & (effects["outcome"] == "conversion"),
                "effect",
            ]
            .round(12)
            .to_list()
        )
        if signature not in signatures:
            selected.append(feature)
            signatures.append(signature)
        if len(selected) == 3:
            break
    conversion_summary = summary.loc[summary["outcome"] == "conversion"].set_index("feature")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for axis, feature in zip(axes, selected):
        data = effects.loc[
            (effects["feature"] == feature) & (effects["outcome"] == "conversion")
        ].sort_values("quantile_bin")
        axis.errorbar(
            data["quantile_bin"],
            data["effect"] * 100,
            yerr=np.vstack(
                [
                    (data["effect"] - data["ci_low"]) * 100,
                    (data["ci_high"] - data["effect"]) * 100,
                ]
            ),
            color="#2A9D8F",
            marker="o",
            capsize=4,
        )
        q_value = conversion_summary.loc[feature, "bh_adjusted_p_value"]
        axis.axhline(0, color="#777777", linewidth=1)
        axis.set_title(f"{feature} (BH q={q_value:.2g})")
        axis.set_xlabel("Quantile bin (ties may collapse bins)")
        axis.set_xticks(range(1, 6))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Conversion effect (percentage points)")
    fig.suptitle("Exploratory conversion-effect heterogeneity\n(three non-duplicate effect profiles)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(effects: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    display = summary.copy().sort_values(["outcome", "bh_adjusted_p_value"])
    display["effect_range_pp"] = display["effect_range"] * 100
    display = display[
        ["outcome", "feature", "effect_range_pp", "heterogeneity_p_value", "bh_adjusted_p_value"]
    ]
    report = f"""# Phase 2A: Exploratory treatment-effect heterogeneity

All 12 anonymized features were analyzed using global 20th/40th/60th/80th percentile cutpoints. Adjacent bins are collapsed when tied feature values share a cutpoint. Within each resulting bin, the intent-to-treat effect is the treatment-minus-control difference in outcome rates. Feature-level heterogeneity uses a fixed-effect Cochran Q test across estimable bin effects, followed by Benjamini–Hochberg adjustment across estimable features separately for each outcome.

Three non-duplicate conversion-effect profiles are visualized for exploration only. They were selected after inspecting the results and must not be interpreted as pre-specified confirmatory findings or named customer segments. Features without enough distinct quantile bins remain in the output as non-testable (`NaN`).

## Feature-level summary

{display.to_markdown(index=False, floatfmt='.6g')}

## Outputs

- `reports/tables/heterogeneity_bins.csv`
- `reports/tables/heterogeneity_tests.csv`
- `figures/exploratory_heterogeneity.png`
"""
    output.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        _, summary = run_heterogeneity(args.data, args.tables_dir, args.figure, args.report)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(summary.sort_values(["outcome", "bh_adjusted_p_value"]).to_string(index=False))
    print(f"Wrote heterogeneity report to {args.report}")


if __name__ == "__main__":
    main()
