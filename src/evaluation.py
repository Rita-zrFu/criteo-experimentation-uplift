"""Out-of-sample uplift and targeting-policy evaluation helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def difference_in_rates(treatment: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    treated = treatment == 1
    control = ~treated
    n1, n0 = int(treated.sum()), int(control.sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("Both randomized arms are required for policy evaluation")
    p1 = float(outcome[treated].mean())
    p0 = float(outcome[control].mean())
    effect = p1 - p0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return {
        "control_n": n0,
        "treatment_n": n1,
        "control_rate": p0,
        "treatment_rate": p1,
        "effect": effect,
        "standard_error": se,
        "ci_low": effect - 1.96 * se,
        "ci_high": effect + 1.96 * se,
    }


def uplift_by_decile(
    treatment: np.ndarray,
    outcome: np.ndarray,
    uplift_score: np.ndarray,
) -> pd.DataFrame:
    ranks = pd.Series(uplift_score).rank(method="first", ascending=False)
    deciles = pd.qcut(ranks, 10, labels=False).to_numpy() + 1
    rows = []
    for decile in range(1, 11):
        mask = deciles == decile
        estimate = difference_in_rates(treatment[mask], outcome[mask])
        rows.append(
            {
                "uplift_decile": decile,
                "users": int(mask.sum()),
                "mean_predicted_uplift": float(uplift_score[mask].mean()),
                **estimate,
            }
        )
    return pd.DataFrame(rows)


def qini_curve(
    treatment: np.ndarray,
    outcome: np.ndarray,
    uplift_score: np.ndarray,
    points: int = 200,
) -> tuple[pd.DataFrame, dict[str, float]]:
    propensity = float(treatment.mean())
    if not 0 < propensity < 1:
        raise ValueError("Treatment propensity must be between zero and one")
    order = np.argsort(-uplift_score, kind="stable")
    ordered_treatment = treatment[order]
    ordered_outcome = outcome[order]
    transformed = ordered_outcome * (
        ordered_treatment / propensity - (1 - ordered_treatment) / (1 - propensity)
    )
    cumulative_gain = np.cumsum(transformed) / len(transformed)
    indices = np.unique(
        np.r_[0, np.linspace(1, len(transformed), points, dtype=int) - 1, len(transformed) - 1]
    )
    fractions = (indices + 1) / len(transformed)
    gains = cumulative_gain[indices]
    total_effect = float(cumulative_gain[-1])
    random_gain = fractions * total_effect
    auuc = float(np.trapezoid(gains, fractions))
    qini = float(np.trapezoid(gains - random_gain, fractions))
    curve = pd.DataFrame(
        {
            "targeted_fraction": np.r_[0.0, fractions],
            "incremental_conversion_rate": np.r_[0.0, gains],
            "random_targeting_rate": np.r_[0.0, random_gain],
        }
    )
    return curve, {
        "treatment_propensity": propensity,
        "auuc": auuc,
        "qini_coefficient": qini,
        "overall_transformed_effect": total_effect,
    }


def _bootstrap_policy_effect(
    treatment: np.ndarray,
    outcome: np.ndarray,
    coverage: float,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    treated = treatment == 1
    control = ~treated
    n1, n0 = int(treated.sum()), int(control.sum())
    y1, y0 = int(outcome[treated].sum()), int(outcome[control].sum())
    boot_p1 = rng.binomial(n1, y1 / n1, size=repetitions) / n1
    boot_p0 = rng.binomial(n0, y0 / n0, size=repetitions) / n0
    values = coverage * (boot_p1 - boot_p0)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate_policies(
    treatment: np.ndarray,
    outcome: np.ndarray,
    uplift_score: np.ndarray,
    conversion_value: float,
    treatment_cost: float,
    bootstrap_repetitions: int = 1000,
    seed: int = 2026,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    policies = [
        ("Treat none", np.zeros(len(outcome), dtype=bool)),
        ("Top 20% uplift", uplift_score >= np.quantile(uplift_score, 0.80)),
        ("Top 30% uplift", uplift_score >= np.quantile(uplift_score, 0.70)),
        ("Treat all", np.ones(len(outcome), dtype=bool)),
    ]
    rows = []
    for name, mask in policies:
        coverage = float(mask.mean())
        if not mask.any():
            effect = ci_low = ci_high = 0.0
            subgroup_effect = math.nan
            control_n = treatment_n = 0
        else:
            estimate = difference_in_rates(treatment[mask], outcome[mask])
            subgroup_effect = estimate["effect"]
            effect = coverage * subgroup_effect
            ci_low, ci_high = _bootstrap_policy_effect(
                treatment[mask],
                outcome[mask],
                coverage,
                bootstrap_repetitions,
                rng,
            )
            control_n = estimate["control_n"]
            treatment_n = estimate["treatment_n"]
        incremental_per_100k = effect * 100_000
        treated_per_100k = coverage * 100_000
        net_value = incremental_per_100k * conversion_value - treated_per_100k * treatment_cost
        rows.append(
            {
                "policy": name,
                "coverage": coverage,
                "control_n": control_n,
                "treatment_n": treatment_n,
                "subgroup_effect": subgroup_effect,
                "incremental_conversion_rate": effect,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "incremental_conversions_per_100k": incremental_per_100k,
                "ci_low_per_100k": ci_low * 100_000,
                "ci_high_per_100k": ci_high * 100_000,
                "assumed_net_value_per_100k": net_value,
            }
        )
    return pd.DataFrame(rows)
