"""Small deterministic tests for the core uplift evaluation functions."""

from __future__ import annotations

import unittest

import numpy as np

from src.evaluation import difference_in_rates, evaluate_policies, qini_curve


class EvaluationTests(unittest.TestCase):
    def test_difference_in_rates(self) -> None:
        treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
        outcome = np.array([0, 0, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
        estimate = difference_in_rates(treatment, outcome)
        self.assertAlmostEqual(estimate["control_rate"], 0.25)
        self.assertAlmostEqual(estimate["treatment_rate"], 0.50)
        self.assertAlmostEqual(estimate["effect"], 0.25)

    def test_qini_rewards_useful_ranking(self) -> None:
        rng = np.random.default_rng(42)
        users = 20_000
        score = rng.uniform(size=users)
        treatment = rng.binomial(1, 0.5, size=users).astype(np.uint8)
        response_probability = 0.02 + treatment * (score > 0.8) * 0.20
        outcome = rng.binomial(1, response_probability).astype(np.uint8)
        _, metrics = qini_curve(treatment, outcome, score)
        self.assertGreater(metrics["qini_coefficient"], 0)

    def test_policy_table_contains_expected_strategies(self) -> None:
        rng = np.random.default_rng(7)
        users = 10_000
        score = rng.normal(size=users)
        treatment = rng.binomial(1, 0.5, size=users).astype(np.uint8)
        outcome = rng.binomial(1, 0.05 + treatment * (score > 1) * 0.10).astype(np.uint8)
        policies = evaluate_policies(
            treatment,
            outcome,
            score,
            conversion_value=100,
            treatment_cost=0.05,
            bootstrap_repetitions=100,
            seed=7,
        )
        self.assertEqual(
            policies["policy"].tolist(),
            ["Treat none", "Top 20% uplift", "Top 30% uplift", "Treat all"],
        )
        self.assertAlmostEqual(policies.iloc[-1]["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()

