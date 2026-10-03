import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sca3_compass.advanced_analytics import (
    bayesian_random_effects_meta,
    simulate_longitudinal_trial,
)
from sca3_compass.analytics import StudyEstimate


def estimates() -> list[StudyEstimate]:
    return [
        StudyEstimate("a", "A", 1.49, 0.08, 247, "https://example/a"),
        StudyEstimate("b", "B", 1.56, 0.08, 122, "https://example/b"),
        StudyEstimate("c", "C", 1.60, 0.14, 118, "https://example/c"),
        StudyEstimate("d", "D", 0.65, 0.24, 138, "https://example/d"),
    ]


class AdvancedAnalyticsTests(unittest.TestCase):
    def test_bayesian_meta_is_seeded_and_reports_sensitivity(self) -> None:
        first = bayesian_random_effects_meta(
            estimates(), posterior_draws=4_000, grid_points=1001, seed=91
        )
        second = bayesian_random_effects_meta(
            estimates(), posterior_draws=4_000, grid_points=1001, seed=91
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["prior_sensitivity"]), 4)
        self.assertLess(first["pooled_effect"]["low"], 1.5)
        self.assertGreater(first["pooled_effect"]["high"], 1.0)
        self.assertLess(first["diagnostics"]["posterior_boundary_mass"], 1e-4)

    def test_longitudinal_simulation_reports_bias_and_coverage(self) -> None:
        result = simulate_longitudinal_trial(
            annual_slope=1.5,
            participants_per_arm=80,
            treatment_reduction=0.4,
            simulations=120,
            seed=17,
        )
        self.assertTrue(math.isfinite(result["bias"]))
        self.assertGreater(result["coverage_95"], 0.75)
        self.assertEqual(result["assumptions"]["missingness"], "MAR")
        self.assertEqual(
            result["provenance_tier"],
            "SIMULATION_OUTPUT_FROM_PUBLIC_REAL_PARAMETERS",
        )


if __name__ == "__main__":
    unittest.main()
