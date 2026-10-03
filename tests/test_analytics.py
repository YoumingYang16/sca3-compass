import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sca3_compass.analytics import (
    StudyEstimate,
    random_effects_meta,
    simulate_parallel_trial,
)


class AnalyticsTests(unittest.TestCase):
    def test_meta_analysis_returns_finite_interval_and_influence(self) -> None:
        estimates = [
            StudyEstimate("a", "A", 1.49, 0.08, 247, "https://example/a"),
            StudyEstimate("b", "B", 1.56, 0.08, 122, "https://example/b"),
            StudyEstimate("c", "C", 1.60, 0.14, 118, "https://example/c"),
            StudyEstimate("d", "D", 0.65, 0.24, 138, "https://example/d"),
        ]
        result = random_effects_meta(estimates)
        self.assertEqual(result["k"], 4)
        self.assertEqual(len(result["leave_one_out"]), 4)
        self.assertTrue(math.isfinite(result["pooled_effect"]))
        self.assertLess(result["ci_low"], result["pooled_effect"])
        self.assertGreater(result["ci_high"], result["pooled_effect"])

    def test_trial_simulation_is_seeded_and_detects_large_effect(self) -> None:
        params = {
            "annual_slope": 1.5,
            "participants_per_arm": 150,
            "followup_months": 24,
            "treatment_reduction": 0.5,
            "slope_sd": 1.2,
            "measurement_sd": 0.5,
            "attrition_rate": 0.1,
            "simulations": 500,
            "seed": 202709,
        }
        first = simulate_parallel_trial(**params)
        second = simulate_parallel_trial(**params)
        self.assertEqual(first, second)
        self.assertGreater(first["power"], 0.8)
        self.assertEqual(
            first["provenance_tier"], "SIMULATION_OUTPUT_FROM_PUBLIC_REAL_PARAMETERS"
        )


if __name__ == "__main__":
    unittest.main()
