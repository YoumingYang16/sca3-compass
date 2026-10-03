import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sca3_compass.learning_engine import (
    evaluate_response,
    initial_mastery,
    select_next_item,
    simulate_policy_comparison,
    update_bkt,
)


def model() -> dict:
    return json.loads((ROOT / "configs" / "learning_model.json").read_text())


class LearningEngineTests(unittest.TestCase):
    def test_correct_response_increases_mastery(self) -> None:
        result = update_bkt(
            0.3, correct=True, p_learn=0.15, p_guess=0.2, p_slip=0.1
        )
        self.assertGreater(result["posterior_after_learning"], 0.3)

    def test_selector_respects_prerequisites(self) -> None:
        selected = select_next_item(model(), initial_mastery(model()), [])
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["concept_id"], "mental_model")

    def test_response_returns_misconception_and_next_item(self) -> None:
        result = evaluate_response(
            model(), mastery=None, history=[], item_id="mm-01", selected_index=1
        )
        self.assertFalse(result["correct"])
        self.assertEqual(result["misconception"], "genetic_determinism")
        self.assertIn("mastery", result)

    def test_policy_simulation_is_seeded_and_labeled(self) -> None:
        first = simulate_policy_comparison(model(), learners=120, steps=8, seed=9)
        second = simulate_policy_comparison(model(), learners=120, steps=8, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(first["provenance_tier"], "SIMULATION_OUTPUT")
        self.assertIn("same prerequisites", first["comparison_design"])
        interval = first["paired_difference_intervals"]["expected_posttest_accuracy"]
        self.assertLessEqual(interval["low"], interval["mean"])
        self.assertGreaterEqual(interval["high"], interval["mean"])


if __name__ == "__main__":
    unittest.main()
