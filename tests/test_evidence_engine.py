import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sca3_compass.evidence_engine import (
    build_claim_index,
    evaluate_benchmark,
    query_evidence,
)


class EvidenceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = build_claim_index()

    def test_answer_is_extractive_and_cited(self) -> None:
        result = query_evidence("What genetic change causes SCA3?", index=self.index)
        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["generation"], "extractive only")
        self.assertEqual(result["citations"][0]["claim_id"], "sca3-genetic-cause")

    def test_personal_treatment_query_is_blocked(self) -> None:
        result = query_evidence("Which medicine should I take for my SCA3?", index=self.index)
        self.assertEqual(result["action"], "safety_abstain")
        self.assertEqual(result["citations"], [])

    def test_unsupported_query_abstains(self) -> None:
        result = query_evidence("What is the best hospital in my city?", index=self.index)
        self.assertEqual(result["action"], "evidence_abstain")

    def test_benchmark_meets_baseline_retrieval_gate(self) -> None:
        benchmark = json.loads(
            (ROOT / "configs" / "evidence_benchmark.json").read_text(encoding="utf-8")
        )
        report = evaluate_benchmark(benchmark, self.index)
        self.assertGreaterEqual(report["metrics"]["mrr"], 0.9)
        self.assertGreaterEqual(report["metrics"]["recall_at_3"], 0.9)
        self.assertGreaterEqual(report["metrics"]["action_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
