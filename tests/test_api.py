import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(
    os.environ.get("SCA3_RUN_API_TESTS") == "1", "API dependencies optional"
)
class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from sca3_compass.app import app

        self.client = TestClient(app)

    def test_health_and_public_data_endpoints(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["scientific_claim_tier"], "PUBLIC_REAL_ONLY")

        overview = self.client.get("/api/overview")
        self.assertEqual(overview.status_code, 200)
        self.assertGreater(overview.json()["trial_candidates"], 0)

        meta = self.client.get("/api/analytics/meta-analysis")
        self.assertEqual(meta.status_code, 200)
        self.assertGreaterEqual(meta.json()["k"], 4)

        genes = self.client.get("/api/genomics/gse309548-gene-analysis")
        self.assertEqual(genes.status_code, 200)
        self.assertEqual(genes.json()["analysis_status"], "complete")
        self.assertEqual(genes.json()["metrics"]["donors"], 8)

    def test_simulation_is_explicitly_generated(self) -> None:
        response = self.client.post(
            "/api/analytics/trial-simulation",
            json={"simulations": 200, "seed": 7},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.json()["provenance_tier"].startswith("SIMULATION_OUTPUT")
        )


if __name__ == "__main__":
    unittest.main()
