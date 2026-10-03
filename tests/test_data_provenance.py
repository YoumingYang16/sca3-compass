import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sca3_compass.data_provenance import (
    DataAsset,
    DataTier,
    IntendedUse,
    validate_registry,
)


class DataProvenanceTests(unittest.TestCase):
    def test_registered_sources_pass_policy_validation(self) -> None:
        registry = json.loads(
            (ROOT / "configs" / "data_sources.json").read_text(encoding="utf-8")
        )
        assets = validate_registry(registry["sources"])
        self.assertGreaterEqual(len(assets), 6)
        self.assertTrue(all(asset.tier is DataTier.PUBLIC_REAL for asset in assets))

    def test_fixture_cannot_support_scientific_inference(self) -> None:
        asset = DataAsset(
            id="forbidden_fixture",
            tier=DataTier.TEST_FIXTURE,
            intended_uses=(IntendedUse.SCIENTIFIC_INFERENCE,),
            public_url=None,
            license_or_terms="internal",
        )
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            asset.validate()

    def test_human_study_data_cannot_replace_public_evidence(self) -> None:
        asset = DataAsset(
            id="learning_pilot",
            tier=DataTier.NEW_HUMAN_STUDY,
            intended_uses=(IntendedUse.SCIENTIFIC_INFERENCE,),
            public_url=None,
            license_or_terms="consent required",
        )
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            asset.validate()


if __name__ == "__main__":
    unittest.main()
