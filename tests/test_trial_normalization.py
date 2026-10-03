import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sca3_compass.ingest_public_data import normalize_trial


class TrialNormalizationTests(unittest.TestCase):
    def test_normalizes_nested_clinicaltrials_record(self) -> None:
        study = {
            "hasResults": True,
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000000",
                    "briefTitle": "A study in SCA3",
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "startDateStruct": {"date": "2024-01"},
                },
                "conditionsModule": {"conditions": ["Spinocerebellar Ataxia Type 3"]},
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "phases": ["PHASE2"],
                    "enrollmentInfo": {"count": 42, "type": "ACTUAL"},
                },
                "armsInterventionsModule": {
                    "interventions": [{"type": "DRUG", "name": "Example"}]
                },
                "outcomesModule": {"primaryOutcomes": [{"measure": "SARA change"}]},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Example University"}
                },
                "contactsLocationsModule": {
                    "locations": [{"country": "United States"}]
                },
            },
        }
        row = normalize_trial(study)
        self.assertEqual(row["nct_id"], "NCT00000000")
        self.assertEqual(row["enrollment"], 42)
        self.assertEqual(row["primary_outcomes"], "SARA change")
        self.assertTrue(row["exact_sca3_text_match"])
        self.assertTrue(row["has_results"])

    def test_marks_broad_candidate_as_not_exact_match(self) -> None:
        study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000001",
                    "briefTitle": "General movement disorder study",
                },
                "conditionsModule": {"conditions": ["Ataxia"]},
                "designModule": {},
            }
        }
        row = normalize_trial(study)
        self.assertFalse(row["exact_sca3_text_match"])


if __name__ == "__main__":
    unittest.main()
