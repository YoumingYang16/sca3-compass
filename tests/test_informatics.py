import tempfile
import unittest
from pathlib import Path

from sca3_compass.audit_ledger import AuditLedger
from sca3_compass.fhir_service import (
    build_research_bundle,
    capability_statement,
    validate_resource,
    validation_passed,
)


class InformaticsTests(unittest.TestCase):
    def test_hash_chained_ledger_detects_consistent_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = AuditLedger(Path(directory) / "audit.sqlite3")
            ledger.append(
                actor="test",
                action="execute",
                resource_type="Analysis",
                resource_id="run-1",
                recorded_at="2026-09-14T00:00:00+00:00",
                event_id="event-1",
            )
            ledger.append(
                actor="test",
                action="validate",
                resource_type="Bundle",
                resource_id="bundle-1",
                recorded_at="2026-09-14T00:01:00+00:00",
                event_id="event-2",
            )
            report = ledger.verify()
            self.assertTrue(report["valid"])
            self.assertEqual(report["events"], 2)

    def test_bundle_contains_study_and_provenance_and_validates(self) -> None:
        trial = {
            "nct_id": "NCT00000001",
            "record_url": "https://clinicaltrials.gov/study/NCT00000001",
            "brief_title": "Research study",
            "official_title": "A public research study",
            "overall_status": "COMPLETED",
            "phases": "PHASE2",
            "conditions": "Spinocerebellar Ataxia Type 3",
            "enrollment": 20,
        }
        bundle = build_research_bundle([trial])
        self.assertEqual(len(bundle["entry"]), 2)
        self.assertNotIn("total", bundle)
        self.assertEqual(bundle["entry"][1]["resource"]["resourceType"], "Provenance")
        self.assertTrue(validation_passed(validate_resource(bundle)))

    def test_invalid_study_returns_operation_outcome_error(self) -> None:
        outcome = validate_resource({"resourceType": "ResearchStudy", "status": "active"})
        self.assertFalse(validation_passed(outcome))
        self.assertGreaterEqual(len(outcome["issue"]), 1)

    def test_capability_statement_declares_r5_and_validate(self) -> None:
        capability = capability_statement()
        self.assertEqual(capability["fhirVersion"], "5.0.0")
        self.assertEqual(capability["rest"][0]["operation"][0]["name"], "validate")


if __name__ == "__main__":
    unittest.main()
