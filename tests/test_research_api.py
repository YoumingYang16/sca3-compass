"""API contracts; mutating tests use temporary stores, never public artifacts."""
import pytest
from fastapi.testclient import TestClient
from fhir.resources import get_fhir_model_class

import sca3_compass.app as api
from sca3_compass.audit_ledger import AuditLedger
from sca3_compass.experiment_store import ExperimentStore
from sca3_compass.learning_sessions import LearningSessions


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "EXPERIMENTS", ExperimentStore(tmp_path / "experiments.sqlite3"))
    monkeypatch.setattr(api, "LEARNING_SESSIONS", LearningSessions(tmp_path / "learning.sqlite3"))
    monkeypatch.setattr(api, "AUDIT_LEDGER", AuditLedger(tmp_path / "audit.sqlite3"))
    return TestClient(api.app)


def test_real_bundle_passes_independent_r5_model_and_focused_checks(client):
    bundle = client.get('/api/fhir/research-bundle').json()
    get_fhir_model_class('Bundle').model_validate(bundle)
    outcome = client.post('/api/fhir/$validate', json=bundle).json()
    get_fhir_model_class('OperationOutcome').model_validate(outcome)
    assert all(issue['severity'] not in {'error','fatal'} for issue in outcome['issue'])
    for entry in bundle['entry']:
        resource = entry['resource']
        response = client.get(f"/api/fhir/{resource['resourceType']}/{resource['id']}")
        assert response.status_code == 200
        assert response.json()['id'] == resource['id']


def test_study_status_and_registry_progress_are_distinct(client):
    studies = client.get('/api/fhir/ResearchStudy?_count=2').json()
    assert len(studies['entry']) == 2
    assert studies['total'] >= 2
    for entry in studies['entry']:
        assert entry['resource']['status'] == 'active'
        assert entry['resource']['progressStatus'][0]['state']['coding'][0]['code']
    token = studies['entry'][0]['resource']['identifier'][0]['value']
    assert client.get('/api/fhir/ResearchStudy', params={'identifier':token}).json()['total'] == 1


@pytest.mark.parametrize('resource', [
    {'resourceType':'ResearchStudy','status':'completed'},
    {'resourceType':'Evidence','id':'broken','status':'active','title':'Missing required variableDefinition'},
    {'resourceType':'Bundle','type':'collection','total':4},
    {'resourceType':'Bundle','type':'collection','entry':[42]},
    {'resourceType':'NotAResource'},
])
def test_invalid_fhir_returns_conformant_operation_outcome(client, resource):
    response = client.post('/api/fhir/$validate', json=resource)
    assert response.status_code == 200
    outcome = response.json()
    get_fhir_model_class('OperationOutcome').model_validate(outcome)
    assert any(issue['severity'] == 'error' for issue in outcome['issue'])


def test_learning_api_does_not_accept_caller_mastery_or_duplicate_answers(client):
    state = client.post('/api/learning/sessions').json()
    item = state['next_item']
    assert 'correct_index' not in item and 'rationale' not in item
    payload = {'session_id': state['session_id'], 'question_token': item['question_token'], 'selected_index': 0}
    assert client.post('/api/learning/respond', json={**payload,'mastery':{'mental_model':1}}).status_code == 422
    assert client.post('/api/learning/respond', json=payload).status_code == 200
    assert client.post('/api/learning/respond', json=payload).status_code == 409
    client.delete(f"/api/learning/sessions/{state['session_id']}")
    assert client.get(f"/api/learning/sessions/{state['session_id']}").status_code == 404


@pytest.mark.parametrize('payload', [{'followup_months':12,'visit_interval_months':12}, {'visit_interval_months':7}])
def test_invalid_longitudinal_schedule_is_422_not_500(client, payload):
    assert client.post('/api/analytics/longitudinal-trial-simulation', json=payload).status_code == 422


def test_design_run_persistence_and_concurrency_gate(client):
    payload = {'sample_sizes':[40], 'durations':[24], 'simulations':2000, 'calibration_simulations':10000}
    first = client.post('/api/analytics/design-assurance', json=payload)
    assert first.status_code == 200
    envelope = first.json()
    assert client.get('/api/experiments/'+envelope['run_id']).json() == envelope
    assert envelope['integrity_verified']
    api.DESIGN_SLOT.acquire()
    try:
        assert client.post('/api/analytics/design-assurance', json=payload).status_code == 429
    finally:
        api.DESIGN_SLOT.release()
