import json
import sqlite3

import pytest

from sca3_compass.learning_engine import initial_mastery, select_next_item
from sca3_compass.learning_sessions import LearningSessions, SessionConflict
from sca3_compass.repository import load_learning_model


def test_no_answer_key_or_rationale_in_unanswered_item(tmp_path):
    sessions = LearningSessions(tmp_path / "learning.sqlite3")
    state = sessions.start(load_learning_model())
    assert "correct_index" not in json.dumps(state)
    assert "rationale" not in json.dumps(state)
    assert state == sessions.get(state["session_id"], load_learning_model())


def test_single_use_tokens_prevent_duplicate_mastery_updates(tmp_path):
    sessions = LearningSessions(tmp_path / "learning.sqlite3")
    model = load_learning_model()
    state = sessions.start(model)
    question = state["next_item"]
    canonical = next(item for item in model["items"] if item["id"] == question["id"])
    index = question["options"].index(canonical["options"][canonical["correct_index"]])
    updated = sessions.respond(
        state["session_id"], model, question["question_token"], index
    )
    assert updated["feedback"]["correct"]
    assert updated["completed_items"] == 1
    assert (
        updated["mastery"][question["concept_id"]]
        > state["mastery"][question["concept_id"]]
    )
    with pytest.raises(SessionConflict):
        sessions.respond(state["session_id"], model, question["question_token"], index)


def test_exhausted_prerequisite_offers_review():
    model = load_learning_model()
    selected = select_next_item(model, initial_mastery(model), ["mm-01", "mm-02"])
    assert selected["concept_id"] == "mental_model"
    assert selected["selection"]["review"]


def test_wrong_answer_feedback_and_deletion(tmp_path):
    sessions = LearningSessions(tmp_path / "learning.sqlite3")
    model = load_learning_model()
    state = sessions.start(model)
    item = state["next_item"]
    canonical = next(x for x in model["items"] if x["id"] == item["id"])
    wrong = item["options"].index(
        canonical["options"][(canonical["correct_index"] + 1) % len(item["options"])]
    )
    result = sessions.respond(state["session_id"], model, item["question_token"], wrong)
    assert not result["feedback"]["correct"]
    assert result["feedback"]["misconception"]
    sessions.delete(state["session_id"])
    with pytest.raises(KeyError):
        sessions.get(state["session_id"], model)


def test_expired_session_is_inaccessible(tmp_path):
    sessions = LearningSessions(tmp_path / "learning.sqlite3")
    model = load_learning_model()
    state = sessions.start(model)
    with sqlite3.connect(sessions.path) as conn:
        conn.execute("UPDATE learning_sessions SET expires=0")
    with pytest.raises(KeyError):
        sessions.get(state["session_id"], model)
