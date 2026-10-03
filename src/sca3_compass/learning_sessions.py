"""Local, anonymous assessment state with single-use question tokens."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from .learning_engine import evaluate_response, initial_mastery, select_next_item


class SessionConflict(ValueError):
    pass


def public_item(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        key: item[key]
        for key in ("id", "concept_id", "prompt", "options", "selection")
        if key in item
    }


class LearningSessions:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS learning_sessions (id TEXT PRIMARY KEY, expires REAL NOT NULL, state TEXT NOT NULL)"
            )

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=15)

    @staticmethod
    def model_digest(model: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()

    def _offer(self, model: dict[str, Any], state: dict[str, Any]) -> None:
        next_item = select_next_item(model, state["mastery"], state["history"])
        state["completion"] = None
        if len(state["history"]) >= 30:
            next_item = None
            state["completion"] = "practice_limit_reached"
        elif next_item is None:
            state["completion"] = "model_thresholds_reached"
        if next_item is None:
            state["offered"] = None
            return
        order = list(range(len(next_item["options"])))
        secrets.SystemRandom().shuffle(order)
        offered = public_item(next_item)
        assert offered is not None
        offered["options"] = [next_item["options"][i] for i in order]
        offered["question_token"] = secrets.token_urlsafe(24)
        state["option_order"] = order
        state["offered"] = offered

    @staticmethod
    def _public(
        session_id: str, model: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "next_item": state["offered"],
            "mastery": state["mastery"],
            "completed_items": len(state["history"]),
            "completion": state["completion"],
            "concepts": model["concepts"],
            "model_sha256": state["model_sha256"],
            "feedback": state.get("feedback"),
            "privacy": "Anonymous local practice: selected option and modeled mastery only. Sessions expire after 7 days; expired records are purged when a new session is created. You can delete now. No patient details or free text are collected.",
            "interpretation_boundary": "Mastery is a model belief, not a validated measurement of competence. Parameters and repeated-item corrections are design assumptions.",
        }

    def start(self, model: dict[str, Any]) -> dict[str, Any]:
        session_id = secrets.token_urlsafe(32)
        state = {
            "mastery": initial_mastery(model),
            "history": [],
            "model_sha256": self.model_digest(model),
        }
        self._offer(model, state)
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM learning_sessions WHERE expires<?", (time.time(),)
            )
            if (
                conn.execute("SELECT COUNT(*) FROM learning_sessions").fetchone()[0]
                >= 1000
            ):
                raise SessionConflict(
                    "Local session capacity reached; delete unused sessions"
                )
            conn.execute(
                "INSERT INTO learning_sessions VALUES (?, ?, ?)",
                (session_id, time.time() + 7 * 86400, json.dumps(state)),
            )
        return self._public(session_id, model, state)

    def _load(
        self, conn: sqlite3.Connection, session_id: str, model: dict[str, Any]
    ) -> dict[str, Any]:
        row = conn.execute(
            "SELECT state FROM learning_sessions WHERE id=? AND expires>?",
            (session_id, time.time()),
        ).fetchone()
        if row is None:
            raise KeyError("Session not found or expired")
        state = json.loads(row[0])
        if state["model_sha256"] != self.model_digest(model):
            raise SessionConflict(
                "Item bank changed; delete this session and start a new one"
            )
        return state

    def get(self, session_id: str, model: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as conn:
            state = self._load(conn, session_id, model)
        return self._public(session_id, model, state)

    def respond(
        self,
        session_id: str,
        model: dict[str, Any],
        question_token: str,
        selected_index: int,
    ) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            state = self._load(conn, session_id, model)
            offered = state["offered"]
            if offered is None or not secrets.compare_digest(
                offered["question_token"], question_token
            ):
                raise SessionConflict(
                    "Question already answered or stale; reload the session"
                )
            if not 0 <= selected_index < len(state["option_order"]):
                raise ValueError("Selected option is out of bounds")
            result = evaluate_response(
                model,
                mastery=state["mastery"],
                history=state["history"],
                item_id=offered["id"],
                selected_index=state["option_order"][selected_index],
            )
            item = next(item for item in model["items"] if item["id"] == offered["id"])
            state["feedback"] = {
                key: result[key]
                for key in (
                    "item_id",
                    "concept_id",
                    "correct",
                    "rationale",
                    "misconception",
                    "update",
                    "review",
                    "review_assumption",
                )
            }
            state["feedback"].update(
                {
                    "prompt": item["prompt"],
                    "selected_option": offered["options"][selected_index],
                    "correct_option": item["options"][item["correct_index"]],
                    "evidence_claim_id": item["evidence_claim_id"],
                }
            )
            state["mastery"], state["history"] = result["mastery"], result["history"]
            self._offer(model, state)
            conn.execute(
                "UPDATE learning_sessions SET state=? WHERE id=?",
                (json.dumps(state), session_id),
            )
        return self._public(session_id, model, state)

    def delete(self, session_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM learning_sessions WHERE id=?", (session_id,))
