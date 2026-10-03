"""Append-only SQLite audit ledger with a verifiable SHA-256 hash chain."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class AuditLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    recorded_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def append(
        self,
        *,
        actor: str,
        action: str,
        resource_type: str,
        resource_id: str,
        metadata: dict[str, Any] | None = None,
        recorded_at: str | None = None,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": event_id or str(uuid.uuid4()),
            "recorded_at": recorded_at or datetime.now(UTC).isoformat(),
            "actor": actor,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "metadata": metadata or {},
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = previous["event_hash"] if previous else "GENESIS"
            digest_input = {**event, "previous_hash": previous_hash}
            event_hash = hashlib.sha256(_canonical(digest_input).encode()).hexdigest()
            cursor = connection.execute(
                """
                INSERT INTO audit_events (
                    event_id, recorded_at, actor, action, resource_type, resource_id,
                    metadata_json, previous_hash, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    event["recorded_at"],
                    actor,
                    action,
                    resource_type,
                    resource_id,
                    _canonical(event["metadata"]),
                    previous_hash,
                    event_hash,
                ),
            )
            sequence = int(cursor.lastrowid)
        return {
            "sequence": sequence,
            **event,
            "previous_hash": previous_hash,
            "event_hash": event_hash,
        }

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY sequence DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "event_id": row["event_id"],
                "recorded_at": row["recorded_at"],
                "actor": row["actor"],
                "action": row["action"],
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "metadata": json.loads(row["metadata_json"]),
                "previous_hash": row["previous_hash"],
                "event_hash": row["event_hash"],
            }
            for row in rows
        ]

    def verify(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY sequence"
            ).fetchall()
        expected_previous = "GENESIS"
        invalid_sequences = []
        for row in rows:
            event = {
                "event_id": row["event_id"],
                "recorded_at": row["recorded_at"],
                "actor": row["actor"],
                "action": row["action"],
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "metadata": json.loads(row["metadata_json"]),
                "previous_hash": row["previous_hash"],
            }
            expected_hash = hashlib.sha256(_canonical(event).encode()).hexdigest()
            if row["previous_hash"] != expected_previous or row["event_hash"] != expected_hash:
                invalid_sequences.append(row["sequence"])
            expected_previous = row["event_hash"]
        return {
            "valid": not invalid_sequences,
            "events": len(rows),
            "invalid_sequences": invalid_sequences,
            "head_hash": expected_previous,
            "algorithm": "SHA-256 chained canonical JSON",
        }
