"""Content-addressed, local experiment records. No patient information."""

from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class ExperimentStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL)"
            )

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def save(self, result: dict[str, Any]) -> dict[str, Any]:
        package = Path(__file__).parent
        filenames = [
            "design_assurance.py",
            "advanced_analytics.py",
            "analytics.py",
            "experiment_store.py",
        ]
        envelope = {
            "result": result,
            "reproducibility": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "source_sha256": {
                    name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                    for name in filenames
                },
                "input_sha256": hashlib.sha256(
                    canonical_json(result["sources"]).encode()
                ).hexdigest(),
                "scope": "Content hash proves local artifact integrity, not truth, preregistration or external timestamping.",
            },
        }
        payload = canonical_json(envelope)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        run_id = digest[:24]
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO experiments VALUES (?, ?, ?, ?)",
                (run_id, datetime.now(UTC).isoformat(), payload, digest),
            )
        return self.get(run_id)

    def get(self, run_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT created_at, payload, sha256 FROM experiments WHERE id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        actual = hashlib.sha256(row[1].encode()).hexdigest()
        if actual != row[2]:
            raise ValueError("Stored experiment digest mismatch")
        return {
            "run_id": run_id,
            "created_at": row[0],
            "sha256": row[2],
            "integrity_verified": True,
            **json.loads(row[1]),
        }

    def recent(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id,created_at,sha256 FROM experiments ORDER BY created_at DESC LIMIT 30"
            ).fetchall()
        return [
            {"run_id": row[0], "created_at": row[1], "sha256": row[2]} for row in rows
        ]
