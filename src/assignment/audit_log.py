"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """TODO: store input + start timestamp keyed by request_id/user_id."""
        key = request_id or user_id
        start_time = datetime.now().timestamp()
        self._open[key] = start_time
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "request_id": request_id,
            "type": "input",
            "content": text,
            "blocked": False,
            "layer": None,
        }
        self.logs.append(log_entry)
        return log_entry

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """TODO: store output, layer decision, latency; append to self.logs."""
        key = request_id or user_id
        duration_ms = 0.0
        if key in self._open:
            start_time = self._open.pop(key)
            duration_ms = (datetime.now().timestamp() - start_time) * 1000

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "request_id": request_id,
            "type": "output",
            "content": text,
            "blocked": blocked,
            "layer": layer,
            "response_preview": text[:100] if text else "",
            "duration_ms": round(duration_ms, 2)
        }
        self.logs.append(log_entry)

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        # TODO: path = filepath or default_audit_log_path()
        #       ensure parent dirs exist, dump self.logs with indent=2
        path = filepath or default_audit_log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
