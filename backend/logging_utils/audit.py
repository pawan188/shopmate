"""
Append-only structured audit logger (JSON Lines) shared by the agent, validator, and payment layers.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_DIR = Path(__file__).resolve().parent.parent / "audit_log"


class AuditLogger:
    """Append-only JSONL logger. Safe to share across threads (FastAPI later)."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory) if directory else AUDIT_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._path = self.directory / f"audit-{datetime.now().strftime('%Y%m%d')}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def log(
        self,
        event: str,
        *,
        session_id: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one event line. Returns the entry that was written."""
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "event": event,
        }
        if data:
            entry.update(data)
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return entry

    def read(self) -> list[dict[str, Any]]:
        """Replay the whole file, oldest first (for the demo screen)."""
        if not self._path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(self._path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        entries.append({"event": "unparseable_line", "raw": line})
        return entries
