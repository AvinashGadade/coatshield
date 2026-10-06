"""Tamper-evident audit trail: append-only JSONL, each entry chained to the last by SHA-256.

Every entry records who, what, when and why (the audit-trail elements of 21 CFR Part 11),
the model and config hashes in force, the previous entry's hash and its own.
verify_chain() detects any edited, removed or reordered entry.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
_FIELDS = ("seq", "time_utc", "user", "role", "action", "object", "old", "new", "reason",
           "model_hash", "config_hash", "signature", "prev_hash")


def _digest(entry: dict) -> str:
    body = {k: entry.get(k) for k in _FIELDS}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class ChainCheck:
    ok: bool
    entries: int
    first_bad_seq: int | None = None
    problem: str = ""


class AuditTrail:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def latest_hash(self) -> str:
        entries = self.entries()
        return entries[-1]["hash"] if entries else GENESIS

    def append(self, user: str, role: str, action: str, obj: str, old: Any = None,
               new: Any = None, reason: str = "", model_hash: str = "", config_hash: str = "",
               signature: dict | None = None) -> dict:
        """Add one entry. Nothing is ever rewritten: the file is opened for append only."""
        entries = self.entries()
        entry = {
            "seq": len(entries) + 1,
            "time_utc": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
            "user": user,
            "role": role,
            "action": action,
            "object": obj,
            "old": old,
            "new": new,
            "reason": reason,
            "model_hash": model_hash,
            "config_hash": config_hash,
            "signature": signature,
            "prev_hash": entries[-1]["hash"] if entries else GENESIS,
        }
        entry["hash"] = _digest(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        return entry

    def verify_chain(self, expected_latest: str | None = None) -> ChainCheck:
        """Recompute every hash and link.

        expected_latest: a hash noted elsewhere (for example in a batch record); with it,
        removing entries from the end of the log is detected too.
        """
        try:
            entries = self.entries()
        except json.JSONDecodeError as err:
            return ChainCheck(False, 0, None, f"log is not valid JSON lines: {err}")
        prev = GENESIS
        for i, entry in enumerate(entries, 1):
            if entry.get("seq") != i:
                return ChainCheck(False, len(entries), i, "sequence number out of order")
            if entry.get("prev_hash") != prev:
                return ChainCheck(False, len(entries), i, "link to the previous entry is broken")
            if entry.get("hash") != _digest(entry):
                return ChainCheck(False, len(entries), i, "entry content does not match its hash")
            prev = entry["hash"]
        if expected_latest is not None and prev != expected_latest:
            return ChainCheck(False, len(entries), len(entries),
                              "latest hash differs from the one on record")
        return ChainCheck(True, len(entries))
