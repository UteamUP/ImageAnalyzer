"""Checkpoint system for resume capability — atomic writes + lock file."""

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


class CheckpointLockError(Exception):
    """Raised when another process holds the checkpoint lock."""


@dataclass
class Checkpoint:
    """Persists processing state for resume-after-interruption."""

    checkpoint_path: str
    processed: dict[str, dict] = field(default_factory=dict)  # sha256 -> result_dict
    started_at: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()
        self._lock_path = self.checkpoint_path + ".lock"

    def acquire_lock(self) -> None:
        """Acquire lock file to prevent concurrent runs."""
        lock = Path(self._lock_path)
        if lock.exists():
            try:
                with open(lock) as f:
                    lock_data = json.load(f)
                pid = lock_data.get("pid", 0)
                # Check if the process is still running
                try:
                    os.kill(pid, 0)
                    raise CheckpointLockError(
                        f"Another process (PID {pid}) is using this checkpoint. "
                        f"Delete {self._lock_path} if the process is not running."
                    )
                except OSError:
                    logger.warning("Stale lock file found, removing", pid=pid)
                    lock.unlink()
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt lock file found, removing")
                lock.unlink()

        with open(lock, "w") as f:
            json.dump({"pid": os.getpid(), "started": datetime.now().isoformat()}, f)

    def release_lock(self) -> None:
        """Release the lock file."""
        lock = Path(self._lock_path)
        if lock.exists():
            lock.unlink()

    def is_processed(self, file_hash: str) -> bool:
        """Check if a file has already been processed."""
        return file_hash in self.processed

    def add_result(self, file_hash: str, result_dict: dict) -> None:
        """Add a processed result and atomically save."""
        self.processed[file_hash] = result_dict
        self.last_updated = datetime.now().isoformat()
        self._atomic_save()

    def get_results(self) -> list[dict]:
        """Get all processed results."""
        return list(self.processed.values())

    def _atomic_save(self) -> None:
        """Write to temp file, then rename (atomic on POSIX)."""
        data = {
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "processed_count": len(self.processed),
            "processed": self.processed,
        }
        checkpoint_path = Path(self.checkpoint_path)
        tmp_path = checkpoint_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp_path.rename(checkpoint_path)

    def delete(self) -> None:
        """Delete checkpoint and lock files on successful completion."""
        for path in [self.checkpoint_path, self._lock_path]:
            p = Path(path)
            if p.exists():
                p.unlink()

    @classmethod
    def load(cls, checkpoint_path: str) -> "Checkpoint":
        """Load existing checkpoint or create new one."""
        path = Path(checkpoint_path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            cp = cls(
                checkpoint_path=checkpoint_path,
                processed=data.get("processed", {}),
                started_at=data.get("started_at", ""),
                last_updated=data.get("last_updated", ""),
            )
            logger.info(
                "Loaded checkpoint",
                processed_count=len(cp.processed),
                started_at=cp.started_at,
            )
            return cp
        return cls(checkpoint_path=checkpoint_path)

    def get_status(self) -> dict:
        """Get processing status for the status command."""
        type_counts: dict[str, int] = {}
        flagged = 0
        for result in self.processed.values():
            classification = result.get("classification", {})
            entity_type = classification.get("primary_type", "unknown")
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            if result.get("flagged_for_review"):
                flagged += 1

        return {
            "processed_count": len(self.processed),
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "type_breakdown": type_counts,
            "flagged_for_review": flagged,
        }
