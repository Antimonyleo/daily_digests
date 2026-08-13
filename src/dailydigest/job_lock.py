"""Cross-process lock for local embedding and brew jobs."""

from __future__ import annotations

from pathlib import Path

from filelock import FileLock, Timeout


class ComputeBusyError(RuntimeError):
    """Raised when another DailyDigest process owns the compute lock."""


def acquire_compute_lock(db_path: str, *, timeout: float = 0.0) -> FileLock:
    """Acquire the one-machine compute lock next to the configured database."""
    lock_path = Path(db_path).expanduser().resolve().parent / ".compute.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=timeout)
    except Timeout as exc:
        raise ComputeBusyError(
            "Another DailyDigest brew or ranking job is already running."
        ) from exc
    return lock
