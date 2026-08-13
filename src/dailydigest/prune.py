"""Bounded retention maintenance for local state."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import load_settings
from .store import prune as _prune
from .store import prune_old_digests as _prune_digests


def prune_old() -> int:
    """Delete items older than retention_days (protecting voted items). Returns count deleted."""
    settings = load_settings()
    return _prune(settings.retention_days)


def prune_old_digests() -> int:
    """Delete digest metadata older than retention_days. Returns count deleted."""
    settings = load_settings()
    return _prune_digests(settings.retention_days)


def prune_vote_reasons() -> int:
    """Remove vote_reasons.json entries for items that no longer exist in the DB.

    Returns the number of stale entries removed.
    """
    from sqlalchemy import select

    from .store import ItemRow, init_db, session_scope
    from .votes import _VOTE_REASONS_LOCK, _load_vote_reasons, _write_vote_reasons

    init_db()
    with session_scope() as s:
        existing_ids = {
            str(row_id)
            for (row_id,) in s.execute(select(ItemRow.id)).all()
        }

    with _VOTE_REASONS_LOCK:
        data = _load_vote_reasons()
        stale_keys = [k for k in data if k not in existing_ids]
        if stale_keys:
            for k in stale_keys:
                del data[k]
            _write_vote_reasons(data)
    return len(stale_keys)


def prune_previews(days: int) -> int:
    """Remove generated HTML previews older than the retention window."""
    settings = load_settings()
    preview_dir = Path(settings.db_path).expanduser().resolve().parent
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    removed = 0
    for path in preview_dir.glob("digest-*.html"):
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            if modified < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def run_maintenance() -> dict[str, int]:
    """Prune dependent history first, then unreferenced items and previews."""
    settings = load_settings()
    digests = _prune_digests(settings.retention_days)
    items = _prune(settings.retention_days)
    reasons = prune_vote_reasons()
    previews = prune_previews(settings.retention_days)
    return {
        "digests": digests,
        "items": items,
        "vote_reasons": reasons,
        "previews": previews,
    }
