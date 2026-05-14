"""Prune wrapper for CLI."""

from .config import load_settings
from .store import prune as _prune, prune_old_digests as _prune_digests


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
    from .store import ItemRow, init_db, session_scope
    from .votes import _VOTE_REASONS_LOCK, _load_vote_reasons, _write_vote_reasons
    from sqlalchemy import select

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
