"""Cross-day near-duplicate suppression.

Identifier/URL/same-day-title dedupe (``dedupe.py``) collapses duplicates within
a single candidate set, but the same paper or story often re-surfaces days later
through a *different* source (a journal RSS item later indexed by OpenAlex, a
news story re-covered by another outlet) as a distinct row with no shared
identifier. This module drops a candidate when its title+abstract embedding is
near an item already shown in a recent sent digest — content-level dedupe across
days. It deliberately ignores same-id matches (governed by the existing
previously-shown policy) and targets only genuinely different rows.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import select

from ..store import DigestItemRow, DigestRow, ItemRow, session_scope
from .embedding_cache import embed_item_rows

logger = logging.getLogger(__name__)


def _recently_shown_rows(days_lookback: int, exclude_ids: set[int]) -> list[ItemRow]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_lookback)
    with session_scope() as s:
        shown_ids = {
            int(item_id)
            for item_id in s.execute(
                select(DigestItemRow.item_id)
                .join(DigestRow, DigestRow.id == DigestItemRow.digest_id)
                .where(DigestRow.sent_at.isnot(None), DigestRow.sent_at >= cutoff)
            ).scalars()
        }
        shown_ids -= exclude_ids
        if not shown_ids:
            return []
        rows = (
            s.execute(select(ItemRow).where(ItemRow.id.in_(list(shown_ids))))
            .scalars()
            .all()
        )
        for r in rows:
            s.expunge(r)
        return list(rows)


def _audit(row: ItemRow, max_sim: float) -> dict[str, Any]:
    return {
        "stage": "cross_day_near_dup",
        "reason": "near-duplicate of a recently shown item",
        "item_id": int(row.id) if isinstance(row.id, int) else None,
        "source": row.source or "",
        "section": row.section or "",
        "title": row.title or "",
        "url": row.url or "",
        "max_similarity": round(float(max_sim), 4),
    }


def exclude_recent_near_duplicates(
    rows: list[ItemRow],
    *,
    settings: object | None = None,
    days_lookback: int | None = None,
    threshold: float | None = None,
) -> tuple[list[ItemRow], list[dict[str, Any]]]:
    """Return ``(kept_rows, dropped_audit)`` removing recent near-duplicates."""
    if not rows:
        return rows, []
    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    if not getattr(settings, "cross_day_dedupe", True):
        return rows, []

    days_lookback = (
        days_lookback
        if days_lookback is not None
        else int(getattr(settings, "cross_day_dedupe_days", 7))
    )
    threshold = (
        threshold
        if threshold is not None
        else float(getattr(settings, "cross_day_dedupe_threshold", 0.93))
    )

    candidate_ids = {int(r.id) for r in rows if isinstance(getattr(r, "id", None), int)}
    shown = _recently_shown_rows(days_lookback, candidate_ids)
    if not shown:
        return rows, []

    try:
        shown_vecs = embed_item_rows(shown)
        cand_vecs = embed_item_rows(rows)
    except Exception as e:  # noqa: BLE001
        logger.warning("near-dup embedding failed (%s); skipping", e)
        return rows, []
    if shown_vecs.size == 0 or cand_vecs.size == 0 or cand_vecs.shape[0] != len(rows):
        return rows, []

    sv = shown_vecs / (np.linalg.norm(shown_vecs, axis=1, keepdims=True) + 1e-9)
    cv = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
    sims = cv @ sv.T  # (n_candidates, n_shown)
    max_sim = sims.max(axis=1) if sims.shape[1] > 0 else np.zeros(len(rows))

    kept: list[ItemRow] = []
    dropped: list[dict[str, Any]] = []
    for row, ms in zip(rows, max_sim, strict=True):
        if float(ms) >= threshold:
            dropped.append(_audit(row, float(ms)))
        else:
            kept.append(row)
    if dropped:
        logger.info("cross_day_near_dup: dropped %d re-surfaced items", len(dropped))
    return kept, dropped
