"""Cross-day near-duplicate suppression.

Identifier/URL/same-day-title dedupe (``dedupe.py``) collapses duplicates within
a single candidate set, but the same paper or story often re-surfaces days later
through a *different* source (a journal RSS item later indexed by OpenAlex, a
news story re-covered by another outlet) as a distinct row with no shared
identifier. This module drops a candidate when its title+abstract embedding is
near an item already published in a recent local, email or browser digest.
Research additionally has a fourteen-day identity/version cooldown, independent
of cosine, so a preprint and its journal version cannot use separate seats.
Same-day rebrews may upgrade versions. Same-id matches remain governed by the
existing previously-shown policy.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import or_, select

from ..dedupe import are_publication_versions, candidate_identity_keys, publication_version_title_key
from ..store import DigestItemRow, DigestRow, ImpressionRow, ItemRow, session_scope
from .embedding_cache import embed_item_rows

logger = logging.getLogger(__name__)
_RESEARCH_VERSION_COOLDOWN_DAYS = 14


def _recently_shown_rows(
    days_lookback: int, exclude_ids: set[int], exclude_digest_id: str | None = None,
) -> list[ItemRow]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_lookback)
    with session_scope() as s:
        published_stmt = (
            select(DigestItemRow.item_id)
            .join(DigestRow, DigestRow.id == DigestItemRow.digest_id)
            .where(or_(DigestRow.created_at >= cutoff, DigestRow.sent_at >= cutoff))
        )
        impression_stmt = select(ImpressionRow.item_id).where(
            ImpressionRow.selected.is_(True),
            ImpressionRow.viewed.is_(True),
            ImpressionRow.created_at >= cutoff,
        )
        if exclude_digest_id is not None:
            published_stmt = published_stmt.where(DigestItemRow.digest_id != exclude_digest_id)
            impression_stmt = impression_stmt.where(ImpressionRow.digest_id != exclude_digest_id)
        published_ids = {
            int(item_id)
            for item_id in s.execute(published_stmt).scalars()
        }
        browser_ids = {
            int(item_id)
            for item_id in s.execute(impression_stmt).scalars()
        }
        shown_ids = published_ids | browser_ids
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


def _audit(row: ItemRow, max_sim: float | None = None) -> dict[str, Any]:
    result = {
        "stage": "cross_day_near_dup",
        "reason": "near-duplicate of a recently shown item",
        "item_id": int(row.id) if isinstance(row.id, int) else None,
        "source": row.source or "",
        "section": row.section or "",
        "title": row.title or "",
        "url": row.url or "",
    }
    if max_sim is not None:
        result["max_similarity"] = round(float(max_sim), 4)
    return result


def _exclude_recent_research_versions(rows, shown):
    identifiers: dict[str, ItemRow] = {}
    titles: dict[str, list[ItemRow]] = {}
    for previous in shown:
        if previous.section != "research":
            continue
        for key in candidate_identity_keys(previous):
            if not key.startswith("title_day:"):
                identifiers[key] = previous
        title = publication_version_title_key(previous)
        if title:
            titles.setdefault(title, []).append(previous)
    kept, dropped = [], []
    for row in rows:
        matched = None
        match_type = "identifier"
        if row.section == "research":
            matched = next((identifiers[key] for key in candidate_identity_keys(row)
                            if not key.startswith("title_day:") and key in identifiers), None)
            if matched is None:
                match_type = "preprint_journal_title"
                matched = next((previous for previous in titles.get(publication_version_title_key(row), [])
                                if are_publication_versions(row, previous)), None)
        if matched is None:
            kept.append(row)
        else:
            audit = _audit(row)
            audit.update(reason="same research work in a recent digest", match_type=match_type,
                         matched_item_id=int(matched.id), cooldown_days=_RESEARCH_VERSION_COOLDOWN_DAYS)
            dropped.append(audit)
    return kept, dropped


def exclude_recent_near_duplicates(
    rows: list[ItemRow],
    *,
    settings: object | None = None,
    days_lookback: int | None = None,
    threshold: float | None = None,
    exclude_digest_id: str | None = None,
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
    version_history = _recently_shown_rows(_RESEARCH_VERSION_COOLDOWN_DAYS, candidate_ids, exclude_digest_id)
    rows, dropped = _exclude_recent_research_versions(rows, version_history)
    if not rows:
        return rows, dropped
    shown = (version_history if days_lookback == _RESEARCH_VERSION_COOLDOWN_DAYS
             else _recently_shown_rows(days_lookback, candidate_ids, exclude_digest_id))
    if not shown:
        return rows, dropped

    try:
        shown_vecs = embed_item_rows(shown)
        cand_vecs = embed_item_rows(rows)
    except Exception as e:  # noqa: BLE001
        logger.warning("near-dup embedding failed (%s); skipping", e)
        return rows, dropped
    if shown_vecs.size == 0 or cand_vecs.size == 0 or cand_vecs.shape[0] != len(rows):
        return rows, dropped

    sv = shown_vecs / (np.linalg.norm(shown_vecs, axis=1, keepdims=True) + 1e-9)
    cv = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
    sims = cv @ sv.T  # (n_candidates, n_shown)
    max_sim = sims.max(axis=1) if sims.shape[1] > 0 else np.zeros(len(rows))

    kept: list[ItemRow] = []
    for row, ms in zip(rows, max_sim, strict=True):
        if float(ms) >= threshold:
            dropped.append(_audit(row, float(ms)))
        else:
            kept.append(row)
    if dropped:
        logger.info("cross_day_near_dup: dropped %d re-surfaced items", len(dropped))
    return kept, dropped
