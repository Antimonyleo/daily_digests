"""End-to-end orchestration: ingest -> rank -> summarize -> render -> send."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from . import health
from .config import SETTINGS, load_profile, load_sources
from .dedupe import dedupe_by_url, filter_english
from .email_render import SECTION_ORDER, render_digest
from .email_send import send_digest
from .health import IngestStats
from .ingest import dispatch_source
from .models import Item
from .rank.profile import build_profile_vector
from .rank.ranker import pick_top_per_section, score_items
from .store import (
    DigestRow,
    ItemRow,
    init_db,
    mark_sent,
    recent_items,
    session_scope,
    upsert_items,
    write_digest,
)
from .summarize import summarize_items

logger = logging.getLogger(__name__)

SECTION_LABEL_PREFIX: dict[str, str] = {
    "research": "R",
    "industry": "I",
    "regulatory": "G",
    "world": "W",
}

# Type alias for the optional progress callback used by run_all().
# Stages emitted (in order): "ingest_start", "ingest_done", "dedupe_done",
# "rank_done", "summarize_start", "summarize_done", "render_done", "done".
ProgressCallback = Callable[[str, dict[str, Any]], None]


def _emit(cb: ProgressCallback | None, stage: str, payload: dict[str, Any]) -> None:
    """Best-effort progress emission; never raises."""
    if cb is None:
        return
    try:
        cb(stage, payload)
    except Exception as e:  # noqa: BLE001
        logger.warning("progress_callback failed at stage=%s: %s", stage, e)


def ingest_all(progress_callback: ProgressCallback | None = None) -> int:
    """Fetch all sources, dedupe + langdetect filter, upsert. Returns rows inserted."""
    init_db()
    specs = load_sources()
    _emit(progress_callback, "ingest_start", {"sources": len(specs)})
    all_items: list[Item] = []
    stats: list[IngestStats] = []
    for spec in specs:
        t0 = time.monotonic()
        try:
            src = dispatch_source(spec)
            fetched = src.fetch(spec)
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.info("ingest %s: %d items (%dms)", spec.name, len(fetched), duration_ms)
            all_items.extend(fetched)
            stats.append(
                IngestStats(
                    source=spec.name,
                    items=len(fetched),
                    ok=True,
                    duration_ms=duration_ms,
                )
            )
        except Exception as e:  # noqa: BLE001
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.warning("ingest %s failed: %s", spec.name, e)
            stats.append(
                IngestStats(
                    source=spec.name,
                    items=0,
                    ok=False,
                    error=f"{type(e).__name__}: {e}"[:200],
                    duration_ms=duration_ms,
                )
            )

    try:
        health.record(stats)
    except Exception as e:  # noqa: BLE001
        logger.warning("health.record failed: %s", e)

    _emit(
        progress_callback,
        "ingest_done",
        {"sources": len(specs), "raw_items": len(all_items)},
    )
    deduped = dedupe_by_url(all_items)
    en_only = filter_english(deduped)
    logger.info(
        "ingest aggregate: %d raw -> %d deduped -> %d english",
        len(all_items),
        len(deduped),
        len(en_only),
    )
    _emit(
        progress_callback,
        "dedupe_done",
        {
            "raw_items": len(all_items),
            "deduped": len(deduped),
            "english": len(en_only),
        },
    )
    return upsert_items(en_only)


def _section_caps() -> dict[str, int]:
    return {
        "research": SETTINGS.top_research,
        "industry": SETTINGS.top_industry,
        "regulatory": SETTINGS.top_regulatory,
        "world": SETTINGS.top_world,
    }


def _assign_labels(
    picked: list[tuple[ItemRow, float]],
) -> list[tuple[ItemRow, float, str]]:
    """Assign R1..Rn / I1..In / G1..Gn / W1..Wn within each section, in score order."""
    counters: dict[str, int] = {}
    out: list[tuple[ItemRow, float, str]] = []
    for row, score in picked:
        prefix = SECTION_LABEL_PREFIX.get(row.section or "", "X")
        counters[prefix] = counters.get(prefix, 0) + 1
        label = f"{prefix}{counters[prefix]}"
        row.item_label = label
        out.append((row, score, label))
    return out


def _digest_id() -> str:
    """Compute the digest id in the user's local TZ.

    We anchor the date to the user's wall clock so that "today's digest"
    matches the day they expect — not whatever UTC date happens to coincide
    with their local 8am.
    """
    try:
        tz: ZoneInfo | timezone = ZoneInfo(SETTINGS.user_tz)
    except Exception:  # noqa: BLE001 - bad TZ string falls back to UTC
        tz = timezone.utc
    return datetime.now(tz).strftime("%Y-%m-%d")


def _should_write_digest(digest_id: str) -> bool:
    """Return True if `write_digest` should be called for this digest_id.

    Avoids clobbering an existing `sent_at` via SQLAlchemy `merge`. If the
    row already exists and was already sent, we skip the write entirely.
    Otherwise (missing row, or row exists but not yet sent) we proceed.
    """
    with session_scope() as s:
        row = s.get(DigestRow, digest_id)
        if row is None:
            return True
        return row.sent_at is None


def run_all(
    dry_run: bool = False,
    backfill_days: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Run ingest + rank + summarize + render + send. Returns the digest id.

    ``backfill_days`` widens the recency window used for ranking (default 2).
    Useful for catching up after a missed run, e.g. ``run_all(backfill_days=7)``.
    """
    init_db()

    # Idempotency check: if today's digest has already been sent and this is
    # not a dry-run, skip everything (no re-ingest, no re-render, no resend).
    # Dry-runs are always allowed to proceed so the user can preview.
    digest_id_early = _digest_id()
    if not dry_run:
        with session_scope() as s:
            existing = s.get(DigestRow, digest_id_early)
            if existing is not None and existing.sent_at is not None:
                logger.info(
                    "digest %s already sent at %s; skipping resend",
                    digest_id_early,
                    existing.sent_at,
                )
                return digest_id_early

    inserted = ingest_all(progress_callback=progress_callback)
    logger.info("upserted %d new items", inserted)

    profile = load_profile()
    profile_vec = build_profile_vector(profile)

    days = backfill_days if backfill_days and backfill_days > 0 else 2
    items = recent_items(days=days)
    logger.info("ranking %d recent items (window=%d days)", len(items), days)
    scored = score_items(items, profile_vec, profile.downweight)
    # Note: do NOT pre-truncate `scored` to a global top-K before per-section picking;
    # a single-domain profile (e.g. biotech-heavy) starves industry/regulatory/world.
    # `pick_top_per_section` already caps per-section, so summary cost is bounded.
    picked = pick_top_per_section(scored, _section_caps())
    labeled = _assign_labels(picked)
    _emit(
        progress_callback,
        "rank_done",
        {"candidates": len(items), "picked": len(labeled)},
    )

    selected_rows = [row for row, _, _ in labeled]
    _emit(
        progress_callback,
        "summarize_start",
        {"items": len(selected_rows), "backend": SETTINGS.llm_backend},
    )
    summaries = summarize_items(selected_rows)
    _emit(progress_callback, "summarize_done", {"items": len(summaries)})

    sections: dict[str, list[tuple[ItemRow, float, str]]] = {k: [] for k in SECTION_ORDER}
    for row, score, _label in labeled:
        if row.section in sections:
            sections[row.section].append((row, score, summaries.get(row.id, "")))

    digest_id = _digest_id()
    # Only call write_digest if there is no already-sent row for this id.
    # store.write_digest uses session.merge, which would clobber `sent_at` to
    # NULL on a re-run. Skipping the write preserves the original send timestamp.
    if _should_write_digest(digest_id):
        write_digest(digest_id, [(row.item_label, row.id) for row, _, _ in labeled])
    else:
        logger.info(
            "digest %s already has a sent row; skipping write_digest to preserve sent_at",
            digest_id,
        )

    health_summary: list[dict] | None = None
    try:
        summary = health.weekly_summary()
        if summary and health.should_show(summary):
            health_summary = summary
    except Exception as e:  # noqa: BLE001
        logger.warning("health.weekly_summary failed: %s", e)

    html = render_digest(digest_id, sections, health_summary=health_summary)
    _emit(
        progress_callback,
        "render_done",
        {"digest_id": digest_id, "html_bytes": len(html)},
    )
    subject = f"DailyDigest {digest_id}"
    send_digest(html, subject, dry_run=dry_run)

    if not dry_run and SETTINGS.resend_api_key and SETTINGS.digest_to:
        mark_sent(digest_id)

    total_items = sum(len(v) for v in sections.values())
    _emit(
        progress_callback,
        "done",
        {"digest_id": digest_id, "total_items": total_items, "dry_run": dry_run},
    )
    return digest_id
