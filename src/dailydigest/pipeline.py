"""End-to-end orchestration: ingest -> rank -> summarize -> render -> send."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from . import health, votes as votes_mod
from .config import get_settings, load_profile, load_sources
from .dedupe import dedupe_by_url, dedupe_ranking_candidates, filter_english
from .email_render import SECTION_ORDER, render_digest
from .email_send import send_digest
from .health import IngestStats
from .ingest import dispatch_source
from .models import Item
from .rank.profile import build_profile_matrix
from .rank.ranker import pick_top_per_section, score_items, score_items_with_features
from .rank.source_quality import (
    breakdown_payload,
    is_high_quality_journal_source,
    source_bucket,
)
from .store import (
    DigestRow,
    ItemRow,
    days_since_last_sent,
    exclude_previously_shown,
    exclude_reviewed_items,
    init_db,
    mark_sent,
    recent_items,
    session_scope,
    upsert_items,
    write_digest_audit,
    write_digest,
    write_digest_features,
    write_summaries,
)
from .summarize import summarize_items

try:
    from .rank.profile import build_profile_matrix_with_rocchio as _build_profile_with_rocchio
except ImportError:
    _build_profile_with_rocchio = None  # type: ignore[assignment]

try:
    from .rank.profile import build_negative_centroid as _build_neg_centroid
except ImportError:
    _build_neg_centroid = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
_ORIGINAL_SCORE_ITEMS = score_items

_TITLE_BLOCKLIST = re.compile(
    r"^(?:volume\s+\d|issue\s+\d|editorial\b|correspondence\b|correction\b|"
    r"erratum\b|in\s+this\s+issue|table\s+of\s+contents|show\s+hn:|ask\s+hn:|"
    r"sponsored:?|webinar:|save\s+the\s+date)",
    re.IGNORECASE,
)


def _quality_gate(rows: list[ItemRow]) -> list[ItemRow]:
    """Pre-filter obviously low-quality candidates before scoring.

    Drops items that are clearly not useful: very short titles, editorial
    metadata, content-farm patterns, and research items with no abstract.
    Skipped when the pool is < 10 items to preserve test/backfill behaviour.
    """
    if len(rows) < 10:
        return rows
    out: list[ItemRow] = []
    for r in rows:
        title = (r.title or "").strip()
        abstract = (r.abstract or "").strip()
        if len(title) < 15:
            continue
        if _TITLE_BLOCKLIST.match(title):
            continue
        if (r.section or "").lower() == "research" and len(abstract) < 30:
            continue
        out.append(r)
    dropped = len(rows) - len(out)
    if dropped:
        logger.info("quality_gate: dropped %d low-quality candidates", dropped)
    return out

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
    s = get_settings()
    return {
        "research": s.top_research,
        "industry": s.top_industry,
        "regulatory": s.top_regulatory,
        "world": s.top_world,
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
        tz: ZoneInfo | timezone = ZoneInfo(get_settings().user_tz)
    except Exception:  # noqa: BLE001 - bad TZ string falls back to UTC
        tz = timezone.utc
    return datetime.now(tz).strftime("%Y-%m-%d")


def _row_feature_key(row: ItemRow) -> int:
    return int(row.id) if isinstance(row.id, int) else id(row)


def _selection_reason(row: ItemRow, features: dict[str, Any]) -> str:
    bucket = str(features.get("source_bucket") or source_bucket(row))
    if row.section == "research":
        if bucket == "published_journal":
            return "protected published-journal slot"
        if bucket == "published_database":
            return "published-paper database slot"
        if bucket == "arxiv_cs":
            return "capped arXiv CS slot"
        if bucket in {"arxiv_other", "bio_med_preprint"}:
            return "capped preprint slot"
        if bucket == "aggregator":
            return "capped aggregator slot"
    return "score-ranked slot"


def _build_top_journal_audit(
    scored: list[tuple[ItemRow, float]],
    picked: list[tuple[ItemRow, float]],
    score_features: dict[int, dict[str, Any]],
    cap: int = 20,
) -> list[dict[str, Any]]:
    selected_ids = {_row_feature_key(row) for row, _score in picked}
    missed: list[dict[str, Any]] = []
    for row, score in scored:
        if row.section != "research" or not is_high_quality_journal_source(row):
            continue
        key = _row_feature_key(row)
        if key in selected_ids:
            continue
        features = score_features.get(key, {})
        missed.append(
            {
                "item_id": int(row.id) if isinstance(row.id, int) else None,
                "title": row.title or "",
                "source": row.source or "",
                "url": row.url or "",
                "score": round(float(score), 4),
                "topic_score": round(float(features.get("topic_score", score) or 0.0), 4),
                "source_bucket": features.get("source_bucket") or source_bucket(row),
                "reason": (
                    "below the final cutoff after topic fit, diversity caps, "
                    "and feedback penalties"
                ),
            }
        )
        if len(missed) >= cap:
            break
    return missed


def _score_items_for_pipeline(
    items: list[ItemRow],
    profile_vec,
    downweight: list[str],
) -> tuple[list[tuple[ItemRow, float]], dict[int, dict[str, Any]]]:
    """Score with feature snapshots, falling back for tests that monkeypatch score_items."""
    try:
        reason_penalties = votes_mod.reason_penalty_map(items)
    except TypeError:
        reason_penalties = votes_mod.reason_penalty_map()
    if score_items is not _ORIGINAL_SCORE_ITEMS:
        scored = score_items(
            items,
            profile_vec,
            downweight,
            reason_penalty_map=reason_penalties,
        )
        features = {
            _row_feature_key(row): {
                "topic_score": float(score),
                "learned_score": 0.0,
                "final_score": float(score),
                "reason_penalty": 0.0,
                "source_bucket": source_bucket(row),
                "scoring_mode": "legacy",
            }
            for row, score in scored
        }
        return scored, features
    try:
        return score_items_with_features(
            items,
            profile_vec,
            downweight,
            reason_penalty_map=reason_penalties,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("feature-scoring path failed, falling back to legacy scorer: %s", e)
        scored = score_items(
            items,
            profile_vec,
            downweight,
            reason_penalty_map=reason_penalties,
        )
        features = {
            _row_feature_key(row): {
                "topic_score": float(score),
                "learned_score": 0.0,
                "final_score": float(score),
                "reason_penalty": 0.0,
                "source_bucket": source_bucket(row),
                "scoring_mode": "legacy",
            }
            for row, score in scored
        }
        return scored, features


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

    ``backfill_days`` widens the recency window used for ranking. When omitted,
    the window is set automatically: if the last sent digest was N days ago,
    ``days = N + 1`` (capped at 7), so no recent day's content is missed.
    Pass an explicit value to override.
    """
    init_db()

    # Idempotency check: if today's digest has already been sent and this is
    # not a dry-run, skip everything (no re-ingest, no re-render, no resend).
    # Dry-runs are always allowed to proceed so the user can preview.
    digest_id = _digest_id()
    if not dry_run:
        with session_scope() as s:
            existing = s.get(DigestRow, digest_id)
            if existing is not None and existing.sent_at is not None:
                logger.info(
                    "digest %s already sent at %s; skipping resend",
                    digest_id,
                    existing.sent_at,
                )
                return digest_id

    inserted = ingest_all(progress_callback=progress_callback)
    logger.info("upserted %d new items", inserted)

    # Auto-retrain LR when model is stale (> 7 days old) and enough votes exist
    try:
        from .votes import MIN_VOTES_FOR_LR as _min_lr
        from .votes import signed_vote_count as _svc

        _current_votes = _svc()
        if _current_votes >= _min_lr:
            from pathlib import Path as _Path

            _lr_path = _Path(get_settings().db_path).parent / "lr_ranker.npz"
            _needs_retrain = not _lr_path.exists()
            if not _needs_retrain and _lr_path.exists():
                _lr_age_days = (time.time() - _lr_path.stat().st_mtime) / 86400
                _needs_retrain = _lr_age_days > 7
            if _needs_retrain:
                from .votes import train_lr_ranker as _train_lr

                _retrain_result = _train_lr()
                if _retrain_result.get("trained"):
                    logger.info(
                        "auto-retrained LR ranker on %d votes",
                        _retrain_result.get("trained_votes", 0),
                    )
                else:
                    logger.info(
                        "auto-retrain skipped: %s",
                        _retrain_result.get("message", "unknown"),
                    )
    except Exception as _e:  # noqa: BLE001
        logger.warning("auto-retrain check failed: %s", _e)

    profile = load_profile()

    # Use Rocchio-blended profile when available; fall back to static profile matrix.
    _vote_count_now = 0
    try:
        from .votes import signed_vote_count as _vc_now
        _vote_count_now = _vc_now()
    except Exception:  # noqa: BLE001
        pass
    if _build_profile_with_rocchio is not None:
        try:
            profile_vec = _build_profile_with_rocchio(profile, vote_count=_vote_count_now)
        except Exception as _e:  # noqa: BLE001
            logger.warning("Rocchio blend failed, using static profile: %s", _e)
            profile_vec = build_profile_matrix(profile)
    else:
        profile_vec = build_profile_matrix(profile)
    logger.info("ranker: profile_rows=%d votes=%d", len(profile_vec), _vote_count_now)

    if backfill_days and backfill_days > 0:
        days = backfill_days
    else:
        gap = days_since_last_sent(exclude_digest_id=digest_id)
        days = max(2, min(gap + 1, 7)) if gap >= 0 else 2
    recent = exclude_previously_shown(exclude_reviewed_items(recent_items(days=days)))
    recent = _quality_gate(recent)
    items = dedupe_ranking_candidates(recent)
    logger.info(
        "ranking %d recent items after cross-source dedupe (%d before, window=%d days)",
        len(items),
        len(recent),
        days,
    )
    scored, score_features = _score_items_for_pipeline(items, profile_vec, profile.downweight)

    # Apply negative-interest penalty when the profile has configured negative interests
    if _build_neg_centroid is not None:
        try:
            _neg_centroid = _build_neg_centroid(profile)
            if _neg_centroid is not None:
                from .rank.embedding_cache import embed_item_rows as _embed_rows
                _neg_vecs = _embed_rows([row for row, _ in scored])
                _neg_sims = _neg_vecs @ _neg_centroid
                scored = [
                    (row, score - 0.28 * max(0.0, float(_neg_sims[i])))
                    for i, (row, score) in enumerate(scored)
                ]
                scored.sort(key=lambda t: t[1], reverse=True)
        except Exception as _e:  # noqa: BLE001
            logger.warning("negative interest penalty failed: %s", _e)
    # Note: do NOT pre-truncate `scored` to a global top-K before per-section picking;
    # a single-domain profile (e.g. biotech-heavy) starves industry/regulatory/world.
    # `pick_top_per_section` already caps per-section, so summary cost is bounded.
    picked = pick_top_per_section(scored, _section_caps())
    top_journal_audit = _build_top_journal_audit(scored, picked, score_features)
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
        {"items": len(selected_rows), "backend": get_settings().llm_backend},
    )
    summaries = {
        item_id: summary
        for item_id, summary in summarize_items(selected_rows).items()
        if item_id in {int(row.id) for row in selected_rows}
    }
    write_summaries(summaries)
    _emit(progress_callback, "summarize_done", {"items": len(summaries)})

    sections: dict[str, list[tuple[ItemRow, float, str]]] = {k: [] for k in SECTION_ORDER}
    for row, score, _label in labeled:
        if row.section in sections:
            sections[row.section].append((row, score, summaries.get(row.id, "")))

    # Dry-runs should refresh the local preview even if today's email was
    # already sent. write_digest preserves sent_at for existing rows.
    if dry_run or _should_write_digest(digest_id):
        def _feature(row: ItemRow) -> dict[str, Any]:
            return score_features.get(_row_feature_key(row), {})

        write_digest(digest_id, [(row.item_label, row.id, score) for row, score, _ in labeled])
        write_digest_features(
            digest_id,
            [
                (
                    row.item_label or label,
                    int(row.id),
                    float(score),
                    breakdown_payload(
                        row,
                        float(_feature(row).get("topic_score", score)),
                        learned_score=float(_feature(row).get("learned_score", 0.0)),
                        reason_penalty=float(_feature(row).get("reason_penalty", 0.0)),
                        final_score=float(score),
                        selection_reason=_selection_reason(row, _feature(row)),
                        scoring_mode=str(_feature(row).get("scoring_mode", "cosine")),
                    ),
                )
                for row, score, label in labeled
            ],
        )
        write_digest_audit(digest_id, "missed_top_journals", top_journal_audit)
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
    sent = send_digest(html, subject, dry_run=dry_run)

    s = get_settings()
    if sent and not dry_run and s.resend_api_key and s.digest_to:
        mark_sent(digest_id)

    total_items = sum(len(v) for v in sections.values())
    _emit(
        progress_callback,
        "done",
        {"digest_id": digest_id, "total_items": total_items, "dry_run": dry_run},
    )
    return digest_id
