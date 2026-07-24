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
from .rank.ranker import (
    ScoreFeatureMap,
    _row_feature_key,
    pick_top_per_section,
    score_items,
    score_items_with_features,
)
from .rank.source_quality import (
    RANKER_VERSION,
    breakdown_payload,
    is_high_quality_journal_source,
    source_bucket,
    venue_relevance_credit,
    should_skip_item,
)
from .store import (
    DigestRow,
    ItemRow,
    days_since_last_digest,
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
    write_impressions,
    write_summaries,
)
from .summarize import summarize_items, synthesize_catch_up

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


def _audit_item(row: ItemRow, stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "reason": reason,
        "item_id": int(row.id) if isinstance(row.id, int) else None,
        "source": row.source or "",
        "section": row.section or "",
        "title": row.title or "",
        "url": row.url or "",
        "source_bucket": source_bucket(row),
    }


def _quality_gate(rows: list[ItemRow]) -> tuple[list[ItemRow], list[dict[str, Any]]]:
    """Pre-filter obviously low-quality candidates before scoring.

    Drops items that are clearly not useful: very short titles, editorial
    metadata, content-farm patterns, and research items with no abstract.
    Skipped when the pool is < 10 items to preserve test/backfill behaviour.
    """
    if len(rows) < 10:
        return rows, []
    out: list[ItemRow] = []
    drops: list[dict[str, Any]] = []
    for r in rows:
        title = (r.title or "").strip()
        abstract = (r.abstract or "").strip()
        high_quality = is_high_quality_journal_source(r)
        if len(title) < 15 and not high_quality:
            drops.append(_audit_item(r, "quality_gate", "short title"))
            continue
        if _TITLE_BLOCKLIST.match(title):
            drops.append(_audit_item(r, "quality_gate", "metadata or low-information title"))
            continue
        if should_skip_item(r):
            drops.append(_audit_item(r, "quality_gate", "low-information commentary or cover item"))
            continue
        if (r.section or "").lower() == "research" and len(abstract) < 30 and not high_quality:
            drops.append(_audit_item(r, "quality_gate", "thin abstract from non-protected source"))
            continue
        out.append(r)
    dropped = len(rows) - len(out)
    if dropped:
        logger.info("quality_gate: dropped %d low-quality candidates", dropped)
    return out, drops[:100]

SECTION_LABEL_PREFIX: dict[str, str] = {
    "research": "R",
    "industry": "I",
    "ai": "A",
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


def ingest_all(
    progress_callback: ProgressCallback | None = None,
    days: int = 2,
) -> int:
    """Fetch all sources, dedupe + langdetect filter, upsert. Returns rows inserted.

    ``days`` is the look-back window passed to date-capable adapters so a usage
    gap can be backfilled (bounded by each API and the ranking recency window).
    """
    init_db()
    specs = load_sources()
    _emit(progress_callback, "ingest_start", {"sources": len(specs)})
    all_items: list[Item] = []
    stats: list[IngestStats] = []
    for spec in specs:
        t0 = time.monotonic()
        try:
            src = dispatch_source(spec)
            fetched = src.fetch(spec, days=days)
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


def _catch_up_window(digest_id: str, backfill_days: int | None) -> int:
    """Look-back window (days) for ingest + ranking, widened after a usage gap.

    An explicit ``backfill_days`` wins. Otherwise the gap since the last digest
    of any kind (brewed or sent — so local/dry-run use counts too) sets the
    window to gap+1, floored at 2 and capped at ``max_backfill_days``. So after
    a week away the digest reaches back a week instead of only yesterday.
    """
    if backfill_days and backfill_days > 0:
        return backfill_days
    s = get_settings()
    cap = int(getattr(s, "max_backfill_days", 21))
    gap = days_since_last_digest(exclude_digest_id=digest_id)
    if gap < 0:
        return 2
    return max(2, min(gap + 1, cap))


def _research_ceiling_for_window(window_days: int) -> int:
    """Scale the research ceiling up on a catch-up run.

    After a gap, journals have accumulated a backlog of relevant papers, so the
    ceiling grows from ``top_research`` toward ``max_research_backlog`` roughly in
    proportion to the days being covered. Normal daily runs (window <= 2) are
    unchanged. News sections are NOT scaled — stale week-old news isn't wanted.
    """
    s = get_settings()
    base = int(s.top_research)
    ceiling = int(getattr(s, "max_research_backlog", base))
    if window_days <= 2 or ceiling <= base:
        return base
    grown = base + base * (window_days - 2) // 2  # +~half the daily rate per extra day
    return max(base, min(grown, ceiling))


def _section_caps(research_ceiling: int | None = None) -> dict[str, int]:
    s = get_settings()
    return {
        "research": s.top_research if research_ceiling is None else research_ceiling,
        "industry": s.top_industry,
        "ai": s.top_ai,
        "regulatory": s.top_regulatory,
        "world": s.top_world,
    }


def _dynamic_section_caps(
    scored: list[tuple[ItemRow, float]],
    features: ScoreFeatureMap,
    research_ceiling: int | None = None,
) -> dict[str, int]:
    """Size each section to the day's supply of genuinely on-topic items.

    The cap for a section is the count of its candidates whose *topic relevance*
    (true profile cosine) clears ``min_topic_relevance``, clamped to [min_*,
    top_*]. This gates on relevance directly: off-topic-but-prestigious items
    (a high-impact paper outside the user's field) score low on topic cosine and
    are excluded outright rather than filling the section. A day with many
    on-topic items shows more; a quiet day shows fewer — never below the floor
    (digest is never empty) nor above the ceiling (cost/reading load bounded).

    Topic cosine is a stable absolute scale (a true unit cosine), so the bar is a
    fixed threshold rather than a per-run relative one. Falls back to the fused
    rank score only when a per-item topic snapshot is missing, and to fixed
    ``top_*`` caps when adaptive sizing is disabled.
    """
    s = get_settings()
    maxima = _section_caps(research_ceiling=research_ceiling)
    if not getattr(s, "adaptive_section_sizes", False):
        return maxima

    minima = {
        "research": int(getattr(s, "min_research", 0)),
        "industry": int(getattr(s, "min_industry", 0)),
        "ai": int(getattr(s, "min_ai", 0)),
        "regulatory": int(getattr(s, "min_regulatory", 0)),
        "world": int(getattr(s, "min_world", 0)),
    }
    topic_floor = float(getattr(s, "min_topic_relevance", 0.65))
    news_floor = float(getattr(s, "min_news_quality", 0.45))

    # Count the items that clear each section's floor (research on topic cosine,
    # news on final confidence). Regulatory has no floor — it keeps its fixed cap.
    counts: dict[str, int] = {k: 0 for k in maxima}
    for row, fused in scored:
        section = row.section or ""
        feat = features.get(_row_feature_key(row), {})
        if section == "research":
            v = feat.get("topic_score", feat.get("confidence_score", fused))
            neg = float(feat.get("negative_interest_penalty", 0.0) or 0.0)
            if (
                v is not None
                and float(v) + venue_relevance_credit(row) - neg >= topic_floor
            ):
                counts[section] += 1
        elif section in ("industry", "world"):
            v = feat.get("confidence_score", fused)
            if v is not None and float(v) >= news_floor:
                counts[section] += 1

    caps: dict[str, int] = {}
    for section, mx in maxima.items():
        # Regulatory and AI are exempt from the profile-relevance floors — FDA
        # actions and AI-tooling content are wanted whether or not they match the
        # biotech profile cosine — so they keep their fixed caps.
        if section in ("regulatory", "ai"):
            caps[section] = mx
        else:
            floor = min(minima.get(section, 0), mx)
            caps[section] = max(floor, min(mx, counts.get(section, 0)))
    return caps


def _filter_off_topic(
    scored: list[tuple[ItemRow, float]],
    features: ScoreFeatureMap,
) -> list[tuple[ItemRow, float]]:
    """Hard-drop items that fall below their section's relevance/quality floor.

    Research is gated on topic relevance (``min_topic_relevance``, a true profile
    cosine — its abstracts cluster high). Industry/world are gated on final
    confidence (``min_news_quality``) instead, since news cosine is inherently
    lower; this is where opinion columns, paywalled "STAT+:" teasers and weak
    filler get removed so a section shrinks to what's worth reading rather than
    padding to the cap. Regulatory (FDA/EMA) is exempt — those are wanted
    regardless of how they score.

    The size cap alone only bounds the *count*; selection still fills slots by
    score, so without this gate an off-topic-but-high-prestige item could grab a
    slot it never earned. No-op when adaptive sizing is off or the relevant
    snapshot is missing (so a scoring fallback never silently empties a section).
    """
    s = get_settings()
    if not getattr(s, "adaptive_section_sizes", False):
        return scored
    topic_floor = float(getattr(s, "min_topic_relevance", 0.65))
    news_floor = float(getattr(s, "min_news_quality", 0.45))
    out: list[tuple[ItemRow, float]] = []
    for row, score in scored:
        section = row.section or ""
        feat = features.get(_row_feature_key(row), {})
        if section == "research":
            t = feat.get("topic_score")
            neg = float(feat.get("negative_interest_penalty", 0.0) or 0.0)
            # Effective relevance = topic cosine + venue credit − negative-interest
            # penalty. The venue credit rescues borderline top-venue work; the
            # negative penalty excludes off-field items whose cosine clears the floor.
            if (
                t is not None
                and float(t) + venue_relevance_credit(row) - neg < topic_floor
            ):
                continue
        elif section in ("industry", "world"):
            c = feat.get("confidence_score")
            if c is not None and float(c) < news_floor:
                continue
        out.append((row, score))
    return out


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
        if bucket in {"arxiv_other", "bio_med_preprint", "preprint_other"}:
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
                "rank_score": float(score),
                "confidence_score": float(score),
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
                "rank_score": float(score),
                "confidence_score": float(score),
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

    # Widen the look-back once, after any usage gap, and use it for BOTH the
    # ingest fetch window (so the backlog is actually retrieved) and the ranking
    # recency window below.
    window_days = _catch_up_window(digest_id, backfill_days)
    if window_days > 2:
        logger.info("catch-up: widening window to %d days after usage gap", window_days)
    inserted = ingest_all(progress_callback=progress_callback, days=window_days)
    logger.info("upserted %d new items", inserted)

    # Auto-retrain LR when there are new/changed votes since the model was trained
    # (feedback must apply promptly, not after a 7-day timer), with a 7-day fallback
    # so recency-decayed features refresh even on a quiet week.
    try:
        from .votes import MIN_VOTES_FOR_LR as _min_lr
        from .votes import latest_vote_timestamp as _latest_vote_ts
        from .votes import signed_vote_count as _svc

        _current_votes = _svc()
        if _current_votes >= _min_lr:
            from pathlib import Path as _Path

            _lr_path = _Path(get_settings().db_path).parent / "lr_ranker.npz"
            _needs_retrain = not _lr_path.exists()
            if not _needs_retrain and _lr_path.exists():
                _lr_mtime = _lr_path.stat().st_mtime
                _vote_ts = _latest_vote_ts()
                _new_feedback = _vote_ts is not None and _vote_ts > _lr_mtime
                _lr_age_days = (time.time() - _lr_mtime) / 86400
                _needs_retrain = _new_feedback or _lr_age_days > 7
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

    # Refit the score→probability calibrator when stale (> 7 days) so the
    # adaptive relevance floor tracks recent feedback.
    try:
        from pathlib import Path as _Path

        from .rank.calibrate import fit_calibrator as _fit_calib

        _calib_path = _Path(get_settings().db_path).parent / "calibrator.json"
        _calib_stale = (
            not _calib_path.exists()
            or (time.time() - _calib_path.stat().st_mtime) / 86400 > 7
        )
        if _calib_stale and _fit_calib() is not None:
            logger.info("auto-fit score calibrator")
    except Exception as _e:  # noqa: BLE001
        logger.warning("calibrator auto-fit check failed: %s", _e)

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

    days = window_days
    recent_raw = recent_items(days=days)
    after_reviewed = exclude_reviewed_items(recent_raw)
    after_shown = exclude_previously_shown(after_reviewed, exclude_digest_id=digest_id)
    deduped_candidates = dedupe_ranking_candidates(after_shown)   # within-set dedupe FIRST
    # Cross-day content dedupe: drop items re-surfaced from a recent digest.
    try:
        from .rank.near_dup import exclude_recent_near_duplicates

        items, near_dup_drops = exclude_recent_near_duplicates(deduped_candidates)
    except Exception as _e:  # noqa: BLE001
        logger.warning("cross-day near-dup suppression failed: %s", _e)
        items, near_dup_drops = deduped_candidates, []
    quality_rows, quality_drops = _quality_gate(items)      # then gate
    items = quality_rows
    funnel_audit = {
        "ranker_version": RANKER_VERSION,
        "window_days": days,
        "recent_items": len(recent_raw),
        "after_reviewed_filter": len(after_reviewed),
        "after_previously_shown_filter": len(after_shown),
        "after_cross_source_dedupe": len(deduped_candidates),
        "after_cross_day_near_dup": len(deduped_candidates) - len(near_dup_drops),
        "cross_day_near_dup_drops": near_dup_drops[:100],
        "after_quality_gate": len(items),
        "quality_gate_drops": quality_drops,
    }
    logger.info(
        "ranking %d recent items after quality gate (%d before, window=%d days)",
        len(items),
        len(quality_rows) + len(quality_drops),
        days,
    )
    scored, score_features = _score_items_for_pipeline(items, profile_vec, profile.downweight)

    # Apply negative-interest penalty when the profile has configured negative interests
    if _build_neg_centroid is not None:
        try:
            import numpy as np
            from .rank.embedding_cache import embed_item_rows as _embed_rows
            _neg_vecs = _embed_rows([row for row, _ in scored])

            _neg_scale = 1.0
            _neg_settings = get_settings()
            # DISCRIMINATIVE penalty: penalize by how much an item is CLOSER to a
            # negative topic than to the reader's own profile (neg − topic). An
            # absolute similarity threshold cannot work here — bge-small scores
            # both the reader's field and off-field biomedical content ~0.6 against
            # the negative phrases, so any cut taxes everything or nothing. The
            # relative signal separates them cleanly (clinical/epi/GWAS are
            # neg-dominant; materials/design are profile-dominant).
            _neg_margin = float(getattr(_neg_settings, "negative_interest_margin", 0.0))
            _neg_w = float(getattr(_neg_settings, "negative_interest_weight", 0.80))
            # Per-item topic relevance (true profile cosine) captured at scoring.
            _neg_topic = np.array(
                [
                    float(score_features.get(_row_feature_key(row), {}).get("topic_score", 0.0) or 0.0)
                    for row, _ in scored
                ],
                dtype=np.float32,
            )

            # Try per-axis penalty (max similarity to any individual negative interest)
            _neg_sims = None
            try:
                from .rank.profile import build_negative_vectors
                _neg_axes = build_negative_vectors(profile)  # list of individual vectors
            except ImportError:
                _neg_axes = None

            if _neg_axes is not None and len(_neg_axes) > 0:
                _neg_mat = np.array(_neg_axes, dtype=np.float32)  # (n_neg, embed_dim)
                # Normalize neg vectors
                _neg_norms = np.linalg.norm(_neg_mat, axis=1, keepdims=True)
                _neg_mat_n = _neg_mat / (_neg_norms + 1e-9)
                _neg_vecs_n = _neg_vecs / (np.linalg.norm(_neg_vecs, axis=1, keepdims=True) + 1e-9)
                all_neg_sims = _neg_vecs_n @ _neg_mat_n.T  # (n_items, n_neg)
                try:
                    from .rank import profile as profile_mod
                    _neg_weights = np.array(profile_mod.get_negative_interest_weights(profile), dtype=np.float32)
                    if len(_neg_weights) == all_neg_sims.shape[1]:
                        all_neg_sims = all_neg_sims * _neg_weights  # broadcast: scale each axis
                except Exception:  # noqa: BLE001
                    pass
                # Penalize by (nearest-negative similarity − topic relevance),
                # so only items closer to a negative topic than to the profile pay.
                _neg_sims = np.maximum(0.0, all_neg_sims.max(axis=1) - _neg_topic - _neg_margin)
            else:
                _neg_centroid = _build_neg_centroid(profile)
                if _neg_centroid is not None:
                    _neg_vecs_n = _neg_vecs / (np.linalg.norm(_neg_vecs, axis=1, keepdims=True) + 1e-9)
                    _neg_sims = np.maximum(
                        0.0, (_neg_vecs_n @ _neg_centroid) - _neg_topic - _neg_margin
                    )

            if _neg_sims is not None:
                # Capture penalties BEFORE creating the new scored list
                _penalty_list = []
                for i, (row, score) in enumerate(scored):
                    if i < len(_neg_sims):
                        raw_sim = float(_neg_sims[i])
                        penalty = (_neg_w * _neg_scale) * max(0.0, raw_sim)
                    else:
                        penalty = 0.0
                    _penalty_list.append(penalty)

                # Apply penalties to scores
                scored = [(row, score - _penalty_list[i]) for i, (row, score) in enumerate(scored)]

                # Update score_features using captured penalties
                for i, (row, score) in enumerate(scored):
                    key = _row_feature_key(row)
                    if key in score_features:
                        penalty = _penalty_list[i]
                        score_features[key]["negative_interest_penalty"] = round(penalty, 4)
                        score_features[key]["confidence_score"] = round(
                            score_features[key]["confidence_score"] - penalty, 4
                        )
                        score_features[key]["final_score"] = round(score, 4)
                scored.sort(key=lambda t: t[1], reverse=True)
        except Exception as _e:  # noqa: BLE001
            logger.warning("negative interest penalty failed: %s", _e)

    # Author/lab match boost: items whose byline matches the profile watchlist
    # are pushed up. Applied in both the cosine and hybrid paths so it works
    # before any votes exist; the learner additionally sees author_match.
    try:
        from .rank.authors import author_match_score, load_watchlist

        _watchlist = load_watchlist(profile)
        if _watchlist:
            _author_boost = 0.12
            _reboosted: list[tuple[ItemRow, float]] = []
            for row, score in scored:
                m = author_match_score(getattr(row, "authors", "") or "", _watchlist)
                if m > 0:
                    score = float(score) + _author_boost * m
                    key = _row_feature_key(row)
                    if key in score_features:
                        score_features[key]["author_match"] = round(float(m), 4)
                        score_features[key]["author_boost"] = round(_author_boost * m, 4)
                        score_features[key]["final_score"] = round(float(score), 4)
                _reboosted.append((row, score))
            scored = _reboosted
            scored.sort(key=lambda t: t[1], reverse=True)
    except Exception as _e:  # noqa: BLE001
        logger.warning("author match boost failed: %s", _e)

    # Optional live citation-velocity boost via OpenAlex (no-op unless enabled).
    try:
        from .rank.enrich import enrich_scored

        scored = enrich_scored(scored)
    except Exception as _e:  # noqa: BLE001
        logger.warning("citation enrichment failed: %s", _e)

    # Note: do NOT pre-truncate `scored` to a global top-K before per-section picking;
    # a single-domain profile (e.g. biotech-heavy) starves industry/regulatory/world.
    # `pick_top_per_section` already caps per-section, so summary cost is bounded.
    # Hard-gate off-topic research/industry first so prestige can't fill a slot an
    # item's topic relevance never earned, then size + pick from what remains.
    pickable = _filter_off_topic(scored, score_features)
    research_ceiling = _research_ceiling_for_window(window_days)
    picked = pick_top_per_section(
        pickable,
        _dynamic_section_caps(pickable, score_features, research_ceiling),
        catch_up=window_days > 2,
    )

    # Active-learning exploration: swap a few low-scored picks for high-quality
    # items the learned ranker is most uncertain about. Only when enabled and the
    # LR is actually driving scores (uncertainty is meaningful).
    _explore = int(getattr(get_settings(), "exploration_slots", 0) or 0)
    if _explore > 0:
        try:
            from .rank.ranker import apply_exploration
            from .rank.source_quality import is_high_quality_journal_source

            uncertainty = {
                key: 1.0 - 2.0 * abs(float(feat.get("learned_score", 0.0)) - 0.5)
                for key, feat in score_features.items()
                if str(feat.get("scoring_mode", "")) == "hybrid_lr"
            }
            if uncertainty:
                picked = apply_exploration(
                    picked,
                    scored,
                    uncertainty,
                    slots=_explore,
                    eligible=is_high_quality_journal_source,
                )
        except Exception as _e:  # noqa: BLE001
            logger.warning("exploration slot failed: %s", _e)

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
    # Summary cache: only (re)summarize items that don't already have one, so a
    # re-brew or a catch-up run doesn't re-spend the LLM budget on items already
    # summarized. Existing summaries are reused for rendering.
    selected_ids = {int(row.id) for row in selected_rows}
    existing_summaries = {
        int(row.id): (row.summary or "").strip() for row in selected_rows
    }
    to_summarize = [row for row in selected_rows if not existing_summaries.get(int(row.id))]
    new_summaries = {
        item_id: summary
        for item_id, summary in summarize_items(to_summarize, profile=profile).items()
        if item_id in selected_ids
    }
    summaries = {
        rid: (new_summaries.get(rid) or existing_summaries.get(rid) or "")
        for rid in selected_ids
    }
    write_summaries({rid: s for rid, s in new_summaries.items() if s})
    _emit(progress_callback, "summarize_done", {"items": len(new_summaries)})

    sections: dict[str, list[tuple[ItemRow, float, str]]] = {k: [] for k in SECTION_ORDER}
    for row, score, _label in labeled:
        if row.section in sections:
            sections[row.section].append((row, score, summaries.get(int(row.id), "")))

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
                        rank_score=float(score),
                        confidence_score=float(
                            _feature(row).get("confidence_score", score)
                        ),
                        negative_interest_penalty=float(
                            _feature(row).get("negative_interest_penalty", 0.0)
                        ),
                        selection_reason=_selection_reason(row, _feature(row)),
                        scoring_mode=str(_feature(row).get("scoring_mode", "cosine")),
                    ),
                )
                for row, score, label in labeled
            ],
        )
        # Immutable per-run impression log: APPEND this run's slate (never
        # overwrites prior runs, unlike write_digest). This is the A/B substrate,
        # so it records more than the final slate:
        #  * For the RESEARCH section we log the scored candidate POOL (top
        #    RESEARCH_CANDIDATE_POOL_CAP by final score) with selected=True when the
        #    item made the final `labeled` slate and False otherwise — so an
        #    alternative ranker can be scored against the identical candidate set.
        #  * For the OTHER sections we log only the selected slate (selected=True),
        #    matching the prior behavior.
        # Position is the 0-based rank; within research it is the rank in the
        # score-ordered candidate pool, elsewhere the rank within the section slate.
        RESEARCH_CANDIDATE_POOL_CAP = 100
        _selected_ids = {int(row.id) for row, _s, _l in labeled}
        _impressions: list[tuple[str, int, int, float | None, bool]] = []
        # `scored` is (row, score) sorted by final score desc; filter to research
        # and cap the pool so the row count stays bounded.
        _research_scored = [
            (row, score)
            for row, score in scored
            if (getattr(row, "section", "") or "") == "research"
        ]
        # Log the top-CAP research candidates by final score PLUS every selected
        # research item, even if it ranks below the cap (source balancing /
        # exploration / last-resort fill can select items past position CAP). This
        # preserves the invariant that every displayed research item has an
        # impression row with selected=True. Appended tail items keep score order.
        _research_pool = _research_scored[:RESEARCH_CANDIDATE_POOL_CAP]
        _pool_ids = {int(row.id) for row, _s in _research_pool}
        for row, score in _research_scored[RESEARCH_CANDIDATE_POOL_CAP:]:
            if int(row.id) in _selected_ids and int(row.id) not in _pool_ids:
                _research_pool.append((row, score))
                _pool_ids.add(int(row.id))
        for pos, (row, score) in enumerate(_research_pool):
            _impressions.append(
                (
                    "research",
                    int(row.id),
                    pos,
                    float(score),
                    int(row.id) in _selected_ids,
                )
            )
        # Selected non-research items, positioned within their section slate.
        _section_pos: dict[str, int] = {}
        for row, score, _label in labeled:
            section = row.section or ""
            if section == "research":
                continue  # already covered by the research candidate pool above
            pos = _section_pos.get(section, 0)
            _section_pos[section] = pos + 1
            _impressions.append((section, int(row.id), pos, float(score), True))
        write_impressions(digest_id, _impressions, model_version=RANKER_VERSION)
        write_digest_audit(digest_id, "missed_top_journals", top_journal_audit)
        write_digest_audit(digest_id, "candidate_funnel", [funnel_audit])
        # Catch-up briefing: only on gap runs, grouping the backlog into themes.
        if window_days > 2:
            try:
                _synth = synthesize_catch_up(selected_rows, window_days, profile=profile)
                if _synth:
                    write_digest_audit(
                        digest_id, "catch_up_synthesis",
                        [{"days": window_days, "text": _synth}],
                    )
            except Exception as _e:  # noqa: BLE001
                logger.warning("catch-up synthesis write failed: %s", _e)
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
