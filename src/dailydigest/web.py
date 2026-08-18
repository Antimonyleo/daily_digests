"""Local FastAPI UI for browsing today's digest, voting, and onboarding setup.

This is an *alternative* delivery surface; the email path is unchanged.
The app binds to 127.0.0.1 by default — no remote access, no auth.

Routes:
- ``GET /``                — render today's digest (or redirect to /setup if no profile)
- ``POST /vote/{id}/{v}``  — record a +1/0/-1 vote for an item
- ``POST|DELETE /bookmark/{id}`` — save or remove an item from the reading archive
- ``GET /saved``           — search the local saved-reading archive
- ``POST|DELETE /known/{id}`` — manually flag an item as known so future digests skip it
- ``POST /overflow/save``  — pin today's reading-mode overflow for tomorrow's brew
- ``POST /vote/{id}/reason/{reason}`` — record qualitative feedback reason
- ``DELETE /vote/{id}/reason/{reason}`` — remove qualitative feedback reason
- ``GET /ranking/status``  — return vote counts and LR ranker status
- ``POST /ranking/train``  — refresh vote-derived ranking calibration on demand
- ``POST /refresh``        — kick off a dry-run pipeline in the background
- ``GET /healthz``         — liveness probe
- ``GET /setup``           — onboarding wizard (weighted interests, LLM backend)
- ``POST /setup``          — write local profile YAML and .env, redirect to /run
- ``GET /run``             — brewing page with live SSE progress feed
- ``POST /run/start``      — kick off pipeline.run_all in a background thread
- ``GET /run/stream``      — Server-Sent Events stream of pipeline progress
- ``GET /done``            — celebration page after a successful brew
- ``GET /manifest.webmanifest|/sw.js`` — local installable-app shell
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import queue as std_queue
import re
import secrets
import tempfile
import threading
import uuid
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError
from sqlalchemy import func, select

from . import health
from . import votes as votes_mod
from .config import SETTINGS, get_settings, reload_settings, section_enabled
from .email_render import (
    SECTION_META,
    SECTION_ORDER,
    _is_high_profile,
    content_type_label,
    reason_line,
    safe_url,
)
from .job_lock import ComputeBusyError, acquire_compute_lock
from .opportunities import (
    OpportunityProfile,
    load_opportunity_profile,
    opportunity_display,
)
from .pipeline import _digest_id, normalize_reading_mode, run_all
from .rank.embed import release_encoder
from .rank.source_quality import display_breakdown, source_bucket
from .store import (
    DigestItemRow,
    DigestRow,
    ItemRow,
    VoteRow,
    add_carryover_items,
    bookmarked_item_ids,
    carryover_item_ids,
    init_db,
    item_metadata,
    known_item_ids,
    load_digest_audit,
    load_digest_features,
    mark_impressions_viewed,
    search_bookmarks,
    session_scope,
    set_bookmark,
    set_item_known,
)
from .tea_break import daily_tea_deck

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(os.environ.get("DD_APP_ROOT") or Path.cwd()).resolve()
_PACKAGED_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE_DIR = (
    _PACKAGED_TEMPLATE_DIR
    if _PACKAGED_TEMPLATE_DIR.is_dir()
    else _REPO_ROOT / "templates"
)
_ENV_PATH = _REPO_ROOT / ".env"


def _get_profile_path() -> Path:
    from .config import get_settings
    candidate = (_REPO_ROOT / get_settings().profile_path).resolve()
    try:
        candidate.relative_to(_REPO_ROOT)
    except ValueError:
        raise ValueError(f"PROFILE_PATH escapes repo root: {candidate}") from None
    return candidate


def _get_opportunity_profile_path() -> Path:
    from .config import get_settings

    candidate = (_REPO_ROOT / get_settings().opportunity_profile_path).resolve()
    try:
        candidate.relative_to(_REPO_ROOT)
    except ValueError:
        raise ValueError(
            f"OPPORTUNITY_PROFILE_PATH escapes repo root: {candidate}"
        ) from None
    return candidate

app = FastAPI(title="DailyDigest")
templates = Jinja2Templates(
    env=Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)
)

# Per-section sort order so we can ORDER BY (section_order, label_number).
_SECTION_RANK = {name: idx for idx, name in enumerate(SECTION_ORDER)}

# Per-run progress queues.  Uses stdlib queue.Queue (thread-safe) so the
# producer thread can put_nowait without needing the event loop reference.
# The SSE consumer drains via asyncio.to_thread so it never blocks the loop.
_RUN_QUEUES: dict[str, std_queue.Queue[dict[str, Any]]] = {}
_RUN_STARTED: set[str] = set()
_MAX_RETAINED_RUNS = 32
_RUN_LOCK = threading.Lock()
_TRAIN_LOCK = threading.Lock()
_TRAIN_JOB: dict[str, Any] = {"running": False, "last_result": None}
_BREW_LOCK = threading.Lock()
_BREW_JOB: dict[str, Any] = {"running": False, "run_id": None}
# Brewing and manual ranker training both run local embeddings. Keeping one
# compute job at a time prevents them from multiplying CPU and peak RAM.
_COMPUTE_LOCK = threading.Lock()
_CSRF_TOKEN = secrets.token_urlsafe(32)
_ENV_SAFE_RE = re.compile(r"^[A-Za-z0-9_./:@+-]*$")
_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# Existing digest viewer helpers (unchanged behavior).
# ---------------------------------------------------------------------------


def _label_number(label: str | None) -> int:
    if not label:
        return 0
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 0


def _format_date(row: ItemRow) -> str:
    if row.published_at is None:
        return ""
    return row.published_at.strftime("%Y-%m-%d")


def _bar_pct(value: float) -> int:
    if value <= 0:
        return 0
    return max(3, min(100, int(round(value * 100))))


def _breakdown_payload(row: ItemRow, persisted: dict | None = None) -> dict:
    if persisted:
        score = persisted.get("score", row.score or 0.0)
        rank_score = persisted.get("rank_score", score)
        confidence_score = persisted.get("confidence_score", score)
        return {
            "score": f"{float(score):.2f}" if isinstance(score, (int, float)) else str(score),
            "rank_score": float(rank_score) if isinstance(rank_score, (int, float)) else 0.0,
            "confidence_score": (
                float(confidence_score) if isinstance(confidence_score, (int, float)) else 0.0
            ),
            "tags": list(persisted.get("tags") or []),
            "why_shown": list(persisted.get("why_shown") or []),
            "content_type": str(persisted.get("content_type") or "article"),
            "primary_facet": str(persisted.get("primary_facet") or ""),
            "source_bucket": str(persisted.get("source_bucket") or source_bucket(row)),
            "selection_reason": str(persisted.get("selection_reason") or ""),
            "ranker_version": str(persisted.get("ranker_version") or ""),
            "components": {
                "topic": _bar_pct(float(persisted.get("topic") or 0.0)),
                "source": _bar_pct(float(persisted.get("source") or 0.0)),
                "novelty": _bar_pct(float(persisted.get("novelty") or 0.0)),
                "penalty": _bar_pct(float(persisted.get("penalty") or 0.0)),
            },
        }
    breakdown = display_breakdown(row)
    return {
        "score": f"{breakdown.final:.2f}",
        "rank_score": float(breakdown.final),
        "confidence_score": float(breakdown.final),
        "tags": list(breakdown.tags),
        "why_shown": list(breakdown.why_shown),
        "content_type": breakdown.content_type,
        "primary_facet": "",
        "source_bucket": source_bucket(row),
        "selection_reason": "",
        "ranker_version": "",
        "components": {
            "topic": _bar_pct(breakdown.topic),
            "source": _bar_pct(breakdown.source),
            "novelty": _bar_pct(breakdown.novelty),
            "penalty": _bar_pct(breakdown.penalty),
        },
    }


def _entry_confidence(entry: dict) -> float:
    ranking = entry.get("ranking") or {}
    value = ranking.get("confidence_score", entry.get("score_raw") or 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _digest_overview(sections: list[dict]) -> dict:
    entries = [entry for section in sections for entry in section["entries"]]
    source_counts: dict[str, int] = {}
    for entry in entries:
        source = entry.get("source") or "Unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
    source_mix = sorted(source_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))[:6]
    topic_counts = Counter(
        str(entry.get("ranking", {}).get("primary_facet") or "").strip()
        for entry in entries
        if str(entry.get("ranking", {}).get("primary_facet") or "").strip()
    )
    digest_words = sum(
        len(re.findall(r"\b\w+\b", f"{entry.get('title', '')} {entry.get('summary', '')}"))
        for entry in entries
    )
    reading_minutes = (
        max(1, (digest_words + 25 * len(entries) + 219) // 220) if entries else 0
    )
    latest = health.latest_snapshot()
    scanned = sum(int(row.get("items") or 0) for row in latest)
    return {
        "selected": len(entries),
        "scanned": scanned,
        "sources": len(latest),
        "failures": sum(1 for row in latest if not row.get("ok", True)),
        "source_mix": source_mix,
        "topic_mix": [
            {"title": topic, "count": count}
            for topic, count in topic_counts.most_common(4)
        ],
        "reading_minutes": reading_minutes,
        "top_journals_shown": sum(
            1
            for entry in entries
            if entry.get("ranking", {}).get("source_bucket") == "published_journal"
        ),
    }


def _prepare_candidate_funnel(raw: list[dict]) -> dict:
    if not raw:
        return {}
    funnel = dict(raw[0] or {})
    drops = list(funnel.get("quality_gate_drops") or [])
    reason_counts = Counter(str(drop.get("reason") or "other") for drop in drops)
    funnel["quality_drop_reason_counts"] = [
        {"reason": reason, "count": count}
        for reason, count in reason_counts.most_common(5)
    ]
    funnel["quality_gate_drops"] = drops[:6]
    return funnel


# Match the field labels anywhere (not only at line start) so a summary whose
# fields are space-separated on one line still splits into three rows.
_SUMMARY_FIELD_RE = re.compile(r"(Key finding|Why read|Caveat)\s*:\s*")


def _summary_fields(summary: str) -> dict[str, str]:
    """Parse stored three-field summaries for denser UI rendering."""
    if not summary:
        return {}
    matches = list(_SUMMARY_FIELD_RE.finditer(summary))
    if not matches:
        return {}
    out: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(summary)
        value = summary[start:end].strip()
        if value:
            out[label] = value
    return out


def _load_today(digest_id: str) -> tuple[list[dict], dict[int, int]]:
    """Return (rendered_sections, current_vote_per_item)."""
    init_db()
    with session_scope() as s:
        rows = (
            s.execute(select(ItemRow).where(ItemRow.digest_id == digest_id))
            .scalars()
            .all()
        )
        if not rows:
            return [], {}

        rows = [r for r in rows if section_enabled(SETTINGS, r.section or "")]
        if not rows:
            return [], {}

        item_ids = [r.id for r in rows]
        saved_item_ids = bookmarked_item_ids(int(item_id) for item_id in item_ids)
        known_ids = known_item_ids(int(item_id) for item_id in item_ids)
        vote_rows = s.execute(
            select(VoteRow.item_id, VoteRow.value, VoteRow.grade)
            .where(VoteRow.item_id.in_(item_ids))
            .order_by(VoteRow.item_id, VoteRow.created_at.desc(), VoteRow.id.desc())
        ).all()
        current_vote: dict[int, int] = {}
        current_grade: dict[int, int] = {}
        for item_id, value, grade in vote_rows:
            iid = int(item_id)
            if iid not in current_vote:
                current_vote[iid] = int(value)
                current_grade[iid] = int(grade) if grade is not None else votes_mod.value_to_grade(int(value))
        # Load all vote reasons in one read instead of N+1 reads
        _all_reasons = votes_mod._load_vote_reasons()
        current_reasons = {
            int(item_id): _all_reasons.get(str(int(item_id)), [])
            for item_id in item_ids
        }
        persisted_features = load_digest_features(digest_id)
        opportunity_profile = None
        if any((row.section or "") in {"opportunities", "events"} for row in rows):
            try:
                opportunity_profile = load_opportunity_profile(
                    get_settings().opportunity_profile_path
                )
            except Exception:  # noqa: BLE001
                pass

        rows.sort(
            key=lambda r: (
                _SECTION_RANK.get(r.section or "", 99),
                _label_number(r.item_label),
            )
        )

        rendered_sections: list[dict] = []
        seen_keys: list[str] = []
        by_section: dict[str, list[dict]] = {}
        for row in rows:
            key = row.section or "other"
            ranking = _breakdown_payload(row, persisted_features.get(int(row.id)))
            confidence_score = _entry_confidence(
                {"ranking": ranking, "score_raw": float(row.score or 0.0)}
            )
            reason = reason_line(
                ranking.get("primary_facet"),
                high_profile=_is_high_profile(
                    ranking.get("source_bucket"), ranking.get("tags") or []
                ),
                journal=row.source or "",
                why_shown=ranking.get("why_shown") or [],
                tags=ranking.get("tags") or [],
            )
            type_label = content_type_label(ranking.get("content_type"))
            if key not in by_section:
                by_section[key] = []
                seen_keys.append(key)
            opportunity = (
                opportunity_display(item_metadata(row), opportunity_profile)
                if key in {"opportunities", "events"}
                else None
            )
            if opportunity is not None:
                opportunity["calendar_url"] = f"/calendar/{int(row.id)}.ics"
            by_section[key].append(
                {
                    "id": row.id,
                    "label": row.item_label or "",
                    "title": row.title or "",
                    "url": safe_url(row.url),
                    "source": row.source or "",
                    "published": _format_date(row),
                    "summary": row.summary or "",
                    "summary_fields": _summary_fields(row.summary or ""),
                    "score_raw": float(row.score or 0.0),
                    "confidence_score": confidence_score,
                    "ranking": ranking,
                    "reason_line": reason,
                    "type_label": type_label,
                    "opportunity": opportunity,
                    "current_vote": current_vote.get(int(row.id)),
                    "current_grade": current_grade.get(int(row.id)),
                    "current_reasons": current_reasons.get(int(row.id), []),
                    "bookmarked": int(row.id) in saved_item_ids,
                    "known": int(row.id) in known_ids,
                }
            )

        for r in rows:
            s.expunge(r)

    for key in SECTION_ORDER:
        if key in by_section:
            meta = SECTION_META.get(key, {"title": key.title(), "emoji": ""})
            rendered_sections.append(
                {
                    "key": key,
                    "title": meta["title"],
                    "emoji": meta["emoji"],
                    "entries": by_section[key],
                }
            )
    for key in seen_keys:
        if key not in SECTION_ORDER:
            rendered_sections.append(
                {"key": key, "title": key.title(), "emoji": "", "entries": by_section[key]}
            )

    return rendered_sections, current_vote


def _digest_exists(digest_id: str) -> bool:
    init_db()
    with session_scope() as s:
        return s.get(DigestRow, digest_id) is not None


def _saved_date(value: date | None) -> str:
    if value is None:
        return ""
    return value.strftime("%b %d, %Y").replace(" 0", " ")


def _summarizer_label(digest_id: str | None = None) -> str:
    backend = (SETTINGS.llm_backend or "extractive").lower()
    if backend == "extractive":
        return "Extractive (local, no AI)"
    names = {
        "api": "OpenAI-compatible API",
        "anthropic": "Anthropic API",
        "claude_cli": "Claude Code",
        "codex_cli": "Codex",
    }
    name = names.get(backend, "Extractive (local, no AI)")
    model = (SETTINGS.llm_model or "").strip()
    label = f"{name} · {model}" if model else name
    if digest_id and backend != "extractive":
        init_db()
        with session_scope() as s:
            fallback_count = int(
                s.execute(
                    select(func.count(ItemRow.id))
                    .join(DigestItemRow, DigestItemRow.item_id == ItemRow.id)
                    .where(
                        DigestItemRow.digest_id == digest_id,
                        ItemRow.summary_backend == "extractive_fallback",
                    )
                ).scalar_one()
            )
        if fallback_count:
            label += f" · {fallback_count} extractive fallback"
    return label


def _host_name(host_header: str | None) -> str:
    host = (host_header or "").strip()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end].lower() if end != -1 else host.lower()
    return host.rsplit(":", 1)[0].lower()


def _is_local_host(host: str) -> bool:
    return host == "localhost" or host == "testserver" or host == "::1" or host.startswith("127.")


def _require_local_origin(request: Request) -> None:
    host = _host_name(request.headers.get("host"))
    if not _is_local_host(host):
        raise HTTPException(status_code=403, detail="local UI only accepts loopback hosts")

    origin = request.headers.get("origin")
    if not origin:
        return
    origin_host = _host_name(urlsplit(origin).netloc)
    if origin_host != host or not _is_local_host(origin_host):
        raise HTTPException(status_code=403, detail="cross-origin writes are not allowed")


@app.middleware("http")
async def _loopback_host_only(request: Request, call_next):
    """Reject non-loopback Host headers for every UI and asset route."""
    if not _is_local_host(_host_name(request.headers.get("host"))):
        return JSONResponse(
            {"detail": "local UI only accepts loopback hosts"}, status_code=403
        )
    return await call_next(request)


def _ics_text(value: object) -> str:
    # A bare CR or LF inside a property value terminates the line and corrupts
    # the file, so both newline forms must become the literal ``\n`` escape.
    return (
        str(value or "")
        .replace("\\", "\\\\")
        .replace("\r\n", "\\n")
        .replace("\r", "\\n")
        .replace("\n", "\\n")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )


@app.get("/calendar/{item_id}.ics")
def calendar_item(request: Request, item_id: int) -> Response:
    """Export an opportunity deadline or event as a portable calendar file."""
    _require_local_origin(request)
    init_db()
    with session_scope() as s:
        row = s.get(ItemRow, item_id)
        if row is None or (row.section or "") not in {"opportunities", "events"}:
            raise HTTPException(status_code=404, detail="opportunity or event not found")
        metadata = item_metadata(row)
        raw_start = metadata.get("event_start") or metadata.get("deadline")
        raw_end = metadata.get("event_end") or raw_start
        try:
            start = date.fromisoformat(str(raw_start)[:10])
            end = date.fromisoformat(str(raw_end)[:10])
        except ValueError:
            raise HTTPException(status_code=422, detail="this item has no usable date") from None
        title = row.title or "DailyDigest opportunity"
        url = safe_url(row.url)
        description = row.abstract or ""
    # DATE-valued DTEND is exclusive under RFC 5545.
    exclusive_end = max(start, end) + timedelta(days=1)
    body = "\r\n".join(
        [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//DailyDigest//Opportunity Calendar//EN",
            "CALSCALE:GREGORIAN",
            "BEGIN:VEVENT",
            f"UID:dailydigest-{item_id}@localhost",
            f"DTSTART;VALUE=DATE:{start:%Y%m%d}",
            f"DTEND;VALUE=DATE:{exclusive_end:%Y%m%d}",
            f"SUMMARY:{_ics_text(title)}",
            f"DESCRIPTION:{_ics_text(description[:1000])}",
            f"URL:{_ics_text(url)}",
            "END:VEVENT",
            "END:VCALENDAR",
            "",
        ]
    )
    return Response(
        body,
        media_type="text/calendar",
        headers={"Content-Disposition": f'attachment; filename="dailydigest-{item_id}.ics"'},
    )


def _require_csrf(request: Request, form: dict[str, str] | None = None) -> None:
    _require_local_origin(request)
    supplied = request.headers.get("x-csrf-token")
    if form is not None:
        supplied = supplied or form.get("_csrf_token")
    if not supplied or not hmac.compare_digest(str(supplied), _CSRF_TOKEN):
        raise HTTPException(status_code=403, detail="invalid CSRF token")


# ---------------------------------------------------------------------------
# Profile / env helpers (onboarding).
# ---------------------------------------------------------------------------


def _profile_exists() -> bool:
    return _get_profile_path().exists()


def _profile_data() -> dict[str, Any]:
    if not _get_profile_path().exists():
        return {}
    try:
        return yaml.safe_load(_get_profile_path().read_text()) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("could not parse profile.yaml: %s", e)
        return {}


def _profile_name() -> str:
    return str(_profile_data().get("name") or "").strip()


async def _read_urlencoded_form(request: Request) -> dict[str, str]:
    """Parse browser form posts without requiring python-multipart.

    The setup and name forms use the default
    ``application/x-www-form-urlencoded`` encoding, so Starlette's multipart
    dependency is unnecessary here.
    """
    raw = await request.body()
    parsed = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
    return {k: values[-1] if values else "" for k, values in parsed.items()}


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def _parse_ranked_topics(value: str | None) -> tuple[list[tuple[str, float]], list[str]]:
    """Parse one ``topic | relative weight`` entry per non-empty line."""
    topics: list[tuple[str, float]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate((value or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        topic, sep, raw_weight = line.partition("|")
        topic = topic.strip()
        if not sep or not topic or not raw_weight.strip():
            errors.append(f"Topic line {line_number} must use: topic | relative weight.")
            continue
        try:
            weight = float(raw_weight.strip())
        except ValueError:
            errors.append(f"Topic line {line_number} has an invalid weight.")
            continue
        if not 0.0 < weight <= 100.0:
            errors.append(f"Topic line {line_number} weight must be greater than 0 and at most 100.")
            continue
        key = topic.casefold()
        if key in seen:
            errors.append(f"Topic line {line_number} duplicates {topic!r}.")
            continue
        seen.add(key)
        topics.append((topic, weight))
    if not errors and not topics:
        errors.append("Add at least one weighted research interest.")
    if len(topics) > 10:
        errors.append("Add at most 10 weighted research interests.")
    return topics, errors


def _keywords_after_topic_edit(
    profile: dict[str, Any], topics: list[tuple[str, float]]
) -> list[str]:
    """Preserve specific retrieval phrases when visible facet names are unchanged."""
    submitted = [topic for topic, _weight in topics]
    canonical = profile.get("canonical_facets") or {}
    existing_names = list(canonical) if isinstance(canonical, dict) else []
    if {name.casefold() for name in submitted} == {
        str(name).casefold() for name in existing_names
    }:
        existing = [str(value).strip() for value in profile.get("keywords") or []]
        existing = list(dict.fromkeys(value for value in existing if value))
        if existing:
            return existing
    return submitted


def _load_existing_form_defaults() -> dict[str, str]:
    """Pre-populate the form from any existing profile.yaml + .env values."""
    backend = (SETTINGS.llm_backend or "extractive").lower()
    if backend not in {
        "api",
        "anthropic",
        "claude_cli",
        "codex_cli",
        "extractive",
    }:
        backend = "extractive"
    out: dict[str, str] = {
        "name": "",
        "bio": "",
        "topics": "",
        "downweight": "",
        "llm_backend": backend,
        "llm_base_url": SETTINGS.llm_base_url,
        "llm_api_key": "***" if SETTINGS.llm_api_key else "",
        "remove_llm_api_key": "false",
        "llm_model": SETTINGS.llm_model,
        "user_tz": SETTINGS.user_tz,
        "top_research": str(SETTINGS.top_research),
        "include_industry": str(section_enabled(SETTINGS, "industry")).lower(),
        "top_industry": str(SETTINGS.top_industry),
        "include_ai": str(section_enabled(SETTINGS, "ai")).lower(),
        "top_ai": str(SETTINGS.top_ai),
        "include_regulatory": str(section_enabled(SETTINGS, "regulatory")).lower(),
        "top_regulatory": str(SETTINGS.top_regulatory),
        "include_world": str(section_enabled(SETTINGS, "world")).lower(),
        "top_world": str(SETTINGS.top_world),
        "include_opportunities": str(
            section_enabled(SETTINGS, "opportunities")
        ).lower(),
        "top_opportunities": str(SETTINGS.top_opportunities),
        "include_events": str(section_enabled(SETTINGS, "events")).lower(),
        "top_events": str(SETTINGS.top_events),
        "opportunity_career_stage": "",
        "opportunity_institution_type": "",
        "opportunity_country": "",
        "opportunity_applicant_role": "",
        "opportunity_types": "",
        "event_types": "",
        "event_regions": "",
        "event_formats": "",
        "minimum_lead_days": "7",
    }
    if _get_profile_path().exists():
        try:
            data = yaml.safe_load(_get_profile_path().read_text()) or {}
            out["name"] = str(data.get("name") or "").strip()
            out["bio"] = (data.get("bio") or "").strip()
            canonical = data.get("canonical_facets") or {}
            if isinstance(canonical, dict) and canonical:
                lines = []
                for topic, spec in canonical.items():
                    priority = spec.get("priority", 1) if isinstance(spec, dict) else 1
                    if priority is None:
                        priority = 1
                    lines.append(f"{topic} | {priority}")
                out["topics"] = "\n".join(lines)
            else:
                priorities = data.get("topic_priorities") or {}
                out["topics"] = "\n".join(
                    f"{topic} | {priorities.get(topic, 1)}"
                    for topic in (data.get("keywords") or [])
                )
            out["downweight"] = ", ".join(data.get("downweight") or [])
        except Exception as e:  # noqa: BLE001
            logger.warning("could not parse existing profile.yaml: %s", e)
    if _get_opportunity_profile_path().exists():
        try:
            data = yaml.safe_load(_get_opportunity_profile_path().read_text()) or {}
            out.update(
                {
                    "opportunity_career_stage": str(data.get("career_stage") or ""),
                    "opportunity_institution_type": str(
                        data.get("institution_type") or ""
                    ),
                    "opportunity_country": str(data.get("country") or ""),
                    "opportunity_applicant_role": str(
                        data.get("applicant_role") or ""
                    ),
                    "opportunity_types": ",".join(
                        data.get("opportunity_types") or []
                    ),
                    "event_types": ",".join(data.get("event_types") or []),
                    "event_regions": ",".join(data.get("event_regions") or []),
                    "event_formats": ",".join(data.get("event_formats") or []),
                    "minimum_lead_days": str(data.get("minimum_lead_days", 7)),
                }
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("could not parse existing opportunities.yaml: %s", e)
    return out


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=value .env file. Preserves order is unnecessary here
    because we re-emit via _write_env_file with merged keys."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = _parse_env_value(v)
    return out


def _parse_env_value(value: str) -> str:
    val = value.strip()
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        body = val[1:-1]
        return body.replace('\\"', '"').replace("\\\\", "\\")
    return val


def _write_env_file(path: Path, updates: dict[str, str]) -> None:
    """Update KEY=value lines in place; append new keys at end. Preserves
    comments and blank lines from the original file."""
    if path.exists():
        original_lines = path.read_text().splitlines()
    else:
        original_lines = []

    seen: set[str] = set()
    new_lines: list[str] = []
    for raw in original_lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        k, _, _v = stripped.partition("=")
        key = k.strip()
        if key in updates:
            new_lines.append(f"{key}={_format_env_value(updates[key])}")
            seen.add(key)
        else:
            new_lines.append(line)

    for key, val in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={_format_env_value(val)}")

    _atomic_private_write(path, "\n".join(new_lines) + "\n")


def _write_private_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write local profile data with owner-only permissions on POSIX."""
    _atomic_private_write(path, yaml.safe_dump(payload, sort_keys=False))


def _atomic_private_write(path: Path, content: str) -> None:
    """Replace a private text file atomically when the filesystem permits it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        if os.name == "posix":
            temporary.chmod(0o600)
        try:
            os.replace(temporary, path)
            temporary = None
        except OSError:
            # Docker bind-mounted individual files may reject rename(2). Keep
            # that supported, with the smallest possible non-atomic fallback.
            path.write_text(content, encoding="utf-8")
        if os.name == "posix":
            path.chmod(0o600)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _format_env_value(value: str) -> str:
    val = str(value)
    if any(ch in val for ch in ("\r", "\n", "\0")):
        raise ValueError("environment values cannot contain line breaks")
    if val == "" or _ENV_SAFE_RE.match(val):
        return val
    escaped = val.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _env_value_has_control_chars(value: str | None) -> bool:
    return any(ch in str(value or "") for ch in ("\r", "\n", "\0"))


def _validate_setup(form: dict[str, str]) -> list[str]:
    errors: list[str] = []
    backend = form.get("llm_backend", "extractive")
    if backend not in (
        "extractive",
        "api",
        "anthropic",
        "claude_cli",
        "codex_cli",
    ):
        errors.append(f"Unknown backend: {backend}")
    _topics, topic_errors = _parse_ranked_topics(form.get("topics"))
    errors.extend(topic_errors)
    if backend in {"api", "anthropic"}:
        if not (form.get("llm_base_url") or "").strip():
            errors.append("API backend requires a base URL.")
        entered_key = (form.get("llm_api_key") or "").strip()
        removing_key = form.get("remove_llm_api_key") == "true"
        has_submitted_key = bool(entered_key and entered_key != "***")
        has_saved_key = bool(SETTINGS.llm_api_key and not removing_key)
        if not has_submitted_key and not has_saved_key:
            errors.append("API backend requires an API key.")
        if not (form.get("llm_model") or "").strip():
            errors.append("API backend requires a model name.")
    for key, label in (
        ("llm_base_url", "API base URL"),
        ("llm_api_key", "API key"),
        ("llm_model", "API model"),
    ):
        if _env_value_has_control_chars(form.get(key)):
            errors.append(f"{label} cannot contain line breaks.")
    for key, label, enabled_key in (
        ("top_research", "Research items", None),
        ("top_industry", "Industry items", "include_industry"),
        ("top_ai", "AI tools and methods items", "include_ai"),
        ("top_regulatory", "Clinical and regulatory items", "include_regulatory"),
        ("top_world", "World news items", "include_world"),
        ("top_opportunities", "Funding and opportunities items", "include_opportunities"),
        ("top_events", "Events and calls items", "include_events"),
    ):
        minimum = 1 if enabled_key is None or form.get(enabled_key) == "true" else 0
        raw = (form.get(key) or "").strip()
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if value < minimum or value > 30:
            errors.append(f"{label} must be between {minimum} and 30.")
    user_tz = (form.get("user_tz") or "").strip()
    if user_tz:
        try:
            ZoneInfo(user_tz)
        except (ZoneInfoNotFoundError, ValueError):
            errors.append("Browser timezone is not a valid IANA timezone.")
    if form.get("include_opportunities") == "true" or form.get("include_events") == "true":
        required = (
            ("opportunity_career_stage", "Career stage"),
            ("opportunity_institution_type", "Institution type"),
            ("opportunity_country", "Current country"),
            ("opportunity_applicant_role", "Applicant role"),
        )
        for key, label in required:
            if not (form.get(key) or "").strip():
                errors.append(f"{label} is required when opportunities or events are enabled.")
        lead_days_valid = True
        try:
            lead_days = int((form.get("minimum_lead_days") or "7").strip())
            if not 0 <= lead_days <= 365:
                raise ValueError
        except ValueError:
            lead_days_valid = False
            errors.append("Minimum preparation time must be between 0 and 365 days.")
        if lead_days_valid and not any(
            not (form.get(key) or "").strip() for key, _label in required
        ):
            try:
                _opportunity_profile_from_form(form)
            except ValidationError as exc:
                labels = {
                    "career_stage": "Career stage",
                    "institution_type": "Institution type",
                    "country": "Current country",
                    "applicant_role": "Applicant role",
                }
                for issue in exc.errors():
                    field = str(issue.get("loc", ("profile",))[-1])
                    label = labels.get(field, field.replace("_", " ").capitalize())
                    errors.append(f"{label}: {issue.get('msg', 'invalid value')}.")
    return errors


def _opportunity_profile_from_form(form: dict[str, str]) -> OpportunityProfile:
    return OpportunityProfile(
        career_stage=form["opportunity_career_stage"],
        institution_type=form["opportunity_institution_type"],
        country=form["opportunity_country"],
        applicant_role=form["opportunity_applicant_role"],
        opportunity_types=_parse_csv(form.get("opportunity_types")),
        event_types=_parse_csv(form.get("event_types")),
        event_regions=_parse_csv(form.get("event_regions")),
        event_formats=_parse_csv(form.get("event_formats")),
        minimum_lead_days=int(form.get("minimum_lead_days") or 7),
    )


def _int_form(form: dict[str, str], key: str, default: int) -> str:
    try:
        return str(int((form.get(key) or "").strip()))
    except ValueError:
        return str(default)


def _bool_form(form: dict[str, str], key: str, default: bool) -> str:
    if key not in form:
        return "true" if default else "false"
    return "true" if str(form.get(key, "")).lower() == "true" else "false"


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    if not _profile_exists():
        return RedirectResponse(url="/setup", status_code=302)
    digest_id = _digest_id()
    tea_notes = daily_tea_deck(date.fromisoformat(digest_id))
    sections, current_vote = _load_today(digest_id)
    brewed = bool(sections) or _digest_exists(digest_id)
    if brewed:
        # The reader is viewing this digest: flag its latest run's impressions as
        # viewed. Measurement-only; best-effort so a logging hiccup never blocks
        # the page render.
        try:
            visible_item_ids = [
                int(entry["id"])
                for section in sections
                for entry in section.get("entries", [])
            ]
            mark_impressions_viewed(digest_id, visible_item_ids)
        except Exception as _e:  # noqa: BLE001
            logger.warning("mark_impressions_viewed failed: %s", _e)
    overview = _digest_overview(sections)
    top_journal_audit = load_digest_audit(digest_id, "missed_top_journals") if brewed else []
    candidate_funnel = (
        _prepare_candidate_funnel(load_digest_audit(digest_id, "candidate_funnel"))
        if brewed
        else {}
    )
    _synth_rows = load_digest_audit(digest_id, "catch_up_synthesis") if brewed else []
    catch_up_synthesis = (_synth_rows[0].get("text") or "") if _synth_rows else ""
    catch_up_days = (_synth_rows[0].get("days") or 0) if _synth_rows else 0
    for audit in top_journal_audit:
        audit["url"] = safe_url(str(audit.get("url") or ""))
    # Qualified picks the reading mode trimmed today, with their pin state so a
    # reloaded page keeps showing "saved for tomorrow" after the click.
    overflow_audit = (
        load_digest_audit(digest_id, "reading_mode_overflow") if brewed else []
    )
    for audit in overflow_audit:
        audit["url"] = safe_url(str(audit.get("url") or ""))
    _overflow_ids = {
        int(audit["item_id"])
        for audit in overflow_audit
        if audit.get("item_id") is not None
    }
    overflow_saved = bool(_overflow_ids) and _overflow_ids <= carryover_item_ids()
    response = templates.TemplateResponse(
        request,
        "digest_web.html.j2",
        {
            "digest_id": digest_id,
            "profile_name": _profile_name(),
            "salutation": "Welcome back",
            "daily_note": tea_notes[0],
            "daily_notes": tea_notes,
            "summarizer_label": _summarizer_label(digest_id),
            "sections": sections,
            "overview": overview,
            "top_journal_audit": top_journal_audit,
            "overflow_audit": overflow_audit,
            "overflow_saved": overflow_saved,
            "candidate_funnel": candidate_funnel,
            "catch_up_synthesis": catch_up_synthesis,
            "catch_up_days": catch_up_days,
            "current_vote_per_item": current_vote,
            "ranking_status": votes_mod.lr_training_status(),
            "empty": len(sections) == 0,
            "brewed": brewed,
            "csrf_token": _CSRF_TOKEN,
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/saved", response_class=HTMLResponse)
def saved_items(request: Request, q: str = "") -> Response:
    if not _profile_exists():
        return RedirectResponse(url="/setup", status_code=302)
    query = str(q or "").strip()[:200]
    items = [
        {
            "id": row.item_id,
            "title": row.title,
            "url": safe_url(row.url),
            "source": row.source,
            "section": SECTION_META.get(
                row.section, {"title": row.section.title() or "Other"}
            )["title"],
            "summary": row.summary,
            "published": _saved_date(row.published_at),
            "saved": _saved_date(row.saved_at),
        }
        for row in search_bookmarks(query)
    ]
    response = templates.TemplateResponse(
        request,
        "saved.html.j2",
        {
            "items": items,
            "query": query,
            "csrf_token": _CSRF_TOKEN,
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/vote/{item_id}/{grade}")
def vote(request: Request, item_id: int, grade: int) -> JSONResponse:
    _require_csrf(request)
    # 4-level preference grades (must-read 100, relevant 70, hmmm 40, not-for-me
    # 10). Legacy -1/0/1 signs are still accepted for older clients.
    if grade in (100, 70, 40, 10):
        value = votes_mod.grade_to_value(grade)
        eff_grade: int | None = grade
    elif grade in (-1, 0, 1):
        value = grade
        eff_grade = votes_mod.value_to_grade(grade)
    else:
        raise HTTPException(status_code=400, detail="grade must be one of 100, 70, 40, 10")
    ok = votes_mod.record_vote_by_id(item_id, value, eff_grade)
    if not ok:
        raise HTTPException(status_code=404, detail=f"item {item_id} not found")
    return JSONResponse(
        {
            "ok": True,
            "item_id": item_id,
            "new_value": value,
            "new_grade": eff_grade,
            "ranking_status": votes_mod.lr_training_status(),
        }
    )


@app.post("/bookmark/{item_id}")
def bookmark_add(request: Request, item_id: int) -> JSONResponse:
    _require_csrf(request)
    if not set_bookmark(item_id, True):
        raise HTTPException(status_code=404, detail=f"item {item_id} not found")
    return JSONResponse({"ok": True, "item_id": item_id, "saved": True})


@app.delete("/bookmark/{item_id}")
def bookmark_remove(request: Request, item_id: int) -> JSONResponse:
    _require_csrf(request)
    if not set_bookmark(item_id, False):
        raise HTTPException(status_code=404, detail=f"item {item_id} not found")
    return JSONResponse({"ok": True, "item_id": item_id, "saved": False})


@app.post("/known/{item_id}")
def known_add(request: Request, item_id: int) -> JSONResponse:
    """Flag an item as already known so future digests skip it."""
    _require_csrf(request)
    if not set_item_known(item_id, True):
        raise HTTPException(status_code=404, detail=f"item {item_id} not found")
    return JSONResponse({"ok": True, "item_id": item_id, "known": True})


@app.delete("/known/{item_id}")
def known_remove(request: Request, item_id: int) -> JSONResponse:
    _require_csrf(request)
    if not set_item_known(item_id, False):
        raise HTTPException(status_code=404, detail=f"item {item_id} not found")
    return JSONResponse({"ok": True, "item_id": item_id, "known": False})


@app.post("/overflow/save")
def overflow_save(request: Request) -> JSONResponse:
    """Pin today's reading-mode overflow for one more evaluation at the next brew."""
    _require_csrf(request)
    digest_id = _digest_id()
    overflow = load_digest_audit(digest_id, "reading_mode_overflow")
    item_ids = [
        int(audit["item_id"]) for audit in overflow if audit.get("item_id") is not None
    ]
    if not item_ids:
        raise HTTPException(status_code=404, detail="no reading-mode overflow today")
    added = add_carryover_items(item_ids)
    return JSONResponse(
        {"ok": True, "digest_id": digest_id, "saved": len(item_ids), "added": added}
    )


@app.post("/vote/{item_id}/reason/{reason}")
def vote_reason(request: Request, item_id: int, reason: str) -> JSONResponse:
    _require_csrf(request)
    ok = votes_mod.record_vote_reason(item_id, reason)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="choose Seen or Not for me before adding a reason",
        )
    return JSONResponse(
        {
            "ok": True,
            "item_id": item_id,
            "reason": reason,
            "reasons": votes_mod.get_vote_reasons(item_id),
        }
    )


@app.delete("/vote/{item_id}/reason/{reason}")
def vote_reason_delete(request: Request, item_id: int, reason: str) -> JSONResponse:
    _require_csrf(request)
    ok = votes_mod.remove_vote_reason(item_id, reason)
    if not ok:
        raise HTTPException(status_code=400, detail="unknown item or reason")
    return JSONResponse(
        {
            "ok": True,
            "item_id": item_id,
            "reason": reason,
            "reasons": votes_mod.get_vote_reasons(item_id),
        }
    )


@app.get("/ranking/status")
def ranking_status(request: Request) -> JSONResponse:
    _require_local_origin(request)
    with _TRAIN_LOCK:
        training_job = dict(_TRAIN_JOB)
    return JSONResponse(
        {
            "ok": True,
            "status": votes_mod.lr_training_status(),
            "training_job": training_job,
        }
    )


@app.post("/ranking/train")
def ranking_train(request: Request) -> JSONResponse:
    _require_csrf(request)
    status = votes_mod.lr_training_status()
    if not status["can_train"]:
        return JSONResponse(
            {
                "ok": False,
                "started": False,
                "running": False,
                "message": (
                    f"Need {status['remaining_votes_for_lr']} more signed votes "
                    "before LR training."
                ),
                "status": status,
            }
        )

    with _TRAIN_LOCK:
        if _TRAIN_JOB["running"]:
            return JSONResponse(
                {
                    "ok": True,
                    "started": False,
                    "running": True,
                    "message": "Ranking training is already running.",
                    "status": status,
                    "training_job": dict(_TRAIN_JOB),
                }
            )
        if not _COMPUTE_LOCK.acquire(blocking=False):
            return JSONResponse(
                {
                    "ok": False,
                    "started": False,
                    "running": True,
                    "message": "Another brew or ranking job is already running.",
                    "status": status,
                }
            )
        _TRAIN_JOB["running"] = True
        _TRAIN_JOB["last_result"] = None

    def _target() -> None:
        process_lock = None
        try:
            process_lock = acquire_compute_lock(SETTINGS.db_path)
            result = votes_mod.train_lr_ranker()
        except ComputeBusyError as e:
            result = {
                "ok": False,
                "trained": False,
                "reason": "compute_busy",
                "message": str(e),
                "status": status,
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("ranking training failed")
            try:
                status_after_error = votes_mod.lr_training_status()
            except Exception:
                status_after_error = status
            result = {
                "ok": False,
                "trained": False,
                "reason": "training_error",
                "message": f"Ranking training failed: {type(e).__name__}: {e}",
                "status": status_after_error,
            }
        try:
            release_encoder()
        finally:
            if process_lock is not None:
                process_lock.release()
            with _TRAIN_LOCK:
                _TRAIN_JOB["running"] = False
                _TRAIN_JOB["last_result"] = result
            _COMPUTE_LOCK.release()

    threading.Thread(target=_target, daemon=True).start()
    return JSONResponse(
        {
            "ok": True,
            "started": True,
            "running": True,
            "message": "Ranking training started.",
            "status": status,
            "training_job": dict(_TRAIN_JOB),
        }
    )


def _run_pipeline_dry_run() -> None:
    if not _BREW_LOCK.acquire(blocking=False):
        logger.info("background refresh skipped because another brew is running")
        return
    if not _COMPUTE_LOCK.acquire(blocking=False):
        _BREW_LOCK.release()
        logger.info("background refresh skipped because another compute job is running")
        return
    process_lock = None
    try:
        process_lock = acquire_compute_lock(SETTINGS.db_path)
        with _RUN_LOCK:
            _BREW_JOB["running"] = True
            _BREW_JOB["run_id"] = "refresh"
        try:
            run_all(dry_run=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("background refresh failed: %s", e)
    except ComputeBusyError:
        logger.info("background refresh skipped because another process is computing")
    finally:
        try:
            release_encoder()
        finally:
            if process_lock is not None:
                process_lock.release()
            with _RUN_LOCK:
                _BREW_JOB["running"] = False
                _BREW_JOB["run_id"] = None
            _COMPUTE_LOCK.release()
            _BREW_LOCK.release()


@app.post("/refresh")
def refresh(request: Request) -> JSONResponse:
    _require_csrf(request)
    digest_id = _digest_id()
    if _BREW_LOCK.locked():
        return JSONResponse(
            {
                "ok": True,
                "digest_id": digest_id,
                "running": True,
                "message": "A brew is already running.",
            }
        )
    if _COMPUTE_LOCK.locked():
        return JSONResponse(
            {
                "ok": False,
                "digest_id": digest_id,
                "running": True,
                "message": "Another brew or ranking job is already running.",
            }
        )
    threading.Thread(target=_run_pipeline_dry_run, daemon=True).start()
    return JSONResponse({"ok": True, "digest_id": digest_id, "running": True})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/manifest.webmanifest")
def web_manifest(request: Request) -> Response:
    _require_local_origin(request)
    return Response(
        json.dumps(
            {
                "name": "DailyDigest",
                "short_name": "DailyDigest",
                "description": "A private, personalized research and news digest.",
                "start_url": "/",
                "scope": "/",
                "display": "standalone",
                "background_color": "#f6f7f4",
                "theme_color": "#256f52",
                "icons": [
                    {
                        "src": "/app-icon.svg",
                        "sizes": "192x192",
                        "type": "image/svg+xml",
                        "purpose": "any",
                    },
                    {
                        "src": "/app-icon.svg",
                        "sizes": "512x512",
                        "type": "image/svg+xml",
                        "purpose": "any maskable",
                    },
                    {
                        "src": "/app-icon.svg",
                        "sizes": "any",
                        "type": "image/svg+xml",
                        "purpose": "any",
                    },
                ],
                "prefer_related_applications": False,
            }
        ),
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/app-icon.svg")
def app_icon(request: Request) -> Response:
    _require_local_origin(request)
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
<rect width="512" height="512" rx="112" fill="#256f52"/>
<path d="M135 194h220v102c0 70-50 116-110 116s-110-46-110-116V194z" fill="#fff"/>
<path d="M355 225h27c44 0 66 27 66 59s-22 59-66 59h-33" fill="none" stroke="#fff" stroke-width="28" stroke-linecap="round"/>
<path d="M196 153c-24-31 20-43 0-75M256 153c-24-31 20-43 0-75M316 153c-24-31 20-43 0-75" fill="none" stroke="#f0c779" stroke-width="18" stroke-linecap="round"/>
</svg>"""
    return Response(
        svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/sw.js")
def service_worker(request: Request) -> Response:
    _require_local_origin(request)
    script = """const CACHE_PREFIX = "dailydigest-shell-";
const CACHE_NAME = `${CACHE_PREFIX}v1`;
const SAFE_ASSETS = ["/manifest.webmanifest", "/app-icon.svg"];
self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(SAFE_ASSETS)));
  self.skipWaiting();
});
self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys().then((keys) => Promise.all(
    keys.filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE_NAME)
      .map((key) => caches.delete(key))
  )).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin || !SAFE_ASSETS.includes(url.pathname)) return;
  event.respondWith(caches.open(CACHE_NAME).then(async (cache) => {
    const cached = await cache.match(url.pathname);
    if (cached) return cached;
    const response = await fetch(event.request);
    if (response.ok) cache.put(url.pathname, response.clone());
    return response;
  }));
});
"""
    return Response(
        script,
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"},
    )


# --- Onboarding -------------------------------------------------------------


@app.get("/setup", response_class=HTMLResponse)
def setup_get(request: Request) -> Response:
    form = _load_existing_form_defaults()
    response = templates.TemplateResponse(
        request,
        "setup.html.j2",
        {
            "form": form,
            "errors": [],
            "csrf_token": _CSRF_TOKEN,
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/setup")
async def setup_post(request: Request) -> Response:
    raw = await _read_urlencoded_form(request)
    _require_csrf(request, raw)
    form: dict[str, str] = {
        "name": str(raw.get("name", "")),
        "bio": str(raw.get("bio", "")),
        "topics": str(raw.get("topics", "")),
        "downweight": str(raw.get("downweight", "")),
        "llm_backend": str(raw.get("llm_backend", "extractive")),
        "llm_base_url": str(raw.get("llm_base_url", "")),
        "llm_api_key": str(raw.get("llm_api_key", "")),
        "remove_llm_api_key": _bool_form(raw, "remove_llm_api_key", False),
        "llm_model": str(raw.get("llm_model", "")),
        "user_tz": str(raw.get("user_tz", SETTINGS.user_tz)),
        "top_research": str(raw.get("top_research", SETTINGS.top_research)),
        "include_industry": _bool_form(
            raw, "include_industry", section_enabled(SETTINGS, "industry")
        ),
        "top_industry": str(raw.get("top_industry", SETTINGS.top_industry)),
        "include_ai": _bool_form(raw, "include_ai", section_enabled(SETTINGS, "ai")),
        "top_ai": str(raw.get("top_ai", SETTINGS.top_ai)),
        "include_regulatory": _bool_form(
            raw, "include_regulatory", section_enabled(SETTINGS, "regulatory")
        ),
        "top_regulatory": str(raw.get("top_regulatory", SETTINGS.top_regulatory)),
        "include_world": _bool_form(
            raw, "include_world", section_enabled(SETTINGS, "world")
        ),
        "top_world": str(raw.get("top_world", SETTINGS.top_world)),
        "include_opportunities": _bool_form(
            raw,
            "include_opportunities",
            section_enabled(SETTINGS, "opportunities")
            and _get_opportunity_profile_path().exists(),
        ),
        "top_opportunities": str(
            raw.get("top_opportunities", SETTINGS.top_opportunities)
        ),
        "include_events": _bool_form(
            raw,
            "include_events",
            section_enabled(SETTINGS, "events")
            and _get_opportunity_profile_path().exists(),
        ),
        "top_events": str(raw.get("top_events", SETTINGS.top_events)),
        "opportunity_career_stage": str(
            raw.get("opportunity_career_stage", "")
        ),
        "opportunity_institution_type": str(
            raw.get("opportunity_institution_type", "")
        ),
        "opportunity_country": str(raw.get("opportunity_country", "")),
        "opportunity_applicant_role": str(
            raw.get("opportunity_applicant_role", "")
        ),
        "opportunity_types": str(raw.get("opportunity_types", "")),
        "event_types": str(raw.get("event_types", "")),
        "event_regions": str(raw.get("event_regions", "")),
        "event_formats": str(raw.get("event_formats", "")),
        "minimum_lead_days": str(raw.get("minimum_lead_days", "7")),
    }

    errors = _validate_setup(form)
    if errors:
        # Never reflect a newly submitted secret into the response body. A
        # previously saved key can remain masked/reusable; a new key must be
        # entered again after the other validation errors are corrected.
        render_form = dict(form)
        display_errors = list(errors)
        submitted_key = render_form.get("llm_api_key", "").strip()
        if submitted_key:
            render_form["llm_api_key"] = "***" if SETTINGS.llm_api_key else ""
            if submitted_key != "***":
                display_errors.append(
                    "Re-enter the API key after correcting the form; "
                    "submitted keys are not returned for security."
                )
        response = templates.TemplateResponse(
            request,
            "setup.html.j2",
            {
                "form": render_form,
                "errors": display_errors,
                "csrf_token": _CSRF_TOKEN,
            },
            status_code=400,
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    topics, _topic_errors = _parse_ranked_topics(form["topics"])
    # Write the private user profile. Keywords drive retrieval; unchanged facet
    # edits preserve any more-specific retrieval phrases already stored, while
    # facets carry relative priority for attribution and selection.
    profile = _profile_data()
    existing_canonical = profile.get("canonical_facets") or {}
    existing_by_name = (
        {str(name).casefold(): spec for name, spec in existing_canonical.items()}
        if isinstance(existing_canonical, dict)
        else {}
    )
    canonical_facets: dict[str, dict[str, Any]] = {}
    for topic, weight in topics:
        previous = existing_by_name.get(topic.casefold())
        spec = dict(previous) if isinstance(previous, dict) else {}
        if not spec.get("anchors"):
            spec["anchors"] = [topic]
        spec["priority"] = weight
        canonical_facets[topic] = spec
    # The weighted-topic editor supersedes these older parallel weighting maps.
    # Keeping them would let removed topics continue influencing the profile.
    for legacy_key in ("topic_priorities", "interest_weights", "facet_weights"):
        profile.pop(legacy_key, None)
    profile.update(
        {
            "name": form["name"].strip(),
            "bio": form["bio"].strip() or "General reader.",
            "keywords": _keywords_after_topic_edit(profile, topics),
            "canonical_facets": canonical_facets,
            "downweight": _parse_csv(form["downweight"]),
        }
    )
    _write_private_yaml(_get_profile_path(), profile)

    if (
        form["include_opportunities"] == "true"
        or form["include_events"] == "true"
    ):
        opportunity_profile = _opportunity_profile_from_form(form)
        opportunity_path = _get_opportunity_profile_path()
        payload = opportunity_profile.model_dump(
            exclude={
                "description",
                "citizenship_or_residency",
                "requires_travel_support",
            }
        )
        # The simplified form no longer asks for these legacy values, but an
        # ordinary Settings save must not silently erase private profile data a
        # prior version already stored.
        if opportunity_path.exists():
            try:
                existing = yaml.safe_load(opportunity_path.read_text()) or {}
            except (OSError, yaml.YAMLError):
                existing = {}
            if isinstance(existing, dict):
                for key in (
                    "description",
                    "citizenship_or_residency",
                    "requires_travel_support",
                ):
                    if key in existing:
                        payload[key] = existing[key]
        _write_private_yaml(opportunity_path, payload)

    # Write/update .env.
    default_base_url = (
        "https://api.anthropic.com/v1"
        if form["llm_backend"] == "anthropic"
        else "https://api.openai.com/v1"
    )
    default_model = (
        "claude-haiku-4-5-20251001"
        if form["llm_backend"] == "anthropic"
        else ("" if form["llm_backend"] in {"claude_cli", "codex_cli"} else "gpt-4o-mini")
    )
    env_updates: dict[str, str] = {
        "LLM_BACKEND": form["llm_backend"],
        "LLM_BASE_URL": form["llm_base_url"].strip()
        or default_base_url,
        "LLM_MODEL": form["llm_model"].strip() or default_model,
        "USER_TZ": form["user_tz"].strip() or SETTINGS.user_tz,
        "TOP_RESEARCH": _int_form(form, "top_research", SETTINGS.top_research),
        "INCLUDE_INDUSTRY": form["include_industry"],
        "TOP_INDUSTRY": _int_form(form, "top_industry", SETTINGS.top_industry),
        "INCLUDE_AI": form["include_ai"],
        "TOP_AI": _int_form(form, "top_ai", SETTINGS.top_ai),
        "INCLUDE_REGULATORY": form["include_regulatory"],
        "TOP_REGULATORY": _int_form(form, "top_regulatory", SETTINGS.top_regulatory),
        "INCLUDE_WORLD": form["include_world"],
        "TOP_WORLD": _int_form(form, "top_world", SETTINGS.top_world),
        "INCLUDE_OPPORTUNITIES": form["include_opportunities"],
        "TOP_OPPORTUNITIES": _int_form(
            form, "top_opportunities", SETTINGS.top_opportunities
        ),
        "INCLUDE_EVENTS": form["include_events"],
        "TOP_EVENTS": _int_form(form, "top_events", SETTINGS.top_events),
    }
    # Only update API key if it's not the masked "***" value
    api_key = form.get("llm_api_key", "").strip()
    if form.get("remove_llm_api_key") == "true":
        env_updates["LLM_API_KEY"] = ""
    elif api_key and api_key != "***":
        env_updates["LLM_API_KEY"] = api_key
    _write_env_file(_ENV_PATH, env_updates)
    # Propagate changes to os.environ so pydantic-settings picks them up,
    # then refresh the in-memory SETTINGS singleton.
    for k, v in env_updates.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    reload_settings()
    globals()["SETTINGS"] = get_settings()

    response = RedirectResponse(url="/run", status_code=303)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/profile/name")
async def profile_name_post(request: Request) -> Response:
    raw = await _read_urlencoded_form(request)
    _require_csrf(request, raw)
    name = str(raw.get("name", "")).strip()
    if name:
        data = _profile_data()
        if not data:
            data = {"bio": "General reader.", "keywords": [], "downweight": []}
        data["name"] = name
        _write_private_yaml(_get_profile_path(), data)
    response = RedirectResponse(url="/", status_code=303)
    response.headers["Cache-Control"] = "no-store"
    return response


# --- Run / brewing flow -----------------------------------------------------


def _ensure_run(run_id: str) -> std_queue.Queue[dict[str, Any]]:
    """Get-or-create the stdlib Queue for a run."""
    with _RUN_LOCK:
        q = _RUN_QUEUES.get(run_id)
        if q is None:
            active_run = _BREW_JOB.get("run_id")
            while len(_RUN_QUEUES) >= _MAX_RETAINED_RUNS:
                stale_id = next(
                    (candidate for candidate in _RUN_QUEUES if candidate != active_run),
                    None,
                )
                if stale_id is None:
                    break
                _RUN_QUEUES.pop(stale_id, None)
                _RUN_STARTED.discard(stale_id)
            q = std_queue.Queue()
            _RUN_QUEUES[run_id] = q
        return q


def _kick_off_run(run_id: str, reading_mode: str) -> None:
    """Run pipeline.run_all in a background thread; always emits a terminal event."""

    def _push(evt: dict[str, Any]) -> None:
        q = _RUN_QUEUES.get(run_id)
        if q is not None:
            q.put_nowait(evt)

    def _target() -> None:
        terminal_sent = False
        process_lock = None
        acquired = _BREW_LOCK.acquire(blocking=False)
        if not acquired:
            _push(
                {
                    "stage": "error",
                    "payload": {"message": "Another brew is already running. Please wait for it to finish."},
                }
            )
            return
        compute_acquired = _COMPUTE_LOCK.acquire(blocking=False)
        if not compute_acquired:
            _BREW_LOCK.release()
            _push(
                {
                    "stage": "error",
                    "payload": {
                        "message": "Another brew or ranking job is already running. Please wait for it to finish."
                    },
                }
            )
            return
        with _RUN_LOCK:
            _BREW_JOB["running"] = True
            _BREW_JOB["run_id"] = run_id

        def cb(stage: str, payload: dict[str, Any]) -> None:
            nonlocal terminal_sent
            if stage in ("done", "error"):
                terminal_sent = True
            _push({"stage": stage, "payload": payload})

        try:
            process_lock = acquire_compute_lock(SETTINGS.db_path)
            run_all(
                dry_run=True,
                progress_callback=cb,
                reading_mode=reading_mode,
            )
        except ComputeBusyError as e:
            _push({"stage": "error", "payload": {"message": str(e)}})
            terminal_sent = True
        except Exception as e:  # noqa: BLE001
            logger.exception("pipeline failed in run %s", run_id)
            _push({"stage": "error", "payload": {"message": f"{type(e).__name__}: {e}"}})
            terminal_sent = True
        finally:
            try:
                release_encoder()
            finally:
                if process_lock is not None:
                    process_lock.release()
                with _RUN_LOCK:
                    _BREW_JOB["running"] = False
                    _BREW_JOB["run_id"] = None
                _COMPUTE_LOCK.release()
                _BREW_LOCK.release()
                if not terminal_sent:
                    _push({"stage": "done", "payload": {"forced": True}})

    threading.Thread(target=_target, daemon=True).start()


@app.get("/run", response_class=HTMLResponse)
def run_get(
    request: Request, reading_mode: str = "usual", autostart: bool = False
) -> Response:
    if not _profile_exists():
        return RedirectResponse(url="/setup", status_code=302)
    try:
        selected_mode = normalize_reading_mode(reading_mode)
    except ValueError:
        selected_mode = "usual"
    run_id = uuid.uuid4().hex[:12]
    response = templates.TemplateResponse(
        request,
        "run.html.j2",
        {
            "run_id": run_id,
            "csrf_token": _CSRF_TOKEN,
            "reading_mode": selected_mode,
            "autostart": bool(autostart),
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/run/start")
async def run_start(request: Request) -> JSONResponse:
    _require_csrf(request)
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    run_id = str(body.get("run_id") or uuid.uuid4().hex[:12])
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="invalid run id")
    try:
        reading_mode = normalize_reading_mode(
            str(body.get("reading_mode") or "usual")
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _ensure_run(run_id)
    with _RUN_LOCK:
        if run_id in _RUN_STARTED:
            return JSONResponse({"ok": True, "run_id": run_id, "already_started": True})
        _RUN_STARTED.add(run_id)
    _kick_off_run(run_id, reading_mode)
    return JSONResponse({"ok": True, "run_id": run_id})


@app.get("/run/stream")
async def run_stream(request: Request, run_id: str) -> StreamingResponse:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="invalid run id")
    with _RUN_LOCK:
        q = _RUN_QUEUES.get(run_id)
    if q is None:
        raise HTTPException(
            status_code=404,
            detail="brew run not found; it may have ended or the server restarted",
        )

    async def event_gen():
        terminal_seen = False
        try:
            yield f"data: {json.dumps({'stage': 'connected', 'run_id': run_id})}\n\n"
            terminal = {"done", "error"}
            while True:
                try:
                    evt = await asyncio.to_thread(q.get, True, 5.0)
                except Exception:  # queue.Empty or similar
                    yield ": heartbeat\n\n"
                    continue
                yield f"data: {json.dumps(evt)}\n\n"
                if evt.get("stage") in terminal:
                    terminal_seen = True
                    break
        finally:
            if terminal_seen:
                with _RUN_LOCK:
                    _RUN_QUEUES.pop(run_id, None)
                    _RUN_STARTED.discard(run_id)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/done", response_class=HTMLResponse)
def done(request: Request, digest_id: str = "", n: int = 0) -> Response:
    if not digest_id:
        digest_id = _digest_id()
    if not n:
        # Best-effort lookup if the URL didn't carry the count.
        sections, _ = _load_today(digest_id)
        n = sum(len(s["entries"]) for s in sections)
    response = templates.TemplateResponse(
        request,
        "done.html.j2",
        {"digest_id": digest_id, "total_items": n},
    )
    response.headers["Cache-Control"] = "no-store"
    return response
