"""Local FastAPI UI for browsing today's digest, voting, and onboarding setup.

This is an *alternative* delivery surface; the email path is unchanged.
The app binds to 127.0.0.1 by default — no remote access, no auth.

Routes:
- ``GET /``                — render today's digest (or redirect to /setup if no profile)
- ``POST /vote/{id}/{v}``  — record a +1/0/-1 vote for an item
- ``POST|DELETE /bookmark/{id}`` — save or remove an item from the reading archive
  (``?versions=1,2`` also covers that card's other versions of the same work)
- ``GET /saved``           — search the local saved-reading archive
- ``POST|DELETE /known/{id}`` — manually flag an item as known so future digests skip it
- ``POST|DELETE /interested/{id}`` — track (or stop tracking) a funding, call or event deadline
- ``POST /ui-prefs``       — remember theme, density and Pip in ``data/ui_prefs.json``
- ``POST /overflow/save``  — pin today's reading-mode overflow for tomorrow's brew
- ``POST /overflow/save/{id}`` — pin one overflow item (used when rating it positively)
- ``POST /vote/{id}/reason/{reason}`` — record qualitative feedback reason
- ``DELETE /vote/{id}/reason/{reason}`` — remove qualitative feedback reason
- ``GET /ranking/status``  — return vote counts and LR ranker status
- ``POST /ranking/train``  — refresh vote-derived ranking calibration on demand
- ``POST /refresh``        — kick off a dry-run pipeline in the background
- ``GET /healthz``         — liveness probe
- ``GET /update/status``   — newer release and what changed (checks twice a day)
- ``POST /update/apply``   — install the newer release and restart the local app
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
import errno
import hmac
import ipaddress
import json
import logging
import os
import math
import re
import secrets
import tempfile
import threading
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
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

from . import health, updates
from .article_figure import display_figure
from . import votes as votes_mod
from .config import SETTINGS, get_settings, reload_settings, section_enabled, settings_env_path
from .email_render import (
    SECTION_META,
    SECTION_ORDER,
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
from .brew import BrewCancelled, _digest_id, normalize_reading_mode
from .publications import publication_label
from .rank.source_quality import display_breakdown, display_source, source_bucket
from .store import (
    DigestItemRow,
    DigestRow,
    ItemRow,
    VoteRow,
    add_carryover_items,
    bookmarked_item_ids,
    carryover_item_ids,
    exclude_publication_companions,
    init_db,
    interested_item_ids,
    interested_items,
    item_metadata,
    known_item_ids,
    load_digest_audit,
    load_digest_features,
    load_publication_companions,
    mark_impressions_viewed,
    search_bookmarks,
    session_scope,
    set_bookmark,
    set_item_interested,
    set_item_known,
)
from .tea_break import daily_tea_deck, preview_tea_deck

logger = logging.getLogger(__name__)


def run_all(*args, **kwargs):
    """Load the brew pipeline only when a reader starts a job."""
    from .pipeline import run_all as run
    return run(*args, **kwargs)


def release_encoder():
    from .rank.embed import release_encoder as release
    return release()


@contextmanager
def _compute_write_guard():
    """Keep settings and selection-sensitive writes out of an active brew."""
    if not _COMPUTE_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A brew or ranking job is running. Try again when it finishes.")
    process_lock = None
    try:
        try:
            process_lock = acquire_compute_lock(get_settings().db_path)
        except ComputeBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        yield
    finally:
        if process_lock is not None:
            process_lock.release()
        _COMPUTE_LOCK.release()


def _settings_write(action):
    @wraps(action)
    async def guarded(*args, **kwargs):
        with _compute_write_guard():
            return await action(*args, **kwargs)
    return guarded


def _feedback_mutation(action):
    @wraps(action)
    def guarded(*args, **kwargs):
        try:
            return action(*args, **kwargs)
        except votes_mod.FeedbackStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return guarded


def _feedback_reasons():
    try:
        return votes_mod._load_vote_reasons(), ""
    except votes_mod.FeedbackStateError as exc:
        return {}, str(exc)


def _feedback_status():
    try:
        return votes_mod.lr_training_status()
    except votes_mod.FeedbackStateError as exc:
        return {"can_train": False, "blocked": True, "message": str(exc)}


def _overflow_signature(digest_id: str, audits: list[dict]) -> str:
    payload = json.dumps([digest_id, audits], sort_keys=True, default=str).encode()
    return hmac.new(_CSRF_TOKEN.encode(), payload, "sha256").hexdigest()


def _validated_overflow(digest_id: str, audit_signature: str) -> list[dict]:
    if not digest_id or not re.fullmatch(r"[0-9a-f]{64}", audit_signature):
        raise HTTPException(status_code=409, detail="Refresh the digest before saving these picks.")
    audits = load_digest_audit(digest_id, "reading_mode_overflow")
    if not audits:
        raise HTTPException(status_code=404, detail="This saved digest has no overflow picks.")
    if not hmac.compare_digest(_overflow_signature(digest_id, audits), audit_signature):
        raise HTTPException(status_code=409, detail="These picks changed after a rebrew. Refresh the digest.")
    return audits

_REPO_ROOT = Path(os.environ.get("DD_APP_ROOT") or Path.cwd()).resolve()
_PACKAGED_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE_DIR = (
    _PACKAGED_TEMPLATE_DIR
    if _PACKAGED_TEMPLATE_DIR.is_dir()
    else _REPO_ROOT / "templates"
)
_ENV_PATH = settings_env_path(_REPO_ROOT)


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
templates.env.globals["pip_notes"] = lambda: preview_tea_deck(date.fromisoformat(_digest_id()))
templates.env.globals["pip_day"] = lambda: _digest_id()
# The browser build has its own release notice; only native installs show this one.
templates.env.globals["local_updates_enabled"] = lambda: updates.install_mode() != "browser"
templates.env.globals["update_csrf_token"] = lambda: _CSRF_TOKEN
templates.env.globals["ui_prefs"] = lambda: _load_ui_prefs()

# Per-section sort order so we can ORDER BY (section_order, label_number).
_SECTION_RANK = {name: idx for idx, name in enumerate(SECTION_ORDER)}

# Per-run progress event logs. They are retained until the bounded run cache
# evicts them, so a refreshed page can replay the same brew instead of starting
# another one or racing the old page for queue entries.
_RUN_QUEUES: dict[str, list[dict[str, Any]]] = {}
_RUN_STARTED: set[str] = set()
_RUN_CANCEL: dict[str, threading.Event] = {}
_RUN_PUBLISHED: set[str] = set()
_MAX_RETAINED_RUNS = 32
_RUN_LOCK = threading.Lock()
_RUN_START_PERMITS: dict[str, tuple[str, float]] = {}
BROWSER_MODE = False
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


# ---------------------------------------------------------------------------
# Reader display transforms. These shape what a card shows; stored fields
# (raw authors, ``source`` ids, summaries) stay untouched for search,
# follow-matching, training and source caps.
# ---------------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _friendly_date(value: object, reference: date | None = None) -> dict[str, str]:
    """Return ``{"label": "Oct 12", "iso": "2026-10-12"}`` without a timezone shift.

    The stored calendar date is shown as-is: a date-only publication value is
    never converted to the reader's zone, where it could become the previous
    day. The year is added only when it differs from ``reference``.
    """
    day: date | None = None
    if isinstance(value, datetime):
        day = value.date()
    elif isinstance(value, date):
        day = value
    elif isinstance(value, str) and _ISO_DATE_RE.match(value.strip()):
        try:
            day = date.fromisoformat(value.strip())
        except ValueError:
            day = None
    if day is None:
        text = str(value or "").strip()
        return {"label": text, "iso": ""}
    label = f"{day:%b} {day.day}"
    if reference is None or day.year != reference.year:
        label += f", {day.year}"
    return {"label": label, "iso": day.isoformat()}


def _friendly_span(value: object, reference: date | None = None) -> str:
    """Format ``start–end`` ISO spans (event dates) and leave other text alone."""
    text = str(value or "").strip()
    parts = text.split("–") if "–" in text else [text]
    if len(parts) == 2 and all(_ISO_DATE_RE.match(part.strip()) for part in parts):
        try:
            start, end = (date.fromisoformat(part.strip()) for part in parts)
        except ValueError:
            return text  # an impossible calendar date is shown as stored, never a page error
        first = _friendly_date(start, end)["label"]
        last = _friendly_date(end, reference)["label"]
        if start.year == end.year and start.month == end.month:
            last = last.split(" ", 1)[1]  # "Nov 3–5"
        return f"{first}–{last}" if start.year == end.year else (
            f"{_friendly_date(start, None)['label']}–{_friendly_date(end, None)['label']}"
        )
    if _ISO_DATE_RE.match(text):
        return _friendly_date(text, reference)["label"]
    return text


# Feed-registry qualifiers that describe how a source is polled, not what it is.
_INTERNAL_SOURCE_SUFFIX_RE = re.compile(
    r"\s*\((?:recent|recent updates|front|your topics)\)\s*$", re.IGNORECASE
)


# Feed names that abbreviate a journal title; readers see the title. Publisher
# qualifiers such as ``Chem (Cell Press)`` stay, since they disambiguate.
_JOURNAL_DISPLAY_NAMES = {
    "angew. chem. int. ed.": "Angewandte Chemie International Edition",
    "chem. soc. rev.": "Chemical Society Reviews",
    "energy and environmental science": "Energy & Environmental Science",
}
_TITLE_SMALL_WORDS = frozenset({"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "&"})


def _title_case_venue(text: str) -> str:
    """Restore capitals to a sentence-cased journal title from an index.

    PubMed often lists ``Genomics, proteomics & bioinformatics``. Only titles
    of three or more words whose later words are all lowercase change, so
    names such as ``eLife``, ``bioRxiv`` or ``arXiv q-bio.QM`` and feed names
    with a qualifier in parentheses keep their form.
    """
    words = text.split(" ")
    if len(words) < 3 or "(" in text or not all(word.islower() or word == "&" for word in words[1:]):
        return text
    return " ".join(
        [words[0], *(word if word in _TITLE_SMALL_WORDS else word[:1].upper() + word[1:] for word in words[1:])]
    )


def _display_source_name(source: str) -> str:
    """Drop internal polling qualifiers such as ``bioRxiv (recent)`` and show
    a journal's own title rather than a feed abbreviation."""
    text = str(source or "").strip()
    cleaned = _INTERNAL_SOURCE_SUFFIX_RE.sub("", text).strip() or text
    return _JOURNAL_DISPLAY_NAMES.get(cleaned.lower()) or _title_case_venue(cleaned)


def _display_publication_label(label: str) -> str:
    """Hide the uninformative fallback; never relabel unknown as a research article."""
    return "" if not label or label == "Publication" else label


_NON_PERSON_WORDS = frozenset({
    "university", "universität", "université", "universidad", "department", "dept",
    "institute", "institutes", "institut", "instituto", "school", "hospital", "center",
    "centre", "laboratory", "laboratories", "lab", "college", "faculty", "inc", "ltd",
    "llc", "group", "consortium", "collaboration", "division", "national", "foundation",
    "program", "programme", "council", "agency", "ministry", "office", "committee",
    "society", "association", "network", "team", "investigators", "authors", "study",
    "street", "road", "avenue", "province", "china", "usa", "france", "germany",
    "switzerland", "japan", "kingdom",
})


def _is_initials(segment: str) -> bool:
    tokens = [token for token in re.split(r"[\s.\-‐]+", segment) if token]
    if not tokens or len(tokens) > 4:
        return False
    if all(len(token) <= 2 and token.isupper() for token in tokens):
        return True
    return "." in segment and all(len(token) <= 2 and token[0].isupper() for token in tokens)


def _looks_like_person(segment: str) -> bool:
    if not segment or len(segment) > 60 or any(ch.isdigit() for ch in segment):
        return False
    if any(ch in segment for ch in "@()[]{}<>/\\|:"):
        return False
    tokens = segment.split()
    if not 1 <= len(tokens) <= 6:
        return False
    if any(token.strip(".,'’-").casefold() in _NON_PERSON_WORDS for token in tokens):
        return False
    return all(token[0].isalpha() for token in tokens)


def _parse_author_names(raw: str) -> list[str] | None:
    """Split an author string only when its structure is unambiguous.

    Returns ``None`` when the list looks truncated, mixes in affiliations, or
    could be either ``Last, First`` pairs or separate people.
    """
    text = " ".join(str(raw or "").split())
    if not text:
        return []
    if re.search(r"\bet\.?\s+al\b|…|\.\.\.", text, re.IGNORECASE):
        return None
    if ";" in text:
        names: list[str] = []
        for part in (piece.strip() for piece in text.split(";")):
            if not part:
                continue
            if "," in part:
                last, _, given = (piece.strip() for piece in part.partition(","))
                if "," in given or not given or not _looks_like_person(last):
                    return None
                part = f"{given} {last}"
            if not _looks_like_person(part):
                return None
            names.append(part)
        return names
    parts = [piece.strip() for piece in re.split(r",|\s+(?:and|&)\s+", text) if piece.strip()]
    if len(parts) == 1:
        return parts if _looks_like_person(parts[0]) else None
    odd, even = parts[1::2], parts[0::2]
    if len(parts) % 2 == 0 and all(map(_is_initials, odd)) and not any(map(_is_initials, even)):
        names = [f"{given} {last}" for last, given in zip(even, odd, strict=True)]
    elif any(map(_is_initials, parts)) or any(len(part.split()) < 2 for part in parts):
        # "Smith, John, Doe, Jane" cannot be told apart from four people.
        return None
    else:
        names = parts
    return names if all(map(_looks_like_person, names)) else None


_AUTHOR_DETAILS_LIMIT = 6000


def _author_display(raw: object) -> dict[str, Any]:
    """Shorten authors for the source line; the complete string stays in Details.

    Shows ``First … Last (N authors)`` only when parsing is reliable. Anything
    ambiguous falls back to ``First et al.`` without a count. OpenAlex lists may
    stop at 100 authorships, so a count at that size is not trusted either.
    """
    full = " ".join(str(raw or "").split())
    if not full:
        return {"short": "", "full": "", "count": None}
    names = _parse_author_names(full)
    count: int | None = None
    if names and len(names) < 100:
        count = len(names)
        if count == 1:
            short = names[0]
        elif count == 2:
            short = f"{names[0]} and {names[1]}"
        else:
            short = f"{names[0]} … {names[-1]} ({count} authors)"
    else:
        first = (names[0] if names else re.split(r"[;,]|\s+and\s+", full, maxsplit=1)[0].strip())
        if names is None and len(full) <= 80 and not re.search(r"[;,]|\s(?:and|&)\s", full):
            short = full  # One named entity, such as a funding agency.
        elif first and not _is_initials(first) and len(first) > 1 and (
            names or _looks_like_person(first)
        ):
            short = f"{first} et al."
        else:
            short = ""
    if len(full) > _AUTHOR_DETAILS_LIMIT:
        full = full[:_AUTHOR_DETAILS_LIMIT].rstrip() + " …"
    return {"short": short, "full": full, "count": count}


def _highlight_segments(text: str, term: str | list[str], limit: int = 12) -> list[dict[str, Any]]:
    """Split text into plain/matched runs for autoescaped template rendering.

    Uses the same Unicode-aware whole-phrase boundaries as the topic evidence,
    so a term never highlights inside a longer word. Several terms match
    longest first, so "DNA origami" is one mark rather than "origami".
    """
    from .text_matching import term_pattern

    text = str(text or "")
    if not text:
        return []
    terms = [term] if isinstance(term, str) else list(term)
    terms = sorted({t.strip() for t in terms if isinstance(t, str) and t.strip() and len(t.strip()) <= 160},
                   key=lambda t: (-len(t), t.casefold()))
    if not terms:
        return [{"text": text, "match": False}]
    pattern = "|".join(f"(?:{term_pattern(t)})" for t in terms)
    segments: list[dict[str, Any]] = []
    position = 0
    for count, found in enumerate(re.finditer(pattern, text, re.IGNORECASE)):
        if count >= limit:
            break
        if found.start() > position:
            segments.append({"text": text[position:found.start()], "match": False})
        segments.append({"text": found.group(0), "match": True})
        position = found.end()
    if position < len(text):
        segments.append({"text": text[position:], "match": False})
    return segments


# A finding marks at most this many reader terms; more would be noise.
_FINDING_MARKS = 4


_EVIDENCE_DIMENSIONS = (
    ("validation", "Stated validation"),
    ("data_code", "Data/code availability"),
    ("limitations", "Stated limitations"),
)


def _complete_quote(quote: str, text: str) -> str:
    """Finish a quote that a sentence split cut inside brackets ("(66.5% pos.")."""
    start = text.find(quote) if text and quote.count("(") > quote.count(")") else -1
    close = text.find(")", start + len(quote)) if start >= 0 else -1
    if close < 0:
        return quote
    stop = re.search(r"[.!?;](?=\s|$)", text[close:])
    return text[start:close + stop.end() if stop else len(text)].strip()


def _evidence_view(ranking: dict, text: str = "") -> dict[str, Any] | None:
    """Show supplied quotes; collapse every unknown dimension into one line."""
    rows: list[tuple[str, list[str]]] = []

    def quotes(claims: object) -> list[str]:
        out: list[str] = []
        for claim in claims if isinstance(claims, list) else []:
            quote = str((claim or {}).get("quote") or "").strip() if isinstance(claim, dict) else ""
            quote = _complete_quote(quote, text)
            if quote and quote not in out:
                out.append(quote)
        return out

    contribution = ranking.get("contribution_claims")
    if contribution is not None:
        claims = contribution.get("claims") if isinstance(contribution, dict) else None
        rows.append(("Stated contribution", quotes(claims)))
    article = ranking.get("article_evidence")
    if article is not None:
        dimensions = article.get("dimensions") if isinstance(article, dict) else None
        dimensions = dimensions if isinstance(dimensions, dict) else {}
        for key, label in _EVIDENCE_DIMENSIONS:
            dimension = dimensions.get(key)
            rows.append((label, quotes(dimension.get("claims") if isinstance(dimension, dict) else None)))
    if not rows:
        return None
    unknown = [label for label, found in rows if not found]
    return {
        "known": [{"label": label, "quotes": found} for label, found in rows if found],
        "unknown": unknown,
        "unknown_text": _join_labels(unknown),
        "attributed": article is not None,
    }


def _join_labels(labels: list[str]) -> str:
    if len(labels) <= 2:
        return " and ".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


_SELECTION_NOTES = {
    "protected published-journal slot": "One of today’s published-journal picks.",
    "published-paper database slot": "Found through a published-paper database.",
    "capped arxiv cs slot": "One of a few AI and computing preprints shown each day.",
    "capped preprint slot": "One of a few preprints shown each day.",
    "capped aggregator slot": "One of a few aggregator picks shown each day.",
    "score-ranked slot": "",
}


def _selection_note(reason: object) -> str:
    """Translate internal slot names into plain reader wording."""
    text = " ".join(str(reason or "").split())
    if not text:
        return ""
    note = _SELECTION_NOTES.get(text.casefold())
    if note is not None:
        return note
    return text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")


def _bar_pct(value: float) -> int:
    if value <= 0:
        return 0
    return max(3, min(100, int(round(value * 100))))


def _breakdown_payload(row: ItemRow, persisted: dict | None = None) -> dict:
    if persisted:
        score = persisted.get("score", row.score or 0.0)
        rank_score = persisted.get("rank_score", score)
        confidence_score = persisted.get("confidence_score", score)
        payload = {
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
        if isinstance(persisted.get("contribution_rerank"), dict):
            payload["contribution_rerank"] = dict(persisted["contribution_rerank"])
        for field in ('contribution_claims', 'article_evidence', 'topic_alignment', 'topic_support'):
            if isinstance(persisted.get(field), dict):
                payload[field] = dict(persisted[field])
        return payload
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
            for topic, count in topic_counts.most_common()
        ],
        "reading_minutes": reading_minutes,
        "top_journals_shown": sum(
            1
            for entry in entries
            if entry.get("ranking", {}).get("source_bucket") == "published_journal"
        ),
    }


def _learned_profile_note(digest_id: str) -> str:
    """One quiet line when this brew rebuilt or could not use the learned profile.

    ``ok``/``static`` brews, and older digests without the audit, show nothing.
    """
    try:
        raw = load_digest_audit(digest_id, "learned_profile")
    except Exception:  # noqa: BLE001 - diagnostics never block reading.
        return ""
    audit = raw[-1] if isinstance(raw, list) and raw else raw
    if not isinstance(audit, dict) or audit.get("status") not in {"rebuilt", "unavailable"}:
        return ""
    return " ".join(str(audit.get("message") or "").split())[:500]


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


def _display_summary(summary: str | None, section: str = "", row: Any = None) -> str:
    """Read a stored summary under the current field contract (display only).

    Older digests carry generic Why read/Caveat boilerplate; the summarizer's
    normalizer drops it for display, checking a caveat against the item's own
    title and abstract when ``row`` is given. Nothing is written back to the store.
    """
    text = summary or ""
    if not text:
        return ""
    try:
        from .summarize import normalize_cached_summary

        context = f"{getattr(row, 'title', '') or ''} {getattr(row, 'abstract', '') or ''}"
        return normalize_cached_summary(text, section or "", context=context)
    except Exception:  # noqa: BLE001 - an unreadable summary still shows as stored.
        logger.debug("summary normalization failed", exc_info=True)
        return text


def _finding_label(section: str, publication: str = "") -> str:
    """Name the lead sentence by what it is; storage keeps "Key finding"."""
    if section in {"opportunities", "events"}:
        return "Purpose"
    if publication in {"Review", "Perspective", "Primer", "Commentary", "Editorial",
                       "News & Views", "Research Highlight"}:
        return "Scope"
    return "Key finding"


def _summary_fields(summary: str) -> dict[str, str]:
    """Parse stored three-field summaries for denser UI rendering.

    One parser for the field contract: the summarizer's, which also matches a
    label anywhere on a line ("Key finding: … Why read: …").
    """
    from .summarize import _summary_parts

    return _summary_parts(summary or "")


_GRADE_LABELS = {100: "Very relevant", 70: "Relevant", 40: "Less relevant", 10: "Not relevant"}


def _version_link_view(link: dict[str, Any]) -> dict[str, Any]:
    """One quiet "Also published in" line, from that version's own fields only.

    The status shown is the linked record's own withdrawal/retraction; nothing
    here is copied from, or onto, the card's primary record.
    """
    venue = _display_source_name(str(link.get("venue") or link.get("source") or "").strip())
    doi = str(link.get("doi") or "").strip()
    href = safe_url(f"https://doi.org/{doi}" if doi else str(link.get("url") or ""))
    if bool(link.get("is_preprint")):
        lead = "Preprint version on" if venue else "Preprint version"
    else:
        lead = "Also published in" if venue else "Also published"
    item_id = link.get("item_id")
    return {
        "lead": lead,
        "venue": venue,
        "href": "" if href == "#" else href,
        "link_text": f"DOI {doi}" if doi else "Open this version",
        "doi": doi,
        "status": str(link.get("retraction") or ""),
        "item_id": item_id if isinstance(item_id, int) and not isinstance(item_id, bool) else None,
        "in_digest": bool(link.get("in_digest")),
    }


def _apply_work_group(entry: dict, group: Any) -> dict:
    """Decorate the primary's card with its work group, for display only.

    Ratings stay per record: the card's controls and ``current_grade`` belong
    to ``group.primary``; another version's differing rating is a note, never
    merged or averaged. Saved/known are shown when ANY version carries them,
    and the card lists every carrying ID so clearing reaches all of them.
    """
    links = [_version_link_view(link) for link in group.links]
    venue_by_id = {link["item_id"]: link["venue"] for link in links if link["item_id"] is not None}
    notes = []
    for item_id, grade in group.other_version_ratings:
        if grade == group.primary_rating or grade not in _GRADE_LABELS:
            continue
        notes.append({"item_id": int(item_id), "venue": venue_by_id.get(item_id) or "other",
                      "grade": int(grade), "grade_label": _GRADE_LABELS[grade]})
    search_terms = [str(getattr(row, "title", "") or "") for row in group.versions]
    search_terms += [f"{link['venue']} {link['doi']}".strip() for link in links]
    entry.update({
        "version_links": links,
        "version_item_ids": [int(item_id) for item_id in group.item_ids[1:]],
        "version_rating_notes": notes,
        "version_search": " ".join(term for term in search_terms if term),
        "bookmarked": group.saved,
        "saved_item_ids": [int(item_id) for item_id in group.saved_item_ids],
        "known": group.known,
        "known_item_ids": [int(item_id) for item_id in group.known_item_ids],
    })
    return entry


def _group_entries(
    rows: list[Any],
    entries: dict[int, dict],
    *,
    ratings: dict[int, Any],
    saved: set[int],
    known: set[int],
    recorded_links: dict[int, list[dict]] | None = None,
) -> list[dict]:
    """One card per unique work, in the rows' order; no row is added or removed."""
    from .dedupe import group_publication_versions

    groups = group_publication_versions(
        rows, ratings=ratings, saved=saved, known=known, recorded_links=recorded_links or {},
    )
    return [_apply_work_group(entries[int(group.primary.id)], group) for group in groups]


def _recorded_version_links(features: dict[int, dict]) -> dict[int, list[dict]]:
    """Brew-time ``publication_versions`` links recorded in digest features."""
    out: dict[int, list[dict]] = {}
    for item_id, payload in features.items():
        links = payload.get("publication_versions") if isinstance(payload, dict) else None
        if isinstance(links, list) and links:
            out[int(item_id)] = [link for link in links if isinstance(link, dict)]
    return out


def _parse_version_ids(raw: str, primary: int) -> list[int]:
    """Other version IDs a card asks to clear together with its own ID."""
    ids: list[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or len(ids) >= 20:
            raise HTTPException(status_code=400, detail="versions must be up to 20 item ids")
        value = int(part)
        if value != primary and value not in ids:
            ids.append(value)
    return ids


def _load_today(
    digest_id: str, *, include_disabled_sections: bool = False
) -> tuple[list[dict], dict[int, int]]:
    """Return (rendered_sections, current_vote_per_item).

    The slate comes from ``digest_items``, the authoritative per-digest record,
    not from ``ItemRow.digest_id``: that back-pointer holds a single value and a
    later brew reassigns it whenever it re-selects the same item, which would
    silently rewrite an archived day. ``ItemRow.digest_id`` is only a fallback
    for digests stored before ``digest_items`` recorded the slate.
    ``include_disabled_sections`` renders an archived day as it was served, even
    if the reader has since switched one of its sections off.
    """
    init_db()
    with session_scope() as s:
        slate = s.execute(
            select(
                DigestItemRow.item_id,
                DigestItemRow.item_label,
                DigestItemRow.score,
            ).where(DigestItemRow.digest_id == digest_id)
        ).all()
        slate_labels = {int(item_id): (label or "") for item_id, label, _ in slate}
        slate_scores = {
            int(item_id): score for item_id, _, score in slate if score is not None
        }
        if slate_labels:
            rows = (
                s.execute(select(ItemRow).where(ItemRow.id.in_(list(slate_labels))))
                .scalars()
                .all()
            )
        else:
            rows = (
                s.execute(select(ItemRow).where(ItemRow.digest_id == digest_id))
                .scalars()
                .all()
            )
        if not rows:
            return [], {}

        if not include_disabled_sections:
            rows = [r for r in rows if section_enabled(SETTINGS, r.section or "")]
        # A linked commentary belongs to its paper, including when it arrives
        # on a later day or an older slate selected it as a standalone item.
        rows = exclude_publication_companions(rows)
        if not rows:
            return [], {}

        companions = load_publication_companions(rows)
        item_ids = [r.id for r in rows]
        saved_item_ids = bookmarked_item_ids(int(item_id) for item_id in item_ids)
        known_ids = known_item_ids(int(item_id) for item_id in item_ids)
        tracked_ids = interested_item_ids(int(item_id) for item_id in item_ids)
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
        _all_reasons, _reason_warning = _feedback_reasons()
        current_reasons = {
            int(item_id): _all_reasons.get(str(int(item_id)), [])
            for item_id in item_ids
        }
        persisted_features = load_digest_features(digest_id)
        try:
            reference_day: date | None = date.fromisoformat(digest_id)
        except ValueError:
            reference_day = None
        from .reader_insights import match_evidence
        from .config import load_profile

        try:
            reader_profile = load_profile(_get_profile_path())
            facets = reader_profile.canonical_facets
            reader_terms = list(reader_profile.keywords or [])
        except Exception:  # Historical digests remain readable without a profile.
            facets = {}
            reader_terms = []
        opportunity_profile = None
        if any((row.section or "") in {"opportunities", "events"} for row in rows):
            try:
                opportunity_profile = load_opportunity_profile(
                    get_settings().opportunity_profile_path
                )
            except Exception:  # noqa: BLE001
                pass

        def _slate_label(row: ItemRow) -> str:
            return slate_labels.get(int(row.id), row.item_label or "")

        def _slate_score(row: ItemRow) -> float:
            return float(slate_scores.get(int(row.id), row.score or 0.0) or 0.0)

        rows.sort(
            key=lambda r: (
                _SECTION_RANK.get(r.section or "", 99),
                _label_number(_slate_label(r)),
            )
        )

        rendered_sections: list[dict] = []
        seen_keys: list[str] = []
        by_section: dict[str, list[dict]] = {}
        section_rows: dict[str, list[ItemRow]] = {}
        for row in rows:
            key = row.section or "other"
            source = _display_source_name(display_source(row))
            ranking = _breakdown_payload(row, persisted_features.get(int(row.id)))
            genre_label = _display_publication_label(publication_label(row))
            confidence_score = _entry_confidence(
                {"ranking": ranking, "score_raw": _slate_score(row)}
            )
            reason = reason_line(
                ranking.get("primary_facet"),
                why_shown=ranking.get("why_shown") or [],
                tags=ranking.get("tags") or [],
            )
            type_label = content_type_label(ranking.get("content_type"))
            topic = ranking.get('primary_facet') or ''
            facet = facets.get(topic)
            evidence = match_evidence(row, topic, getattr(facet, 'aliases', [])) if topic else None
            if key not in by_section:
                by_section[key] = []
                section_rows[key] = []
                seen_keys.append(key)
            section_rows[key].append(row)
            opportunity = (
                opportunity_display(item_metadata(row), opportunity_profile, key)
                if key in {"opportunities", "events"}
                else None
            )
            if opportunity is not None:
                opportunity["calendar_url"] = f"/calendar/{int(row.id)}.ics"
                # A deadline always names its year; "Dec 2" alone is ambiguous.
                deadline = _friendly_date(opportunity.get("deadline"), None)
                opportunity["deadline_label"] = deadline["label"] or "Not stated"
                opportunity["deadline_iso"] = deadline["iso"]
                opportunity["event_dates_label"] = _friendly_span(
                    opportunity.get("event_dates"), reference_day
                )
                kind = opportunity.get("kind")
                is_event = kind == "event"
                opportunity["tracked"] = {
                    **_tracked_date(opportunity, is_event, item_metadata(row), reference_day),
                    "kind": "Event" if is_event else "Call" if kind == "call" else "Funding",
                }
            published = _friendly_date(row.published_at, reference_day)
            display_summary = _display_summary(row.summary, key, row)
            # Feed headers ("Journal, Published online: …; doi:…") are not the
            # abstract; the summary path already reads the same cleaned text.
            from .ingest.rss import strip_feed_metadata  # Setup must not load feed adapters.

            display_abstract = strip_feed_metadata(row.abstract or "")
            authors = _author_display(row.authors)
            summary_fields = _summary_fields(display_summary)
            finding_terms = [*reader_terms, topic, *getattr(facet, "aliases", [])]
            by_section[key].append(
                {
                    "id": row.id,
                    "label": _slate_label(row),
                    "title": row.title or "",
                    "url": safe_url(row.url),
                    "source": source,
                    "published": _format_date(row),
                    "published_label": published["label"],
                    "published_iso": published["iso"],
                    "summary": display_summary,
                    "abstract": display_abstract,
                    "summary_fields": summary_fields,
                    "finding_segments": _highlight_segments(
                        summary_fields.get("Key finding") or display_summary, finding_terms, _FINDING_MARKS
                    ),
                    "finding_label": _finding_label(key, genre_label),
                    "figure": display_figure(item_metadata(row)),
                    "score_raw": _slate_score(row),
                    "confidence_score": confidence_score,
                    "ranking": ranking,
                    "reason_line": reason,
                    "match_evidence": evidence,
                    "abstract_segments": _highlight_segments(
                        display_abstract, (evidence or {}).get("term", "")
                    ),
                    "evidence": _evidence_view(ranking, display_abstract),
                    "selection_note": _selection_note(ranking.get("selection_reason")),
                    "authors": authors["short"],
                    "authors_full": authors["full"],
                    "publication_label": genre_label,
                    "related_commentaries": [
                        {**child, "url": safe_url(child.get("url", ""))}
                        for child in companions.get(int(row.id), [])
                    ],
                    "type_label": type_label,
                    "opportunity": opportunity,
                    "current_vote": current_vote.get(int(row.id)),
                    "current_grade": current_grade.get(int(row.id)),
                    "current_reasons": current_reasons.get(int(row.id), []),
                    "bookmarked": int(row.id) in saved_item_ids,
                    "known": int(row.id) in known_ids,
                }
            )

        # One card per unique work (preprint + journal version of one paper).
        # Display only: every row, vote, bookmark and known flag keeps its ID,
        # and a history is never refilled with another item.
        recorded_links = _recorded_version_links(persisted_features)
        for key, section_entries in by_section.items():
            by_section[key] = _group_entries(
                section_rows[key], {int(entry["id"]): entry for entry in section_entries},
                ratings=current_grade, saved=saved_item_ids, known=known_ids,
                recorded_links=recorded_links,
            )
            for entry in by_section[key]:
                carrying = [iid for iid in (int(entry["id"]), *entry.get("version_item_ids", [])) if iid in tracked_ids]
                entry["interested"] = bool(carrying)
                entry["interested_item_ids"] = carrying
            _mark_top_picks(by_section[key])

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


def _tracked_date(opportunity: dict, is_event: bool, metadata: dict | None = None,
                  today: date | None = None) -> dict[str, str]:
    """The date a tracked item is waiting on.

    That is the next stated deadline (a call's later stages and an event's
    other registration routes included), else an upcoming event's start.
    Once everything has passed, the last of those dates is shown as passed.
    """
    today = today or date.fromisoformat(_digest_id())
    stated = [opportunity.get("deadline")] + [
        row.get("date") for row in (metadata or {}).get("deadlines") or [] if isinstance(row, dict)]
    deadlines = sorted({when["iso"]: when for when in (_friendly_date(value, None) for value in stated)
                        if when["iso"]}.values(), key=lambda when: when["iso"])
    start = _friendly_date(str(opportunity.get("event_dates") or "").split("–")[0].strip(), None)
    word = "Deadline" if is_event else "Closes"
    upcoming = [when for when in deadlines if when["iso"] >= today.isoformat()]
    if upcoming:
        return {"iso": upcoming[0]["iso"], "label": upcoming[0]["label"], "word": word}
    if is_event and start["iso"] and (start["iso"] >= today.isoformat() or not deadlines
                                      or start["iso"] > deadlines[-1]["iso"]):
        return {"iso": start["iso"], "label": start["label"], "word": "Starts"}
    if deadlines:
        return {"iso": deadlines[-1]["iso"], "label": deadlines[-1]["label"], "word": word}
    return {"iso": "", "label": str(opportunity.get("deadline") or "Not stated"), "word": "Deadline"}


def _tracked_view(row: ItemRow, profile: Any = None) -> dict:
    """One Interested item as the deadline strip and the Saved page show it."""
    opportunity = opportunity_display(item_metadata(row), profile, row.section or "")
    kind = opportunity.get("kind")
    is_event = kind == "event"
    when = _tracked_date(opportunity, is_event, item_metadata(row))
    return {
        "id": int(row.id), "title": row.title or "", "url": safe_url(row.url),
        "source": _display_source_name(display_source(row)),
        "kind": "Event" if is_event else "Call" if kind == "call" else "Funding",
        "date_iso": when["iso"], "date_label": when["label"], "date_word": when["word"],
        "event_dates_label": _friendly_span(opportunity.get("event_dates"), None),
        "location": opportunity.get("location") or "", "amount": opportunity.get("amount") or "",
        "eligibility": opportunity.get("eligibility") or "",
        "eligibility_reason": opportunity.get("eligibility_reason") or "",
        "eligibility_note": opportunity.get("eligibility_note") or "",
        "calendar_url": f"/calendar/{int(row.id)}.ics",
    }


def _tracked_items(profile: Any = None) -> list[dict]:
    """Every Interested funding, call or event item, soonest date first, undated last."""
    rows = [row for row in interested_items() if (row.section or "") in {"opportunities", "events"}]
    views = [_tracked_view(row, profile) for row in rows]
    return sorted(views, key=lambda view: (not view["date_iso"], view["date_iso"]))


def _tracked_deadlines(sections: list[dict], today: date, limit: int = 6) -> dict:
    """Upcoming dates of the items the reader marked Interested, for the top of the page.

    Only an explicit Interested mark puts an item here, and it stays tracked
    across digests; a passed or undated item is listed on the Saved page instead.
    """
    on_page = {int(entry["id"]) for section in sections for entry in section.get("entries", [])}
    tracked = _tracked_items()
    upcoming = [view for view in tracked if view["date_iso"] and view["date_iso"] >= today.isoformat()]
    for view in upcoming:
        view["on_page"] = view["id"] in on_page
    return {"rows": upcoming[:limit], "more": max(0, len(upcoming) - limit), "total": len(tracked)}


# Top picks are the strongest of every section, in slate order: about a third
# of each section and at least one. Sections score on different scales, so one
# absolute cutoff would show only research.
_TOP_PICK_SHARE = 0.3


def _mark_top_picks(entries: list[dict]) -> None:
    count = math.ceil(len(entries) * _TOP_PICK_SHARE)
    for index, entry in enumerate(entries):
        entry["top_pick"] = index < max(1, count)


def _digest_exists(digest_id: str) -> bool:
    init_db()
    with session_scope() as s:
        return s.get(DigestRow, digest_id) is not None


def _recent_digest_ids(today_id: str, limit: int | None = 3) -> list[str]:
    """Return retained editions by date, independent of later rebrews."""
    init_db()
    with session_scope() as session:
        query = (select(DigestRow.id).where(DigestRow.id <= today_id)
                 .order_by(DigestRow.id.desc(), DigestRow.created_at.desc()))
        if limit is not None:
            query = query.limit(max(1, min(int(limit), 30)))
        return [str(value) for value in session.execute(query).scalars()]


def _as_int(value: object) -> int | None:
    """Coerce an audit field to int, or None when it is missing/garbage."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


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
    try:
        parsed = urlsplit(f"//{host_header or ''}")
        # Validate the port as well as the hostname; urlsplit defers this check.
        _ = parsed.port
        if parsed.username is not None or parsed.path or parsed.query or parsed.fragment:
            return ""
        return (parsed.hostname or "").lower()
    except ValueError:
        return ""


def _is_local_host(host: str) -> bool:
    if host in {"localhost", "testserver"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_local_origin(request: Request) -> None:
    host = _host_name(request.headers.get("host"))
    if not _is_local_host(host):
        raise HTTPException(status_code=403, detail="local UI only accepts loopback hosts")

    origin = request.headers.get("origin")
    if not origin:
        return
    try:
        supplied = urlsplit(origin)
        expected = urlsplit(str(request.url))
        default_ports = {"http": 80, "https": 443}
        same_origin = (
            supplied.scheme in default_ports
            and supplied.username is None
            and (supplied.scheme, supplied.hostname, supplied.port or default_ports[supplied.scheme])
            == (expected.scheme, expected.hostname, expected.port or default_ports[expected.scheme])
        )
    except (ValueError, KeyError):
        same_origin = False
    if not same_origin:
        raise HTTPException(status_code=403, detail="cross-origin writes are not allowed")


@app.middleware("http")
async def _loopback_host_only(request: Request, call_next):
    """Reject non-loopback Host headers for every UI and asset route."""
    if not _is_local_host(_host_name(request.headers.get("host"))):
        response = JSONResponse(
            {"detail": "local UI only accepts loopback hosts"}, status_code=403
        )
    else:
        response = await call_next(request)
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


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
    if not supplied or not hmac.compare_digest(str(supplied).encode(), _CSRF_TOKEN.encode()):
        raise HTTPException(status_code=403, detail="invalid CSRF token")


# ---------------------------------------------------------------------------
# Profile / env helpers (onboarding).
# ---------------------------------------------------------------------------


def _profile_exists() -> bool:
    return _get_profile_path().exists()


def _profile_data(*, strict: bool = False, path: Path | None = None) -> dict[str, Any]:
    path = path or _get_profile_path()
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text())
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("profile must be a YAML mapping")
        return data
    except Exception as e:  # noqa: BLE001
        logger.warning("could not parse %s: %s", path.name, e)
        if strict:
            raise HTTPException(
                status_code=409,
                detail=f"Existing {path.name} cannot be read; repair it before saving settings.",
            ) from e
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
    try:
        parsed = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="form must use UTF-8") from exc
    return {k: values[-1] if values else "" for k, values in parsed.items()}


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def _parse_ranked_topics(value: str | None) -> tuple[list[tuple[str, float]], list[str]]:
    """Parse one topic per line, with an optional ``| relative weight``."""
    topics: list[tuple[str, float]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate((value or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        topic, sep, raw_weight = line.partition("|")
        topic = topic.strip()
        if not sep:
            raw_weight = "1"
        if not topic or not raw_weight.strip():
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
        errors.append("Add at least one research interest.")
    if len(topics) > 10:
        errors.append("Add at most 10 research interests.")
    return topics, errors


def _parse_topic_details(value: str | None, topics: list[tuple[str, float]]) -> tuple[dict, list[str]]:
    """Validate optional enhanced-editor details; legacy forms omit this field."""
    if not value:
        return {}, []
    try:
        details = json.loads(value)
    except (TypeError, ValueError):
        return {}, ["Interest details must be valid JSON."]
    if not isinstance(details, dict):
        return {}, ["Interest details must be an object."]
    names = {name for name, _ in topics}
    clean = {}
    for name, detail in details.items():
        if name not in names or not isinstance(detail, dict):
            return {}, ["Interest details must match the listed interests."]
        aliases, description = detail.get("aliases", []), detail.get("description", "")
        if (not isinstance(aliases, list) or len(aliases) > 3
                or any(not isinstance(alias, str) or not alias.strip() or len(alias) > 160 or "|" in alias or "\n" in alias for alias in aliases)
                or not isinstance(description, str) or len(description) > 600):
            return {}, ["Use up to 3 alternative terms (160 characters each) and a description of up to 600 characters."]
        clean[name] = {"aliases": list(dict.fromkeys(alias.strip() for alias in aliases)),
                       "description": description.strip()}
        previous_name = detail.get("previous_name")
        if previous_name is not None:
            if (not isinstance(previous_name, str) or not previous_name.strip()
                    or len(previous_name) > 160 or "\n" in previous_name or "|" in previous_name
                    or (previous_name != name and previous_name in names)):
                return {}, ["A corrected interest must refer to one original interest."]
            clean[name]["previous_name"] = previous_name.strip()
    return clean, []


def _interest_coverage(sections: list[dict], audits: list[dict]) -> list[dict]:
    profile = _profile_data()
    names = list(profile.get("canonical_facets") or {}) or list(profile.get("keywords") or [])
    recorded = {str(row.get("title", "")).casefold(): row for row in audits}
    facets = [str(entry.get("ranking", {}).get("primary_facet") or "").casefold()
              for section in sections if section.get("key", "research") == "research"
              for entry in section.get("entries", [])]
    selected = Counter(facets)
    complete = bool(facets) and all(facets)
    rows = []
    for name in names:
        row = dict(recorded.get(str(name).casefold(), {}))
        count = selected[str(name).casefold()]
        row.update(title=name, selected=count if count or complete else row.get("selected"))
        row.setdefault("candidates", None)
        row.setdefault("eligible", None)
        row.setdefault("explanation", "Candidate attribution was not recorded for this saved digest.")
        rows.append(row)
    return rows


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
        "topic_details": "",
        "downweight": "",
        "authors_of_interest": "",
        "seed_works": "",
        "context_keywords": "",
        "llm_backend": backend,
        "llm_base_url": SETTINGS.llm_base_url,
        "llm_api_key": "***" if SETTINGS.llm_api_key else "",
        "remove_llm_api_key": "false",
        "llm_model": SETTINGS.llm_model,
        "user_tz": SETTINGS.user_tz,
        "top_research": str(SETTINGS.top_research),
        "publication_enrichment": str(SETTINGS.publication_enrichment).lower(),
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
    if not _get_profile_path().exists():
        for key in out:
            if key.startswith('include_'):
                out[key] = 'false'
    if _get_profile_path().exists():
        try:
            data = yaml.safe_load(_get_profile_path().read_text()) or {}
            out["name"] = str(data.get("name") or "").strip()
            out["bio"] = (data.get("bio") or "").strip()
            canonical = data.get("canonical_facets") or {}
            if isinstance(canonical, dict) and canonical:
                resolved_priorities = []
                legacy_priorities = {str(name).casefold(): value
                                     for name, value in (data.get("topic_priorities") or {}).items()}
                for topic, spec in canonical.items():
                    priority = spec.get("priority") if isinstance(spec, dict) else None
                    if priority is None:
                        names = [topic, *(spec.get("aliases") or [])] if isinstance(spec, dict) else [topic]
                        matched = []
                        for name in names:
                            try:
                                value = float(legacy_priorities.get(str(name).casefold(), 0))
                            except (TypeError, ValueError):
                                continue
                            if 0 < value <= 100:
                                matched.append(value)
                        priority = max(matched) if matched else None
                    resolved_priorities.append((topic, priority))
                positive_priorities = [float(priority) for _topic, priority in resolved_priorities
                                       if isinstance(priority, (int, float)) and 0 < priority <= 100]
                # Attribution supplies 0.5 for an unlisted facet once another
                # explicit priority exists. Preserve that effective ratio on a
                # name/provider-only Save instead of silently replacing it by 1.
                unlisted_priority = max(positive_priorities) / 2 if positive_priorities else 1
                lines = [f"{topic} | {priority if priority is not None else unlisted_priority}"
                         for topic, priority in resolved_priorities]
                out["topics"] = "\n".join(lines)
                out["topic_details"] = json.dumps({str(name): {"aliases": spec.get("aliases", []), "description": spec.get("description", "")} for name, spec in canonical.items() if isinstance(spec, dict)})
            else:
                priorities = data.get("topic_priorities") or {}
                out["topics"] = "\n".join(
                    f"{topic} | {priorities.get(topic, 1)}"
                    for topic in (data.get("keywords") or [])
                )
            out["downweight"] = ", ".join(data.get("downweight") or [])
            for key in ("authors_of_interest", "seed_works", "context_keywords"):
                out[key] = "\n".join(str(value) for value in data.get(key) or [])
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


def _updated_env_content(path: Path, updates: dict[str, str]) -> str:
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

    return "\n".join(new_lines) + "\n"


def _write_env_file(path: Path, updates: dict[str, str]) -> None:
    _atomic_private_write(path, _updated_env_content(path, updates))


def _write_private_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write local profile data with owner-only permissions on POSIX."""
    _atomic_private_write(path, yaml.safe_dump(payload, sort_keys=False))


def _atomic_private_write(path: Path, content: str | bytes) -> None:
    """Replace a private file atomically when the filesystem permits it."""
    encoded = content.encode('utf-8') if isinstance(content, str) else content
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            temporary.chmod(0o600)
        try:
            os.replace(temporary, path)
            temporary = None
        except OSError as exc:
            # Docker bind-mounted individual files may reject rename(2). Keep
            # that supported, with the smallest possible non-atomic fallback.
            from .settings_transaction import settings_transaction_active
            if exc.errno != errno.EBUSY or settings_transaction_active():
                raise
            path.write_bytes(encoded)
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


def _parse_advanced_interest_list(value: str) -> list[str]:
    return list(dict.fromkeys(line.strip() for line in value.splitlines() if line.strip()))


def _saved_api_key_matches(backend: str, base_url: str) -> bool:
    from .provider_check import api_provider_binding

    selected = api_provider_binding(backend, base_url)
    saved = api_provider_binding(SETTINGS.llm_backend, SETTINGS.llm_base_url)
    return selected is not None and selected == saved


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
    if form.get("topic_weight_mode") == "percentage" and (
        any(not weight.is_integer() for _, weight in _topics)
        or sum(weight for _, weight in _topics) != 100
    ):
        errors.append("Use whole-number percentages totaling 100% for your research interests.")
    _details, detail_errors = _parse_topic_details(form.get("topic_details"), _topics)
    errors.extend(detail_errors)
    for key, label in (
        ("authors_of_interest", "Authors, labs and institutions"),
        ("seed_works", "Representative works"),
        ("context_keywords", "Background interests"),
    ):
        value = form.get(key, "")
        entries = _parse_advanced_interest_list(value)
        if len(entries) > 20 or any(len(entry) > 300 for entry in entries) or "\0" in value:
            errors.append(f"{label}: use at most 20 lines of 300 characters each, without null characters.")
    if backend in {"api", "anthropic"}:
        if not (form.get("llm_base_url") or "").strip():
            errors.append("API backend requires a base URL.")
        entered_key = (form.get("llm_api_key") or "").strip()
        removing_key = form.get("remove_llm_api_key") == "true"
        has_submitted_key = bool(entered_key and entered_key != "***")
        has_saved_key = bool(SETTINGS.llm_api_key and not removing_key
                             and _saved_api_key_matches(backend, form.get('llm_base_url', '')))
        if not has_submitted_key and not has_saved_key:
            errors.append("API backend requires an API key.")
            if SETTINGS.llm_api_key and not removing_key:
                errors.append("Enter a key for this provider and URL; saved keys cannot be reused at a different API endpoint.")
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
def index(request: Request, digest_id: str = "") -> Response:
    if not _profile_exists():
        return RedirectResponse(url="/setup", status_code=302)
    today_id = _digest_id()
    today_date = date.fromisoformat(today_id)
    requested_missing = False
    edition_ids = _recent_digest_ids(today_id, limit=None)
    if digest_id and digest_id != today_id and digest_id not in edition_ids:
        try:
            valid_date = date.fromisoformat(digest_id).isoformat() == digest_id
        except ValueError:
            valid_date = False
        if valid_date and _digest_exists(digest_id):
            # Only a date after today can be absent from the list (a timezone
            # change); keep the list newest-first so navigation stays in order.
            edition_ids = sorted({*edition_ids, digest_id}, reverse=True)
        else:
            requested_missing = True
    recent_brews = []
    for brew_id in edition_ids:
        try:
            brew_date = date.fromisoformat(brew_id)
        except ValueError:
            continue
        offset = (today_date - brew_date).days
        if offset == 0:
            label, cup_kicker = "Today", "Today’s cup"
        elif offset == 1:
            label, cup_kicker = "Yesterday", "Yesterday’s cup"
        elif offset == -1:
            label, cup_kicker = "Tomorrow", "Tomorrow’s cup"
        elif offset < 0:
            label, cup_kicker = f"In {-offset} days", f"Cup dated {-offset} days ahead"
        else:
            label, cup_kicker = f"{offset} days ago", f"Cup from {offset} days ago"
        recent_brews.append(
            {
                "id": brew_id,
                "label": label,
                "cup_kicker": cup_kicker,
                "date_label": brew_date.strftime("%b %d" if brew_date.year == today_date.year else "%b %d, %Y").replace(" 0", " "),
                "href": "/" if offset == 0 else f"/?digest_id={brew_id}",
            }
        )
    available_brew_ids = {str(brew["id"]) for brew in recent_brews}
    if not digest_id or digest_id not in available_brew_ids:
        digest_id = today_id
    brew_ids = [str(brew["id"]) for brew in recent_brews]
    current_position = brew_ids.index(digest_id) if digest_id in brew_ids else -1
    previous_brew = (
        recent_brews[current_position + 1]
        if current_position + 1 < len(recent_brews) else None
    )
    cup_kicker = "Today’s cup"
    for brew in recent_brews:
        brew["active"] = brew["id"] == digest_id
        if brew["active"]:
            cup_kicker = str(brew["cup_kicker"])
    tea_notes = daily_tea_deck(date.fromisoformat(digest_id))
    if digest_id == today_id:
        sections, current_vote = _load_today(digest_id)
    else:
        sections, current_vote = _load_today(
            digest_id, include_disabled_sections=True
        )
    brewed = bool(sections) or _digest_exists(digest_id)
    shown_section_keys = {str(section.get("key") or "") for section in sections}
    empty_enabled_sections = []
    if digest_id == today_id and brewed:
        for section_key in ("opportunities", "events"):
            if section_key in shown_section_keys or not section_enabled(
                get_settings(), section_key
            ):
                continue
            section_meta = SECTION_META[section_key]
            empty_enabled_sections.append(
                {
                    "key": section_key,
                    "title": section_meta["title"],
                    "emoji": section_meta["emoji"],
                }
            )
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
    overflow_signature = _overflow_signature(digest_id, overflow_audit)
    # `load_digest_audit` yields {} for a malformed row, so every field here is
    # normalized before it reaches the template. Without this a single bad row
    # 500s the entire digest page, not just the shelf.
    overflow_audit = [
        {
            **audit,
            "url": safe_url(str(audit.get("url") or "")),
            "title": str(audit.get("title") or "Untitled"),
            "source": str(audit.get("source") or ""),
            "section": str(audit.get("section") or ""),
            "topic_score": _as_float(audit.get("topic_score")),
            "item_id": _as_int(audit.get("item_id")),
        }
        for audit in overflow_audit
    ]
    overflow_audit = [a for a in overflow_audit if a["item_id"] is not None]
    _overflow_ids = {int(audit["item_id"]) for audit in overflow_audit}
    overflow_saved = bool(_overflow_ids) and _overflow_ids <= carryover_item_ids()
    from .reader_insights import usefulness_summary

    feedback_summary = usefulness_summary([entry for section in sections for entry in section.get('entries', [])])
    response = templates.TemplateResponse(
        request,
        "digest_web.html.j2",
        {
            "digest_id": digest_id,
            "viewing_today": digest_id == today_id,
            "requested_digest_missing": requested_missing,
            "today_label": today_date.strftime("%b %d").replace(" 0", " "),
            "cup_kicker": cup_kicker,
            "profile_name": _profile_name(),
            "salutation": "Welcome back",
            "recent_brews": recent_brews,
            "previous_brew": previous_brew,
            "latest_saved_brew": next(
                (brew for brew in recent_brews if brew["id"] != today_id), None
            ),
            "overflow_signature": overflow_signature,
            "feedback_warning": _feedback_reasons()[1],
            "daily_note": tea_notes[0],
            "daily_notes": tea_notes,
            "summarizer_label": _summarizer_label(digest_id),
            "sections": sections,
            "coming_deadlines": _tracked_deadlines(sections, today_date),
            "empty_enabled_sections": empty_enabled_sections,
            "overview": overview,
            "source_coverage": health.coverage_summary(),
            "interest_coverage": _interest_coverage(sections, load_digest_audit(digest_id, "interest_coverage") if brewed else []),
            "top_journal_audit": top_journal_audit,
            "learned_profile_note": _learned_profile_note(digest_id) if brewed else "",
            "overflow_audit": overflow_audit,
            "overflow_saved": overflow_saved,
            "candidate_funnel": candidate_funnel,
            "catch_up_synthesis": catch_up_synthesis,
            "catch_up_days": catch_up_days,
            "current_vote_per_item": current_vote,
            "ranking_status": _feedback_status(),
            "feedback_summary": feedback_summary,
            "empty": len(sections) == 0,
            "brewed": brewed,
            "csrf_token": _CSRF_TOKEN,
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/reading/study", response_class=HTMLResponse)
def reading_study(request: Request) -> Response:
    from .reader_insights import recent_usefulness
    response = templates.TemplateResponse(request, 'reading_study.html.j2', {'days': recent_usefulness()})
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.get("/saved", response_class=HTMLResponse)
def saved_items(request: Request, q: str = "", page: int = 1) -> Response:
    if not _profile_exists():
        return RedirectResponse(url="/setup", status_code=302)
    query = str(q or "").strip()[:200]
    page_size = 50
    # Group the whole matching library first so the count and the pages are
    # unique works: a saved preprint and its saved journal version are one card.
    saved_rows: list[Any] = []
    while True:
        batch = search_bookmarks(query, limit=500, offset=len(saved_rows))
        saved_rows.extend(batch)
        if len(batch) < 500:
            break
    saved_by_id = {int(row.item_id): row for row in saved_rows}
    reasons, warning = _feedback_reasons()
    with session_scope() as session:
        item_rows = {int(row.id): row for row in session.execute(
            select(ItemRow).where(ItemRow.id.in_(list(saved_by_id)))).scalars()} if saved_by_id else {}
        ratings: dict[int, int] = {}
        for iid, value, grade in session.execute(
            select(VoteRow.item_id, VoteRow.value, VoteRow.grade)
            .where(VoteRow.item_id.in_(list(saved_by_id)))
            .order_by(VoteRow.created_at.desc(), VoteRow.id.desc())
        ):
            ratings.setdefault(int(iid), grade if grade is not None else votes_mod.value_to_grade(value))
        for row in item_rows.values():
            session.expunge(row)
    from .dedupe import group_publication_versions

    ordered = [item_rows[iid] for iid in saved_by_id if iid in item_rows]
    groups = group_publication_versions(ordered, ratings=ratings, saved=saved_by_id, known=())
    total = len(groups)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(int(page), pages))
    page_groups = groups[(page - 1) * page_size:page * page_size]
    rows = [saved_by_id[int(group.primary.id)] for group in page_groups]
    companions = load_publication_companions(rows)
    page_ids = {int(item_id) for group in page_groups for item_id in group.item_ids}
    recorded: dict[int, list[dict]] = {}
    if page_ids:
        from .store import DigestItemFeatureRow

        with session_scope() as session:
            for item_id, raw in session.execute(
                select(DigestItemFeatureRow.item_id, DigestItemFeatureRow.features_json)
                .where(DigestItemFeatureRow.item_id.in_(list(page_ids)))
                .order_by(DigestItemFeatureRow.digest_id.desc())
            ):
                try:
                    payload = json.loads(raw or "{}")
                except json.JSONDecodeError:
                    continue
                for link_item_id, links in _recorded_version_links({int(item_id): payload}).items():
                    recorded.setdefault(link_item_id, links)
    if recorded:
        # Brew-time links name versions that were not saved; attach them to the
        # card without counting them as saved items.
        regrouped = group_publication_versions(
            [item_rows[iid] for group in page_groups for iid in group.item_ids],
            ratings=ratings, saved=saved_by_id, known=(), recorded_links=recorded)
        # Same rows and relations, so the same works; keep the page if not.
        if [group.item_ids for group in regrouped] == [group.item_ids for group in page_groups]:
            page_groups = regrouped
    try:
        today: date | None = date.fromisoformat(_digest_id())
    except ValueError:
        today = None
    items = [_apply_work_group({
        "id": row.item_id, "label": row.title, "title": row.title, "url": safe_url(row.url),
        "source": _display_source_name(display_source(row)), "figure": display_figure(row.metadata),
        "publication_label": (
            _display_publication_label(publication_label(row)) if row.section == "research" else ""
        ),
        "related_commentaries": [
            {**child, "url": safe_url(child.get("url", ""))}
            for child in companions.get(int(row.item_id), [])
        ],
        "section": SECTION_META.get(row.section, {"title": row.section.title() or "Other"})["title"],
        "summary": (display_summary := _display_summary(row.summary, row.section, row)),
        "summary_fields": _summary_fields(display_summary),
        "finding_label": _finding_label(row.section, publication_label(row)),
        "published": _saved_date(row.published_at),
        "published_label": _friendly_date(row.published_at, today)["label"],
        "published_iso": _friendly_date(row.published_at, today)["iso"],
        "saved": _saved_date(row.saved_at), "current_grade": ratings.get(row.item_id),
        "current_reasons": reasons.get(str(row.item_id), []),
    }, group) for row, group in zip(rows, page_groups, strict=True)]
    from urllib.parse import urlencode
    def page_url(number):
        return "/saved?" + urlencode({"page": number, **({"q": query} if query else {})})
    tracked: list[dict] = []
    if not query and page == 1:
        try:
            tracked_profile = load_opportunity_profile(get_settings().opportunity_profile_path)
        except Exception:  # noqa: BLE001 - eligibility then stays "check the official page"
            tracked_profile = None
        tracked = _tracked_items(tracked_profile)
    response = templates.TemplateResponse(request, "saved.html.j2", {
        "items": items, "query": query, "csrf_token": _CSRF_TOKEN, "tracked": tracked,
        "total": total, "page": page, "pages": pages, "feedback_warning": warning,
        "previous_page": page_url(page - 1) if page > 1 else "",
        "next_page": page_url(page + 1) if page < pages else "",
    })
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/vote/{item_id}/{grade}")
@_feedback_mutation
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
            "ranking_status": _feedback_status(),
        }
    )


@app.delete("/vote/{item_id}")
@_feedback_mutation
def vote_clear(request: Request, item_id: int) -> JSONResponse:
    _require_csrf(request)
    with _compute_write_guard():
        if not votes_mod.clear_rating_by_id(item_id):
            raise HTTPException(status_code=404, detail="Item not found.")
    return JSONResponse({"ok": True, "item_id": item_id, "new_grade": None,
                         "ranking_status": _feedback_status()})


def _set_flag_on_versions(setter, item_id: int, versions: str, value: bool) -> list[int]:
    """Set one flag on a card's own ID and, when named, its other versions.

    Each flag stays on its original item ID; ``versions`` only lets one card
    clear (or restore) the flags its work already carries on other versions.
    """
    others = _parse_version_ids(versions, item_id)
    if not setter(item_id, value):
        raise HTTPException(status_code=404, detail=f"item {item_id} not found")
    return [item_id, *(other for other in others if setter(other, value))]


def _versions_payload(changed: list[int]) -> dict[str, list[int]]:
    return {"item_ids": changed} if len(changed) > 1 else {}


@app.post("/bookmark/{item_id}")
def bookmark_add(request: Request, item_id: int, versions: str = "") -> JSONResponse:
    _require_csrf(request)
    changed = _set_flag_on_versions(set_bookmark, item_id, versions, True)
    return JSONResponse({"ok": True, "item_id": item_id, "saved": True, **_versions_payload(changed)})


@app.delete("/bookmark/{item_id}")
def bookmark_remove(request: Request, item_id: int, versions: str = "") -> JSONResponse:
    _require_csrf(request)
    changed = _set_flag_on_versions(set_bookmark, item_id, versions, False)
    return JSONResponse({"ok": True, "item_id": item_id, "saved": False, **_versions_payload(changed)})


@app.post("/known/{item_id}")
def known_add(request: Request, item_id: int, versions: str = "") -> JSONResponse:
    """Flag an item as already known so future digests skip it."""
    _require_csrf(request)
    changed = _set_flag_on_versions(set_item_known, item_id, versions, True)
    return JSONResponse({"ok": True, "item_id": item_id, "known": True, **_versions_payload(changed)})


@app.post("/interested/{item_id}")
def interested_add(request: Request, item_id: int, versions: str = "") -> JSONResponse:
    """Track an item's deadline in Coming deadlines and on the Saved page."""
    _require_csrf(request)
    changed = _set_flag_on_versions(set_item_interested, item_id, versions, True)
    return JSONResponse({"ok": True, "item_id": item_id, "interested": True, **_versions_payload(changed)})


@app.delete("/interested/{item_id}")
def interested_remove(request: Request, item_id: int, versions: str = "") -> JSONResponse:
    _require_csrf(request)
    changed = _set_flag_on_versions(set_item_interested, item_id, versions, False)
    return JSONResponse({"ok": True, "item_id": item_id, "interested": False, **_versions_payload(changed)})


# Display choices each page also keeps in this browser's localStorage. The file
# copy travels with backups and restores, so a new device starts the same way.
_UI_PREFS = {"theme": {"light", "dark"}, "display": {"comfortable", "compact"},
             "pip": {"visible", "hidden"}}


def _ui_prefs_path() -> Path:
    return Path(get_settings().db_path).parent / "ui_prefs.json"


def _load_ui_prefs() -> dict[str, str]:
    try:
        stored = json.loads(_ui_prefs_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    return {key: value for key, value in stored.items()
            if isinstance(value, str) and value in _UI_PREFS.get(key, ())}


@app.post("/ui-prefs")
def ui_prefs_save(request: Request, theme: str = "", display: str = "", pip: str = "") -> JSONResponse:
    """Remember display choices beside the reader's other data."""
    _require_csrf(request)
    chosen = {key: value for key, value in (("theme", theme), ("display", display), ("pip", pip)) if value}
    if any(value not in _UI_PREFS[key] for key, value in chosen.items()):
        raise HTTPException(status_code=422, detail="unknown display choice")
    prefs = _load_ui_prefs()
    if chosen and {**prefs, **chosen} != prefs:
        prefs.update(chosen)
        _atomic_private_write(_ui_prefs_path(), json.dumps(prefs, sort_keys=True) + "\n")
    return JSONResponse(prefs)


@app.delete("/known/{item_id}")
def known_remove(request: Request, item_id: int, versions: str = "") -> JSONResponse:
    _require_csrf(request)
    changed = _set_flag_on_versions(set_item_known, item_id, versions, False)
    return JSONResponse({"ok": True, "item_id": item_id, "known": False, **_versions_payload(changed)})


@app.post("/overflow/save")
def overflow_save(request: Request, digest_id: str = "", audit_signature: str = "") -> JSONResponse:
    """Pin the displayed overflow for evaluation after the current local day."""
    _require_csrf(request)
    with _compute_write_guard():
        audits = _validated_overflow(digest_id, audit_signature)
        ids = [iid for audit in audits if (iid := _as_int(audit.get("item_id"))) is not None]
        if not ids:
            raise HTTPException(status_code=404, detail="No overflow picks in this saved digest.")
        added = add_carryover_items(ids, pinned_digest_id=_digest_id())
    return JSONResponse({"ok": True, "digest_id": digest_id, "saved": len(ids), "added": added})


@app.post("/overflow/save/{item_id}")
def overflow_save_one(request: Request, item_id: int, digest_id: str = "", audit_signature: str = "") -> JSONResponse:
    _require_csrf(request)
    with _compute_write_guard():
        audits = _validated_overflow(digest_id, audit_signature)
        if item_id not in {_as_int(audit.get("item_id")) for audit in audits}:
            raise HTTPException(status_code=404, detail="This item is not in the displayed overflow.")
        added = add_carryover_items([item_id], pinned_digest_id=_digest_id())
    return JSONResponse({"ok": True, "item_id": item_id, "added": added})


@app.post("/vote/{item_id}/reason/{reason}")
@_feedback_mutation
def vote_reason(request: Request, item_id: int, reason: str) -> JSONResponse:
    _require_csrf(request)
    ok = votes_mod.record_vote_reason(item_id, reason)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="choose Less relevant or Not relevant before adding a reason",
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
@_feedback_mutation
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
            "status": _feedback_status(),
            "training_job": training_job,
        }
    )


@app.post("/ranking/train")
def ranking_train(request: Request) -> JSONResponse:
    _require_csrf(request)
    status = _feedback_status()
    if status.get("blocked"):
        raise HTTPException(status_code=409, detail=status["message"])
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
                status_after_error = _feedback_status()
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


def _local_jobs_busy() -> bool:
    return _BREW_LOCK.locked() or _COMPUTE_LOCK.locked()


@app.get("/update/status")
def update_status(request: Request) -> JSONResponse:
    _require_local_origin(request)
    updates.UPDATER.maybe_check(get_settings().check_for_updates)
    return JSONResponse(updates.UPDATER.status(busy=_local_jobs_busy()),
                        headers={"Cache-Control": "no-store"})


@app.post("/update/apply")
def update_apply(request: Request) -> JSONResponse:
    _require_csrf(request)
    # Hold both job locks from here until the process restarts, so no brew or
    # ranking job starts against half-replaced program files.
    brew = _BREW_LOCK.acquire(blocking=False)
    compute = brew and _COMPUTE_LOCK.acquire(blocking=False)

    def release() -> None:
        if compute:
            _COMPUTE_LOCK.release()
        if brew:
            _BREW_LOCK.release()

    address = (request.url.hostname or "127.0.0.1", request.url.port or 8765)
    state = updates.UPDATER.apply(busy=not compute, address=address, on_failure=release)
    return JSONResponse(state, headers={"Cache-Control": "no-store"})


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


@app.post("/setup/check-wording")
def setup_check_wording(request: Request, topic: str = Query(min_length=1, max_length=160)) -> Response:
    from .interest_spelling import SpellingUnavailable, suggest_topic_spelling

    _require_csrf(request)
    topic = topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Enter an interest first.")
    try:
        suggestion = suggest_topic_spelling(topic)
    except SpellingUnavailable:
        return JSONResponse(
            {"detail": "Couldn't check wording right now. Your interest is unchanged. Try again later."},
            status_code=503, headers={"Cache-Control": "no-store"},
        )
    return JSONResponse({"topic": topic, "suggestion": suggestion}, headers={"Cache-Control": "no-store"})


@app.post("/setup/test-connection")
async def setup_test_connection(request: Request) -> Response:
    from starlette.concurrency import run_in_threadpool

    from .provider_check import check_api_connection

    raw = await _read_urlencoded_form(request)
    _require_csrf(request, raw)
    backend = raw.get("llm_backend", "")
    base_url = raw.get("llm_base_url", "").strip()
    api_key = raw.get("llm_api_key", "").strip()
    if api_key == "***":
        # A masked saved secret may only be reused at its configured provider.
        api_key = (SETTINGS.llm_api_key or "") if (
            _saved_api_key_matches(backend, base_url)
        ) else ""
    result = await run_in_threadpool(
        check_api_connection, backend=backend, base_url=base_url,
        api_key=api_key, model=raw.get("llm_model", "").strip(),
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.get("/setup", response_class=HTMLResponse)
def setup_get(request: Request) -> Response:
    form = _load_existing_form_defaults()
    response = templates.TemplateResponse(
        request,
        "setup.html.j2",
        {
            "form": form,
            "first_run": not _profile_exists(),
            "errors": [],
            "csrf_token": _CSRF_TOKEN,
            "browser_mode": BROWSER_MODE,
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/setup")
@_settings_write
async def setup_post(request: Request) -> Response:
    raw = await _read_urlencoded_form(request)
    _require_csrf(request, raw)
    form: dict[str, str] = {
        "name": str(raw.get("name", "")),
        "bio": str(raw.get("bio", "")),
        "topics": str(raw.get("topics", "")),
        "topic_details": str(raw.get("topic_details", "")),
        "topic_weight_mode": str(raw.get("topic_weight_mode", "")),
        "downweight": str(raw.get("downweight", "")),
        "llm_backend": str(raw.get("llm_backend", "extractive")),
        "llm_base_url": str(raw.get("llm_base_url", "")),
        "llm_api_key": str(raw.get("llm_api_key", "")),
        "remove_llm_api_key": _bool_form(raw, "remove_llm_api_key", False),
        "llm_model": str(raw.get("llm_model", "")),
        "user_tz": str(raw.get("user_tz", SETTINGS.user_tz)),
        "top_research": str(raw.get("top_research", SETTINGS.top_research)),
        "publication_enrichment": _bool_form(raw, "publication_enrichment", SETTINGS.publication_enrichment),
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

    # Missing controls from older clients preserve existing private values;
    # an explicitly submitted empty field clears that list.
    existing_form = _load_existing_form_defaults()
    for key in ("authors_of_interest", "seed_works", "context_keywords"):
        form[key] = str(raw.get(key, existing_form[key]))

    validation_form = dict(form)
    for key in ("authors_of_interest", "seed_works", "context_keywords"):
        if form[key] == existing_form[key]:
            validation_form[key] = ""
    errors = _validate_setup(validation_form)
    if errors:
        # Never reflect a newly submitted secret into the response body. A
        # previously saved key can remain masked/reusable; a new key must be
        # entered again after the other validation errors are corrected.
        render_form = dict(form)
        display_errors = list(errors)
        submitted_key = render_form.get("llm_api_key", "").strip()
        if submitted_key:
            render_form["llm_api_key"] = "***" if (SETTINGS.llm_api_key and _saved_api_key_matches(
                form["llm_backend"], form["llm_base_url"])) else ""
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
                "first_run": not _profile_exists(),
                "errors": display_errors,
                "csrf_token": _CSRF_TOKEN,
                "browser_mode": BROWSER_MODE,
            },
            status_code=400,
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    topics, _topic_errors = _parse_ranked_topics(form["topics"])
    details, _detail_errors = _parse_topic_details(form.get("topic_details"), topics)
    # Write the private user profile. Keywords drive retrieval; unchanged facet
    # edits preserve any more-specific retrieval phrases already stored, while
    # facets carry relative priority for attribution and selection.
    profile = _profile_data(strict=True)
    existing_canonical = profile.get("canonical_facets") or {}
    existing_by_name = (
        {str(name).casefold(): spec for name, spec in existing_canonical.items()}
        if isinstance(existing_canonical, dict)
        else {}
    )
    canonical_facets: dict[str, dict[str, Any]] = {}
    for topic, weight in topics:
        detail = dict(details.get(topic, {}))
        previous_name = detail.pop("previous_name", topic)
        previous = existing_by_name.get(previous_name.casefold())
        spec = dict(previous) if isinstance(previous, dict) else {}
        if previous_name != topic and isinstance(spec.get("anchors"), list):
            spec["anchors"] = [topic if str(anchor).strip().casefold() == previous_name.casefold()
                               else anchor for anchor in spec["anchors"]]
        if not spec.get("anchors"):
            spec["anchors"] = [topic]
        if topic in details:
            spec.update(detail)
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
            **{key: _parse_advanced_interest_list(form[key])
               for key in ("authors_of_interest", "seed_works", "context_keywords")
               if key in raw and form[key] != existing_form[key]},
        }
    )
    opportunity_payload = None
    if (
        form["include_opportunities"] == "true"
        or form["include_events"] == "true"
    ):
        opportunity_profile = _opportunity_profile_from_form(form)
        opportunity_path = _get_opportunity_profile_path()
        # Start from the stored file, so a Settings save never erases what the
        # form doesn't show: values the simplified form no longer asks for, or
        # fields a newer version added before the reader went back to this one.
        existing = _profile_data(path=opportunity_path, strict=True)
        opportunity_payload = {
            **existing,
            **opportunity_profile.model_dump(
                exclude={
                    "description",
                    "citizenship_or_residency",
                    "requires_travel_support",
                }
            ),
        }

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
        "PUBLICATION_ENRICHMENT": form["publication_enrichment"],
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
    # Journal exact old/new bytes before touching any member. Startup recovery
    # runs before .env parsing, so an interrupted group is never used silently.
    from .settings_transaction import SettingsRecoveryError, settings_transaction

    profile_path = _get_profile_path()
    try:
        changes = {
            profile_path: yaml.safe_dump(profile, sort_keys=False).encode(),
            _ENV_PATH: _updated_env_content(_ENV_PATH, env_updates).encode(),
        }
        if opportunity_payload is not None:
            changes[opportunity_path] = yaml.safe_dump(opportunity_payload, sort_keys=False).encode()
        # Native/browser journals live beside root .env. An explicit writable
        # override (Compose) stores the journal in its persistent directory,
        # while targets still use app-relative names, including custom profiles.
        transaction_root = _REPO_ROOT if os.environ.get("DD_ENV_PATH") else _ENV_PATH.parent
        with settings_transaction(transaction_root, changes, writer=_atomic_private_write,
                                  journal_root=_ENV_PATH.parent):
            _write_private_yaml(profile_path, profile)
            if opportunity_payload is not None:
                _write_private_yaml(opportunity_path, opportunity_payload)
            _write_env_file(_ENV_PATH, env_updates)
    except SettingsRecoveryError as exc:
        raise HTTPException(status_code=500, detail=(
            "Settings could not be saved, and some configuration files could not be restored. "
            "Configuration loading is blocked until the recovery journal is resolved."
        )) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Settings could not be saved. Previous settings were restored; please try again.") from exc
    # Propagate changes to os.environ so pydantic-settings picks them up,
    # then refresh the in-memory SETTINGS singleton.
    for k, v in env_updates.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    reload_settings()
    globals()["SETTINGS"] = get_settings()

    after_save = str(raw.get("after_save", ""))
    destination = "/" if after_save == "save" else "/run"
    if after_save == "brew":
        destination = _authorized_run_url("usual")
    response = RedirectResponse(url=destination, status_code=303)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/profile/name")
@_settings_write
async def profile_name_post(request: Request) -> Response:
    raw = await _read_urlencoded_form(request)
    _require_csrf(request, raw)
    name = str(raw.get("name", "")).strip()
    if name:
        data = _profile_data(strict=True)
        if not data:
            data = {"bio": "General reader.", "keywords": [], "downweight": []}
        data["name"] = name
        _write_private_yaml(_get_profile_path(), data)
    response = RedirectResponse(url="/", status_code=303)
    response.headers["Cache-Control"] = "no-store"
    return response


# --- Run / brewing flow -----------------------------------------------------


def _ensure_run(run_id: str) -> list[dict[str, Any]]:
    """Get or create the retained progress-event log for a run."""
    with _RUN_LOCK:
        events = _RUN_QUEUES.get(run_id)
        if events is None:
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
                _RUN_CANCEL.pop(stale_id, None)
                _RUN_PUBLISHED.discard(stale_id)
            events = []
            _RUN_QUEUES[run_id] = events
        return events


def _kick_off_run(run_id: str, reading_mode: str) -> None:
    """Run pipeline.run_all in a background thread; always emits a terminal event."""
    with _RUN_LOCK:
        cancel_event = _RUN_CANCEL.setdefault(run_id, threading.Event())

    def _push(evt: dict[str, Any]) -> None:
        with _RUN_LOCK:
            events = _RUN_QUEUES.get(run_id)
            if events is not None:
                events.append(evt)

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
            if stage in ("done", "error", "cancelled"):
                terminal_sent = True
            if stage == "published":
                with _RUN_LOCK:
                    _RUN_PUBLISHED.add(run_id)
            _push({"stage": stage, "payload": payload})

        try:
            process_lock = acquire_compute_lock(SETTINGS.db_path)
            run_all(
                dry_run=True,
                progress_callback=cb,
                reading_mode=reading_mode,
                should_cancel=cancel_event.is_set,
            )
        except BrewCancelled as e:
            _push({"stage": "cancelled", "payload": {"message": str(e)}})
            terminal_sent = True
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


def _authorized_run_url(reading_mode: str) -> str:
    """Authorize one automatic start only after a CSRF-checked reader action."""
    token = secrets.token_urlsafe(24)
    now = monotonic()
    with _RUN_LOCK:
        for key, (_, expires) in list(_RUN_START_PERMITS.items()):
            if expires <= now:
                del _RUN_START_PERMITS[key]
        while len(_RUN_START_PERMITS) >= _MAX_RETAINED_RUNS:
            del _RUN_START_PERMITS[next(iter(_RUN_START_PERMITS))]
        _RUN_START_PERMITS[token] = (reading_mode, now + 300)
    return f"/run?reading_mode={reading_mode}&autostart=1&start={token}"


def _consume_run_start(token: str, reading_mode: str) -> bool:
    with _RUN_LOCK:
        permit = _RUN_START_PERMITS.get(token)
        if permit is None:
            return False
        mode, expires = permit
        if expires <= monotonic():
            del _RUN_START_PERMITS[token]
            return False
        if mode != reading_mode:
            return False
        del _RUN_START_PERMITS[token]
        return True


@app.post("/run")
async def run_prepare(request: Request) -> Response:
    raw = await _read_urlencoded_form(request)
    _require_csrf(request, raw)
    if not _profile_exists():
        return RedirectResponse(url="/setup", status_code=303)
    try:
        mode = normalize_reading_mode(raw.get("reading_mode", "usual"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = RedirectResponse(url=_authorized_run_url(mode), status_code=303)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/run", response_class=HTMLResponse)
def run_get(
    request: Request,
    reading_mode: str = "usual",
    autostart: bool = False,
    run_id: str = "",
    resume: bool = False,
    start: str = "",
) -> Response:
    if not _profile_exists():
        return RedirectResponse(url="/setup", status_code=302)
    try:
        selected_mode = normalize_reading_mode(reading_mode)
    except ValueError:
        selected_mode = "usual"
    retained_identity = bool(_RUN_ID_RE.fullmatch(run_id))
    # Old refresh URLs used autostart=1 with a run id. They are recovery links,
    # even after a restart loses the retained in-memory status.
    resume = bool(resume or (autostart and retained_identity))
    if not retained_identity:
        run_id = uuid.uuid4().hex[:12]
    authorized_start = bool(autostart and not resume and _consume_run_start(start, selected_mode))
    latest = _recent_digest_ids(_digest_id(), limit=1)
    response = templates.TemplateResponse(
        request,
        "run.html.j2",
        {
            "run_id": run_id,
            "csrf_token": _CSRF_TOKEN,
            "reading_mode": selected_mode,
            "autostart": authorized_start,
            "browser_mode": BROWSER_MODE,
            "has_saved_digest": bool(latest),
            "resume": bool(resume),
            "resume_url": (
                f"/run?reading_mode={selected_mode}&resume=1&run_id={run_id}"
            ),
            "restart_url": f"/run?reading_mode={selected_mode}",
            "saved_reading_url": (
                f"/?digest_id={latest[0]}" if latest else "/"
            ),
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/run/start")
async def run_start(request: Request) -> JSONResponse:
    _require_csrf(request)
    try:
        body = await request.json() if await request.body() else {}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
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


@app.post("/run/cancel")
def run_cancel(request: Request, run_id: str) -> JSONResponse:
    _require_csrf(request)
    with _RUN_LOCK:
        event = _RUN_CANCEL.get(run_id)
        events = _RUN_QUEUES.get(run_id, [])
        if event is None:
            raise HTTPException(status_code=404, detail="Brew run not found. Reconnect to check its status.")
        if run_id in _RUN_PUBLISHED or any(
            row.get("stage") in {"published", "done", "error", "cancelled"} for row in events
        ):
            raise HTTPException(status_code=409, detail="This brew is already published or finished.")
        event.set()
    return JSONResponse({"ok": True, "pending": True,
                         "message": "Stopping at the next safe point before publication."})


@app.get("/run/stream")
async def run_stream(request: Request, run_id: str) -> StreamingResponse:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="invalid run id")
    with _RUN_LOCK:
        events = _RUN_QUEUES.get(run_id)
    if events is None:
        raise HTTPException(
            status_code=404,
            detail="brew run not found; it may have ended or the server restarted",
        )

    async def event_gen():
        yield f"data: {json.dumps({'stage': 'connected', 'run_id': run_id})}\n\n"
        terminal = {"done", "error", "cancelled"}
        cursor = 0
        while True:
            with _RUN_LOCK:
                batch = list(events[cursor:])
            if not batch:
                await asyncio.sleep(1.0)
                yield ": heartbeat\n\n"
                continue
            for evt in batch:
                cursor += 1
                yield f"data: {json.dumps(evt)}\n\n"
                if evt.get("stage") in terminal:
                    return

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/done", response_class=HTMLResponse)
def done(request: Request, digest_id: str = "", n: int = 0, preview_warning: bool = False) -> Response:
    if not digest_id:
        digest_id = _digest_id()
    if not n:
        # Best-effort lookup if the URL didn't carry the count.
        sections, _ = _load_today(digest_id)
        n = sum(len(s["entries"]) for s in sections)
    response = templates.TemplateResponse(
        request,
        "done.html.j2",
        {"digest_id": digest_id, "total_items": n, "preview_warning": preview_warning},
    )
    response.headers["Cache-Control"] = "no-store"
    return response
