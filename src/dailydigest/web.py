"""Local FastAPI UI for browsing today's digest, voting, and onboarding setup.

This is an *alternative* delivery surface; the email path is unchanged.
The app binds to 127.0.0.1 by default — no remote access, no auth.

Routes:
- ``GET /``                — render today's digest (or redirect to /setup if no profile)
- ``POST /vote/{id}/{v}``  — record a +1/0/-1 vote for an item
- ``POST /vote/{id}/reason/{reason}`` — record qualitative feedback reason
- ``DELETE /vote/{id}/reason/{reason}`` — remove qualitative feedback reason
- ``GET /ranking/status``  — return vote counts and LR ranker status
- ``POST /ranking/train``  — train the local LR ranker when enough votes exist
- ``POST /refresh``        — kick off a dry-run pipeline in the background
- ``GET /healthz``         — liveness probe
- ``GET /setup``           — onboarding wizard (bio, keywords, LLM backend)
- ``POST /setup``          — write local profile YAML and .env, redirect to /run
- ``GET /run``             — brewing page with live SSE progress feed
- ``POST /run/start``      — kick off pipeline.run_all in a background thread
- ``GET /run/stream``      — Server-Sent Events stream of pipeline progress
- ``GET /done``            — celebration page after a successful brew
"""

from __future__ import annotations

import asyncio
from collections import Counter
import hmac
import json
import logging
import os
import queue as std_queue
import re
import secrets
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

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
from sqlalchemy import select

from . import health, votes as votes_mod
from .config import SETTINGS, get_settings, reload_settings
from .email_render import SECTION_META, SECTION_ORDER, safe_url
from .pipeline import _digest_id, run_all
from .rank.source_quality import display_breakdown, source_bucket
from .store import (
    DigestRow,
    ItemRow,
    VoteRow,
    init_db,
    load_digest_audit,
    load_digest_features,
    session_scope,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_DIR = _REPO_ROOT / "templates"
_ENV_PATH = _REPO_ROOT / ".env"


def _get_profile_path() -> Path:
    from .config import get_settings
    candidate = (_REPO_ROOT / get_settings().profile_path).resolve()
    try:
        candidate.relative_to(_REPO_ROOT)
    except ValueError:
        raise ValueError(f"PROFILE_PATH escapes repo root: {candidate}")
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
_RUN_LOCK = threading.Lock()
_TRAIN_LOCK = threading.Lock()
_TRAIN_JOB: dict[str, Any] = {"running": False, "last_result": None}
_BREW_LOCK = threading.Lock()
_BREW_JOB: dict[str, Any] = {"running": False, "run_id": None}
_CSRF_TOKEN = secrets.token_urlsafe(32)
_ENV_SAFE_RE = re.compile(r"^[A-Za-z0-9_./:@+-]*$")


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


def _ranking_phrase(value: str | None) -> str:
    text = " ".join(str(value or "").replace("_", " ").split()).rstrip(".")
    if not text:
        return ""
    if text[:1].islower() and any(ch.isupper() for ch in text[1:]):
        return text
    return text[:1].upper() + text[1:]


def _ranking_reason(value: str | None) -> str:
    return " ".join(str(value or "").replace("_", " ").split()).rstrip(".")


def _ranking_sentence(ranking: dict) -> str:
    signals: list[str] = []
    for raw in list(ranking.get("tags") or []) + list(ranking.get("why_shown") or []):
        phrase = _ranking_phrase(raw)
        if phrase and phrase not in signals:
            signals.append(phrase)
        if len(signals) == 2:
            break

    reason = _ranking_reason(str(ranking.get("selection_reason") or ""))
    if signals and reason:
        return f"Ranked for {', '.join(signals)}; selected via {reason}."
    if signals:
        return f"Ranked for {', '.join(signals)}."
    if reason:
        return f"Selected via {reason}."
    return "Ranked for profile fit and source signals."


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
    latest = health.latest_snapshot()
    scanned = sum(int(row.get("items") or 0) for row in latest)
    return {
        "selected": len(entries),
        "scanned": scanned,
        "sources": len(latest),
        "failures": sum(1 for row in latest if not row.get("ok", True)),
        "section_mix": [
            {"title": s["title"], "count": len(s["entries"])}
            for s in sections
            if s.get("entries")
        ],
        "source_mix": source_mix,
        "must_read": sorted(
            [entry for entry in entries if _entry_confidence(entry) >= 0.65],
            key=_entry_confidence,
            reverse=True,
        )[:5],
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

        item_ids = [r.id for r in rows]
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
            if key not in by_section:
                by_section[key] = []
                seen_keys.append(key)
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
                    "ranking_sentence": _ranking_sentence(ranking),
                    "current_vote": current_vote.get(int(row.id)),
                    "current_grade": current_grade.get(int(row.id)),
                    "current_reasons": current_reasons.get(int(row.id), []),
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


def _detect_clis() -> dict[str, bool]:
    return {
        "claude": shutil.which("claude") is not None,
        "codex": shutil.which("codex") is not None,
    }


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def _load_existing_form_defaults() -> dict[str, str]:
    """Pre-populate the form from any existing profile.yaml + .env values."""
    out: dict[str, str] = {
        "name": "",
        "bio": "",
        "keywords": "",
        "downweight": "",
        "llm_backend": SETTINGS.llm_backend or "extractive",
        "llm_base_url": SETTINGS.llm_base_url,
        "llm_api_key": "***" if SETTINGS.llm_api_key else "",
        "llm_model": SETTINGS.llm_model,
        "claude_cli_model": "",
        "codex_cli_model": "",
        "top_research": str(SETTINGS.top_research),
        "top_industry": str(SETTINGS.top_industry),
        "top_regulatory": str(SETTINGS.top_regulatory),
        "top_world": str(SETTINGS.top_world),
    }
    if _get_profile_path().exists():
        try:
            data = yaml.safe_load(_get_profile_path().read_text()) or {}
            out["name"] = str(data.get("name") or "").strip()
            out["bio"] = (data.get("bio") or "").strip()
            out["keywords"] = ", ".join(data.get("keywords") or [])
            out["downweight"] = ", ".join(data.get("downweight") or [])
        except Exception as e:  # noqa: BLE001
            logger.warning("could not parse existing profile.yaml: %s", e)
    # LLM_CLI_MODEL is read by the config agent; we just preserve whatever the
    # current .env has so the user doesn't lose their pin if they revisit /setup.
    env = _read_env_file(_ENV_PATH)
    cli_model = env.get("LLM_CLI_MODEL", "")
    if out["llm_backend"] == "claude_code":
        out["claude_cli_model"] = cli_model
    elif out["llm_backend"] == "codex":
        out["codex_cli_model"] = cli_model
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

    path.write_text("\n".join(new_lines) + "\n")


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
    if backend not in ("extractive", "claude_code", "codex", "api"):
        errors.append(f"Unknown backend: {backend}")
    bio = (form.get("bio") or "").strip()
    keywords = _parse_csv(form.get("keywords"))
    if not bio and not keywords:
        errors.append("Please provide either a bio or at least one keyword.")
    if backend == "api":
        if not (form.get("llm_base_url") or "").strip():
            errors.append("API backend requires a base URL.")
        if not (form.get("llm_model") or "").strip():
            errors.append("API backend requires a model name.")
    clis = _detect_clis()
    if backend == "claude_code" and not clis["claude"]:
        errors.append("`claude` CLI not found in PATH. Install Claude Code first.")
    if backend == "codex" and not clis["codex"]:
        errors.append("`codex` CLI not found in PATH. Install Codex first.")
    for key, label in (
        ("llm_base_url", "API base URL"),
        ("llm_api_key", "API key"),
        ("llm_model", "API model"),
        ("claude_cli_model", "Claude model"),
        ("codex_cli_model", "Codex model"),
    ):
        if _env_value_has_control_chars(form.get(key)):
            errors.append(f"{label} cannot contain line breaks.")
    for key, label in (
        ("top_research", "Research items"),
        ("top_industry", "Industry items"),
        ("top_regulatory", "Regulatory items"),
        ("top_world", "World items"),
    ):
        raw = (form.get(key) or "").strip()
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if value < 0 or value > 30:
            errors.append(f"{label} must be between 0 and 30.")
    return errors


def _int_form(form: dict[str, str], key: str, default: int) -> str:
    try:
        return str(int((form.get(key) or "").strip()))
    except ValueError:
        return str(default)


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    if not _profile_exists():
        return RedirectResponse(url="/setup", status_code=302)
    digest_id = _digest_id()
    sections, current_vote = _load_today(digest_id)
    brewed = bool(sections) or _digest_exists(digest_id)
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
    response = templates.TemplateResponse(
        request,
        "digest_web.html.j2",
        {
            "digest_id": digest_id,
            "profile_name": _profile_name(),
            "sections": sections,
            "overview": overview,
            "top_journal_audit": top_journal_audit,
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
        _TRAIN_JOB["running"] = True
        _TRAIN_JOB["last_result"] = None

    def _target() -> None:
        try:
            result = votes_mod.train_lr_ranker()
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
        with _TRAIN_LOCK:
            _TRAIN_JOB["running"] = False
            _TRAIN_JOB["last_result"] = result

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
    try:
        with _RUN_LOCK:
            _BREW_JOB["running"] = True
            _BREW_JOB["run_id"] = "refresh"
        try:
            run_all(dry_run=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("background refresh failed: %s", e)
    finally:
        with _RUN_LOCK:
            _BREW_JOB["running"] = False
            _BREW_JOB["run_id"] = None
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
    threading.Thread(target=_run_pipeline_dry_run, daemon=True).start()
    return JSONResponse({"ok": True, "digest_id": digest_id, "running": True})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


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
            "cli_status": _detect_clis(),
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
        "keywords": str(raw.get("keywords", "")),
        "downweight": str(raw.get("downweight", "")),
        "llm_backend": str(raw.get("llm_backend", "extractive")),
        "llm_base_url": str(raw.get("llm_base_url", "")),
        "llm_api_key": str(raw.get("llm_api_key", "")),
        "llm_model": str(raw.get("llm_model", "")),
        "claude_cli_model": str(raw.get("claude_cli_model", "")),
        "codex_cli_model": str(raw.get("codex_cli_model", "")),
        "top_research": str(raw.get("top_research", SETTINGS.top_research)),
        "top_industry": str(raw.get("top_industry", SETTINGS.top_industry)),
        "top_regulatory": str(raw.get("top_regulatory", SETTINGS.top_regulatory)),
        "top_world": str(raw.get("top_world", SETTINGS.top_world)),
    }

    errors = _validate_setup(form)
    if errors:
        response = templates.TemplateResponse(
            request,
            "setup.html.j2",
            {
                "form": form,
                "errors": errors,
                "cli_status": _detect_clis(),
                "csrf_token": _CSRF_TOKEN,
            },
            status_code=400,
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    # Write profile.yaml.
    profile = {
        "name": form["name"].strip(),
        "bio": form["bio"].strip() or "General reader.",
        "keywords": _parse_csv(form["keywords"]),
        "downweight": _parse_csv(form["downweight"]),
    }
    _get_profile_path().parent.mkdir(parents=True, exist_ok=True)
    _get_profile_path().write_text(yaml.safe_dump(profile, sort_keys=False))

    # Write/update .env.
    backend = form["llm_backend"]
    cli_model = ""
    if backend == "claude_code":
        cli_model = form["claude_cli_model"].strip()
    elif backend == "codex":
        cli_model = form["codex_cli_model"].strip()

    env_updates: dict[str, str] = {
        "LLM_BACKEND": backend,
        "LLM_BASE_URL": form["llm_base_url"].strip()
        or (SETTINGS.llm_base_url or "https://api.openai.com/v1"),
        "LLM_MODEL": form["llm_model"].strip() or "gpt-4o-mini",
        # NOTE: LLM_CLI_MODEL is the new env var owned by the config agent;
        # we write it now so when that agent lands their config field, the
        # value is already on disk.
        "LLM_CLI_MODEL": cli_model,
        "TOP_RESEARCH": _int_form(form, "top_research", SETTINGS.top_research),
        "TOP_INDUSTRY": _int_form(form, "top_industry", SETTINGS.top_industry),
        "TOP_REGULATORY": _int_form(form, "top_regulatory", SETTINGS.top_regulatory),
        "TOP_WORLD": _int_form(form, "top_world", SETTINGS.top_world),
    }
    # Only update API key if it's not the masked "***" value
    api_key = form.get("llm_api_key", "").strip()
    if api_key and api_key != "***":
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
        _get_profile_path().parent.mkdir(parents=True, exist_ok=True)
        _get_profile_path().write_text(yaml.safe_dump(data, sort_keys=False))
    response = RedirectResponse(url="/", status_code=303)
    response.headers["Cache-Control"] = "no-store"
    return response


# --- Run / brewing flow -----------------------------------------------------


def _ensure_run(run_id: str) -> std_queue.Queue[dict[str, Any]]:
    """Get-or-create the stdlib Queue for a run."""
    with _RUN_LOCK:
        q = _RUN_QUEUES.get(run_id)
        if q is None:
            q = std_queue.Queue()
            _RUN_QUEUES[run_id] = q
        return q


def _kick_off_run(run_id: str) -> None:
    """Run pipeline.run_all in a background thread; always emits a terminal event."""

    def _push(evt: dict[str, Any]) -> None:
        q = _RUN_QUEUES.get(run_id)
        if q is not None:
            q.put_nowait(evt)

    def _target() -> None:
        terminal_sent = False
        acquired = _BREW_LOCK.acquire(blocking=False)
        if not acquired:
            _push(
                {
                    "stage": "error",
                    "payload": {"message": "Another brew is already running. Please wait for it to finish."},
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
            run_all(dry_run=True, progress_callback=cb)
        except Exception as e:  # noqa: BLE001
            logger.exception("pipeline failed in run %s", run_id)
            _push({"stage": "error", "payload": {"message": f"{type(e).__name__}: {e}"}})
            terminal_sent = True
        finally:
            with _RUN_LOCK:
                _BREW_JOB["running"] = False
                _BREW_JOB["run_id"] = None
            _BREW_LOCK.release()
            if not terminal_sent:
                _push({"stage": "done", "payload": {"forced": True}})

    threading.Thread(target=_target, daemon=True).start()


@app.get("/run", response_class=HTMLResponse)
def run_get(request: Request) -> Response:
    if not _profile_exists():
        return RedirectResponse(url="/setup", status_code=302)
    run_id = uuid.uuid4().hex[:12]
    response = templates.TemplateResponse(
        request,
        "run.html.j2",
        {"run_id": run_id, "csrf_token": _CSRF_TOKEN},
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
    _ensure_run(run_id)
    with _RUN_LOCK:
        if run_id in _RUN_STARTED:
            return JSONResponse({"ok": True, "run_id": run_id, "already_started": True})
        _RUN_STARTED.add(run_id)
    _kick_off_run(run_id)
    return JSONResponse({"ok": True, "run_id": run_id})


@app.get("/run/stream")
async def run_stream(run_id: str) -> StreamingResponse:
    q = _ensure_run(run_id)

    async def event_gen():
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
                    break
        finally:
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
