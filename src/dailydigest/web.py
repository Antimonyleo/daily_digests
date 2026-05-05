"""Local FastAPI UI for browsing today's digest, voting, and onboarding setup.

This is an *alternative* delivery surface; the email path is unchanged.
The app binds to 127.0.0.1 by default — no remote access, no auth.

Routes:
- ``GET /``                — render today's digest (or redirect to /setup if no profile)
- ``POST /vote/{id}/{v}``  — record a +1/0/-1 vote for an item
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

from . import votes as votes_mod
from .config import SETTINGS, get_settings, reload_settings
from .email_render import SECTION_META, SECTION_ORDER, safe_url
from .pipeline import _digest_id, run_all
from .store import DigestRow, ItemRow, VoteRow, init_db, session_scope

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_DIR = _REPO_ROOT / "templates"
_PROFILE_PATH = _REPO_ROOT / SETTINGS.profile_path
_ENV_PATH = _REPO_ROOT / ".env"

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
            select(VoteRow.item_id, VoteRow.value).where(VoteRow.item_id.in_(item_ids))
        ).all()
        current_vote = {int(iid): int(val) for iid, val in vote_rows}

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
                    "current_vote": current_vote.get(int(row.id)),
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
    return _PROFILE_PATH.exists()


def _profile_data() -> dict[str, Any]:
    if not _PROFILE_PATH.exists():
        return {}
    try:
        return yaml.safe_load(_PROFILE_PATH.read_text()) or {}
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
        "llm_api_key": SETTINGS.llm_api_key,
        "llm_model": SETTINGS.llm_model,
        "claude_cli_model": "",
        "codex_cli_model": "",
        "top_research": str(SETTINGS.top_research),
        "top_industry": str(SETTINGS.top_industry),
        "top_regulatory": str(SETTINGS.top_regulatory),
        "top_world": str(SETTINGS.top_world),
    }
    if _PROFILE_PATH.exists():
        try:
            data = yaml.safe_load(_PROFILE_PATH.read_text()) or {}
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
    response = templates.TemplateResponse(
        request,
        "digest_web.html.j2",
        {
            "digest_id": digest_id,
            "profile_name": _profile_name(),
            "sections": sections,
            "current_vote_per_item": current_vote,
            "empty": len(sections) == 0,
            "brewed": brewed,
            "csrf_token": _CSRF_TOKEN,
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/vote/{item_id}/{value}")
def vote(request: Request, item_id: int, value: int) -> JSONResponse:
    _require_csrf(request)
    if value not in (-1, 0, 1):
        raise HTTPException(status_code=400, detail="value must be -1, 0, or 1")
    ok = votes_mod.record_vote_by_id(item_id, value)
    if not ok:
        raise HTTPException(status_code=404, detail=f"item {item_id} not found")
    return JSONResponse({"ok": True, "item_id": item_id, "new_value": value})


def _run_pipeline_dry_run() -> None:
    try:
        run_all(dry_run=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("background refresh failed: %s", e)


@app.post("/refresh")
def refresh(request: Request) -> JSONResponse:
    _require_csrf(request)
    digest_id = _digest_id()
    threading.Thread(target=_run_pipeline_dry_run, daemon=True).start()
    return JSONResponse({"ok": True, "digest_id": digest_id})


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
    _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROFILE_PATH.write_text(yaml.safe_dump(profile, sort_keys=False))

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
        "LLM_API_KEY": form["llm_api_key"].strip(),
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
        _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PROFILE_PATH.write_text(yaml.safe_dump(data, sort_keys=False))
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
        yield f"data: {json.dumps({'stage': 'connected', 'payload': {'run_id': run_id}})}\n\n"
        terminal = {"done", "error"}
        while True:
            try:
                # Block in a thread-pool slot so the event loop stays free.
                evt = await asyncio.to_thread(q.get, True, 30.0)
            except std_queue.Empty:
                yield ": heartbeat\n\n"
                continue
            yield f"data: {json.dumps(evt)}\n\n"
            if evt.get("stage") in terminal:
                with _RUN_LOCK:
                    _RUN_QUEUES.pop(run_id, None)
                    _RUN_STARTED.discard(run_id)
                break

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
