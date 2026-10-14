"""Explicit local companion for browser summaries, email, inbound, and state.

Start: .venv/bin/python browser/companion.py --allowed-origin https://antimonyleo.github.io
The private token is written to a file; only its path is printed. State sync is
disabled unless --state-dir selects a separate companion store. --schedule
explicitly brews an uploaded browser snapshot in an isolated native workspace;
browser requests never forward brewing or ranking to this service.
"""
from __future__ import annotations

import sys

VERSION = "0.1.0"
if __name__ == "__main__" and sys.argv[1:] in (["--version"], ["-v"], ["-V"]):
    print(VERSION)
    raise SystemExit(0)

import argparse
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hmac
import hashlib
from email.utils import parseaddr
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
from io import StringIO
import subprocess
import tempfile
from threading import Event, RLock, Thread
from types import SimpleNamespace
from typing import Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


MAX_BODY = 3 * 1024 * 1024
MAX_STATE = 128 * 1024 * 1024
BACKENDS = Literal["extractive", "api", "anthropic", "claude_cli", "codex_cli"]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Article(Input):
    id: int = Field(gt=0)
    title: str = Field(max_length=2000)
    abstract: str = Field(default="", max_length=4000)
    authors: str = Field(default="", max_length=2000)
    source: str = Field(default="", max_length=500)
    section: str = Field(default="research", max_length=40)
    url: str = Field(default="", max_length=4000)
    published_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)


class Reader(Input):
    bio: str = Field(default="", max_length=1000)
    keywords: list[str] = Field(default_factory=list, max_length=20)


class SummaryInput(Input):
    backend: BACKENDS = "extractive"
    items: list[Article] = Field(max_length=80)
    profile: Reader = Field(default_factory=Reader)


class EmailInput(Input):
    html: str = Field(min_length=1, max_length=2 * 1024 * 1024)
    subject: str = Field(min_length=1, max_length=200)
    digest_id: str | None = Field(default=None, max_length=100)


class StateInput(Input):
    snapshot: str = Field(max_length=MAX_STATE)


class InboundAck(Input):
    mail_ids: list[str] = Field(max_length=1000)


class TextInput(Input):
    backend: Literal["claude_cli", "codex_cli"]
    system: str = Field(max_length=8000)
    user: str = Field(max_length=32000)
    timeout: float = Field(default=90, ge=1, le=120)


def validate_snapshot(payload):
    try:
        if len(payload.encode()) > MAX_STATE:
            raise ValueError("Snapshot exceeds the transfer limit")
        snapshot = json.loads(payload)
        if not isinstance(snapshot, dict) or snapshot.get("format") != "dailydigest-browser-v1":
            raise ValueError("Unsupported snapshot format")
        files = snapshot.get("files")
        if not isinstance(files, dict) or len(files) > 10000:
            raise ValueError("Invalid snapshot files")
        paths = set()
        total = 0
        for name, value in files.items():
            path = PurePosixPath(name)
            if (not isinstance(name, str) or "\\" in name or path.is_absolute()
                    or ".." in path.parts or str(path) != name
                    or (name != ".env" and (len(path.parts) < 2 or path.parts[0] != "data"))
                    or path.name == ".compute.lock"):
                raise ValueError("Snapshot path outside private browser state")
            total += len(base64.b64decode(value, validate=True))
            if total > MAX_STATE:
                raise ValueError("Decoded snapshot too large")
            paths.add(path)
        if any(parent in paths for path in paths for parent in path.parents):
            raise ValueError("Snapshot file and directory conflict")
        return snapshot
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "Invalid browser snapshot") from exc


def _state_store(path):
    if path is None:
        return None
    store = Path(path)
    if store.is_symlink():
        raise ValueError("State directory must not be a symlink")
    marker = store / ".dailydigest-companion"
    if marker.is_symlink() or (marker.is_file() and marker.read_text() != "dailydigest-companion-v1\n"):
        raise ValueError("State directory marker is invalid")
    if store.exists() and any(store.iterdir()) and not marker.is_file():
        raise ValueError("State directory must be new, empty, or an existing companion store")
    store.mkdir(mode=0o700, parents=True, exist_ok=True)
    store.chmod(0o700)
    marker.write_text("dailydigest-companion-v1\n")
    marker.chmod(0o600)
    return store.resolve()


def _write_private(path, payload):
    if path.is_symlink():
        raise ValueError("Companion files must not be symlinks")
    descriptor, temporary = tempfile.mkstemp(prefix=".snapshot-", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _schedule_once(store, settings, lock, status, source_path, *, now=None):
    """Publish isolated native state only after a successful explicit job."""
    from dailydigest.config import Settings
    from dotenv import dotenv_values

    instant = now or datetime.now(timezone.utc)
    with lock:
        path = store / "snapshot.json"
        if not path.is_file() or path.is_symlink():
            status.update(last_status="pending_uploaded_state", error=None, error_type=None)
            return
        status.update(last_status="running", last_attempt=instant.isoformat(), error=None, error_type=None)
        try:
            original = path.read_text()
            snapshot = validate_snapshot(original)
            files = {name: base64.b64decode(value, validate=True)
                     for name, value in snapshot["files"].items()}
            uploaded = dotenv_values(stream=StringIO(files.get(".env", b"").decode()),
                                     interpolate=False)
            defaults = Settings.model_construct().model_dump()
            values = {key.lower(): value for key, value in uploaded.items()
                      if key.lower() in defaults and value is not None}
            configured = Settings(_env_file=None, **{**defaults, **values})
            zone = ZoneInfo(configured.user_tz or "UTC")
            if instant.astimezone(zone).hour != configured.digest_hour:
                status.update(last_status="waiting_for_digest_hour")
                return
            root = Path(tempfile.mkdtemp(prefix="native-reader-", dir=store))
            for name, content in files.items():
                target = root / name
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                target.write_bytes(content)
                target.chmod(0o600)

            def private_path(value):
                name = str(value).removeprefix("/app/")
                relative = PurePosixPath(name)
                if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                        or not relative.parts or relative.parts[0] != "data"):
                    raise ValueError("Scheduled reader paths must remain in browser data")
                return str(root / relative)

            overrides = {key: private_path(getattr(configured, key)) for key in
                         ("db_path", "profile_path", "opportunity_profile_path")}
            for key in ("contribution_reranker_path", "contribution_encoder_dir"):
                value = getattr(configured, key)
                overrides[key] = private_path(value) if value else ""
            overrides.update(sources_path=str(source_path), user_tz=configured.user_tz or "UTC",
                             imap_user="", imap_password="")
            # Browser uploads select the provider; they cannot redirect a server
            # credential to an uploaded URL or replace the configured recipient.
            if configured.llm_backend in {"api", "anthropic"}:
                if (configured.llm_backend != settings.llm_backend or not settings.llm_api_key):
                    raise ValueError("Scheduled summary provider is not configured")
                overrides.update(llm_api_key=settings.llm_api_key,
                                 llm_base_url=settings.llm_base_url, llm_model=settings.llm_model)
            else:
                overrides["llm_api_key"] = ""
            delivery = bool(configured.digest_to and configured.digest_to == settings.digest_to
                            and settings.resend_api_key)
            overrides.update(resend_api_key=settings.resend_api_key if delivery else "",
                             digest_to=settings.digest_to if delivery else "",
                             digest_from=settings.digest_from,
                             reply_to_email=settings.reply_to_email if delivery else "")
            safe_settings = configured.model_copy(update=overrides)
            environment = dict(os.environ)
            for key, value in safe_settings.model_dump().items():
                environment[key.upper()] = str(value).lower() if isinstance(value, bool) else str(value)
            environment["TMPDIR"] = str(root)
            # Keep model downloads/cache inside the explicitly selected store.
            environment["HF_HOME"] = str(store / "model-cache")
            environment["XDG_CACHE_HOME"] = str(store / "model-cache")
            completed = subprocess.run(
                [sys.executable, "-m", "dailydigest.cli", "run-all", "--gate"],
                cwd=root, env=environment, timeout=1800, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if completed.returncode:
                raise RuntimeError("Native scheduled brew did not complete successfully")
            exported = {str(file.relative_to(root)): base64.b64encode(file.read_bytes()).decode()
                        for file in (root / "data").rglob("*")
                        if file.is_file() and not file.is_symlink() and file.name != ".compute.lock"}
            if ".env" in snapshot["files"]:
                exported[".env"] = snapshot["files"][".env"]
            updated = json.dumps({"format": "dailydigest-browser-v1", "files": exported})
            validate_snapshot(updated)
            _write_private(path, updated)
            status.update(last_status="success", last_success=instant.isoformat(), error=None)
        except Exception as exc:
            # Preserve the prior snapshot, and disclose failure without private
            # provider URLs, credentials, reader content, or subprocess output.
            status.update(last_status="failed", error="Scheduled brew failed; previous snapshot preserved",
                          error_type=type(exc).__name__)


def _schedule_loop(stop, job):
    """Attempt the uploaded reader's native time gate at most once per hour."""
    previous_hour = None
    while not stop.is_set():
        hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
        if hour != previous_hour:
            job()
            # An upload during this hour must still be able to run its gate.
            if job.status.get("last_status") != "pending_uploaded_state":
                previous_hour = hour
        stop.wait(60)


def create_app(*, token, allowed_origin="https://antimonyleo.github.io", settings=None,
               state_dir=None, schedule=False):
    origin = urlsplit(allowed_origin)
    if (origin.scheme not in {"http", "https"} or not origin.hostname or origin.path
            or origin.query or origin.fragment or origin.username or origin.password):
        raise ValueError("Allowed origin must be an exact HTTP(S) origin without a path")
    if not token or len(token) < 32:
        raise ValueError("Companion token must contain at least 32 characters")
    if settings is None:
        from dailydigest.config import get_settings

        settings = get_settings()
    store = _state_store(state_dir)
    if schedule and store is None:
        raise ValueError("Scheduling requires --state-dir and an explicit browser state upload")
    lock, stop = RLock(), Event()
    source_path = Path(settings.sources_path).resolve()
    schedule_status = {"last_status": "pending_uploaded_state" if schedule else "disabled",
                       "last_attempt": None, "last_success": None, "error": None}

    def scheduled_job():
        _schedule_once(store, settings, lock, schedule_status, source_path)

    scheduled_job.status = schedule_status

    @asynccontextmanager
    async def lifespan(app):
        thread = None
        if schedule:
            thread = Thread(target=_schedule_loop, args=(stop, scheduled_job), daemon=True)
            thread.start()
        yield
        stop.set()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=[allowed_origin],
                       allow_methods=["GET", "POST", "PUT"],
                       allow_headers=["Content-Type", "X-DailyDigest-Token"],
                       allow_credentials=False)

    @app.middleware("http")
    async def authorized(request: Request, call_next):
        if request.headers.get("origin") != allowed_origin:
            return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
        if request.method != "OPTIONS":
            supplied = request.headers.get("x-dailydigest-token", "")
            if not hmac.compare_digest(supplied.encode(), token.encode()):
                return JSONResponse({"detail": "Invalid companion token"}, status_code=401,
                                    headers={"Access-Control-Allow-Origin": allowed_origin,
                                             "Vary": "Origin", "Cache-Control": "no-store"})
            body = bytearray()
            limit = MAX_STATE + 1024 * 1024 if request.url.path == "/state" else MAX_BODY
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > limit:
                    return JSONResponse({"detail": "Request too large"}, status_code=413,
                                        headers={"Access-Control-Allow-Origin": allowed_origin,
                                                 "Vary": "Origin", "Cache-Control": "no-store"})
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Validation errors must never echo private profile/state content.
        return JSONResponse({"detail": "Invalid request fields", "errors": [
            {"loc": list(item["loc"]), "type": item["type"]} for item in error.errors()
        ]}, status_code=422)

    def backends():
        available = ["extractive"]
        configured = str(getattr(settings, "llm_backend", "extractive")).lower()
        if configured in {"api", "anthropic"} and getattr(settings, "llm_api_key", ""):
            available.append(configured)
        for name, command in (("claude_cli", "claude"), ("codex_cli", "codex")):
            if shutil.which(command):
                available.append(name)
        return available

    @app.get("/health")
    def health():
        return {"ok": True, "capabilities": {
            "summarize": backends(),
            "email": bool(getattr(settings, "resend_api_key", "") and
                          getattr(settings, "digest_to", "")),
            "inbound": bool(getattr(settings, "imap_user", "") and
                            getattr(settings, "imap_password", "")),
            "state": store is not None, "scheduler": schedule,
            "scheduler_mode": "uploaded_browser_state" if schedule else "disabled",
        }, "scheduler_status": dict(schedule_status)}

    @app.post("/summarize")
    def summarize(payload: SummaryInput):
        if payload.backend not in backends():
            raise HTTPException(409, "Summary backend is not configured on the companion")
        if len({item.id for item in payload.items}) != len(payload.items):
            raise HTTPException(422, "Selected article IDs must be unique")
        if any(len(keyword) > 160 for keyword in payload.profile.keywords):
            raise HTTPException(422, "Reader keywords exceed the short profile limit")
        from dailydigest import summarize as native
        from dailydigest.store import ItemRow

        rows = [ItemRow(**item.model_dump(exclude={"metadata"}),
                        metadata_json=json.dumps(item.metadata)) for item in payload.items]
        configured = settings.model_copy(update={"llm_backend": payload.backend})
        diagnostics = {}
        try:
            with lock:
                summaries, provenance = native.summarize_items_with_provenance(
                    rows, SimpleNamespace(**payload.profile.model_dump()),
                    settings=configured, diagnostics=diagnostics,
                )
        except Exception as exc:
            raise HTTPException(502, "Configured summarizer failed") from exc
        # Preserve actual per-item fallback provenance without provider exception
        # strings, which can contain configured endpoints or sensitive details.
        return {"summaries": summaries, "provenance": provenance,
                "diagnostics": {"fallback_items": sum(v != payload.backend for v in provenance.values())}}

    @app.post("/text")
    def text(payload: TextInput):
        if payload.backend not in backends():
            raise HTTPException(409, "CLI backend is not configured on the companion")
        from dailydigest import summarize as native

        configured = settings.model_copy(update={"llm_backend": payload.backend})
        try:
            with lock:
                output = native._cli_text(payload.system, payload.user, payload.backend,
                                          timeout=payload.timeout, settings=configured)
        except Exception as exc:
            raise HTTPException(502, "Configured CLI text generation failed") from exc
        return {"text": output}

    @app.post("/email")
    def email(payload: EmailInput):
        if not getattr(settings, "resend_api_key", "") or not getattr(settings, "digest_to", ""):
            raise HTTPException(409, "Email service and recipient are not configured")
        if "\r" in payload.subject or "\n" in payload.subject:
            raise HTTPException(422, "Email subject must be a single line")
        from dailydigest import email_send

        with lock:
            # Existing sender owns the configured recipient; no recipient is
            # accepted from browser content or arbitrary query parameters.
            subject = payload.subject
            if payload.digest_id:
                from dailydigest.inbound import _parse_digest_id

                if _parse_digest_id(payload.digest_id) != payload.digest_id:
                    raise HTTPException(422, "Digest ID must be its dated browser identifier")
                if _parse_digest_id(subject) != payload.digest_id:
                    subject = f"DailyDigest {payload.digest_id}: {subject}"
            sent = email_send.send_digest(payload.html, subject, dry_run=False)
        if not sent:
            raise HTTPException(502, "Configured email delivery failed")
        return {"ok": True, "sent": True}

    @app.post("/inbound")
    def inbound(payload: Input):
        if not getattr(settings, "imap_user", "") or not getattr(settings, "imap_password", ""):
            raise HTTPException(409, "Inbound mailbox is not configured")
        from dailydigest import inbound as native

        cfg = mailbox_config(native)
        expected_sender = authorized_sender()
        with lock:
            try:
                replies = native.fetch_replies(cfg)  # BODY.PEEK; no acknowledgement
            except Exception as exc:
                raise HTTPException(502, "Configured mailbox read failed") from exc
        votes, unauthorized, missing_digest = [], 0, 0
        for reply in replies[:1000]:
            if parseaddr(reply.get("from", ""))[1].strip().lower() != expected_sender:
                unauthorized += 1
                continue
            line = native.extract_vote_line(reply.get("body", ""))
            if not line:
                continue
            # Return vote tokens rather than surrounding private message text.
            # Never truncate a valid token sequence into a different vote.
            line = " ".join(native._VOTE_TOKEN_RE.findall(line))
            if len(line) > 4000:
                raise HTTPException(422, "Reply contains too many vote tokens; mailbox remains unread")
            digest_id = reply.get("digest_id")
            if not digest_id:
                missing_digest += 1
                continue
            votes.append({"mail_id": reply_id(reply), "line": line, "digest_id": digest_id})
        return {"ok": True, "scope": "browser", "votes": votes, "errors": [],
                "messages": len(replies), "skipped_unauthorized": unauthorized,
                "skipped_missing_digest": missing_digest}

    def mailbox_config(native):
        return native.InboundConfig(
            imap_host=getattr(settings, "imap_host", ""),
            imap_port=getattr(settings, "imap_port", 993),
            imap_user=getattr(settings, "imap_user", ""),
            imap_password=getattr(settings, "imap_password", ""),
            mailbox=getattr(settings, "imap_mailbox", "INBOX"),
        )

    def authorized_sender():
        address = parseaddr(getattr(settings, "digest_to", "") or
                            getattr(settings, "imap_user", ""))[1].strip().lower()
        if not address:
            raise HTTPException(409, "Inbound reader address is not configured")
        return address

    def reply_id(reply):
        # Message-ID alone can be absent or repeated; content identity supports
        # safe browser dedup and a read-only-poll/commit/acknowledge sequence.
        value = {key: reply.get(key, "") for key in
                 ("message_id", "from", "subject", "date", "body", "digest_id")}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    @app.post("/inbound/ack")
    def inbound_ack(payload: InboundAck):
        if not getattr(settings, "imap_user", "") or not getattr(settings, "imap_password", ""):
            raise HTTPException(409, "Inbound mailbox is not configured")
        if any(len(mail_id) != 64 or any(char not in "0123456789abcdef" for char in mail_id)
               for mail_id in payload.mail_ids):
            raise HTTPException(422, "Invalid inbound message identities")
        from dailydigest import inbound as native

        expected_sender, matched = authorized_sender(), []
        identifiers = set(payload.mail_ids)

        def acknowledge(reply):
            if (parseaddr(reply.get("from", ""))[1].strip().lower() == expected_sender
                    and reply.get("digest_id") and reply_id(reply) in identifiers):
                matched.append(reply_id(reply))
                return True
            return False

        with lock:
            try:
                native.fetch_replies(mailbox_config(native), on_reply=acknowledge)
            except Exception as exc:
                raise HTTPException(502, "Configured mailbox acknowledgement failed") from exc
        return {"ok": True, "acknowledgement_attempts": len(matched)}

    def state_path():
        if store is None:
            raise HTTPException(409, "Private state sync requires --state-dir")
        path = store / "snapshot.json"
        if path.is_symlink():
            raise HTTPException(409, "Private state file must not be a symlink")
        return path

    @app.get("/state")
    def read_state():
        with lock:
            path = state_path()
            if not path.is_file():
                raise HTTPException(404, "No private snapshot has been saved")
            payload = path.read_text()
            validate_snapshot(payload)
        return {"snapshot": payload}

    @app.put("/state")
    def write_state(payload: StateInput):
        validate_snapshot(payload.snapshot)
        with lock:
            _write_private(state_path(), payload.snapshot)
        return {"ok": True}

    @app.post("/backup")
    def backup(payload: Input):
        with lock:
            path = state_path()
            if not path.is_file():
                raise HTTPException(404, "No private snapshot has been saved")
            snapshot = path.read_text()
            validate_snapshot(snapshot)
            name = "backup-" + secrets.token_hex(12) + ".json"
            _write_private(store / name, snapshot)
        return {"ok": True, "backup": name}

    return app


class Parser(argparse.ArgumentParser):
    def error(self, message):
        print("error: " + json.dumps(message))
        print('help: "Run browser/companion.py --help for allowed flags and examples"')
        raise SystemExit(2)


def main():
    parser = Parser(description=__doc__, epilog=(
        "Examples: companion.py --allowed-origin https://antimonyleo.github.io; "
        "companion.py --allow-local-preview --allowed-origin http://127.0.0.1:8766 --state-dir /private/new-store"))
    parser.add_argument("--allowed-origin", default="https://antimonyleo.github.io")
    parser.add_argument("--allow-local-preview", action="store_true")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--schedule", "--enable-scheduler", action="store_true",
                        help="Brew uploaded browser state in an isolated workspace; requires --state-dir")
    args = parser.parse_args()
    origin = urlsplit(args.allowed_origin)
    if origin.scheme != "https" and not (args.allow_local_preview and
                                          origin.hostname in {"localhost", "127.0.0.1", "::1"}):
        parser.error("Allowed origin must use HTTPS; local HTTP requires --allow-local-preview")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    if args.schedule and args.state_dir is None:
        parser.error("--schedule requires --state-dir and an explicit browser state upload")
    try:
        if args.token_file is None:
            scratch_root = os.environ.get("TMPDIR")
            if not scratch_root:
                import tomllib

                config_path = Path.home() / ".codex" / "config.toml"
                if config_path.is_file():
                    scratch_root = tomllib.loads(config_path.read_text()).get(
                        "shell_environment_policy", {}).get("set", {}).get("TMPDIR")
            scratch = Path(tempfile.mkdtemp(prefix="dailydigest-companion-", dir=scratch_root))
            args.token_file = scratch / "token"
        if args.token_file.is_symlink():
            raise ValueError("Token file must not be a symlink")
        if args.token_file.exists():
            if args.token_file.stat().st_mode & 0o077:
                raise ValueError("Token file must have private permissions (0600)")
            token = args.token_file.read_text().strip()
        else:
            token = secrets.token_urlsafe(32)
            descriptor = os.open(args.token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                stream.write(token + "\n")
        app = create_app(token=token, allowed_origin=args.allowed_origin,
                         state_dir=args.state_dir, schedule=args.schedule)
        print('status: "Starting explicit local companion"')
        print("endpoint: " + json.dumps(f"http://127.0.0.1:{args.port}"))
        print("token_file: " + json.dumps(str(args.token_file)))
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    except (OSError, ValueError):
        print('error: "Companion startup failed; check paths, token permissions, and allowed origin"')
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
