"""Companion capability tests use fixtures and mocks, never providers/mailboxes."""
import base64
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

from fastapi.testclient import TestClient
import pytest

from dailydigest.config import Settings


module_spec = importlib.util.spec_from_file_location(
    "browser_companion", Path(__file__).resolve().parents[1] / "browser" / "companion.py"
)
companion = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = companion
module_spec.loader.exec_module(companion)
TOKEN = "fixture-private-token-" + "x" * 32
ORIGIN = "https://antimonyleo.github.io"
HEADERS = {"Origin": ORIGIN, "X-DailyDigest-Token": TOKEN}


@pytest.fixture
def settings(tmp_path):
    return Settings.model_construct(db_path=str(tmp_path / "unused.db"),
                                    profile_path=str(tmp_path / "unused-profile.yaml"),
                                    llm_api_key="", resend_api_key="", digest_to="",
                                    imap_user="", imap_password="")


def client(settings, **kwargs):
    return TestClient(companion.create_app(token=TOKEN, settings=settings, **kwargs))


def request(http, method, path, **kwargs):
    return http.request(method, path, headers=HEADERS, **kwargs)


def test_health_requires_token_and_exact_origin_without_mutating_data(settings, tmp_path):
    with client(settings) as http:
        assert http.request("GET", "/health", headers={"Origin": ORIGIN}).status_code == 401
        assert http.request("GET", "/health", headers={**HEADERS, "Origin": ORIGIN + ".evil"}).status_code == 403
        assert http.request("GET", "/health").status_code == 403
        response = request(http, "GET", "/health")
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == ORIGIN
        assert response.headers["cache-control"] == "no-store"
        capabilities = response.json()["capabilities"]
        assert not capabilities["state"]
        assert not capabilities["scheduler"]
        assert not capabilities["email"]
        assert not capabilities["inbound"]
        assert TOKEN not in response.text
    assert list(tmp_path.iterdir()) == []


def test_strict_preflight_allows_token_header_but_not_unknown_headers(settings):
    with client(settings) as http:
        headers = {"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
                   "Access-Control-Request-Headers": "content-type,x-dailydigest-token"}
        response = http.request("OPTIONS", "/summarize", headers=headers)
        assert response.status_code == 200
        assert "access-control-allow-credentials" not in response.headers
        headers["Access-Control-Request-Headers"] = "x-arbitrary-header"
        assert http.request("OPTIONS", "/summarize", headers=headers).status_code == 400
        headers["Origin"] = "https://other.example"
        assert http.request("OPTIONS", "/summarize", headers=headers).status_code == 403


def test_selected_metadata_and_short_profile_use_native_summarizer(settings, monkeypatch):
    from dailydigest import summarize as native

    settings.llm_backend = "api"
    settings.llm_api_key = "private-provider-key"
    calls = []

    def summarize(rows, profile, *, settings, diagnostics):
        calls.append((rows, profile, settings))
        return {4: "Selected article summary"}, {4: "api"}

    monkeypatch.setattr(native, "summarize_items_with_provenance", summarize)
    payload = {"backend": "api", "items": [{"id": 4, "title": "Public article",
               "abstract": "Selected abstract", "metadata": {"doi": "10.123/test"}}],
               "profile": {"bio": "Short reader profile", "keywords": ["materials"]}}
    with client(settings) as http:
        response = request(http, "POST", "/summarize", json=payload)
    assert response.status_code == 200
    assert response.json()["summaries"] == {"4": "Selected article summary"}
    rows, profile, configured = calls[0]
    assert rows[0].title == "Public article"
    assert json.loads(rows[0].metadata_json)["doi"] == "10.123/test"
    assert profile.bio == "Short reader profile"
    assert configured.llm_api_key == "private-provider-key"
    assert "private-provider-key" not in response.text


def test_unconfigured_backend_does_not_contact_provider(settings, monkeypatch):
    from dailydigest import summarize as native

    def forbidden(*args, **kwargs):
        raise AssertionError("Provider must not be contacted")

    monkeypatch.setattr(native, "summarize_items_with_provenance", forbidden)
    with client(settings) as http:
        response = request(http, "POST", "/summarize", json={"backend": "api", "items": []})
    assert response.status_code == 409


def test_summary_validation_never_echoes_private_input(settings):
    with client(settings) as http:
        response = request(http, "POST", "/summarize", json={"items": [],
                           "profile": {"bio": "private-sentinel" * 1000}})
    assert response.status_code == 422
    assert "private-sentinel" not in response.text


def test_explicit_email_uses_native_configured_sender_and_rejects_recipient(settings, monkeypatch):
    from dailydigest import email_send

    settings.resend_api_key = "private-email-key"
    settings.digest_to = "configured@example.test"
    calls = []
    monkeypatch.setattr(email_send, "send_digest", lambda html, subject, dry_run: calls.append(
        (html, subject, dry_run)) or True)
    with client(settings) as http:
        response = request(http, "POST", "/email", json={"html": "<p>Selected digest</p>", "subject": "Digest"})
        rejected = request(http, "POST", "/email", json={"html": "<p>Digest</p>",
                           "subject": "Digest", "to": "arbitrary@example.test"})
    assert response.json() == {"ok": True, "sent": True}
    assert rejected.status_code == 422
    assert calls == [("<p>Selected digest</p>", "Digest", False)]
    assert "private-email-key" not in response.text


def test_missing_email_configuration_never_writes_dry_run(settings, monkeypatch):
    from dailydigest import email_send

    def forbidden(*args, **kwargs):
        raise AssertionError("No dry-run file may be written")

    monkeypatch.setattr(email_send, "send_digest", forbidden)
    with client(settings) as http:
        assert request(http, "POST", "/email", json={"html": "<p>Digest</p>", "subject": "Digest"}).status_code == 409


def test_inbound_reads_browser_votes_and_acknowledges_only_after_commit(settings, monkeypatch):
    from dailydigest import inbound as native

    settings.imap_user = "configured@example.test"
    settings.imap_password = "private-mailbox-password"
    calls = []
    replies = [{"from": "configured@example.test", "message_id": "mail-fixture",
                "subject": "DailyDigest 2026-10-13", "body": "+R3 -R5",
                "date": "fixture", "digest_id": "2026-10-13"},
               {"from": "other@example.test", "body": "+R1", "digest_id": "2026-10-13"},
               {"from": "configured@example.test", "body": "+R1", "digest_id": None}]
    acknowledged = []

    def fetch(cfg, *, on_reply=None):
        calls.append(on_reply)
        if on_reply:
            acknowledged.extend(reply for reply in replies if on_reply(reply))
        return replies

    def forbidden_native_database():
        raise AssertionError("Companion must not write its unrelated local votes")

    monkeypatch.setattr(native, "fetch_replies", fetch)
    monkeypatch.setattr(native, "process_replies", forbidden_native_database)
    with client(settings) as http:
        request(http, "GET", "/health")
        assert calls == []
        response = request(http, "POST", "/inbound", json={})
        assert calls == [None]
        assert acknowledged == []
        result = response.json()
        assert result["scope"] == "browser"
        assert result["skipped_unauthorized"] == 1
        assert result["skipped_missing_digest"] == 1
        vote = result["votes"][0]
        assert vote["line"] == "+R3 -R5"
        assert vote["digest_id"] == "2026-10-13"
        again = request(http, "POST", "/inbound", json={}).json()["votes"][0]
        assert again == vote  # read-only retry remains available after a browser failure
        ack = request(http, "POST", "/inbound/ack", json={"mail_ids": [vote["mail_id"]]})
        assert ack.json()["acknowledgement_attempts"] == 1
    assert acknowledged == [replies[0]]
    assert "private-mailbox-password" not in response.text


def test_email_subject_keeps_browser_digest_identity(settings, monkeypatch):
    from dailydigest import email_send

    settings.resend_api_key = "private-email-key"
    settings.digest_to = "configured@example.test"
    calls = []
    monkeypatch.setattr(email_send, "send_digest", lambda html, subject, dry_run: calls.append(subject) or True)
    with client(settings) as http:
        assert request(http, "POST", "/email", json={"html": "<p>Digest</p>",
                       "subject": "Reading", "digest_id": "2026-10-13"}).status_code == 200
    assert calls == ["DailyDigest 2026-10-13: Reading"]


def test_scheduler_only_starts_with_explicit_opt_in(settings, tmp_path, monkeypatch):
    calls = []
    entered = companion.Event()

    def schedule(stop, job):
        calls.append(job)
        entered.set()
        stop.wait(2)

    monkeypatch.setattr(companion, "_schedule_loop", schedule)
    with client(settings):
        assert calls == []
    with client(settings, schedule=True, state_dir=tmp_path / "scheduler-state") as http:
        assert entered.wait(1)
        health = request(http, "GET", "/health").json()
        assert health["capabilities"]["scheduler"]
        assert health["scheduler_status"]["last_status"] == "pending_uploaded_state"
    assert len(calls) == 1


def test_scheduler_requires_explicit_isolated_state_store(settings):
    with pytest.raises(ValueError, match="requires --state-dir"):
        companion.create_app(token=TOKEN, settings=settings, schedule=True)


def scheduler_snapshot():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE fixture_digest (value TEXT)")
    connection.execute("INSERT INTO fixture_digest VALUES ('original')")
    database = connection.serialize()
    connection.close()
    return snapshot({
        ".env": base64.b64encode(b"USER_TZ=UTC\nDIGEST_HOUR=8\nDB_PATH=/app/data/digest.db\n"
                                  b"PROFILE_PATH=/app/data/profile.yaml\n"
                                  b"SOURCES_PATH=/private/untrusted-sources.yaml\n").decode(),
        "data/profile.yaml": base64.b64encode(b"bio: Uploaded fixture reader\n").decode(),
        "data/digest.db": base64.b64encode(database).decode(),
    })


def test_scheduler_publishes_isolated_updated_browser_snapshot(settings, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    store = companion._state_store(tmp_path / "scheduler-state")
    original = scheduler_snapshot()
    companion._write_private(store / "snapshot.json", original)
    calls = []

    def native_cli(command, *, cwd, env, **kwargs):
        root = Path(cwd)
        calls.append((root, env))
        assert command[-3:] == ["dailydigest.cli", "run-all", "--gate"]
        assert (root / "data/profile.yaml").read_text() == "bio: Uploaded fixture reader\n"
        assert env["DB_PATH"] == str(root / "data/digest.db")
        assert env["PROFILE_PATH"] == str(root / "data/profile.yaml")
        assert env["SOURCES_PATH"] == str(tmp_path / "trusted-public-sources.yaml")
        assert env["RESEND_API_KEY"] == ""
        assert env["DIGEST_TO"] == ""
        assert env["IMAP_PASSWORD"] == ""
        with sqlite3.connect(root / "data/digest.db") as connection:
            connection.execute("UPDATE fixture_digest SET value = 'scheduled'")
        (root / "data/digest.html").write_text("Scheduled fixture digest")
        return companion.SimpleNamespace(returncode=0)

    monkeypatch.setattr(companion.subprocess, "run", native_cli)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "original-user-data.db"))
    monkeypatch.setenv("RESEND_API_KEY", "ambient-private-email-key")
    status = {}
    companion._schedule_once(store, settings, companion.RLock(), status,
                             tmp_path / "trusted-public-sources.yaml",
                             now=datetime(2026, 10, 13, 8, tzinfo=timezone.utc))
    assert status["last_status"] == "success"
    updated = json.loads((store / "snapshot.json").read_text())
    restored = tmp_path / "restored-scheduled.db"
    restored.write_bytes(base64.b64decode(updated["files"]["data/digest.db"]))
    with sqlite3.connect(restored) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("SELECT value FROM fixture_digest").fetchone() == ("scheduled",)
    assert "data/digest.html" in updated["files"]
    assert updated["files"][".env"] == json.loads(original)["files"][".env"]
    assert "ambient-private-email-key" not in (store / "snapshot.json").read_text()
    assert len(calls) == 1
    assert not (tmp_path / "original-user-data.db").exists()


def test_scheduler_failed_job_preserves_published_snapshot(settings, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    store = companion._state_store(tmp_path / "scheduler-state")
    original = scheduler_snapshot()
    companion._write_private(store / "snapshot.json", original)

    def failed_cli(command, *, cwd, **kwargs):
        (Path(cwd) / "data/digest.db").write_bytes(b"failed-stage-only")
        return companion.SimpleNamespace(returncode=1)

    monkeypatch.setattr(companion.subprocess, "run", failed_cli)
    status = {}
    companion._schedule_once(store, settings, companion.RLock(), status, tmp_path / "sources.yaml",
                             now=datetime(2026, 10, 13, 8, tzinfo=timezone.utc))
    assert status["last_status"] == "failed"
    assert "previous snapshot preserved" in status["error"]
    assert (store / "snapshot.json").read_text() == original


def test_scheduler_waits_for_upload_and_reader_time_gate(settings, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    store = companion._state_store(tmp_path / "scheduler-state")
    status = {}

    def forbidden_cli(*args, **kwargs):
        raise AssertionError("No brew before upload or outside reader's hour")

    monkeypatch.setattr(companion.subprocess, "run", forbidden_cli)
    companion._schedule_once(store, settings, companion.RLock(), status, tmp_path / "sources.yaml")
    assert status["last_status"] == "pending_uploaded_state"
    companion._write_private(store / "snapshot.json", scheduler_snapshot())
    companion._schedule_once(store, settings, companion.RLock(), status, tmp_path / "sources.yaml",
                             now=datetime(2026, 10, 13, 9, tzinfo=timezone.utc))
    assert status["last_status"] == "waiting_for_digest_hour"
    assert not list(store.glob("native-reader-*"))


def snapshot(files=None):
    return json.dumps({"format": "dailydigest-browser-v1", "files": files or {
        "data/profile.yaml": base64.b64encode(b"bio: fixture").decode(),
    }})


def test_private_state_is_explicit_atomic_and_backup_is_immutable(settings, tmp_path):
    store = tmp_path / "companion-state"
    with client(settings, state_dir=store) as http:
        assert request(http, "GET", "/state").status_code == 404
        payload = snapshot()
        assert request(http, "PUT", "/state", json={"snapshot": payload}).json() == {"ok": True}
        assert request(http, "GET", "/state").json() == {"snapshot": payload}
        backup = request(http, "POST", "/backup", json={}).json()["backup"]
        assert (store / backup).read_text() == payload
        assert (store / "snapshot.json").stat().st_mode & 0o077 == 0
        assert not (store / "data").exists()
        assert not list(store.glob(".snapshot-*"))


@pytest.mark.parametrize("path", ["../.env", "/tmp/private", "data/../../private",
                                  "data\\private", "data/.compute.lock", "data//private",
                                  "data/a/./private"])
def test_invalid_snapshot_paths_preserve_existing_state(settings, tmp_path, path):
    with client(settings, state_dir=tmp_path / "companion-state") as http:
        existing = snapshot()
        request(http, "PUT", "/state", json={"snapshot": existing})
        invalid = snapshot({path: base64.b64encode(b"fixture").decode()})
        assert request(http, "PUT", "/state", json={"snapshot": invalid}).status_code == 422
        assert request(http, "GET", "/state").json()["snapshot"] == existing


def test_sync_disabled_without_explicit_state_directory(settings):
    with client(settings) as http:
        assert request(http, "GET", "/state").status_code == 409
        assert request(http, "PUT", "/state", json={"snapshot": snapshot()}).status_code == 409


def test_companion_refuses_existing_user_data_directory(settings, tmp_path):
    (tmp_path / "digest.db").write_bytes(b"user sentinel")
    with pytest.raises(ValueError, match="must be new"):
        companion.create_app(token=TOKEN, settings=settings, state_dir=tmp_path)
    assert (tmp_path / "digest.db").read_bytes() == b"user sentinel"


def test_bounded_cli_text_reuses_native_sandbox(settings, monkeypatch):
    from dailydigest import summarize as native

    monkeypatch.setattr(companion.shutil, "which", lambda command: "/fixture/" + command)
    calls = []
    monkeypatch.setattr(native, "_cli_text", lambda system, user, backend, **kwargs: calls.append(
        (system, user, backend, kwargs["timeout"])) or "Generated tea note")
    with client(settings) as http:
        response = request(http, "POST", "/text", json={"backend": "codex_cli",
                           "system": "Selected context only", "user": "Short profile", "timeout": 20})
    assert response.json() == {"text": "Generated tea note"}
    assert calls == [("Selected context only", "Short profile", "codex_cli", 20)]


def test_cli_token_is_private_and_never_printed(settings, tmp_path, monkeypatch, capsys):
    import uvicorn
    from dailydigest import config

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(sys, "argv", ["companion.py"])
    starts = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: starts.append(kwargs))
    companion.main()
    token_file = next(tmp_path.glob("dailydigest-companion-*/token"))
    token = token_file.read_text().strip()
    assert len(token) >= 32
    assert token not in capsys.readouterr().out
    assert token_file.stat().st_mode & 0o077 == 0
    assert starts[0]["host"] == "127.0.0.1"
    assert starts[0]["port"] == 8768
