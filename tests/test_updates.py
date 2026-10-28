"""One-click updates for local installs: release notes, safe replacement,
rollback, and the routes behind the update banner."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from dailydigest import updates

ROOT = Path(__file__).resolve().parents[1]


def _notes(*versions: str) -> dict:
    return {"schema": 1, "releases": [
        {"version": v, "date": "2026-10-27", "title": f"Release {v}", "notes": [f"Change in {v}"]}
        for v in versions]}


def _install(tmp_path: Path, version: str = "2026.10.27") -> Path:
    root = tmp_path / "DailyDigest"
    (root / "src" / "dailydigest").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "dailydigest"\n')
    (root / "src" / "dailydigest" / "release_notes.json").write_text(json.dumps(_notes(version)))
    (root / "src" / "dailydigest" / "web.py").write_text("old = True\n")
    (root / "config").mkdir()
    (root / "config" / "sources.yaml").write_text("research: []\n")
    (root / "data").mkdir()
    (root / "data" / "digest.db").write_bytes(b"reader database")
    (root / ".env").write_text("LLM_API_KEY=secret\n")
    return root


def _archive(files: dict[str, bytes], prefix: str = "daily_digests-main") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(f"{prefix}/{name}", content)
    return buffer.getvalue()


def _release_files(version: str = "2026.11.02") -> dict[str, bytes]:
    return {
        "pyproject.toml": b'[project]\nname = "dailydigest"\n',
        "src/dailydigest/release_notes.json": json.dumps(_notes(version, "2026.10.27")).encode(),
        "src/dailydigest/web.py": b"old = False\n",
        "src/dailydigest/new_module.py": b"added = True\n",
        "data/digest.db": b"attacker replacement",
        ".env": b"LLM_API_KEY=stolen\n",
        ".venv/bin/python": b"not a python",
    }


class _Run:
    def __init__(self, fail_sync: bool = False):
        self.fail_sync, self.calls = fail_sync, []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        failed = self.fail_sync and "sync" in command
        return SimpleNamespace(returncode=1 if failed else 0, stdout="", stderr="resolver failed" if failed else "")


def _updater(root: Path, remote: dict, archive: bytes = b"", run=None, restarts=None):
    def http_get(url, limit):
        return json.dumps(remote).encode() if url == updates.NOTES_URL else archive

    return updates.Updater(root=root, notes_path=root / "src" / "dailydigest" / "release_notes.json",
                           http_get=http_get, run=run or _Run(),
                           restart=lambda host, port: (restarts if restarts is not None else []).append((host, port)),
                           clock=lambda: datetime(2026, 11, 2, 9, tzinfo=timezone.utc))


def test_shipped_release_notes_are_valid_and_newest_first():
    releases = updates.load_release_notes()
    versions = [row["version"] for row in releases]
    assert len(versions) == len(set(versions))
    assert updates.current_version() == versions[0]
    assert all(row["title"] and row["notes"] for row in releases)


def test_release_notes_order_and_newer_releases():
    releases = updates.parse_release_notes(_notes("2026.10.27", "2026.11.2", "bad", "2026.11.2.1"))
    assert [row["version"] for row in releases] == ["2026.11.2.1", "2026.11.2", "2026.10.27"]
    assert [row["version"] for row in updates.releases_since(releases, "2026.11.2")] == ["2026.11.2.1"]
    with pytest.raises(ValueError):
        updates.parse_release_notes({"schema": 2, "releases": []})


def test_install_mode_tells_zip_git_docker_and_package_installs_apart(tmp_path, monkeypatch):
    monkeypatch.delenv("DD_IN_DOCKER", raising=False)
    monkeypatch.setattr(updates.Path, "exists", lambda self, _real=Path.exists: False if str(self) == "/.dockerenv" else _real(self))
    root = _install(tmp_path)
    assert updates.install_mode(root) == "archive"
    (root / ".git").mkdir()
    assert updates.install_mode(root) == "git"
    assert updates.install_mode(tmp_path / "site-packages") == "unsupported"
    monkeypatch.setenv("DD_IN_DOCKER", "1")
    assert updates.install_mode(root) == "docker"


def test_archive_extraction_rejects_unsafe_or_foreign_downloads():
    assert PurePosixPath("src/dailydigest/web.py") in updates.extract_archive(_archive(_release_files()))
    with pytest.raises(ValueError, match="unsafe"):
        updates.extract_archive(_archive({"pyproject.toml": b'name = "dailydigest"', "../escape.py": b""}, prefix="x"))
    with pytest.raises(ValueError, match="not a DailyDigest"):
        updates.extract_archive(_archive({"pyproject.toml": b'[project]\nname = "other"\n'}))


def test_zip_update_replaces_program_files_only_and_restarts(tmp_path, monkeypatch):
    monkeypatch.setattr(updates, "_find_uv", lambda: "uv")
    monkeypatch.setattr(updates.time, "sleep", lambda seconds: None)
    root = _install(tmp_path)
    run, restarts, failures = _Run(), [], []
    updater = _updater(root, _notes("2026.11.02", "2026.10.27"), _archive(_release_files()), run, restarts)
    updater.check_now()
    state = updater.status(busy=False)
    assert state["available"] and state["can_apply"] and state["latest"] == "2026.11.02"
    assert [row["version"] for row in state["releases"]] == ["2026.11.02"]

    updater._apply("archive", "2026.11.02", ("127.0.0.1", 8765), lambda: failures.append(True))

    assert (root / "src/dailydigest/web.py").read_text() == "old = False\n"
    assert (root / "src/dailydigest/new_module.py").is_file()
    assert (root / "data/digest.db").read_bytes() == b"reader database"
    assert (root / ".env").read_text() == "LLM_API_KEY=secret\n"
    assert not (root / ".venv").exists()
    backups = list((root / updates.BACKUP_DIR).iterdir())
    assert len(backups) == 1 and (backups[0] / "files/src/dailydigest/web.py").read_text() == "old = True\n"
    assert any("sync" in call for call in run.calls)
    assert restarts == [("127.0.0.1", 8765)] and failures == []
    assert updates.current_version(root / "src/dailydigest/release_notes.json") == "2026.11.02"


def test_failed_install_restores_every_file_and_keeps_running(tmp_path, monkeypatch):
    monkeypatch.setattr(updates, "_find_uv", lambda: "uv")
    root = _install(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    restarts, failures = [], []
    updater = _updater(root, _notes("2026.11.02"), _archive(_release_files()), _Run(fail_sync=True), restarts)
    updater.check_now()

    updater._apply("archive", "2026.11.02", ("127.0.0.1", 8765), lambda: failures.append(True))

    after = {path: path.read_bytes() for path in root.rglob("*")
             if path.is_file() and updates.BACKUP_DIR not in path.parts}
    assert after == before
    assert restarts == [] and failures == [True]
    state = updater.status(busy=False)
    assert state["phase"] == "failed" and "nothing was changed" in state["message"]
    assert state["can_apply"], "A failed attempt can be retried"


def test_older_download_than_announced_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(updates, "_find_uv", lambda: "uv")
    root = _install(tmp_path)
    updater = _updater(root, _notes("2026.11.09"), _archive(_release_files("2026.11.02")))
    updater.check_now()
    updater._apply("archive", "2026.11.09", ("127.0.0.1", 8765), lambda: None)
    assert (root / "src/dailydigest/web.py").read_text() == "old = True\n"
    assert updater.status(busy=False)["phase"] == "failed"


def test_busy_docker_and_edited_git_copies_explain_instead_of_updating(tmp_path, monkeypatch):
    root = _install(tmp_path)
    updater = _updater(root, _notes("2026.11.02"))
    updater.check_now()
    busy = updater.status(busy=True)
    assert not busy["can_apply"] and "Finish or stop" in busy["blocked_reason"]

    monkeypatch.setenv("DD_IN_DOCKER", "1")
    docker = updater.status(busy=False)
    assert docker["available"] and not docker["can_apply"] and "docker compose up --build -d" in docker["instructions"]
    monkeypatch.delenv("DD_IN_DOCKER")

    (root / ".git").mkdir()
    dirty = SimpleNamespace(returncode=0, stdout=" M config/sources.yaml\n", stderr="")
    updater._run = lambda command, **kwargs: dirty
    edited = updater.status(busy=False)
    assert edited["mode"] == "git" and not edited["can_apply"]
    assert "were changed" in edited["blocked_reason"]


def test_unreachable_update_check_is_quiet(tmp_path):
    root = _install(tmp_path)

    def offline(url, limit):
        raise OSError("offline")

    updater = updates.Updater(root=root, notes_path=root / "src/dailydigest/release_notes.json", http_get=offline)
    updater.check_now()
    state = updater.status(busy=False)
    assert not state["available"] and state["check_error"]


def test_check_is_skipped_when_disabled_or_in_the_browser(tmp_path, monkeypatch):
    root = _install(tmp_path)
    calls = []
    updater = updates.Updater(root=root, http_get=lambda url, limit: calls.append(url) or b"{}")
    updater.maybe_check(False)
    monkeypatch.setattr(updates.sys, "platform", "emscripten")
    updater.maybe_check(True)
    assert calls == [] and updater.checked_at is None


def test_windows_launcher_stays_byte_identical():
    # cmd.exe reads a running .bat from disk by offset. Updating this file while
    # it runs would execute garbage, so the one-click updater relies on it never
    # changing. Change the launch logic in scripts/install.ps1 instead.
    digest = hashlib.sha256((ROOT / "DailyDigest-Windows.bat").read_bytes()).hexdigest()
    assert digest == WINDOWS_LAUNCHER_SHA256


WINDOWS_LAUNCHER_SHA256 = "7e44eb7d1c551befe864ebeccede2d1129839b2a9d99be8667f8b386bfb4f35d"


def test_update_routes_are_local_and_csrf_protected(monkeypatch):
    from dailydigest import web
    from tests.test_web import _request

    class Fake:
        def __init__(self):
            self.applied = []

        def maybe_check(self, enabled):
            self.enabled = enabled

        def status(self, *, busy):
            return {"available": True, "busy": busy}

        def apply(self, *, busy, address, on_failure):
            self.applied.append((busy, address))
            on_failure()
            return {"phase": "idle"}

    fake = Fake()
    monkeypatch.setattr(web.updates, "UPDATER", fake)
    response = web.update_status(_request("GET", "/update/status"))
    assert json.loads(response.body) == {"available": True, "busy": False}
    assert response.headers["cache-control"] == "no-store"
    with pytest.raises(web.HTTPException):
        web.update_apply(_request("POST", "/update/apply"))
    assert fake.applied == []
    web.update_apply(_request("POST", "/update/apply", headers={"x-csrf-token": web._CSRF_TOKEN}))
    assert fake.applied and fake.applied[0][0] is False
    assert not web._BREW_LOCK.locked() and not web._COMPUTE_LOCK.locked(), "Locks return when the update does not start"


def test_native_pages_carry_the_update_notice_and_browser_pages_do_not(monkeypatch):
    from dailydigest import web

    template = web.templates.env.get_template("setup.html.j2")
    assert 'id="dd-update"' in (ROOT / "templates/_update_notice.html.j2").read_text()
    base = (ROOT / "templates/_base.html.j2").read_text()
    assert '{% if local_updates_enabled is defined and local_updates_enabled() %}{% include "_update_notice.html.j2" %}' in base
    monkeypatch.setattr(web.updates.sys, "platform", "emscripten")
    assert web.templates.env.globals["local_updates_enabled"]() is False
    assert template is not None
