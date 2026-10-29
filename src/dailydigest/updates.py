"""Release notes and the one-click update path for local installations.

The check downloads one public file (the release notes on the official
repository) and sends nothing about the reader. Applying an update replaces
program files only: the reader's files in ``data/``, ``.env`` files and the
virtual environment are never changed. A copy of the reader's database and
profiles is first added in ``data/update-backups/``, every replaced or removed
program file is backed up in ``.update-backups/``, and any failure restores the
previous files before the running server is touched.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

logger = logging.getLogger(__name__)

RELEASE_NOTES_PATH = Path(__file__).with_name("release_notes.json")
INSTALL_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = "Antimonyleo/daily_digests"
OFFICIAL_BRANCH = "main"
# A fork, or an end-to-end test, can point both URLs at its own copy.
NOTES_URL = os.environ.get("DD_UPDATE_NOTES_URL") or (
    f"https://raw.githubusercontent.com/{OFFICIAL_REPO}/{OFFICIAL_BRANCH}"
    "/src/dailydigest/release_notes.json"
)
ARCHIVE_URL = (os.environ.get("DD_UPDATE_ARCHIVE_URL")
               or f"https://codeload.github.com/{OFFICIAL_REPO}/zip/refs/heads/{OFFICIAL_BRANCH}")
CHECK_INTERVAL = timedelta(hours=12)
MAX_NOTES_BYTES = 256 * 1024
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
BACKUP_DIR = ".update-backups"
KEEP_BACKUPS = 3
# Reader state and the local environment. An update never writes these paths.
PROTECTED_TOP_LEVEL = {"data", ".venv", ".git", BACKUP_DIR}
# Hashes of the program files the last update installed, kept beside the
# backups. They tell a hand-edited file from an old release's copy, and which
# files a newer release dropped.
INSTALLED_MANIFEST = "installed-files.json"
# Folders only a release writes. Before the first manifest exists, files here
# that the new release no longer has are removed (and backed up).
RELEASE_OWNED = (("src", "dailydigest"), ("templates",))
LOCAL_SOURCES = "data/sources.local.yaml"
# Inside the reader's data folder: copies of their data taken before updates.
DATA_BACKUP_DIR = "update-backups"
DOCKER_UPDATE_STEPS = (
    "Download the new version into your DailyDigest folder (or run git pull there), "
    "then run: docker compose up --build -d"
)

_VERSION_RE = re.compile(r"^\d{4}\.\d{1,2}\.\d{1,2}(?:\.\d{1,3})?$")


def version_key(version: str) -> tuple[int, ...]:
    if not _VERSION_RE.fullmatch(str(version or "")):
        return ()
    return tuple(int(part) for part in version.split("."))


def parse_release_notes(payload: Any) -> list[dict[str, Any]]:
    """Validated releases, newest first. Malformed entries are dropped."""
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ValueError("Unsupported release notes")
    releases = []
    for row in payload.get("releases") or []:
        if not isinstance(row, dict) or not version_key(row.get("version", "")):
            continue
        notes = [str(note).strip()[:300] for note in (row.get("notes") or [])
                 if isinstance(note, str) and note.strip()][:8]
        releases.append({
            "version": row["version"],
            "date": str(row.get("date") or "")[:10],
            "title": str(row.get("title") or "").strip()[:120],
            "notes": notes,
        })
    if not releases:
        raise ValueError("Release notes contain no valid release")
    return sorted(releases, key=lambda row: version_key(row["version"]), reverse=True)


def load_release_notes(path: Path = RELEASE_NOTES_PATH) -> list[dict[str, Any]]:
    return parse_release_notes(json.loads(path.read_text(encoding="utf-8")))


def current_version(path: Path = RELEASE_NOTES_PATH) -> str:
    try:
        return load_release_notes(path)[0]["version"]
    except (OSError, ValueError):
        return ""


def releases_since(releases: list[dict[str, Any]], version: str) -> list[dict[str, Any]]:
    floor = version_key(version)
    return [row for row in releases if version_key(row["version"]) > floor]


def install_mode(root: Path = INSTALL_ROOT) -> str:
    """How this copy was installed, which decides how it can be updated."""
    if sys.platform == "emscripten":
        return "browser"
    if Path("/.dockerenv").exists() or os.environ.get("DD_IN_DOCKER") == "1":
        return "docker"
    if not (root / "pyproject.toml").is_file() or not (root / "src" / "dailydigest").is_dir():
        return "unsupported"
    if (root / ".git").exists():
        return "git"
    return "archive"


def _git(root: Path, *args: str, run: Callable[..., Any] = subprocess.run) -> str:
    result = run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "git failed").strip()[:300])
    return result.stdout.strip()


def git_blocker(root: Path = INSTALL_ROOT, run: Callable[..., Any] = subprocess.run) -> str:
    """Why a Git checkout cannot fast-forward safely, or "" when it can."""
    try:
        changed = _git(root, "status", "--porcelain", "--untracked-files=no", run=run)
        if changed:
            # The output is stripped, so the first line may have lost a space.
            paths = {line.split()[-1] for line in changed.splitlines() if line.strip()}
            if paths <= {"config/sources.yaml"}:
                return ("Your edits to config/sources.yaml stop DailyDigest from updating itself. "
                        f"Move your additions to {LOCAL_SOURCES}, which updates never touch, "
                        "and undo them in config/sources.yaml (git checkout config/sources.yaml).")
            return ("Files in your DailyDigest folder were changed, so it can't update "
                    "itself. Update it the way you installed it.")
        _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", run=run)
    except Exception:  # noqa: BLE001
        return ("This copy doesn't follow the official version, so it can't update itself. "
                "Update it the way you installed it.")
    return ""


def _find_uv() -> str | None:
    candidates = [os.environ.get("UV"), shutil.which("uv")]
    home = Path.home()
    candidates += [str(home / ".local" / "bin" / name) for name in ("uv", "uv.exe")]
    candidates.append(str(home / ".cargo" / "bin" / "uv"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _sync_environment(root: Path, run: Callable[..., Any] = subprocess.run) -> None:
    uv = _find_uv()
    if uv is None:
        raise RuntimeError("The uv tool that runs DailyDigest could not be found.")
    result = run([uv, "sync", "--frozen", "--no-dev", "--inexact"], cwd=str(root),
                 capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError("Installing the new version's components failed: "
                           + (result.stderr or result.stdout or "").strip()[-300:])


def _is_protected(relative: PurePosixPath) -> bool:
    top = relative.parts[0]
    return top in PROTECTED_TOP_LEVEL or top.startswith(".env") or "__pycache__" in relative.parts


def extract_archive(content: bytes) -> dict[PurePosixPath, bytes]:
    """Files of a GitHub branch archive keyed by path inside the repository."""
    files: dict[PurePosixPath, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [info for info in archive.infolist() if not info.is_dir()]
        if not names:
            raise ValueError("The downloaded update is empty.")
        prefix = PurePosixPath(names[0].filename).parts[0]
        total = 0
        for info in names:
            path = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if (path.is_absolute() or ".." in path.parts or len(path.parts) < 2
                    or path.parts[0] != prefix or mode == 0o120000):
                raise ValueError("The downloaded update contains an unsafe path.")
            total += info.file_size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("The downloaded update is unexpectedly large.")
            files[PurePosixPath(*path.parts[1:])] = archive.read(info)
    pyproject = files.get(PurePosixPath("pyproject.toml"), b"").decode("utf-8", "replace")
    if not re.search(r'^name\s*=\s*"dailydigest"', pyproject, flags=re.MULTILINE):
        raise ValueError("The download is not a DailyDigest release.")
    return files


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def disk_hashes(root: Path, paths: Any) -> dict[str, str]:
    """SHA-256 of each listed file that exists, read once per update and
    shared by the edited, dropped and changed-file checks."""
    hashes = {}
    for path in paths:
        target = root.joinpath(*PurePosixPath(path).parts)
        if target.is_file():
            hashes[str(path)] = _sha256(target.read_bytes())
    return hashes


def load_installed_manifest(root: Path) -> dict[str, str] | None:
    """Program-file hashes from the last update, or None before the first one."""
    try:
        payload = json.loads((root / BACKUP_DIR / INSTALLED_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        return None
    return {str(path): str(digest) for path, digest in files.items()}


def write_installed_manifest(root: Path, files: dict[PurePosixPath, bytes]) -> None:
    manifest = {"files": {relative.as_posix(): _sha256(content)
                          for relative, content in sorted(files.items()) if not _is_protected(relative)}}
    target = root / BACKUP_DIR / INSTALLED_MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, target)


def edited_files(root: Path, files: dict[PurePosixPath, bytes],
                 manifest: dict[str, str] | None,
                 on_disk: dict[str, str] | None = None) -> list[PurePosixPath]:
    """Installed files changed by hand that this update will replace.

    Only a manifest from an earlier update can tell; the replaced copies are
    backed up either way.
    """
    if not manifest:
        return []
    if on_disk is None:
        on_disk = disk_hashes(root, [path.as_posix() for path in files if path.as_posix() in manifest])
    edited = []
    for relative, content in sorted(files.items()):
        recorded = manifest.get(relative.as_posix())
        current = on_disk.get(relative.as_posix())
        if _is_protected(relative) or recorded is None or current is None:
            continue
        if current != recorded and current != _sha256(content):
            edited.append(relative)
    return edited


def stale_files(root: Path, files: dict[PurePosixPath, bytes],
                manifest: dict[str, str] | None,
                on_disk: dict[str, str] | None = None) -> list[PurePosixPath]:
    """Program files an earlier release installed that the new one dropped.

    With a manifest, only unedited files it lists are removed. Before the first
    manifest, only the release-owned package and template folders are swept.
    """
    if manifest:
        if on_disk is None:
            on_disk = disk_hashes(root, manifest)
        candidates = [PurePosixPath(path) for path, digest in manifest.items() if on_disk.get(path) == digest]
    else:
        candidates = [PurePosixPath(path.relative_to(root).as_posix())
                      for parts in RELEASE_OWNED if root.joinpath(*parts).is_dir()
                      for path in root.joinpath(*parts).rglob("*") if path.is_file()]
    return sorted(relative for relative in candidates
                  if relative not in files and not _is_protected(relative)
                  and not relative.name.startswith(".dd-update-"))


def replace_files(root: Path, files: dict[PurePosixPath, bytes], backup: Path,
                  remove: list[PurePosixPath] = (),
                  on_disk: dict[str, str] | None = None) -> list[Path]:
    """Write changed program files and remove dropped ones; return the paths
    written. Every replaced or removed file is backed up for rollback."""
    written: list[Path] = []
    try:
        for relative in remove:
            target = root.joinpath(*relative.parts)
            if _is_protected(relative) or not target.is_file():
                continue
            saved = backup.joinpath("files", *relative.parts)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
            with (backup / "removed.txt").open("a", encoding="utf-8") as removed:
                removed.write(relative.as_posix() + "\n")
            target.unlink()
        for relative, content in sorted(files.items()):
            if _is_protected(relative):
                continue
            target = root.joinpath(*relative.parts)
            if (on_disk.get(relative.as_posix()) == _sha256(content) if on_disk is not None
                    else target.is_file() and target.read_bytes() == content):
                continue
            if target.exists():
                saved = backup.joinpath("files", *relative.parts)
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
            else:
                with (backup / "created.txt").open("a", encoding="utf-8") as created:
                    created.write(relative.as_posix() + "\n")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=target.parent, prefix=".dd-update-")
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            if target.exists():
                shutil.copymode(target, temp)
            elif relative.suffix in {".sh", ".command"}:
                os.chmod(temp, 0o755)
            os.replace(temp, target)
            written.append(target)
    except Exception:
        restore_files(root, backup)
        raise
    return written


def restore_files(root: Path, backup: Path) -> None:
    saved_root = backup / "files"
    if saved_root.is_dir():
        for saved in saved_root.rglob("*"):
            if saved.is_file():
                target = root / saved.relative_to(saved_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(saved, target)
    created = backup / "created.txt"
    if created.is_file():
        for line in created.read_text(encoding="utf-8").splitlines():
            if line.strip():
                root.joinpath(*PurePosixPath(line.strip()).parts).unlink(missing_ok=True)


def backup_reader_data(name: str, settings: Any = None) -> Path | None:
    """Copy the reader's database and profiles before an update.

    The copy stays with the reader's other private files, in
    ``data/update-backups/<name>``; the newest copies are kept. Returns None
    when there is no database yet. The brew locks are held during an update,
    so only a rating or save made in the same second can be missed.
    """
    from .backup import _SIDECARS, create_backup

    if settings is None:
        from .config import get_settings

        settings = get_settings()
    database = Path(settings.db_path)
    if not database.is_file():
        return None
    data_dir = database.parent.resolve()
    folder = data_dir / DATA_BACKUP_DIR
    for stale in sorted(path for path in folder.glob("*") if path.is_dir())[:-(KEEP_BACKUPS - 1) or None]:
        shutil.rmtree(stale, ignore_errors=True)
    included = {data_dir / name for name in _SIDECARS}

    def extra(path: str) -> list[Path]:
        # Profiles kept elsewhere, or in data/ under another name, are copied
        # too; the sidecars in data/ are already included.
        candidate = Path(path)
        return [candidate] if candidate.is_file() and candidate.resolve() not in included else []

    # Separate folders, so two profiles with the same file name never collide.
    folder.mkdir(mode=0o700, exist_ok=True)
    return create_backup(data_dir, folder / name, database_name=database.name,
                         profile_paths=extra(settings.profile_path),
                         config_paths=extra(settings.opportunity_profile_path),
                         allow_subfolder=True)


def last_update_summary(root: Path, version: str) -> dict[str, Any] | None:
    """What the update that installed ``version`` kept aside, if it recorded it."""
    folders = sorted((path for path in (root / BACKUP_DIR).glob("*") if path.is_dir()),
                     key=lambda path: path.name, reverse=True)
    for folder in folders:
        try:
            summary = json.loads((folder / "update.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(summary, dict) and summary.get("to") == version:
            return {key: summary.get(key) for key in ("backup", "data_backup", "edited", "removed")}
        return None
    return None


def _prune_backups(root: Path) -> None:
    folders = sorted((path for path in (root / BACKUP_DIR).glob("*") if path.is_dir()),
                     key=lambda path: path.name)
    for stale in folders[:-KEEP_BACKUPS]:
        shutil.rmtree(stale, ignore_errors=True)


class Updater:
    """Background update check plus a single guarded apply at a time."""

    def __init__(self, *, root: Path = INSTALL_ROOT, notes_path: Path = RELEASE_NOTES_PATH,
                 http_get: Callable[[str, int], bytes] | None = None,
                 run: Callable[..., Any] = subprocess.run,
                 restart: Callable[[str, int], None] | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 backup_data: Callable[[str], Path | None] = backup_reader_data):
        self.root, self.notes_path = root, notes_path
        self._backup_data = backup_data
        self._get = http_get or _http_get
        self._run = run
        self._restart = restart or restart_server
        self._clock = clock
        self._lock = threading.Lock()
        self.latest: list[dict[str, Any]] = []
        self.checked_at: datetime | None = None
        self.check_error = ""
        self.checking = False
        self.phase = "idle"  # idle | downloading | installing | restarting | failed
        self.message = ""

    # -- checking ---------------------------------------------------------
    def check_now(self) -> None:
        try:
            payload = json.loads(self._get(NOTES_URL, MAX_NOTES_BYTES))
            releases = parse_release_notes(payload)
            with self._lock:
                self.latest, self.check_error = releases, ""
        except Exception as exc:  # noqa: BLE001
            logger.info("Update check unavailable: %s", exc)
            with self._lock:
                self.check_error = "The update check couldn't reach the DailyDigest website."
        finally:
            with self._lock:
                self.checked_at, self.checking = self._clock(), False

    def maybe_check(self, enabled: bool) -> None:
        if not enabled or install_mode(self.root) in {"browser", "unsupported"}:
            return
        with self._lock:
            due = self.checked_at is None or self._clock() - self.checked_at >= CHECK_INTERVAL
            if self.checking or not due:
                return
            self.checking = True
        threading.Thread(target=self.check_now, name="dd-update-check", daemon=True).start()

    # -- reporting --------------------------------------------------------
    def status(self, *, busy: bool) -> dict[str, Any]:
        mode = install_mode(self.root)
        current = current_version(self.notes_path)
        with self._lock:
            latest = list(self.latest)
            phase, message, error = self.phase, self.message, self.check_error
        try:
            installed = load_release_notes(self.notes_path)[0]
        except (OSError, ValueError):
            installed = None
        newer = releases_since(latest, current)
        available = bool(newer)
        blocker = ""
        instructions = ""
        if mode == "docker":
            instructions = DOCKER_UPDATE_STEPS
        elif mode == "git" and available:
            blocker = git_blocker(self.root, run=self._run)
        elif mode not in {"git", "archive"}:
            instructions = "Update DailyDigest the way you installed it."
        if busy and not blocker:
            blocker = "Finish or stop the current brew, then update."
        return {
            "mode": mode,
            "current": current,
            "latest": newer[0]["version"] if newer else current,
            "available": available,
            "releases": newer[:5],
            "can_apply": available and mode in {"git", "archive"} and not blocker
                         and phase in {"idle", "failed"},
            "blocked_reason": blocker,
            "instructions": instructions,
            "phase": phase,
            "message": message,
            "check_error": error,
            "checking": self.checking,
            "installed": installed,
            "last_update": last_update_summary(self.root, current) if mode in {"git", "archive"} else None,
        }

    # -- applying ---------------------------------------------------------
    def _set(self, phase: str, message: str) -> None:
        with self._lock:
            self.phase, self.message = phase, message

    def apply(self, *, busy: bool, address: tuple[str, int],
              on_failure: Callable[[], None] = lambda: None) -> dict[str, Any]:
        """Start the update in the background; ``on_failure`` runs if it stops.

        The caller holds the brew locks until the process restarts, so no brew
        can begin against half-replaced files.
        """
        state = self.status(busy=busy)
        if not state["can_apply"]:
            on_failure()
            return state
        self._set("downloading", "Downloading the new version…")
        threading.Thread(target=self._apply, args=(state["mode"], state["latest"], address, on_failure),
                         name="dd-update-apply", daemon=True).start()
        return self.status(busy=False)

    def _apply(self, mode: str, target: str, address: tuple[str, int],
               on_failure: Callable[[], None]) -> None:
        root = self.root
        stamp = self._clock().strftime("%Y%m%dT%H%M%S")
        backup = root / BACKUP_DIR / f"{stamp}-from-{current_version(self.notes_path) or 'unknown'}"
        previous_head = ""
        files: dict[PurePosixPath, bytes] = {}
        summary: dict[str, Any] = {"from": current_version(self.notes_path), "to": target,
                                   "backup": backup.relative_to(root).as_posix(),
                                   "data_backup": "", "edited": [], "removed": 0}
        try:
            if mode == "git":
                previous_head = _git(root, "rev-parse", "HEAD", run=self._run)
                _git(root, "fetch", "--quiet", run=self._run)
                self._copy_reader_data(backup, summary)
                self._set("installing", "Installing the new version…")
                _git(root, "merge", "--ff-only", "@{u}", run=self._run)
            else:
                files = extract_archive(self._get(ARCHIVE_URL, MAX_ARCHIVE_BYTES))
                notes = files.get(PurePosixPath("src/dailydigest/release_notes.json"))
                downloaded = parse_release_notes(json.loads(notes or b"{}"))[0]["version"]
                if version_key(downloaded) < version_key(target):
                    raise RuntimeError("The download is older than the announced version. Try again later.")
                manifest = load_installed_manifest(root)
                on_disk = disk_hashes(root, {*(manifest or {}), *(path.as_posix() for path in files)})
                stale = stale_files(root, files, manifest, on_disk)
                summary["edited"] = [path.as_posix() for path in edited_files(root, files, manifest, on_disk)]
                summary["removed"] = len(stale)
                self._copy_reader_data(backup, summary)
                self._set("installing", "Installing the new version…")
                replace_files(root, files, backup, remove=stale, on_disk=on_disk)
            try:
                _sync_environment(root, run=self._run)
            except Exception:
                self._rollback(mode, backup, previous_head)
                raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Update failed: %s", exc)
            self._set("failed", f"The update didn't finish, and nothing was changed. {exc}"[:500])
            on_failure()
            return
        try:
            if files:
                write_installed_manifest(root, files)
            (backup / "update.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not record the update summary: %s", exc)
        _prune_backups(root)
        self._set("restarting", "Restarting DailyDigest…")
        time.sleep(1.0)  # Let the browser read the restarting state first.
        self._restart(*address)

    def _copy_reader_data(self, backup: Path, summary: dict[str, Any]) -> None:
        """Copy the reader's data before any program file changes."""
        self._set("downloading", "Copying your data before updating…")
        backup.mkdir(parents=True, exist_ok=True)
        try:
            copied = self._backup_data(backup.name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Your data couldn't be copied first ({exc}).") from exc
        if copied is not None:
            copied, root = Path(copied).resolve(), self.root.resolve()
            summary["data_backup"] = (copied.relative_to(root).as_posix()
                                      if copied.is_relative_to(root) else str(copied))

    def _rollback(self, mode: str, backup: Path, previous_head: str) -> None:
        try:
            if mode == "git" and previous_head:
                _git(self.root, "reset", "--keep", previous_head, run=self._run)
            elif mode == "archive":
                restore_files(self.root, backup)
            _sync_environment(self.root, run=self._run)
        except Exception as exc:  # noqa: BLE001
            logger.error("Restoring the previous version needs attention: %s", exc)


def _http_get(url: str, limit: int) -> bytes:
    import httpx

    with httpx.Client(timeout=30.0, follow_redirects=True,
                      headers={"User-Agent": "dailydigest-update-check"}) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > limit:
                    raise ValueError("Update response exceeded its size limit")
                chunks.append(chunk)
    return b"".join(chunks)


def restart_server(host: str, port: int) -> None:
    """Start the updated code in place of this process."""
    if os.name == "nt":
        # Windows cannot replace a running process image. Open the launcher in a
        # new window, then leave; the old window closes when this process exits.
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(INSTALL_ROOT / "scripts" / "start.ps1"), "-NoBrowser",
             "-HostAddress", host, "-Port", str(port)],
            cwd=str(INSTALL_ROOT), creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
        os._exit(0)
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    if "start" in argv and "--no-browser" not in argv:
        argv.append("--no-browser")  # The open tab reloads itself.
    os.execv(sys.executable, argv)


UPDATER = Updater()
