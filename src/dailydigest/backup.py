"""Explicit local backups, with a SQLite snapshot and selected private sidecars.

Stop DailyDigest (including scheduled brews) before calling this helper. SQLite's
backup API captures committed WAL data, but cannot atomically capture separate
profile/reason files. This module neither restores nor migrates user state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_SIDECARS = ("profile.yaml", "opportunities.yaml", "vote_reasons.json")


def _private_file(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _copy_private(source: Path, destination: Path) -> None:
    with source.open("rb") as original, _private_file(destination) as output:
        shutil.copyfileobj(original, output)


def _credential_filename(path: Path) -> bool:
    name = path.name.lower()
    return (
        name == ".env" or name.startswith(".env.") or name.endswith(".env")
        or name in {"auth.json", "credentials.json", "credentials", "secrets.yaml", "secrets.yml"}
    )


def create_backup(
    data_dir: str | Path,
    destination: str | Path,
    *,
    database_name: str = "digest.db",
    profile_paths: Iterable[str | Path] = (),
    config_paths: Iterable[str | Path] = (),
    credential_paths: Iterable[str | Path] = (),
) -> Path:
    """Back up stopped-app state into a new directory outside ``data_dir``.

    The database and known profile/reason sidecars are included automatically.
    Custom profile/config paths must be passed explicitly; credentials require
    ``credential_paths``. Arbitrary data files, model caches and environment
    variables are never collected. POSIX directories/files use 0700/0600; on
    Windows choose a private destination with appropriate inherited ACLs.

    A complete backup has ``manifest.json``. Failure leaves an incomplete
    directory for inspection and never overwrites or automatically deletes it.
    """
    source_dir = Path(data_dir).resolve(strict=True)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    requested_destination = Path(destination)
    if requested_destination.is_symlink():
        raise ValueError("Backup destination must not be a symlink")
    target = requested_destination.resolve()
    if target == source_dir or target.is_relative_to(source_dir):
        raise ValueError("Backup destination must be outside the source data directory")
    if target.exists():
        raise FileExistsError(target)
    if not database_name or Path(database_name).name != database_name:
        raise ValueError("database_name must be a filename inside data_dir")

    database = source_dir / database_name
    files: dict[str, Path] = {}

    def include(source: str | Path, relative: str, *, credentials: bool = False) -> None:
        path = Path(source)
        if path.is_symlink():
            raise ValueError(f"Backup inputs must not be symlinks: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if _credential_filename(path) and not credentials:
            raise ValueError("Pass credential files only through credential_paths")
        if relative in files:
            raise ValueError(f"Multiple inputs have the same backup path: {relative}")
        files[relative] = path.resolve()

    include(database, f"data/{database_name}")
    for name in _SIDECARS:
        path = source_dir / name
        if path.exists() or path.is_symlink():
            include(path, f"data/{name}")
    for category, paths in (
        ("profiles", profile_paths), ("config", config_paths), ("credentials", credential_paths),
    ):
        for source in paths:
            include(source, f"{category}/{Path(source).name}", credentials=category == "credentials")

    target.mkdir(mode=0o700)
    copied_database = target / "data" / database_name
    with _private_file(copied_database):
        pass
    # mode=ro avoids creating a missing source database. backup(), unlike a raw
    # copy of digest.db, includes committed pages still in its WAL.
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as original:
        with closing(sqlite3.connect(copied_database)) as copied:
            original.backup(copied)
            copied.execute("PRAGMA journal_mode=DELETE")
            if copied.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite backup integrity check failed")

    records = {}
    for relative, source in files.items():
        output = target / relative
        if output != copied_database:
            _copy_private(source, output)
        with output.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        records[relative] = {"source": str(source), "sha256": checksum, "bytes": output.stat().st_size}
    manifest = {
        "format_version": 1,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "consistency_requirement": "DailyDigest and scheduled writers stopped during capture",
        "credentials_included": any(name.startswith("credentials/") for name in files),
        "files": records,
    }
    pending_manifest = target / "manifest.json.tmp"
    with _private_file(pending_manifest) as output:
        output.write((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        output.flush()
        os.fsync(output.fileno())
    pending_manifest.replace(target / "manifest.json")
    return target
