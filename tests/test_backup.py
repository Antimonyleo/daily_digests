"""Explicit backups preserve a SQLite snapshot and selected private sidecars."""

import json
import os
import shutil
import sqlite3

import pytest


def database(data_dir):
    data_dir.mkdir()
    connection = sqlite3.connect(data_dir / "digest.db")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE example(value TEXT)")
    connection.execute("INSERT INTO example VALUES ('committed in WAL')")
    connection.commit()
    return connection


def test_backup_includes_committed_wal_and_explicit_sidecars_without_secrets(tmp_path):
    from dailydigest.backup import create_backup

    data = tmp_path / "data"
    connection = database(data)
    try:
        assert (data / "digest.db-wal").stat().st_size > 0
        (data / "profile.yaml").write_text("bio: Private reader\n")
        (data / "vote_reasons.json").write_text('{"7": ["too_contextual"]}\n')
        (data / ".env").write_text("LLM_API_KEY=not-for-backup\n")
        (data / "unknown-secret.txt").write_text("not-for-backup")
        config = tmp_path / "sources.yaml"
        config.write_text("research: []\n")
        destination = create_backup(data, tmp_path / "backup", config_paths=[config])
        with sqlite3.connect(destination / "data" / "digest.db") as restored:
            assert restored.execute("SELECT value FROM example").fetchone()[0] == "committed in WAL"
            assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (destination / "data" / "profile.yaml").read_bytes() == (data / "profile.yaml").read_bytes()
        assert (destination / "data" / "vote_reasons.json").read_bytes() == (data / "vote_reasons.json").read_bytes()
        assert (destination / "config" / "sources.yaml").read_bytes() == config.read_bytes()
        assert not list(destination.rglob("*-wal"))
        assert not list(destination.rglob(".env"))
        assert not list(destination.rglob("unknown-secret.txt"))
        manifest = json.loads((destination / "manifest.json").read_text())
        assert manifest["complete"] is True
        assert manifest["credentials_included"] is False
        assert "data/digest.db" in manifest["files"]
    finally:
        connection.close()


@pytest.mark.parametrize("target", ["existing", "nested"])
def test_backup_rejects_overwrite_and_destination_inside_source(tmp_path, target):
    from dailydigest.backup import create_backup

    data = tmp_path / "data"
    database(data).close()
    dest = tmp_path / "existing" if target == "existing" else data / "nested"
    if target == "existing":
        dest.mkdir()
        (dest / "keep.txt").write_text("keep")
    with pytest.raises((ValueError, FileExistsError)):
        create_backup(data, dest)
    if target == "existing":
        assert (dest / "keep.txt").read_text() == "keep"


def test_credentials_require_separate_explicit_argument(tmp_path):
    from dailydigest.backup import create_backup

    data = tmp_path / "data"
    database(data).close()
    credentials = tmp_path / ".env"
    credentials.write_text("LLM_API_KEY=fixture-secret\n")
    with pytest.raises(ValueError, match="credential"):
        create_backup(data, tmp_path / "rejected", config_paths=[credentials])
    assert not (tmp_path / "rejected").exists()
    dest = create_backup(data, tmp_path / "explicit", credential_paths=[credentials])
    assert (dest / "credentials" / ".env").read_bytes() == credentials.read_bytes()
    assert json.loads((dest / "manifest.json").read_text())["credentials_included"] is True


def test_missing_explicit_file_and_name_collisions_fail_before_creation(tmp_path):
    from dailydigest.backup import create_backup

    data = tmp_path / "data"
    database(data).close()
    with pytest.raises(FileNotFoundError):
        create_backup(data, tmp_path / "missing", profile_paths=[tmp_path / "absent.yaml"])
    assert not (tmp_path / "missing").exists()
    (data / "profile.yaml").write_text("bio: reader")
    other = tmp_path / "profile.yaml"
    other.write_text("bio: other")
    with pytest.raises(ValueError, match="same backup path"):
        create_backup(data, tmp_path / "collision", profile_paths=[other, data / "profile.yaml"])


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes are not Windows ACLs")
def test_backup_permissions_are_private(tmp_path):
    from dailydigest.backup import create_backup

    data = tmp_path / "data"
    database(data).close()
    (data / "profile.yaml").write_text("bio: reader")
    dest = create_backup(data, tmp_path / "private")
    for path in [dest, *dest.rglob("*")]:
        assert path.stat().st_mode & 0o077 == 0


def test_failed_backup_has_no_complete_manifest_and_leaves_source_unchanged(tmp_path, monkeypatch):
    from dailydigest import backup

    data = tmp_path / "data"
    database(data).close()
    profile = data / "profile.yaml"
    profile.write_bytes(b"bio: reader\n")
    original = profile.read_bytes()

    def fail(_source, _destination):
        raise OSError("injected disk failure")

    monkeypatch.setattr(backup, "_copy_private", fail)
    dest = tmp_path / "partial"
    with pytest.raises(OSError, match="disk failure"):
        backup.create_backup(data, dest)
    assert not (dest / "manifest.json").exists()
    assert profile.read_bytes() == original


def test_default_sidecar_symlink_cannot_silently_include_credentials(tmp_path):
    from dailydigest.backup import create_backup

    data = tmp_path / "data"
    database(data).close()
    credentials = tmp_path / ".env"
    credentials.write_text("LLM_API_KEY=fixture-secret")
    try:
        (data / "profile.yaml").symlink_to(credentials)
    except OSError:
        pytest.skip("This platform/account cannot create symlinks")
    with pytest.raises(ValueError, match="symlink"):
        create_backup(data, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_restore_to_fresh_temporary_location_preserves_reader_state(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from dailydigest import store
    from dailydigest.backup import create_backup

    data = tmp_path / "original"
    data.mkdir()
    engine = create_engine(f"sqlite:///{data / 'digest.db'}")
    store.Base.metadata.create_all(engine)
    with Session(engine) as session:
        item = store.ItemRow(source="test", section="research", external_id="retained",
                             title="Retained reading", url="https://example.test/retained")
        session.add(item)
        session.flush()
        item_id = item.id
        session.add_all([
            store.DigestRow(id="2026-10-10", item_count=1),
            store.VoteRow(item_id=item_id, value=1, grade=100),
            store.FeedbackEventRow(item_id=item_id, value=1, grade=100),
            store.BookmarkRow(item_id=item_id), store.KnownItemRow(item_id=item_id),
            store.DigestItemRow(item_id=item_id, digest_id="2026-10-10", item_label="R1"),
        ])
        session.commit()
    engine.dispose()
    (data / "profile.yaml").write_text("bio: Reader\nkeywords: [methods]\n")
    (data / "vote_reasons.json").write_text(json.dumps({str(item_id): ["already_known"]}))
    saved = create_backup(data, tmp_path / "backup")
    restored = tmp_path / "restored"
    shutil.copytree(saved / "data", restored)
    with sqlite3.connect(restored / "digest.db") as connection:
        for table in ("items", "votes", "feedback_events", "bookmarks", "known_items", "digests", "digest_items"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
    assert (restored / "profile.yaml").read_bytes() == (data / "profile.yaml").read_bytes()
    assert (restored / "vote_reasons.json").read_bytes() == (data / "vote_reasons.json").read_bytes()
