from __future__ import annotations

import sqlite3

from dailydigest import config as config_mod
from dailydigest import store as store_mod


def _reset_store(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False


def _item(source: str, external_id: str, title: str = "Item") -> store_mod.ItemRow:
    return store_mod.ItemRow(
        source=source,
        section="research",
        external_id=external_id,
        url=f"https://example.com/{external_id}",
        title=title,
        abstract="",
    )


def test_write_digest_replaces_existing_item_assignments(monkeypatch, tmp_path):
    _reset_store(monkeypatch, tmp_path)
    store_mod.init_db()

    with store_mod.session_scope() as s:
        first = _item("Test", "first", "First")
        second = _item("Test", "second", "Second")
        s.add_all([first, second])
        s.flush()
        first_id = int(first.id)
        second_id = int(second.id)

    store_mod.write_digest("2026-05-05", [("R1", first_id), ("R2", second_id)])
    store_mod.write_digest("2026-05-05", [("R1", second_id)])

    with store_mod.session_scope() as s:
        first = s.get(store_mod.ItemRow, first_id)
        second = s.get(store_mod.ItemRow, second_id)
        assert first.digest_id is None
        assert first.item_label is None
        assert second.digest_id == "2026-05-05"
        assert second.item_label == "R1"


def test_write_digest_persists_rank_score(monkeypatch, tmp_path):
    _reset_store(monkeypatch, tmp_path)
    store_mod.init_db()

    with store_mod.session_scope() as s:
        item = _item("Nature", "rank-score", "Ranked")
        s.add(item)
        s.flush()
        item_id = int(item.id)

    store_mod.write_digest("2026-05-05", [("R1", item_id, 0.82)])

    with store_mod.session_scope() as s:
        item = s.get(store_mod.ItemRow, item_id)
        assert item.score == 0.82


def test_write_and_load_digest_features(monkeypatch, tmp_path):
    _reset_store(monkeypatch, tmp_path)
    store_mod.init_db()

    with store_mod.session_scope() as s:
        item = _item("Nature", "feature-row", "Feature row")
        s.add(item)
        s.flush()
        item_id = int(item.id)

    features = {
        "priority": "Must read",
        "why": ["Strong topic match", "Fresh primary result"],
        "components": {"topic": 0.8, "source": 0.9},
    }
    store_mod.write_digest_features("2026-05-05", [("R1", item_id, 0.87, features)])

    loaded = store_mod.load_digest_features("2026-05-05")

    assert loaded[item_id]["priority"] == "Must read"
    assert loaded[item_id]["why"] == ["Strong topic match", "Fresh primary result"]
    assert loaded[item_id]["components"]["source"] == 0.9


def test_init_db_adds_columns_to_legacy_sqlite_database(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            source VARCHAR NOT NULL,
            section VARCHAR NOT NULL,
            external_id VARCHAR NOT NULL,
            url VARCHAR NOT NULL,
            title TEXT NOT NULL
        );
        CREATE TABLE digests (id VARCHAR PRIMARY KEY);
        CREATE TABLE votes (
            id INTEGER PRIMARY KEY,
            item_id INTEGER NOT NULL,
            value INTEGER NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("DB_PATH", str(db_path))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

    store_mod.init_db()

    conn = sqlite3.connect(db_path)
    item_columns = {row[1] for row in conn.execute("PRAGMA table_info(items)")}
    digest_columns = {row[1] for row in conn.execute("PRAGMA table_info(digests)")}
    vote_columns = {row[1] for row in conn.execute("PRAGMA table_info(votes)")}
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()

    assert {"summary", "score", "digest_id", "item_label", "fetched_at"} <= item_columns
    assert {"created_at", "item_count", "sent_at"} <= digest_columns
    assert "created_at" in vote_columns
    assert "digest_items" in tables


def test_write_digest_keeps_label_snapshot_for_old_digest(monkeypatch, tmp_path):
    _reset_store(monkeypatch, tmp_path)
    store_mod.init_db()

    with store_mod.session_scope() as s:
        item = _item("Nature", "repeat", "Repeated")
        s.add(item)
        s.flush()
        item_id = int(item.id)

    store_mod.write_digest("2026-05-05", [("R1", item_id, 0.80)])
    store_mod.write_digest("2026-05-06", [("R2", item_id, 0.90)])

    with store_mod.session_scope() as s:
        old_item_id = s.execute(
            store_mod.select(store_mod.DigestItemRow.item_id).where(
                store_mod.DigestItemRow.digest_id == "2026-05-05",
                store_mod.DigestItemRow.item_label == "R1",
            )
        ).scalar_one()
        current_item = s.get(store_mod.ItemRow, item_id)

    assert old_item_id == item_id
    assert current_item.digest_id == "2026-05-06"
    assert current_item.item_label == "R2"
