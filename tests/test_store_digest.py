from __future__ import annotations

from dailydigest import config as config_mod
from dailydigest import store as store_mod


def _reset_store(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None


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
