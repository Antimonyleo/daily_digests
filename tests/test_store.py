from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _reset_store(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod.init_db()
    return store_mod


def test_recent_items_uses_published_at_before_fetched_at(monkeypatch, tmp_path):
    store_mod = _reset_store(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)

    with store_mod.session_scope() as s:
        s.add_all(
            [
                store_mod.ItemRow(
                    source="RecentFetch",
                    section="research",
                    external_id="old-pub-new-fetch",
                    url="https://example.com/old-pub-new-fetch",
                    title="Old publication fetched today",
                    published_at=now - timedelta(days=30),
                    fetched_at=now,
                ),
                store_mod.ItemRow(
                    source="NoPub",
                    section="research",
                    external_id="no-pub-new-fetch",
                    url="https://example.com/no-pub-new-fetch",
                    title="No publication date fetched today",
                    published_at=None,
                    fetched_at=now,
                ),
                store_mod.ItemRow(
                    source="FreshPub",
                    section="research",
                    external_id="fresh-pub-old-fetch",
                    url="https://example.com/fresh-pub-old-fetch",
                    title="Fresh publication fetched earlier",
                    published_at=now,
                    fetched_at=now - timedelta(days=30),
                ),
            ]
        )

    rows = store_mod.recent_items(days=2)
    external_ids = {row.external_id for row in rows}

    assert "old-pub-new-fetch" not in external_ids
    assert "no-pub-new-fetch" in external_ids
    assert "fresh-pub-old-fetch" in external_ids
