from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone


def _reset_store(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
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


def test_exclude_reviewed_items_removes_saved_feedback(monkeypatch, tmp_path):
    store_mod = _reset_store(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)

    with store_mod.session_scope() as s:
        reviewed = store_mod.ItemRow(
            source="Reviewed",
            section="research",
            external_id="reviewed",
            url="https://example.com/reviewed",
            title="Already reviewed",
            published_at=now,
        )
        fresh = store_mod.ItemRow(
            source="Fresh",
            section="research",
            external_id="fresh",
            url="https://example.com/fresh",
            title="Not reviewed",
            published_at=now,
        )
        s.add_all([reviewed, fresh])
        s.flush()
        s.add(store_mod.VoteRow(item_id=reviewed.id, value=0))

    rows = store_mod.recent_items(days=2)
    filtered = store_mod.exclude_reviewed_items(rows)

    assert {row.external_id for row in filtered} == {"fresh"}


def test_review_filters_use_latest_legacy_vote_rows(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy_store.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            source VARCHAR NOT NULL,
            section VARCHAR NOT NULL,
            external_id VARCHAR NOT NULL,
            url VARCHAR NOT NULL,
            title TEXT NOT NULL,
            abstract TEXT DEFAULT '',
            authors TEXT DEFAULT '',
            published_at DATETIME,
            fetched_at DATETIME,
            summary TEXT DEFAULT '',
            score FLOAT,
            digest_id VARCHAR,
            item_label VARCHAR
        );
        CREATE TABLE votes (
            id INTEGER PRIMARY KEY,
            item_id INTEGER NOT NULL,
            value INTEGER NOT NULL,
            created_at DATETIME
        );
        CREATE TABLE digests (
            id VARCHAR PRIMARY KEY,
            created_at DATETIME,
            item_count INTEGER DEFAULT 0,
            sent_at DATETIME
        );
        CREATE TABLE digest_items (
            id INTEGER PRIMARY KEY,
            digest_id VARCHAR NOT NULL,
            item_id INTEGER NOT NULL,
            item_label VARCHAR NOT NULL,
            score FLOAT,
            created_at DATETIME
        );
        INSERT INTO items (
            id, source, section, external_id, url, title, abstract, authors,
            published_at, fetched_at, summary, score, digest_id, item_label
        ) VALUES
            (1, 'Legacy', 'research', 'old-bad-latest-good', 'https://example.com/good',
             'Old bad latest good', 'Abstract', '', '2026-05-12 00:00:00',
             '2026-05-12 00:00:00', '', NULL, NULL, NULL),
            (2, 'Legacy', 'research', 'old-good-latest-neutral', 'https://example.com/neutral',
             'Old good latest neutral', 'Abstract', '', '2026-05-12 00:00:00',
             '2026-05-12 00:00:00', '', NULL, NULL, NULL),
            (3, 'Legacy', 'research', 'shown-unvoted', 'https://example.com/unvoted',
             'Shown but unvoted', 'Abstract', '', '2026-05-12 00:00:00',
             '2026-05-12 00:00:00', '', NULL, NULL, NULL);
        INSERT INTO votes (id, item_id, value, created_at)
        VALUES
            (1, 1, -1, '2026-05-12 10:00:00'),
            (2, 1, 1, '2026-05-12 11:00:00'),
            (3, 2, 1, '2026-05-12 10:00:00'),
            (4, 2, 0, '2026-05-12 11:00:00');
        INSERT INTO digests (id, item_count, sent_at)
        VALUES ('2026-05-12', 3, '2026-05-12 12:00:00');
        INSERT INTO digest_items (digest_id, item_id, item_label, score, created_at)
        VALUES
            ('2026-05-12', 1, 'R1', 0.9, '2026-05-12 12:00:00'),
            ('2026-05-12', 2, 'R2', 0.8, '2026-05-12 12:00:00'),
            ('2026-05-12', 3, 'R3', 0.7, '2026-05-12 12:00:00');
        """
    )
    conn.commit()
    conn.close()

    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    monkeypatch.setenv("DB_PATH", str(db_path))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    store_mod.init_db()
    with store_mod.session_scope() as s:
        rows = s.execute(store_mod.select(store_mod.ItemRow)).scalars().all()
        for row in rows:
            s.expunge(row)

    reviewed_filtered = store_mod.exclude_reviewed_items(rows)
    shown_filtered = store_mod.exclude_previously_shown(rows, days_lookback=365)

    assert {row.external_id for row in reviewed_filtered} == {
        "old-bad-latest-good",
        "shown-unvoted",
    }
    assert {row.external_id for row in shown_filtered} == {
        "old-bad-latest-good",
        "shown-unvoted",
    }
