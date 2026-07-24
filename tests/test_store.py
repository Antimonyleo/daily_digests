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


def _add_item(store_mod, external_id: str, section: str = "research") -> int:
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Test",
            section=section,
            external_id=external_id,
            url=f"https://example.com/{external_id}",
            title=external_id,
        )
        s.add(row)
        s.flush()
        return int(row.id)


def test_empty_rebrew_deletes_stale_feature_rows(monkeypatch, tmp_path):
    """A same-date rebrew with an EMPTY slate must clear prior feature rows.

    The early ``if not feature_rows: return`` used to skip the stale-row DELETE,
    leaving displayed=0 but features=1 after a zero-result rebrew.
    """
    store_mod = _reset_store(tmp_path, monkeypatch)
    item_id = _add_item(store_mod, "feat-1")

    store_mod.write_digest_features("2026-06-01", [("R1", item_id, 0.9, {})])
    with store_mod.session_scope() as s:
        assert (
            s.query(store_mod.DigestItemFeatureRow)
            .filter_by(digest_id="2026-06-01")
            .count()
            == 1
        )

    # Rebrew the same digest_id with an empty slate.
    store_mod.write_digest_features("2026-06-01", [])
    with store_mod.session_scope() as s:
        assert (
            s.query(store_mod.DigestItemFeatureRow)
            .filter_by(digest_id="2026-06-01")
            .count()
            == 0
        )


def test_write_impressions_is_append_only_across_rebrews(monkeypatch, tmp_path):
    """Two rebrews produce TWO run_ids of immutable impression rows, while
    digest_items only reflects the latest slate."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    a = _add_item(store_mod, "imp-a")
    b = _add_item(store_mod, "imp-b")
    digest_id = "2026-06-02"

    # Run 1: two items.
    store_mod.write_digest(digest_id, [("R1", a, 0.9), ("R2", b, 0.8)])
    run1 = store_mod.write_impressions(
        digest_id,
        [("research", a, 0, 0.9), ("research", b, 1, 0.8)],
        model_version="v-test",
    )
    # Run 2 (rebrew): only one item survives.
    store_mod.write_digest(digest_id, [("R1", a, 0.95)])
    run2 = store_mod.write_impressions(
        digest_id,
        [("research", a, 0, 0.95)],
        model_version="v-test",
    )

    assert run1 != run2
    with store_mod.session_scope() as s:
        impressions = (
            s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).all()
        )
        run_ids = {r.run_id for r in impressions}
        # Append-only: both runs' rows coexist (2 + 1 = 3 rows, 2 run_ids).
        assert run_ids == {run1, run2}
        assert len(impressions) == 3
        # viewed defaults to False; model_version persisted.
        assert all(r.viewed is False for r in impressions)
        assert all(r.model_version == "v-test" for r in impressions)
        # selected defaults to True when the tuple omits the flag (selected slate).
        assert all(r.selected is True for r in impressions)

        # digest_items shows only the LATEST slate (replaced, not appended).
        digest_items = (
            s.query(store_mod.DigestItemRow).filter_by(digest_id=digest_id).all()
        )
        assert {di.item_id for di in digest_items} == {a}


def test_write_impressions_stores_selected_flag_for_candidate_pool(monkeypatch, tmp_path):
    """A 5-tuple carries the ``selected`` flag so unpicked candidates log as False."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    a = _add_item(store_mod, "sel-a")
    b = _add_item(store_mod, "sel-b")
    digest_id = "2026-06-03"

    store_mod.write_impressions(
        digest_id,
        [
            ("research", a, 0, 0.9, True),
            ("research", b, 1, 0.7, False),
        ],
        model_version="v-test",
    )

    with store_mod.session_scope() as s:
        by_item = {
            r.item_id: r
            for r in s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).all()
        }
        assert by_item[a].selected is True
        assert by_item[b].selected is False


def test_mark_impressions_viewed_updates_only_latest_run(monkeypatch, tmp_path):
    store_mod = _reset_store(tmp_path, monkeypatch)
    a = _add_item(store_mod, "view-a")
    digest_id = "2026-06-04"

    run1 = store_mod.write_impressions(
        digest_id, [("research", a, 0, 0.9)], model_version="v-test"
    )
    run2 = store_mod.write_impressions(
        digest_id, [("research", a, 0, 0.95)], model_version="v-test"
    )
    assert run1 != run2

    updated = store_mod.mark_impressions_viewed(digest_id)
    assert updated == 1

    with store_mod.session_scope() as s:
        by_run = {
            r.run_id: r
            for r in s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).all()
        }
        # Only the latest run's rows are flagged viewed.
        assert by_run[run2].viewed is True
        assert by_run[run1].viewed is False

    # No impressions for an unknown digest → 0 rows updated, no error.
    assert store_mod.mark_impressions_viewed("2099-01-01") == 0


def test_mark_impressions_viewed_only_flags_selected_rows(monkeypatch, tmp_path):
    """Unselected candidate-pool rows stay viewed=False; only shown rows flip."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    shown = _add_item(store_mod, "view-shown")
    candidate = _add_item(store_mod, "view-candidate")
    digest_id = "2026-06-05"

    store_mod.write_impressions(
        digest_id,
        [
            ("research", shown, 0, 0.9, True),
            ("research", candidate, 1, 0.4, False),
        ],
        model_version="v-test",
    )

    updated = store_mod.mark_impressions_viewed(digest_id)
    # Only the single selected/displayed row is marked viewed.
    assert updated == 1

    with store_mod.session_scope() as s:
        by_item = {
            r.item_id: r
            for r in s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).all()
        }
    assert by_item[shown].selected is True
    assert by_item[shown].viewed is True
    # Unselected candidate never shown → must remain viewed=False.
    assert by_item[candidate].selected is False
    assert by_item[candidate].viewed is False


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
             '2026-05-12 00:00:00', '', NULL, NULL, NULL),
            (4, 'Legacy', 'research', 'never-shown', 'https://example.com/never',
             'Never shown', 'Abstract', '', '2026-05-12 00:00:00',
             '2026-05-12 00:00:00', '', NULL, NULL, NULL);
        INSERT INTO votes (id, item_id, value, created_at)
        VALUES
            (1, 1, -1, '2026-05-12 10:00:00'),
            (2, 1, 1, '2026-05-12 11:00:00'),
            (3, 2, 1, '2026-05-12 10:00:00'),
            (4, 2, 0, '2026-05-12 11:00:00');
        INSERT INTO digests (id, created_at, item_count, sent_at)
        VALUES ('2026-05-12', '2026-05-12 12:00:00', 3, '2026-05-12 12:00:00');
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

    # exclude_reviewed_items is vote-based: latest vote +1 or unvoted are kept.
    assert {row.external_id for row in reviewed_filtered} == {
        "old-bad-latest-good",
        "shown-unvoted",
        "never-shown",
    }
    # exclude_previously_shown is membership-based: any item that appeared in a
    # prior digest is dropped regardless of vote; only never-shown survives.
    assert {row.external_id for row in shown_filtered} == {"never-shown"}
