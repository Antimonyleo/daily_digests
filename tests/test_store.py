from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


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
                store_mod.ItemRow(
                    source="FuturePub",
                    section="research",
                    external_id="future-pub",
                    url="https://example.com/future-pub",
                    title="Implausibly future publication",
                    published_at=now + timedelta(days=30),
                    fetched_at=now,
                ),
            ]
        )

    rows = store_mod.recent_items(days=2)
    external_ids = {row.external_id for row in rows}

    assert "old-pub-new-fetch" not in external_ids
    assert "no-pub-new-fetch" in external_ids
    assert "fresh-pub-old-fetch" in external_ids
    assert "future-pub" not in external_ids


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


def test_known_flag_is_manual_idempotent_and_reversible(monkeypatch, tmp_path):
    """Only an explicit call flags an item; the flag suppresses and undoes cleanly."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    grant = _add_item(store_mod, "grant-known", section="opportunities")
    other = _add_item(store_mod, "grant-open", section="opportunities")

    # Nothing is known until the reader says so.
    assert store_mod.known_item_ids([grant, other]) == set()
    assert store_mod.set_item_known(grant, True) is True
    assert store_mod.set_item_known(grant, True) is True  # idempotent
    assert store_mod.known_item_ids([grant, other]) == {grant}
    assert store_mod.set_item_known(10**9, True) is False  # unknown item id

    with store_mod.session_scope() as s:
        rows = s.query(store_mod.ItemRow).all()
        kept = store_mod.exclude_known_items(rows)
        assert {int(r.id) for r in kept} == {other}
        for row in rows:
            s.expunge(row)

    assert store_mod.set_item_known(grant, False) is True
    assert store_mod.known_item_ids([grant, other]) == set()


def test_known_items_survive_retention_pruning(monkeypatch, tmp_path):
    """Losing the row would lose the flag and let the grant come back."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    grant = _add_item(store_mod, "grant-old", section="opportunities")
    stale = _add_item(store_mod, "news-old", section="world")
    old = datetime.now(timezone.utc) - timedelta(days=90)
    with store_mod.session_scope() as s:
        for row in s.query(store_mod.ItemRow).all():
            row.fetched_at = old
    store_mod.set_item_known(grant, True)

    store_mod.prune(days=30)
    with store_mod.session_scope() as s:
        remaining = {int(r.id) for r in s.query(store_mod.ItemRow).all()}
    assert remaining == {grant}
    assert stale not in remaining


def test_unshown_candidates_are_not_suppressed_as_previously_shown(monkeypatch, tmp_path):
    """A trimmed near-miss must stay eligible, or "save for tomorrow" is a no-op.

    Research impressions log the whole scored pool, so an overflow item DOES get
    an impression row — with selected=False. exclude_previously_shown must key on
    the selected flag, not on the row's existence.
    """
    store_mod = _reset_store(tmp_path, monkeypatch)
    shown = _add_item(store_mod, "shown-item")
    trimmed = _add_item(store_mod, "trimmed-item")
    digest_id = "2026-06-20"

    store_mod.write_digest(digest_id, [("R1", shown)])
    store_mod.write_impressions(
        digest_id,
        [
            ("research", shown, 0, 0.9, True, "facet", 0.8, 0.8),
            ("research", trimmed, 1, 0.8, False, "facet", 0.8, 0.8),
        ],
    )
    store_mod.mark_impressions_viewed(digest_id)

    with store_mod.session_scope() as s:
        rows = s.query(store_mod.ItemRow).all()
        kept = {int(r.id) for r in store_mod.exclude_previously_shown(rows)}
        for row in rows:
            s.expunge(row)
    assert trimmed in kept
    assert shown not in kept


def test_carryover_items_pin_evaluate_once_and_clear(monkeypatch, tmp_path):
    """Save-for-tomorrow entries add idempotently, load as rows, and consume."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    a = _add_item(store_mod, "carry-a")
    b = _add_item(store_mod, "carry-b")

    assert store_mod.add_carryover_items([a, b, a], pinned_digest_id="2026-06-01") == 2
    assert store_mod.add_carryover_items([a]) == 0  # idempotent
    assert store_mod.add_carryover_items([10**9]) == 0  # unknown ids ignored
    assert store_mod.carryover_item_ids() == {a, b}
    rows = store_mod.carryover_item_rows()
    assert {int(r.id) for r in rows} == {a, b}
    # A same-day re-brew must not spend the pins it just created: the pipeline
    # passes the current digest id, and pins made from it are withheld.
    assert store_mod.carryover_item_rows(exclude_digest_id="2026-06-01") == []
    assert {
        int(r.id) for r in store_mod.carryover_item_rows(exclude_digest_id="2026-06-02")
    } == {a, b}

    store_mod.clear_carryover_items([a])
    assert store_mod.carryover_item_ids() == {b}
    store_mod.clear_carryover_items([b])
    assert store_mod.carryover_item_ids() == set()
    assert store_mod.carryover_item_rows() == []


def test_bookmarks_are_idempotent_and_searchable(monkeypatch, tmp_path):
    store_mod = _reset_store(tmp_path, monkeypatch)
    with store_mod.session_scope() as s:
        rna = store_mod.ItemRow(
            source="Nature Nanotechnology",
            section="research",
            external_id="saved-rna",
            url="https://example.com/saved-rna",
            title="Programmable RNA nanostructures",
            abstract="A modular RNA assembly method.",
            summary="RNA tiles assemble into defined particles.",
        )
        colloid = store_mod.ItemRow(
            source="Science",
            section="research",
            external_id="saved-colloid",
            url="https://example.com/saved-colloid",
            title="Colloidal crystal assembly",
        )
        s.add_all([rna, colloid])
        s.flush()
        rna_id, colloid_id = int(rna.id), int(colloid.id)

    assert store_mod.set_bookmark(rna_id, True) is True
    assert store_mod.set_bookmark(rna_id, True) is True
    assert store_mod.set_bookmark(colloid_id, True) is True
    assert store_mod.bookmarked_item_ids([rna_id, colloid_id, 999]) == {
        rna_id,
        colloid_id,
    }

    matches = store_mod.search_bookmarks("RNA")
    assert [row.item_id for row in matches] == [rna_id]
    assert matches[0].title == "Programmable RNA nanostructures"
    assert matches[0].source == "Nature Nanotechnology"

    assert store_mod.set_bookmark(rna_id, False) is True
    assert store_mod.set_bookmark(rna_id, False) is True
    assert store_mod.bookmarked_item_ids([rna_id, colloid_id]) == {colloid_id}
    assert store_mod.set_bookmark(999, True) is False


def test_prune_preserves_bookmarked_items(monkeypatch, tmp_path):
    store_mod = _reset_store(tmp_path, monkeypatch)
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    with store_mod.session_scope() as s:
        saved = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="old-saved",
            url="https://example.com/old-saved",
            title="Old saved paper",
            fetched_at=old_time,
        )
        disposable = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="old-disposable",
            url="https://example.com/old-disposable",
            title="Old disposable paper",
            fetched_at=old_time,
        )
        s.add_all([saved, disposable])
        s.flush()
        saved_id = int(saved.id)

    assert store_mod.set_bookmark(saved_id, True) is True
    assert store_mod.prune(5) == 1
    assert [row.item_id for row in store_mod.search_bookmarks()] == [saved_id]


def test_prune_preserves_items_referenced_by_retained_digest(monkeypatch, tmp_path):
    store_mod = _reset_store(tmp_path, monkeypatch)
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="old-current-digest",
            url="https://example.com/old-current-digest",
            title="Old item still in a retained digest",
            fetched_at=old_time,
        )
        s.add(row)
        s.flush()
        item_id = int(row.id)

    store_mod.write_digest("2099-01-01", [("R1", item_id)])

    assert store_mod.prune(5) == 0
    with store_mod.session_scope() as s:
        assert s.get(store_mod.ItemRow, item_id) is not None


def test_duplicate_upsert_refreshes_richer_metadata_and_invalidates_summary(
    monkeypatch, tmp_path
):
    from dailydigest.models import Item

    store_mod = _reset_store(tmp_path, monkeypatch)
    first = Item(
        source="Journal",
        section="research",
        external_id="paper-1",
        url="bad-url",
        title="CollapsedTitleWords",
        abstract="Short abstract.",
    )
    assert store_mod.upsert_items([first]) == 1
    with store_mod.session_scope() as s:
        row = s.execute(store_mod.select(store_mod.ItemRow)).scalar_one()
        row.summary = "Summary of stale text"
        row.summary_signature = "old-signature"
        row.summary_backend = "extractive_fallback"

    richer = Item(
        source="Journal",
        section="research",
        external_id="paper-1",
        url="https://example.com/paper-1",
        title="Collapsed Title Words",
        abstract=(
            "A substantially richer abstract with methods, results, limitations, "
            "and enough additional detail to cross the conservative refresh threshold."
        ),
        authors="Ada Example",
        published_at=datetime.now(timezone.utc),
        metadata={"doi": "10.1/example"},
    )

    assert store_mod.upsert_items([richer]) == 0
    with store_mod.session_scope() as s:
        row = s.execute(store_mod.select(store_mod.ItemRow)).scalar_one()
        assert row.title == "Collapsed Title Words"
        assert row.abstract == richer.abstract
        assert row.authors == "Ada Example"
        assert row.published_at is not None
        assert row.url == "https://example.com/paper-1"
        assert row.summary == ""
        assert row.summary_signature is None
        assert row.summary_backend is None
        assert store_mod.item_metadata(row)["doi"] == "10.1/example"


def test_legacy_vote_migration_keeps_latest_and_creates_backup(monkeypatch, tmp_path):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    db_path = tmp_path / "legacy-votes.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE items (
                id INTEGER PRIMARY KEY, source VARCHAR NOT NULL,
                section VARCHAR NOT NULL, external_id VARCHAR NOT NULL,
                url VARCHAR NOT NULL, title TEXT NOT NULL
            );
            CREATE TABLE votes (
                id INTEGER PRIMARY KEY, item_id INTEGER NOT NULL,
                value INTEGER NOT NULL, created_at DATETIME
            );
            INSERT INTO items VALUES
                (1, 'Legacy', 'research', 'one', 'https://example.com/one', 'One');
            INSERT INTO votes VALUES
                (1, 1, 1, '2026-01-01 10:00:00'),
                (2, 1, -1, '2026-01-01 11:00:00');
            """
        )

    monkeypatch.setenv("DB_PATH", str(db_path))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    store_mod.init_db()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT value FROM votes").fetchall() == [(-1,)]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO votes (item_id, value) VALUES (1, 1)")
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert (tmp_path / "backups" / "digest-pre-v1.db").exists()


def test_migration_does_not_downgrade_a_future_schema_version(monkeypatch, tmp_path):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    db_path = tmp_path / "future-version.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version = 7")

    monkeypatch.setenv("DB_PATH", str(db_path))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    store_mod.init_db()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 7


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


def test_write_impressions_persists_facet_and_topic_score(monkeypatch, tmp_path):
    """An 8-tuple keeps facet cosine distinct from the overall topic score."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    a = _add_item(store_mod, "facet-a")
    b = _add_item(store_mod, "facet-b")
    digest_id = "2026-06-04"

    store_mod.write_impressions(
        digest_id,
        [
            ("research", a, 0, 0.9, True, "dna nanotechnology", 0.74, 0.81),
            ("research", b, 1, 0.7, False, "colloidal self-assembly", 0.68, 0.72),
        ],
        model_version="v-test",
    )

    with store_mod.session_scope() as s:
        by_item = {
            r.item_id: r
            for r in s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).all()
        }
        assert by_item[a].primary_facet == "dna nanotechnology"
        assert by_item[a].primary_facet_score == 0.74
        assert by_item[a].topic_score == 0.81
        assert by_item[b].primary_facet == "colloidal self-assembly"
        assert by_item[b].primary_facet_score == 0.68
        assert by_item[b].topic_score == 0.72


def test_write_impressions_facet_topic_default_when_omitted(monkeypatch, tmp_path):
    """Omitting the facet/topic elements defaults to ""/None (5-tuple stays valid)."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    a = _add_item(store_mod, "default-a")
    digest_id = "2026-06-05"

    store_mod.write_impressions(
        digest_id,
        [("research", a, 0, 0.9, True)],
        model_version="v-test",
    )

    with store_mod.session_scope() as s:
        row = s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).one()
        assert row.primary_facet == ""
        assert row.primary_facet_score is None
        assert row.topic_score is None


def test_migration_adds_facet_and_topic_columns_to_legacy_impressions(
    monkeypatch, tmp_path
):
    """A legacy impressions table (no facet/topic columns) gains them via ALTER."""
    db_path = tmp_path / "legacy.db"
    # Hand-build a legacy impressions table WITHOUT the new columns.
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE impressions (
            id INTEGER PRIMARY KEY,
            run_id VARCHAR NOT NULL,
            digest_id VARCHAR NOT NULL,
            item_id INTEGER NOT NULL,
            section VARCHAR NOT NULL,
            position INTEGER NOT NULL,
            final_score FLOAT,
            model_version VARCHAR,
            created_at DATETIME
        )
        """
    )
    conn.commit()
    conn.close()

    cols_before = {r[1] for r in sqlite3.connect(db_path).execute(
        "PRAGMA table_info(impressions)"
    ).fetchall()}
    assert "primary_facet" not in cols_before
    assert "primary_facet_score" not in cols_before
    assert "topic_score" not in cols_before

    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    monkeypatch.setenv("DB_PATH", str(db_path))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    store_mod.init_db()  # runs the ALTER-TABLE migration.

    cols_after = {r[1] for r in sqlite3.connect(db_path).execute(
        "PRAGMA table_info(impressions)"
    ).fetchall()}
    assert "primary_facet" in cols_after
    assert "primary_facet_score" in cols_after
    assert "topic_score" in cols_after


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


def test_mark_impressions_viewed_only_flags_visible_selected_rows(
    monkeypatch, tmp_path
):
    store_mod = _reset_store(tmp_path, monkeypatch)
    visible = _add_item(store_mod, "visible-selected")
    hidden = _add_item(store_mod, "hidden-selected")
    digest_id = "2026-06-06"

    store_mod.write_impressions(
        digest_id,
        [
            ("research", visible, 0, 0.9, True),
            ("industry", hidden, 0, 0.8, True),
        ],
        model_version="v-test",
    )

    assert store_mod.mark_impressions_viewed(digest_id, [visible]) == 1

    with store_mod.session_scope() as s:
        by_item = {
            r.item_id: r
            for r in s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).all()
        }
    assert by_item[visible].viewed is True
    assert by_item[hidden].viewed is False


def test_recent_viewed_facet_dates_uses_only_latest_brew(monkeypatch, tmp_path):
    """A viewed obsolete rebrew must not count after a newer slate replaces it."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    old_item = _add_item(store_mod, "coverage-old")
    new_item = _add_item(store_mod, "coverage-new")
    digest_id = "2026-06-06"

    store_mod.write_digest(digest_id, [("R1", old_item)])
    store_mod.mark_sent(digest_id)
    old_run = store_mod.write_impressions(
        digest_id,
        [("research", old_item, 0, 0.9, True, "old facet", 0.8, 0.8)],
    )
    new_run = store_mod.write_impressions(
        digest_id,
        [("research", new_item, 0, 0.9, True, "new facet", 0.8, 0.8)],
    )
    with store_mod.session_scope() as s:
        # Simulate a reader opening the obsolete slate before the rebrew. The
        # current slate is what they later viewed and the only run that counts.
        for row in s.query(store_mod.ImpressionRow).filter_by(digest_id=digest_id).all():
            row.viewed = row.run_id in {old_run, new_run}

    weak_digest_id = "2026-06-05"
    store_mod.write_digest(weak_digest_id, [("R1", old_item)])
    store_mod.mark_sent(weak_digest_id)
    store_mod.write_impressions(
        weak_digest_id,
        [("research", old_item, 0, 0.9, True, "weak facet", 0.4, 0.8)],
    )
    store_mod.mark_impressions_viewed(weak_digest_id)

    seen = store_mod.recent_viewed_facet_dates(
        before_digest_id="2026-06-07", min_primary_facet_score=0.65
    )
    assert set(seen) == {"new facet"}
    assert seen["new facet"].tzinfo is not None


def test_recent_viewed_facet_dates_counts_unsent_browser_digests(monkeypatch, tmp_path):
    """The browser flow never sets sent_at; coverage must still see viewed brews."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    item_id = _add_item(store_mod, "coverage-browser")
    digest_id = "2026-06-06"

    store_mod.write_digest(digest_id, [("R1", item_id)])
    store_mod.write_impressions(
        digest_id,
        [("research", item_id, 0, 0.9, True, "browser facet", 0.8, 0.8)],
    )
    store_mod.mark_impressions_viewed(digest_id)

    seen = store_mod.recent_viewed_facet_dates(
        before_digest_id="2026-06-07", min_primary_facet_score=0.65
    )
    assert set(seen) == {"browser facet"}


def test_write_impressions_keeps_legacy_seven_tuple_contract(monkeypatch, tmp_path):
    """The old (..., primary_facet, topic_score) tuple remains unambiguous."""
    store_mod = _reset_store(tmp_path, monkeypatch)
    item_id = _add_item(store_mod, "legacy-impression")
    store_mod.write_impressions(
        "2026-06-08",
        [("research", item_id, 0, 0.9, True, "dna nanotechnology", 0.81)],
    )
    with store_mod.session_scope() as s:
        row = s.query(store_mod.ImpressionRow).one()
        assert row.primary_facet_score is None
        assert row.topic_score == 0.81


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


def test_previously_viewed_item_stays_hidden_after_same_day_rebrew(
    monkeypatch, tmp_path
):
    store_mod = _reset_store(tmp_path, monkeypatch)
    old_item = _add_item(store_mod, "old-viewed-slate")
    replacement = _add_item(store_mod, "replacement-slate")
    digest_id = "2026-06-06"

    store_mod.write_digest(digest_id, [("R1", old_item)])
    store_mod.write_impressions(
        digest_id, [("research", old_item, 0, 0.9, True)]
    )
    store_mod.mark_impressions_viewed(digest_id)
    # A rebrew destructively replaces digest_items, but impressions are immutable.
    store_mod.write_digest(digest_id, [("R1", replacement)])
    with store_mod.session_scope() as s:
        row = s.get(store_mod.ItemRow, old_item)
        s.expunge(row)

    filtered = store_mod.exclude_previously_shown(
        [row], days_lookback=30, exclude_digest_id="2026-06-07"
    )

    assert filtered == []
