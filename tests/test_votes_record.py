from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np


def _reset_store_for_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_digest.db"))
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    return store_mod


def _insert_item(store_mod) -> int:
    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="test-1",
            url="https://example.com/test-1",
            title="RNA nanotechnology paper",
            abstract="A useful abstract.",
            published_at=datetime.now(timezone.utc),
            digest_id="2026-05-05",
            item_label="R1",
        )
        s.add(row)
        s.flush()
        return int(row.id)


def test_neutral_vote_is_persisted_for_visible_feedback(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    item_id = _insert_item(store_mod)

    assert votes_mod.record_vote_by_id(item_id, 0) is True

    assert votes_mod.get_vote_value(item_id) == 0


def test_vote_reason_is_persisted_once_per_item(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    item_id = _insert_item(store_mod)

    assert votes_mod.record_vote_reason(item_id, "low_impact") is True
    assert votes_mod.record_vote_reason(item_id, "low_impact") is True
    assert votes_mod.record_vote_reason(item_id, "promotional") is True

    assert votes_mod.get_vote_reasons(item_id) == ["low_impact", "promotional"]

    assert votes_mod.remove_vote_reason(item_id, "low_impact") is True
    assert votes_mod.get_vote_reasons(item_id) == ["promotional"]
    assert votes_mod.remove_vote_reason(item_id, "promotional") is True
    assert votes_mod.get_vote_reasons(item_id) == []


def test_reason_penalty_map_sums_reason_weights(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    item_id = _insert_item(store_mod)

    assert votes_mod.record_vote_reason(item_id, "low_impact") is True
    assert votes_mod.record_vote_reason(item_id, "promotional") is True

    penalties = votes_mod.reason_penalty_map()

    assert abs(penalties[str(item_id)] - 0.24) < 1e-9


def test_vote_reason_rejects_unknown_reason_or_item(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    item_id = _insert_item(store_mod)

    assert votes_mod.record_vote_reason(item_id, "not_a_reason") is False
    assert votes_mod.record_vote_reason(999999, "low_impact") is False
    assert votes_mod.get_vote_reasons(item_id) == []


def test_vote_dataset_ignores_neutral_votes(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    store_mod.init_db()
    with store_mod.session_scope() as s:
        for idx in range(31):
            item = store_mod.ItemRow(
                source="Test",
                section="research",
                external_id=f"neutral-{idx}",
                url=f"https://example.com/neutral-{idx}",
                title=f"Item {idx}",
                abstract="Neutral example.",
                published_at=datetime.now(timezone.utc),
            )
            s.add(item)
            s.flush()
            s.add(store_mod.VoteRow(item_id=item.id, value=0))

    def _should_not_embed(_texts):
        raise AssertionError("neutral-only votes should not be embedded")

    monkeypatch.setattr(votes_mod, "embed_item_rows", _should_not_embed)

    assert votes_mod.vote_dataset() is None


def test_legacy_duplicate_vote_rows_use_latest_vote_for_counts_and_training(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy_digest.db"
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
        INSERT INTO items (
            id, source, section, external_id, url, title, abstract, authors,
            published_at, fetched_at, summary, score, digest_id, item_label
        ) VALUES (
            1, 'Legacy', 'research', 'legacy-1', 'https://example.com/legacy-1',
            'Legacy item', 'Abstract', '', '2026-05-12 00:00:00',
            '2026-05-12 00:00:00', '', NULL, NULL, NULL
        );
        INSERT INTO votes (id, item_id, value, created_at)
        VALUES
            (1, 1, 1, '2026-05-12 10:00:00'),
            (2, 1, -1, '2026-05-12 11:00:00');
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("DB_PATH", str(db_path))
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import votes as votes_mod

    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    monkeypatch.setattr(votes_mod, "MIN_VOTES_FOR_LR", 1)
    monkeypatch.setattr(
        votes_mod,
        "embed_item_rows",
        lambda rows: np.ones((len(rows), 2), dtype=np.float32),
    )

    assert votes_mod.get_vote_value(1) == -1
    assert votes_mod.vote_counts()["bad"] == 1
    assert votes_mod.vote_counts()["good"] == 0

    dataset = votes_mod.vote_dataset()

    assert dataset is not None
    _x, y = dataset
    assert y.tolist() == [-1.0]
