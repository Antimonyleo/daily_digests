from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np


def _reset_store_for_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_digest.db"))
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
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


def _insert_item2(store_mod, suffix: str) -> int:
    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id=f"test-{suffix}",
            url=f"https://example.com/test-{suffix}",
            title=f"RNA nanotechnology paper {suffix}",
            abstract="A useful abstract.",
            published_at=datetime.now(timezone.utc),
            digest_id="2026-05-05",
            item_label=f"R{suffix}",
        )
        s.add(row)
        s.flush()
        return int(row.id)


def test_lr_feature_schema_constants_match_feature_matrix(monkeypatch):
    from dailydigest import votes as votes_mod
    from dailydigest.rank import embedding_cache as cache_mod

    rows = [
        SimpleNamespace(
            source="Nature Biotechnology",
            section="research",
            title="RNA delivery study",
            abstract="Primary research.",
            published_at=datetime.now(timezone.utc),
        ),
        SimpleNamespace(
            source="STAT News",
            section="industry",
            title="Biotech financing update",
            abstract="Independent coverage.",
            published_at=datetime.now(timezone.utc),
        ),
    ]
    monkeypatch.setattr(
        cache_mod,
        "embed_item_rows",
        lambda items: np.ones((len(items), 2), dtype=np.float32),
    )
    # Isolate from the real votes DB: no exemplars -> affinity features are 0.
    monkeypatch.setattr(
        votes_mod,
        "_load_vote_exemplars",
        lambda: (
            (np.zeros(0, dtype=np.int64), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)),
            (np.zeros(0, dtype=np.int64), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)),
        ),
    )

    features = votes_mod._build_item_features(rows, np.ones((1, 2), dtype=np.float32))

    # v6 preference schema: de-confounded aggregate features PLUS memory-based
    # pos/neg exemplar affinity (learns "less LNP delivery, more protein redesign").
    assert votes_mod.LR_FEATURE_DIM == 8
    assert votes_mod.LR_FEATURE_DIM == len(votes_mod.LR_FEATURE_NAMES)
    assert votes_mod.LR_FEATURE_NAMES[0] == "cosine_similarity"
    assert votes_mod.LR_FEATURE_NAMES[-2:] == ("pos_affinity", "neg_affinity")
    assert not any(
        n in votes_mod.LR_FEATURE_NAMES
        for n in ("prestige_score", "source_bucket_score", "cosine_x_prestige")
    )
    assert features.shape == (2, votes_mod.LR_FEATURE_DIM)


def test_neutral_vote_is_persisted_for_visible_feedback(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    item_id = _insert_item(store_mod)

    assert votes_mod.record_vote_by_id(item_id, 0) is True

    assert votes_mod.get_vote_value(item_id) == 0


def _rocchio_setup(tmp_path, monkeypatch, stub_vec):
    """Point the learned-profile path at tmp and stub embeddings.

    ``stub_vec`` may be a single [dim] vector (used for every row) or a
    dict {item_id: vec}. The stub keys embeddings by the ItemRow.id so a rebuild
    over multiple canonical votes gets the right vector per row.
    """
    from pathlib import Path
    from dailydigest.rank import embedding_cache as cache_mod

    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    if isinstance(stub_vec, dict):
        def _embed(rows):
            return np.array([stub_vec[int(r.id)] for r in rows], dtype=np.float32)
    else:
        vec = np.asarray(stub_vec, dtype=np.float32)

        def _embed(rows):
            return np.tile(vec, (len(rows), 1)).astype(np.float32)

    # Both votes.py and the cache module reference embed_item_rows; patch the
    # source so the rebuild's `from .rank.embedding_cache import embed_item_rows`
    # picks it up.
    monkeypatch.setattr(cache_mod, "embed_item_rows", _embed)

    profile_path = Path(str(tmp_path / "learned_profile.npz"))
    monkeypatch.setattr(votes_mod, "_learned_profile_path", lambda: profile_path)
    return store_mod, votes_mod, profile_path


def test_rocchio_rebuild_single_vote_is_grade_weighted_alpha_pull(tmp_path, monkeypatch):
    """Rebuild from one canonical vote: learned = alpha*weight*decay**0*vec."""
    stub = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    store_mod, votes_mod, profile_path = _rocchio_setup(tmp_path, monkeypatch, stub)

    item_id = _insert_item(store_mod)
    # A stale prior profile must be discarded by the rebuild (not decayed-into).
    np.savez(profile_path, profile=np.array([9.0, 9.0, 9.0], dtype=np.float32),
             vote_count=np.array([5], dtype=np.int32))

    # "Must read" (grade 100 -> weight 1.0) applies the full alpha pull.
    assert votes_mod.record_vote_by_id(item_id, 1, grade=100) is True

    learned = np.load(profile_path)["profile"]
    np.testing.assert_allclose(learned, 0.08 * 1.0 * stub, rtol=1e-5, atol=1e-6)


def test_rocchio_rebuild_pull_scales_with_grade(tmp_path, monkeypatch):
    """A weaker preference ('Relevant', grade 70) pulls less than 'Must read'."""
    stub = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    store_mod, votes_mod, profile_path = _rocchio_setup(tmp_path, monkeypatch, stub)
    item_id = _insert_item(store_mod)

    # grade 70 -> weight (70-50)/50 = 0.4 -> 0.4 * alpha pull.
    assert votes_mod.record_vote_by_id(item_id, 1, grade=70) is True
    learned = np.load(profile_path)["profile"]
    np.testing.assert_allclose(learned, 0.08 * 0.4 * stub, rtol=1e-5, atol=1e-6)


def test_rocchio_rebuild_idempotent_on_repeated_vote(tmp_path, monkeypatch):
    """Recording the same vote twice yields the same learned profile."""
    stub = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    store_mod, votes_mod, profile_path = _rocchio_setup(tmp_path, monkeypatch, stub)
    item_id = _insert_item(store_mod)

    assert votes_mod.record_vote_by_id(item_id, 1, grade=100) is True
    once = np.load(profile_path)["profile"].copy()
    assert votes_mod.record_vote_by_id(item_id, 1, grade=100) is True
    twice = np.load(profile_path)["profile"]
    np.testing.assert_allclose(twice, once, rtol=1e-6, atol=1e-8)


def test_rocchio_rebuild_vote_change_matches_final_vote_only(tmp_path, monkeypatch):
    """Voting an item then FLIPPING the vote equals only-the-final-vote existing."""
    stub = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    store_mod, votes_mod, profile_path = _rocchio_setup(tmp_path, monkeypatch, stub)
    item_id = _insert_item(store_mod)

    # Vote up, then change to a down-vote of matching magnitude.
    assert votes_mod.record_vote_by_id(item_id, 1, grade=100) is True
    assert votes_mod.record_vote_by_id(item_id, -1, grade=10) is True
    flipped = np.load(profile_path)["profile"].copy()

    # A fresh DB where only the final (down) vote was ever recorded.
    store2, votes2, path2 = _rocchio_setup(tmp_path / "fresh", monkeypatch, stub)
    fresh_id = _insert_item(store2)
    assert votes2.record_vote_by_id(fresh_id, -1, grade=10) is True
    final_only = np.load(path2)["profile"]

    np.testing.assert_allclose(flipped, final_only, rtol=1e-6, atol=1e-8)
    # Down-vote (weight -0.8) pulls negatively with beta rate.
    np.testing.assert_allclose(flipped, 0.04 * -0.8 * stub, rtol=1e-5, atol=1e-6)


def test_rocchio_rebuild_neutral_removes_contribution(tmp_path, monkeypatch):
    """Voting then setting the vote to neutral removes that item's contribution.

    With a single item, the canonical signed-vote set becomes empty, so the
    learned-profile file is removed. With a second still-signed item present,
    the neutralized item simply drops out of the rebuild.
    """
    stub_map = {}
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod
    from dailydigest.rank import embedding_cache as cache_mod
    from pathlib import Path

    id_a = _insert_item2(store_mod, "a")
    id_b = _insert_item2(store_mod, "b")
    stub_map[id_a] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    stub_map[id_b] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def _embed(rows):
        return np.array([stub_map[int(r.id)] for r in rows], dtype=np.float32)

    monkeypatch.setattr(cache_mod, "embed_item_rows", _embed)
    profile_path = Path(str(tmp_path / "learned_profile.npz"))
    monkeypatch.setattr(votes_mod, "_learned_profile_path", lambda: profile_path)

    assert votes_mod.record_vote_by_id(id_a, 1, grade=100) is True
    assert votes_mod.record_vote_by_id(id_b, 1, grade=100) is True

    # Neutralize A: rebuild should reflect ONLY B's (most-recent) contribution.
    assert votes_mod.record_vote_by_id(id_a, 0) is True
    learned = np.load(profile_path)["profile"]
    np.testing.assert_allclose(learned, 0.08 * 1.0 * stub_map[id_b], rtol=1e-5, atol=1e-6)

    # Neutralize B too: no signed votes remain -> the stale file is removed.
    assert votes_mod.record_vote_by_id(id_b, 0) is True
    assert not profile_path.exists()


def test_vote_reason_is_persisted_once_per_item(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    item_id = _insert_item(store_mod)

    assert votes_mod.record_vote_by_id(item_id, -1) is True
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

    assert votes_mod.record_vote_by_id(item_id, -1) is True
    assert votes_mod.record_vote_reason(item_id, "low_impact") is True
    assert votes_mod.record_vote_reason(item_id, "promotional") is True

    penalties = votes_mod.reason_penalty_map()

    assert abs(penalties[str(item_id)] - 0.24) < 1e-9


def test_vote_reason_requires_seen_or_not_for_me_vote(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    item_id = _insert_item(store_mod)

    assert votes_mod.record_vote_reason(item_id, "low_impact") is False
    assert votes_mod.record_vote_by_id(item_id, 1) is True
    assert votes_mod.record_vote_reason(item_id, "low_impact") is False
    assert votes_mod.record_vote_by_id(item_id, 0) is True
    assert votes_mod.record_vote_reason(item_id, "low_impact") is True
    assert votes_mod.get_vote_reasons(item_id) == ["low_impact"]
    assert votes_mod.record_vote_by_id(item_id, 1) is True
    assert votes_mod.get_vote_reasons(item_id) == []
    assert str(item_id) not in votes_mod.reason_penalty_map()


def test_reason_penalty_map_generalizes_to_matching_future_sources(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    store_mod.init_db()
    with store_mod.session_scope() as s:
        voted = store_mod.ItemRow(
            source="Company Press Release",
            section="industry",
            external_id="old-promo",
            url="https://example.com/old-promo",
            title="Company today announced sponsored product launch",
        )
        future = store_mod.ItemRow(
            source="Company Press Release",
            section="industry",
            external_id="new-promo",
            url="https://example.com/new-promo",
            title="Company today announced another product launch",
        )
        s.add_all([voted, future])
        s.flush()
        voted_id = int(voted.id)
        future_id = int(future.id)

    assert votes_mod.record_vote_by_id(voted_id, -1) is True
    assert votes_mod.record_vote_reason(voted_id, "promotional") is True
    with store_mod.session_scope() as s:
        future = s.get(store_mod.ItemRow, future_id)
        s.expunge(future)

    penalties = votes_mod.reason_penalty_map([future])

    assert penalties[str(future_id)] > 0
    assert penalties[str(future_id)] < votes_mod.REASON_PENALTIES["promotional"]


def test_reason_penalty_map_decays_generalized_feedback_by_vote_recency(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    store_mod.init_db()
    now = datetime.now(timezone.utc)
    with store_mod.session_scope() as s:
        old_voted = store_mod.ItemRow(
            source="Old Feedback Journal",
            section="research",
            external_id="old-feedback",
            url="https://example.com/old-feedback",
            title="Routine RNA research item",
        )
        fresh_voted = store_mod.ItemRow(
            source="Fresh Feedback Journal",
            section="research",
            external_id="fresh-feedback",
            url="https://example.com/fresh-feedback",
            title="Routine RNA research item",
        )
        old_future = store_mod.ItemRow(
            source="Old Feedback Journal",
            section="research",
            external_id="old-future",
            url="https://example.com/old-future",
            title="Another routine RNA research item",
        )
        fresh_future = store_mod.ItemRow(
            source="Fresh Feedback Journal",
            section="research",
            external_id="fresh-future",
            url="https://example.com/fresh-future",
            title="Another routine RNA research item",
        )
        s.add_all([old_voted, fresh_voted, old_future, fresh_future])
        s.flush()
        s.add(
            store_mod.VoteRow(
                item_id=old_voted.id,
                value=-1,
                created_at=now - timedelta(days=180),
            )
        )
        s.add(store_mod.VoteRow(item_id=fresh_voted.id, value=-1, created_at=now))
        old_voted_id = int(old_voted.id)
        fresh_voted_id = int(fresh_voted.id)
        old_future_id = int(old_future.id)
        fresh_future_id = int(fresh_future.id)

    assert votes_mod.record_vote_reason(old_voted_id, "promotional") is True
    assert votes_mod.record_vote_reason(fresh_voted_id, "promotional") is True
    with store_mod.session_scope() as s:
        old_future = s.get(store_mod.ItemRow, old_future_id)
        fresh_future = s.get(store_mod.ItemRow, fresh_future_id)
        s.expunge(old_future)
        s.expunge(fresh_future)

    penalties = votes_mod.reason_penalty_map([old_future, fresh_future])

    assert penalties[str(old_future_id)] > 0
    assert penalties[str(fresh_future_id)] > penalties[str(old_future_id)]


def test_reason_penalty_map_good_votes_counter_source_generalization(tmp_path, monkeypatch):
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    store_mod.init_db()
    now = datetime.now(timezone.utc)
    with store_mod.session_scope() as s:
        bad_only_voted = store_mod.ItemRow(
            source="Bad Only Journal",
            section="research",
            external_id="bad-only-feedback",
            url="https://example.com/bad-only-feedback",
            title="Routine RNA research item",
        )
        mixed_voted = store_mod.ItemRow(
            source="Mixed Signal Journal",
            section="research",
            external_id="mixed-feedback",
            url="https://example.com/mixed-feedback",
            title="Routine RNA research item",
        )
        mixed_good = store_mod.ItemRow(
            source="Mixed Signal Journal",
            section="research",
            external_id="mixed-good",
            url="https://example.com/mixed-good",
            title="Useful RNA research item",
        )
        bad_only_future = store_mod.ItemRow(
            source="Bad Only Journal",
            section="research",
            external_id="bad-only-future",
            url="https://example.com/bad-only-future",
            title="Another routine RNA research item",
        )
        mixed_future = store_mod.ItemRow(
            source="Mixed Signal Journal",
            section="research",
            external_id="mixed-future",
            url="https://example.com/mixed-future",
            title="Another routine RNA research item",
        )
        s.add_all([bad_only_voted, mixed_voted, mixed_good, bad_only_future, mixed_future])
        s.flush()
        s.add(store_mod.VoteRow(item_id=bad_only_voted.id, value=-1, created_at=now))
        s.add(store_mod.VoteRow(item_id=mixed_voted.id, value=-1, created_at=now))
        s.add(store_mod.VoteRow(item_id=mixed_good.id, value=1, created_at=now))
        bad_only_voted_id = int(bad_only_voted.id)
        mixed_voted_id = int(mixed_voted.id)
        bad_only_future_id = int(bad_only_future.id)
        mixed_future_id = int(mixed_future.id)

    assert votes_mod.record_vote_reason(bad_only_voted_id, "promotional") is True
    assert votes_mod.record_vote_reason(mixed_voted_id, "promotional") is True
    with store_mod.session_scope() as s:
        bad_only_future = s.get(store_mod.ItemRow, bad_only_future_id)
        mixed_future = s.get(store_mod.ItemRow, mixed_future_id)
        s.expunge(bad_only_future)
        s.expunge(mixed_future)

    penalties = votes_mod.reason_penalty_map([bad_only_future, mixed_future])

    assert penalties[str(bad_only_future_id)] > 0
    assert penalties[str(mixed_future_id)] < penalties[str(bad_only_future_id)]


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
    store_mod._INITIALIZED = False
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


def test_legacy_duplicate_latest_neutral_is_not_signed_for_training(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy_neutral_digest.db"
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
        ) VALUES
            (1, 'Legacy', 'research', 'legacy-neutral', 'https://example.com/legacy-neutral',
             'Legacy neutral item', 'Abstract', '', '2026-05-12 00:00:00',
             '2026-05-12 00:00:00', '', NULL, NULL, NULL),
            (2, 'Legacy', 'research', 'legacy-good', 'https://example.com/legacy-good',
             'Legacy good item', 'Abstract', '', '2026-05-12 00:00:00',
             '2026-05-12 00:00:00', '', NULL, NULL, NULL);
        INSERT INTO votes (id, item_id, value, created_at)
        VALUES
            (1, 1, 1, '2026-05-12 10:00:00'),
            (2, 1, 0, '2026-05-12 11:00:00'),
            (3, 2, -1, '2026-05-12 10:00:00'),
            (4, 2, 1, '2026-05-12 11:00:00');
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
    store_mod._INITIALIZED = False
    monkeypatch.setattr(votes_mod, "MIN_VOTES_FOR_LR", 1)
    monkeypatch.setattr(
        votes_mod,
        "embed_item_rows",
        lambda rows: np.ones((len(rows), 2), dtype=np.float32),
    )

    counts = votes_mod.vote_counts()
    assert counts["good"] == 1
    assert counts["neutral"] == 1
    assert counts["signed"] == 1
    assert votes_mod.signed_vote_count() == 1

    dataset = votes_mod.vote_dataset()

    assert dataset is not None
    _x, y = dataset
    assert y.tolist() == [1.0]


def test_vote_dataset_returns_pairwise_differences_when_both_signs_present(
    tmp_path, monkeypatch
):
    """vote_dataset() should return pairwise difference vectors when both up and down votes exist."""
    import numpy as np
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import votes as votes_mod

    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

    # Mock embed to return distinct per-item vectors
    n_items = 6
    fake_vecs = np.eye(n_items, 10, dtype=np.float32)
    monkeypatch.setattr(votes_mod, "embed_item_rows", lambda rows: fake_vecs[:len(rows)])

    # Insert items and votes: 3 up, 3 down
    store_mod.init_db()
    item_ids = []
    with store_mod.session_scope() as s:
        for i in range(n_items):
            row = store_mod.ItemRow(
                source="Test", section="research",
                external_id=f"pairwise-{i}", url=f"https://example.com/{i}",
                title=f"Paper {i}", abstract="A" * 50,
            )
            s.add(row)
            s.flush()
            item_ids.append(row.id)

    with store_mod.session_scope() as s:
        for i, item_id in enumerate(item_ids):
            vote_value = 1 if i < 3 else -1
            s.add(store_mod.VoteRow(item_id=item_id, value=vote_value))

    monkeypatch.setattr(votes_mod, "MIN_VOTES_FOR_LR", 1)
    result = votes_mod.vote_dataset()
    assert result is not None, "vote_dataset() returned None"
    X, y = result

    # When both signs present, should return pairwise differences
    # 3 ups × 3 downs = 9 pairs × 2 (+ and - direction) = 18 pairwise rows
    assert X.shape[0] > n_items, f"Expected pairwise augmentation to exceed {n_items} rows, got {X.shape[0]}"
    assert set(y.tolist()) == {1.0, -1.0}, "Should have both positive and negative labels"
    # Pairwise diff vectors should be in [-1, 1] range (differences of [0,1] features)
    assert X.min() >= -1.0 and X.max() <= 1.0, "Pairwise diff vectors should be in [-1, 1]"
