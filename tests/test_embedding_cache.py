from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import select

from dailydigest import config as config_mod
from dailydigest import store as store_mod


def _reset_store(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False


def _row(source: str, external_id: str, title: str, abstract: str = "") -> store_mod.ItemRow:
    return store_mod.ItemRow(
        source=source,
        section="research",
        external_id=external_id,
        url=f"https://example.com/{external_id}",
        title=title,
        abstract=abstract,
        fetched_at=datetime.now(timezone.utc),
    )


def test_embed_item_rows_caches_new_items_and_reuses_vectors(monkeypatch, tmp_path):
    _reset_store(monkeypatch, tmp_path)
    from dailydigest.rank import embedding_cache as cache_mod

    store_mod.init_db()
    with store_mod.session_scope() as s:
        first = _row("Test", "first", "First", "Alpha abstract")
        second = _row("Test", "second", "Second", "Beta abstract")
        s.add_all([first, second])
        s.flush()
        item_ids = [int(first.id), int(second.id)]

    with store_mod.session_scope() as s:
        rows = [s.get(store_mod.ItemRow, item_id) for item_id in item_ids]
        for row in rows:
            s.expunge(row)

    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        return np.asarray(
            [[float(i + 1), float(i + 2), float(i + 3)] for i in range(len(texts))],
            dtype=np.float32,
        )

    monkeypatch.setattr(cache_mod, "embed_texts", fake_embed)

    first_vecs = cache_mod.embed_item_rows(rows)
    second_vecs = cache_mod.embed_item_rows(rows)

    assert calls == [["First. Alpha abstract", "Second. Beta abstract"]]
    np.testing.assert_array_equal(second_vecs, first_vecs)
    with store_mod.session_scope() as s:
        cached = s.execute(select(store_mod.ItemEmbeddingRow)).scalars().all()
        assert len(cached) == 2


def test_embed_item_rows_recomputes_when_item_text_changes(monkeypatch, tmp_path):
    _reset_store(monkeypatch, tmp_path)
    from dailydigest.rank import embedding_cache as cache_mod

    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = _row("Test", "changed", "Original", "Text")
        s.add(row)
        s.flush()
        item_id = int(row.id)
        s.expunge(row)

    calls = 0

    def fake_embed(texts: list[str]) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.asarray([[float(calls), 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(cache_mod, "embed_texts", fake_embed)
    cache_mod.embed_item_rows([row])

    with store_mod.session_scope() as s:
        saved = s.get(store_mod.ItemRow, item_id)
        saved.title = "Updated"

    with store_mod.session_scope() as s:
        updated = s.get(store_mod.ItemRow, item_id)
        s.expunge(updated)

    vecs = cache_mod.embed_item_rows([updated])

    assert calls == 2
    assert vecs[0, 0] == 2.0


def test_embed_item_rows_recomputes_when_embedding_signature_changes(
    monkeypatch, tmp_path
):
    _reset_store(monkeypatch, tmp_path)
    from dailydigest.rank import embedding_cache as cache_mod

    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = _row("Test", "signature", "Signature", "Same text")
        s.add(row)
        s.flush()
        s.expunge(row)

    signature = ["model:doc-prefix-a"]
    calls = 0

    def fake_embed(texts):
        nonlocal calls
        calls += 1
        return np.ones((len(texts), 3), dtype=np.float32) * calls

    monkeypatch.setattr(cache_mod, "active_embedding_signature", lambda: signature[0])
    monkeypatch.setattr(cache_mod, "embed_texts", fake_embed)

    cache_mod.embed_item_rows([row])
    signature[0] = "model:doc-prefix-b"
    vectors = cache_mod.embed_item_rows([row])

    assert calls == 2
    assert vectors[0, 0] == 2.0
    with store_mod.session_scope() as s:
        cached = s.execute(select(store_mod.ItemEmbeddingRow)).scalars().all()
        assert len(cached) == 1


def test_embed_item_rows_handles_repeated_item_in_one_batch(monkeypatch, tmp_path):
    _reset_store(monkeypatch, tmp_path)
    from dailydigest.rank import embedding_cache as cache_mod

    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = _row("Test", "repeated", "Repeated", "Same abstract")
        s.add(row)
        s.flush()
        s.expunge(row)

    monkeypatch.setattr(
        cache_mod,
        "embed_texts",
        lambda texts: np.ones((len(texts), 3), dtype=np.float32),
    )

    vectors = cache_mod.embed_item_rows([row, row])

    assert vectors.shape == (2, 3)
    with store_mod.session_scope() as s:
        cached = s.execute(select(store_mod.ItemEmbeddingRow)).scalars().all()
        assert len(cached) == 1


def test_prune_removes_cached_embeddings_for_deleted_items(monkeypatch, tmp_path):
    _reset_store(monkeypatch, tmp_path)
    from dailydigest.rank import embedding_cache as cache_mod

    store_mod.init_db()
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    with store_mod.session_scope() as s:
        old = _row("Test", "old", "Old")
        old.fetched_at = old_time
        fresh = _row("Test", "fresh", "Fresh")
        s.add_all([old, fresh])
        s.flush()
        rows = [old, fresh]
        for row in rows:
            s.expunge(row)

    monkeypatch.setattr(
        cache_mod,
        "embed_texts",
        lambda texts: np.ones((len(texts), 3), dtype=np.float32),
    )
    cache_mod.embed_item_rows(rows)

    assert store_mod.prune(5) == 1
    with store_mod.session_scope() as s:
        cached = s.execute(select(store_mod.ItemEmbeddingRow)).scalars().all()
        assert len(cached) == 1
        assert cached[0].item.title == "Fresh"


def test_deserialize_rejects_corrupt_or_truncated_blob():
    from dailydigest.rank.embedding_cache import _deserialize, _serialize

    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    raw = _serialize(vec)
    # Round-trips cleanly at the correct dim.
    assert np.allclose(_deserialize(raw, 3), vec)
    # Truncated blob (len != dim*4) -> None (treated as cache miss, re-embed).
    assert _deserialize(raw[:-2], 3) is None
    # Wrong/garbage dim -> None instead of raising mid-run.
    assert _deserialize(raw, 4) is None
    assert _deserialize(raw, 0) is None
    assert _deserialize(raw, 99999) is None
