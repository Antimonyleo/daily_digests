"""Tests for cross-day near-duplicate suppression."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from dailydigest import config as config_mod
from dailydigest import store as store_mod
from dailydigest.rank import near_dup as near_dup_mod
from dailydigest.rank.near_dup import exclude_recent_near_duplicates


def _settings(**overrides):
    return config_mod.load_settings().model_copy(update=overrides)


def _candidate(ext_id: str, item_id: int) -> store_mod.ItemRow:
    return store_mod.ItemRow(
        id=item_id,
        source="OpenAlex (biotech)",
        section="research",
        external_id=ext_id,
        url=f"https://example.com/{ext_id}",
        title=f"Title {ext_id}",
        abstract="abstract",
        published_at=datetime.now(timezone.utc),
    )


def _insert_shown_item() -> int:
    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="shown",
            url="https://nature.com/articles/shown",
            title="A landmark CRISPR result",
            abstract="abstract",
            published_at=datetime.now(timezone.utc),
        )
        s.add(row)
        s.flush()
        item_id = int(row.id)
    store_mod.write_digest("2026-06-01", [("R1", item_id, 0.9)])
    store_mod.mark_sent("2026-06-01")
    return item_id


# Map each row to a vector by external_id so similarity is deterministic.
_VECS = {
    "shown": [1.0, 0.0, 0.0],
    "dup": [1.0, 0.0, 0.0],       # identical direction → cosine 1.0
    "distinct": [0.0, 1.0, 0.0],  # orthogonal → cosine 0.0
}


def _fake_embed(rows):
    return np.array([_VECS[r.external_id] for r in rows], dtype=np.float32)


def test_drops_recent_near_duplicate_from_other_source(monkeypatch):
    _insert_shown_item()
    monkeypatch.setattr(near_dup_mod, "embed_item_rows", _fake_embed)

    candidates = [_candidate("dup", 101), _candidate("distinct", 102)]
    kept, dropped = exclude_recent_near_duplicates(
        candidates, settings=_settings(cross_day_dedupe=True, cross_day_dedupe_threshold=0.93)
    )

    kept_ids = {r.id for r in kept}
    assert kept_ids == {102}              # distinct kept
    assert len(dropped) == 1
    assert dropped[0]["item_id"] == 101    # near-dup dropped
    assert dropped[0]["max_similarity"] >= 0.93


def test_drops_near_duplicate_shown_in_unsent_browser_digest(monkeypatch):
    store_mod.init_db()
    with store_mod.session_scope() as s:
        shown = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="shown",
            url="https://nature.com/articles/browser-shown",
            title="A landmark CRISPR result",
            abstract="abstract",
            published_at=datetime.now(timezone.utc),
        )
        s.add(shown)
        s.flush()
        shown_id = int(shown.id)
    digest_id = "2026-06-02"
    store_mod.write_digest(digest_id, [("R1", shown_id, 0.9)])
    store_mod.write_impressions(
        digest_id,
        [("research", shown_id, 0, 0.9, True, "genome editing", 0.8, 0.8)],
    )
    store_mod.mark_impressions_viewed(digest_id)
    monkeypatch.setattr(near_dup_mod, "embed_item_rows", _fake_embed)

    kept, dropped = exclude_recent_near_duplicates(
        [_candidate("dup", 101)],
        settings=_settings(cross_day_dedupe=True, cross_day_dedupe_threshold=0.93),
    )

    assert kept == []
    assert [row["item_id"] for row in dropped] == [101]


def test_unviewed_browser_candidate_pool_does_not_count_as_shown(monkeypatch):
    store_mod.init_db()
    with store_mod.session_scope() as s:
        candidate_pool_row = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="shown",
            url="https://nature.com/articles/not-shown",
            title="A candidate that was never displayed",
            abstract="abstract",
            published_at=datetime.now(timezone.utc),
        )
        s.add(candidate_pool_row)
        s.flush()
        item_id = int(candidate_pool_row.id)
    store_mod.write_impressions(
        "2026-06-02",
        [("research", item_id, 10, 0.4, False, "genome editing", 0.8, 0.8)],
    )
    monkeypatch.setattr(near_dup_mod, "embed_item_rows", _fake_embed)

    kept, dropped = exclude_recent_near_duplicates(
        [_candidate("dup", 101)], settings=_settings(cross_day_dedupe=True)
    )

    assert [row.id for row in kept] == [101]
    assert dropped == []


def test_disabled_returns_unchanged(monkeypatch):
    _insert_shown_item()
    monkeypatch.setattr(near_dup_mod, "embed_item_rows", _fake_embed)
    candidates = [_candidate("dup", 101)]
    kept, dropped = exclude_recent_near_duplicates(
        candidates, settings=_settings(cross_day_dedupe=False)
    )
    assert len(kept) == 1 and dropped == []


def test_no_recent_digests_is_noop(monkeypatch):
    store_mod.init_db()  # empty store, nothing shown
    monkeypatch.setattr(near_dup_mod, "embed_item_rows", _fake_embed)
    candidates = [_candidate("dup", 101), _candidate("distinct", 102)]
    kept, dropped = exclude_recent_near_duplicates(candidates, settings=_settings())
    assert len(kept) == 2 and dropped == []


def test_empty_candidates():
    kept, dropped = exclude_recent_near_duplicates([], settings=_settings())
    assert kept == [] and dropped == []
