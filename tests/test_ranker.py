"""Tests for dailydigest.rank.ranker.score_items and pick_top_per_section.

The embed model (130 MB sentence-transformer) is replaced with a fast
deterministic fake to avoid loading it in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from dailydigest.rank.ranker import (
    DOWNWEIGHT_PENALTY,
    LRRanker,
    _apply_downweight,
    _lr_weights_path,
    _cosine_score_items,
    pick_top_per_section,
    score_items,
)


# ---------------------------------------------------------------------------
# Fixture: fake embed model
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    """Replace embed_texts with a deterministic hash-based stub.

    Each text maps to a 3-D float32 vector; no disk access, no torch.
    """
    from dailydigest.rank import embed as embed_mod

    def _fake_embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            h = hash(t)
            v = np.array([h % 7, h % 11, h % 13], dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vecs.append(v)
        return np.stack(vecs) if vecs else np.zeros((0, 3), dtype=np.float32)

    monkeypatch.setattr(embed_mod, "embed_texts", _fake_embed)
    # Also patch in ranker module's direct reference
    from dailydigest.rank import ranker as ranker_mod
    from dailydigest.rank import embedding_cache as cache_mod
    monkeypatch.setattr(cache_mod, "embed_texts", _fake_embed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(title: str, section: str = "research", abstract: str = "") -> MagicMock:
    row = MagicMock()
    row.title = title
    row.abstract = abstract
    row.section = section
    return row


def _profile_vec(dim: int = 3) -> np.ndarray:
    v = np.ones(dim, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# _apply_downweight
# ---------------------------------------------------------------------------

class TestApplyDownweight:
    def test_penalty_applied_when_term_in_text(self):
        base = np.array([0.8], dtype=np.float32)
        texts = ["CRISPR and cryptocurrency markets"]
        result = _apply_downweight(base, texts, ["cryptocurrency"])
        assert result[0] == pytest.approx(0.8 - DOWNWEIGHT_PENALTY, abs=1e-6)

    def test_no_penalty_when_no_match(self):
        base = np.array([0.8], dtype=np.float32)
        texts = ["CRISPR gene editing advances"]
        result = _apply_downweight(base, texts, ["cryptocurrency"])
        assert result[0] == pytest.approx(0.8, abs=1e-6)

    def test_penalty_is_exactly_downweight_constant(self):
        assert DOWNWEIGHT_PENALTY == 0.05

    def test_empty_downweight_list_no_change(self):
        base = np.array([0.5, 0.7], dtype=np.float32)
        texts = ["some text", "other text"]
        result = _apply_downweight(base, texts, [])
        assert result == pytest.approx([0.5, 0.7], abs=1e-6)

    def test_case_insensitive_matching(self):
        # downweight terms are lowercased before comparison
        base = np.array([0.9], dtype=np.float32)
        texts = ["CRYPTOCURRENCY news today"]
        result = _apply_downweight(base, texts, ["cryptocurrency"])
        assert result[0] == pytest.approx(0.9 - DOWNWEIGHT_PENALTY, abs=1e-6)


# ---------------------------------------------------------------------------
# _cosine_score_items (uses monkeypatched embed)
# ---------------------------------------------------------------------------

class TestCosineScoreItems:
    def test_returns_sorted_descending(self):
        items = [_make_row(t, "research") for t in ["alpha", "beta", "gamma"]]
        profile = _profile_vec(3)
        scored = _cosine_score_items(items, profile, [])
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_returns_all_items(self):
        items = [_make_row(f"item{i}", "research") for i in range(5)]
        profile = _profile_vec(3)
        scored = _cosine_score_items(items, profile, [])
        assert len(scored) == 5

    def test_empty_items_returns_empty(self):
        result = _cosine_score_items([], _profile_vec(3), [])
        assert result == []

    def test_downweight_reduces_score(self):
        # An item containing a downweight term should score lower than otherwise
        item_penalized = _make_row("CRISPR and cryptocurrency", "research")
        item_normal = _make_row("CRISPR gene therapy", "research")
        profile = _profile_vec(3)

        # Score each individually
        scored_pen = _cosine_score_items([item_penalized], profile, ["cryptocurrency"])
        scored_norm = _cosine_score_items([item_penalized], profile, [])

        penalty_applied = scored_pen[0][1]
        no_penalty = scored_norm[0][1]
        assert no_penalty - penalty_applied == pytest.approx(DOWNWEIGHT_PENALTY, abs=1e-6)


# ---------------------------------------------------------------------------
# score_items (public entry point — falls back to cosine since no LR weights)
# ---------------------------------------------------------------------------

class TestScoreItems:
    def test_smoke_returns_list_of_tuples(self):
        items = [_make_row("alpha", "research"), _make_row("beta", "industry")]
        profile = _profile_vec(3)
        scored = score_items(items, profile, [])
        assert isinstance(scored, list)
        assert all(isinstance(row, MagicMock) and isinstance(s, float) for row, s in scored)

    def test_no_lr_weights_uses_cosine(self):
        # Without an lr_ranker.npz file, score_items falls back to cosine.
        # Result should still be sorted descending.
        items = [_make_row(f"item{i}", "research") for i in range(4)]
        scored = score_items(items, _profile_vec(3), [])
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)


class TestLRRankerPersistence:
    def test_fit_writes_loadable_npz(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
        from dailydigest import config as config_mod

        config_mod.reload_settings()
        X = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
        y = np.asarray([1, 1, -1, -1], dtype=np.float32)

        ranker = LRRanker()
        ranker.fit(X, y)

        assert _lr_weights_path().exists()
        loaded = LRRanker()
        assert loaded.load() is True
        scores = loaded.score(X)
        assert scores.shape == (4,)

    def test_missing_weight_cache_rechecks_when_weights_appear(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
        from dailydigest import config as config_mod
        from dailydigest.rank import ranker as ranker_mod

        config_mod.reload_settings()
        ranker_mod._LR_SINGLETON = False
        path = _lr_weights_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            coef=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            intercept=np.asarray([0.0], dtype=np.float32),
            classes=np.asarray([-1, 1], dtype=np.int32),
        )

        try:
            assert ranker_mod.get_lr_ranker() is not None
        finally:
            ranker_mod.reset_lr_cache()


# ---------------------------------------------------------------------------
# pick_top_per_section
# ---------------------------------------------------------------------------

class TestPickTopPerSection:
    def test_respects_per_section_caps(self):
        scored = [
            (_make_row("R1", "research"), 0.9),
            (_make_row("R2", "research"), 0.8),
            (_make_row("R3", "research"), 0.7),
            (_make_row("I1", "industry"), 0.6),
            (_make_row("I2", "industry"), 0.5),
        ]
        caps = {"research": 2, "industry": 1}
        result = pick_top_per_section(scored, caps)
        sections = [row.section for row, _ in result]
        assert sections.count("research") == 2
        assert sections.count("industry") == 1

    def test_unknown_section_skipped(self):
        scored = [
            (_make_row("X1", "unknown_section"), 0.99),
            (_make_row("R1", "research"), 0.5),
        ]
        caps = {"research": 2}
        result = pick_top_per_section(scored, caps)
        sections = [row.section for row, _ in result]
        assert "unknown_section" not in sections
        assert "research" in sections

    def test_preserves_descending_score_order(self):
        scored = [
            (_make_row("A", "research"), 0.9),
            (_make_row("B", "research"), 0.7),
            (_make_row("C", "research"), 0.5),
        ]
        caps = {"research": 3}
        result = pick_top_per_section(scored, caps)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_scored_returns_empty(self):
        assert pick_top_per_section([], {"research": 5}) == []

    def test_zero_cap_excludes_section(self):
        scored = [(_make_row("R1", "research"), 0.8)]
        caps = {"research": 0}
        result = pick_top_per_section(scored, caps)
        assert result == []

    def test_multiple_sections_mixed(self):
        scored = [
            (_make_row("R1", "research"), 1.0),
            (_make_row("I1", "industry"), 0.9),
            (_make_row("R2", "research"), 0.8),
            (_make_row("I2", "industry"), 0.7),
            (_make_row("G1", "general"), 0.6),
        ]
        caps = {"research": 2, "industry": 1, "general": 1}
        result = pick_top_per_section(scored, caps)
        by_section: dict[str, int] = {}
        for row, _ in result:
            by_section[row.section] = by_section.get(row.section, 0) + 1
        assert by_section["research"] == 2
        assert by_section["industry"] == 1
        assert by_section["general"] == 1
