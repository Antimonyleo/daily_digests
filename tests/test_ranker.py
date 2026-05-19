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
    _apply_quality_adjustments_with_features,
    _lr_weights_path,
    _cosine_score_items,
    pick_top_per_section,
    score_items,
    score_items_with_features,
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

    def _fake_embed(texts: list[str], is_query: bool = False) -> np.ndarray:
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
    row.source = ""
    row.id = None
    return row


def _profile_vec(dim: int = 3) -> np.ndarray:
    v = np.ones(dim, dtype=np.float32)
    return v / np.linalg.norm(v)


def _current_lr_schema() -> tuple[str, int]:
    from dailydigest.votes import LR_FEATURE_DIM, LR_FEATURE_SCHEMA_VERSION

    return LR_FEATURE_SCHEMA_VERSION, LR_FEATURE_DIM


# ---------------------------------------------------------------------------
# downweight via _cosine_score_items
# ---------------------------------------------------------------------------

class TestApplyDownweight:
    def test_penalty_is_exactly_downweight_constant(self):
        assert DOWNWEIGHT_PENALTY == 0.20

    def test_downweight_reduces_score_via_cosine_scorer(self):
        item = _make_row("CRISPR and cryptocurrency markets", "research")
        profile = _profile_vec(3)
        scored_pen = _cosine_score_items([item], profile, ["cryptocurrency"])
        scored_norm = _cosine_score_items([item], profile, [])
        assert scored_norm[0][1] - scored_pen[0][1] == pytest.approx(DOWNWEIGHT_PENALTY, abs=1e-6)

    def test_no_penalty_when_no_match(self):
        item = _make_row("CRISPR gene editing advances", "research")
        profile = _profile_vec(3)
        scored_pen = _cosine_score_items([item], profile, ["cryptocurrency"])
        scored_norm = _cosine_score_items([item], profile, [])
        assert scored_norm[0][1] == pytest.approx(scored_pen[0][1], abs=1e-6)

    def test_empty_downweight_list_no_change(self):
        items = [_make_row("some text", "research"), _make_row("other text", "research")]
        profile = _profile_vec(3)
        scored_pen = _cosine_score_items(items, profile, [])
        scored_norm = _cosine_score_items(items, profile, [])
        for (_, s1), (_, s2) in zip(scored_pen, scored_norm):
            assert s1 == pytest.approx(s2, abs=1e-6)


# ---------------------------------------------------------------------------
# _apply_quality_adjustments_with_features
# ---------------------------------------------------------------------------

def _apply_qa(items, base_scores, downweight_terms=None, reason_penalty_map=None):
    """Helper to call _apply_quality_adjustments_with_features and return just the scores."""
    n = len(items)
    result, _features = _apply_quality_adjustments_with_features(
        items,
        np.asarray(base_scores, dtype=np.float32),
        downweight_terms or [],
        reason_penalty_map,
        learned_scores=np.zeros(n, dtype=np.float32),
        hybrid_scores=np.asarray(base_scores, dtype=np.float32),
        scoring_mode="cosine",
    )
    return result


class TestApplyQualityAdjustments:
    def test_top_journal_bonus_is_only_a_tiebreaker(self):
        nature = _make_row(
            "Base editing delivery study",
            "research",
            "Moderately relevant but from a top journal.",
        )
        nature.source = "Nature"
        minor = _make_row(
            "Base editing delivery study in a niche outlet",
            "research",
            "Slightly closer to the profile but lower reputation.",
        )
        minor.source = "Minor Journal"

        adjusted = _apply_qa([nature, minor], [0.55, 0.58])

        assert adjusted[0] > adjusted[1]

    def test_high_impact_source_breaks_close_tie(self):
        high_impact = _make_row(
            "Base editing delivery study",
            "research",
            "Primary research with efficacy data.",
        )
        high_impact.source = "Nature"
        unknown = _make_row(
            "Base editing delivery study",
            "research",
            "Primary research with efficacy data.",
        )
        unknown.source = "Minor Journal"

        adjusted = _apply_qa([high_impact, unknown], [0.62, 0.62])

        assert adjusted[0] > adjusted[1]

    def test_topic_fit_can_beat_high_prestige_with_large_relevance_gap(self):
        nature = _make_row(
            "Broad cellular observation",
            "research",
            "Primary research in a high-impact journal but weakly matched.",
        )
        nature.source = "Nature"
        relevant = _make_row(
            "RNA delivery mechanism for targeted therapeutics",
            "research",
            "Detailed mechanism and efficacy data closely matching the profile.",
        )
        relevant.source = "Minor Journal"

        adjusted = _apply_qa([nature, relevant], [0.55, 0.70])

        assert adjusted[1] > adjusted[0]

    def test_low_prestige_research_can_surface_when_highly_novel_and_relevant(self):
        novel = _make_row(
            "First-in-class breakthrough CRISPR therapy approved after pivotal phase 3 trial",
            "research",
            "Highly novel and urgent gene editing result.",
        )
        novel.source = "Minor Journal"
        routine = _make_row(
            "Incremental review of known delivery chemistry",
            "research",
            "A useful but routine review.",
        )
        routine.source = "Nature Reviews Drug Discovery"

        adjusted = _apply_qa([novel, routine], [0.82, 0.54])

        assert adjusted[0] > adjusted[1]

    def test_industry_promotional_language_is_penalized_below_independent_news(self):
        promo = _make_row(
            "Company today announced the commercial launch of its AI discovery platform",
            "industry",
            "The company is pleased to announce availability and product features.",
        )
        promo.source = "Company Press Release"
        news = _make_row(
            "FDA approves first-in-class RNA therapy after phase 3 survival benefit",
            "industry",
            "Independent coverage of approval, efficacy, and safety data.",
        )
        news.source = "STAT News"

        adjusted = _apply_qa([promo, news], [0.70, 0.70])

        assert adjusted[1] > adjusted[0]

    def test_research_promotional_language_is_penalized(self):
        promo = _make_row(
            "Sponsored webinar on first-in-class CRISPR delivery",
            "research",
            "Register now for partner content about an AI discovery platform.",
        )
        promo.source = "Nature"
        clean = _make_row(
            "First-in-class CRISPR delivery study",
            "research",
            "Primary research on delivery chemistry and efficacy.",
        )
        clean.source = "Nature"

        adjusted = _apply_qa([promo, clean], [0.70, 0.70])

        assert adjusted[1] > adjusted[0]

    def test_regulatory_promotional_language_is_penalized(self):
        promo = _make_row(
            "Company today announced sponsored FDA submission webinar",
            "regulatory",
            "Register now for partner content about the product launch.",
        )
        promo.source = "Company Press Release"
        clean = _make_row(
            "FDA approves first-in-class therapy after phase 3 survival benefit",
            "regulatory",
            "Approval notice covering efficacy and safety data.",
        )
        clean.source = "FDA Drug Approvals (CDER)"

        adjusted = _apply_qa([promo, clean], [0.70, 0.70])

        assert adjusted[1] > adjusted[0]

    def test_reason_penalty_map_downweights_matching_item_id(self):
        low_impact = _make_row(
            "Routine RNA delivery update",
            "research",
            "Primary research with efficacy data.",
        )
        low_impact.id = 101
        neutral = _make_row(
            "Routine RNA delivery update",
            "research",
            "Primary research with efficacy data.",
        )
        neutral.id = 102

        adjusted = _apply_qa([low_impact, neutral], [0.70, 0.70], reason_penalty_map={101: 0.09})

        assert adjusted[1] - adjusted[0] == pytest.approx(0.09, abs=1e-6)


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

    def test_skips_angew_cover_entries(self):
        cover = _make_row("Front Cover: Molecular Catalysts", "research", "Cover picture.")
        cover.source = "Angew. Chem. Int. Ed."
        article = _make_row("Catalyst mechanism study", "research", "Primary research.")
        article.source = "Angew. Chem. Int. Ed."

        scored = _cosine_score_items([cover, article], _profile_vec(3), [])

        assert [row.title for row, _score in scored] == ["Catalyst mechanism study"]

    def test_skips_editorial_entries_without_new_information(self):
        editorial = _make_row(
            "Editorial: The future of biological research",
            "research",
            "This editorial discusses broad challenges and opportunities.",
        )
        editorial.source = "Nature"
        article = _make_row(
            "Single-cell atlas method reveals immune mechanism",
            "research",
            "Primary research reports a method and dataset.",
        )
        article.source = "Nature"

        scored = _cosine_score_items([editorial, article], _profile_vec(3), [])

        assert [row.title for row, _score in scored] == [article.title]


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

    def test_reason_penalty_map_is_supported_by_public_scorer(self):
        penalized = _make_row("CRISPR delivery update", "research")
        penalized.id = None
        penalized.external_id = "penalized"
        neutral = _make_row("CRISPR delivery update", "research")
        neutral.id = None
        neutral.external_id = "neutral"

        scored = score_items([penalized, neutral], _profile_vec(3), [], {"penalized": 0.15})

        score_by_external_id = {row.external_id: score for row, score in scored}
        assert score_by_external_id["neutral"] > score_by_external_id["penalized"]

    def test_score_items_with_features_returns_versioned_snapshots(self):
        item = _make_row("CRISPR delivery method", "research", "Primary research reports a method.")
        item.id = None
        item.source = "Nature Biotechnology"

        scored, features = score_items_with_features([item], _profile_vec(3), [])

        assert scored[0][0] is item
        payload = features[id(item)]
        assert payload["ranker_version"]
        assert payload["source_bucket"] == "published_journal"
        assert payload["scoring_mode"] in {"cosine", "hybrid_lr"}
        assert isinstance(payload["topic_score"], float)


class TestLRRankerPersistence:
    def test_fit_writes_loadable_npz(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
        from dailydigest import config as config_mod

        config_mod.reload_settings()
        feature_schema_version, feature_dim = _current_lr_schema()
        # Current engineered feature vectors from votes.py (10 features).
        X = np.asarray(
            [
                [0.9, 0.8, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.7, 0.1, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.1, 0.8, 0.5, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.9, 0.6, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        assert X.shape[1] == feature_dim
        y = np.asarray([1, 1, -1, -1], dtype=np.float32)

        ranker = LRRanker()
        ranker.fit(X, y)

        assert _lr_weights_path().exists()
        with np.load(_lr_weights_path()) as data:
            assert str(data["feature_schema_version"][0]) == feature_schema_version
            assert int(data["feature_dim"][0]) == feature_dim
        loaded = LRRanker()
        assert loaded.load() is True
        scores = loaded.score(X)
        assert scores.shape == (4,)

    def test_legacy_nine_dim_weights_without_schema_are_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
        from dailydigest import config as config_mod

        config_mod.reload_settings()
        path = _lr_weights_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            coef=np.ones((1, 9), dtype=np.float32),
            intercept=np.asarray([0.0], dtype=np.float32),
            classes=np.asarray([-1, 1], dtype=np.int32),
            feature_dim=np.asarray([9], dtype=np.int32),
        )

        assert LRRanker().load() is False

    def test_mismatched_feature_schema_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
        from dailydigest import config as config_mod

        config_mod.reload_settings()
        _, feature_dim = _current_lr_schema()
        path = _lr_weights_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            coef=np.ones((1, feature_dim), dtype=np.float32),
            intercept=np.asarray([0.0], dtype=np.float32),
            classes=np.asarray([-1, 1], dtype=np.int32),
            feature_dim=np.asarray([feature_dim], dtype=np.int32),
            feature_schema_version=np.asarray(["old_schema"]),
        )

        assert LRRanker().load() is False

    def test_current_feature_schema_weights_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
        from dailydigest import config as config_mod

        config_mod.reload_settings()
        feature_schema_version, feature_dim = _current_lr_schema()
        coef = np.zeros((1, feature_dim), dtype=np.float32)
        coef[0, 0] = 1.0
        path = _lr_weights_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            coef=coef,
            intercept=np.asarray([0.0], dtype=np.float32),
            classes=np.asarray([-1, 1], dtype=np.int32),
            feature_dim=np.asarray([feature_dim], dtype=np.int32),
            feature_schema_version=np.asarray([feature_schema_version]),
        )

        loaded = LRRanker()
        assert loaded.load() is True
        scores = loaded.score(np.zeros((2, feature_dim), dtype=np.float32))
        assert scores.shape == (2,)

    def test_missing_weight_cache_rechecks_when_weights_appear(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
        from dailydigest import config as config_mod
        from dailydigest.rank import ranker as ranker_mod

        config_mod.reload_settings()
        ranker_mod._LR_SINGLETON = False
        feature_schema_version, feature_dim = _current_lr_schema()
        coef = np.zeros((1, feature_dim), dtype=np.float32)
        coef[0, 0] = 1.0
        path = _lr_weights_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            coef=coef,
            intercept=np.asarray([0.0], dtype=np.float32),
            classes=np.asarray([-1, 1], dtype=np.int32),
            feature_dim=np.asarray([feature_dim], dtype=np.int32),
            feature_schema_version=np.asarray([feature_schema_version]),
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

    def test_research_selection_caps_arxiv_cs_when_journals_are_available(self):
        # arXiv CS items score higher but journals must still appear due to mandatory
        # quality fill. Per-source cap (Fix 7) means a single journal source is also
        # capped, so we check the arXiv ceiling and that journals are present.
        arxiv_items = []
        for idx in range(12):
            row = _make_row(f"arXiv CS method {idx}", "research")
            row.source = "arXiv cs.LG"
            arxiv_items.append((row, 0.95 - idx * 0.01))
        journal_items = []
        journal_sources = ["Nature Biotechnology", "Nature Methods", "Nature Medicine",
                           "Nature Chemistry", "Nature Materials"]
        for idx in range(10):
            row = _make_row(f"Nature family paper {idx}", "research")
            row.source = journal_sources[idx % len(journal_sources)]
            journal_items.append((row, 0.70 - idx * 0.01))

        result = pick_top_per_section(arxiv_items + journal_items, {"research": 10})

        arxiv_count = sum(1 for row, _score in result if row.source == "arXiv cs.LG")
        journal_count = sum(
            1 for row, _score in result
            if row.source in journal_sources
        )
        assert arxiv_count <= 1
        assert journal_count >= 6

    def test_research_selection_protects_top_journal_papers_below_preprints(self):
        scored = []
        for idx in range(8):
            row = _make_row(f"preprint {idx}", "research")
            row.source = "bioRxiv (recent)"
            scored.append((row, 0.92 - idx * 0.01))
        for idx in range(4):
            row = _make_row(f"Cell paper {idx}", "research")
            row.source = "Cell"
            scored.append((row, 0.66 - idx * 0.01))

        result = pick_top_per_section(scored, {"research": 8})

        assert any(row.source == "Cell" for row, _score in result)
        assert sum(1 for row, _score in result if row.source == "bioRxiv (recent)") <= 2

    def test_research_selection_caps_single_source_dominance(self):
        scored = []
        for idx in range(18):
            row = _make_row(f"Advanced Materials paper {idx}", "research")
            row.source = "Advanced Materials"
            scored.append((row, 1.00 - idx * 0.005))
        for idx in range(8):
            row = _make_row(f"Science paper {idx}", "research")
            row.source = "Science"
            scored.append((row, 0.82 - idx * 0.005))
        for idx in range(8):
            row = _make_row(f"Nature paper {idx}", "research")
            row.source = "Nature"
            scored.append((row, 0.78 - idx * 0.005))

        result = pick_top_per_section(scored, {"research": 30})

        source_counts: dict[str, int] = {}
        for row, _score in result:
            source_counts[row.source] = source_counts.get(row.source, 0) + 1
        assert source_counts["Advanced Materials"] <= 6
        assert source_counts["Science"] >= 6
        assert source_counts["Nature"] >= 6

    def test_exceptional_preprint_can_displace_marginal_journal_item(self):
        scored = []
        for idx in range(12):
            row = _make_row(f"Published journal paper {idx}", "research")
            row.source = "Advanced Materials" if idx < 6 else "Science"
            scored.append((row, 1.00 - idx * 0.01))
        strong_preprint = _make_row(
            "BioRxiv breakthrough RNA delivery preprint",
            "research",
            "Highly relevant method and mechanism.",
        )
        strong_preprint.source = "bioRxiv (recent)"
        scored.append((strong_preprint, 0.905))

        result = pick_top_per_section(scored, {"research": 10})

        assert strong_preprint in [row for row, _score in result]
        assert sum(1 for row, _score in result if row.source == "bioRxiv (recent)") == 1


# ---------------------------------------------------------------------------
# _multi_cosine (profile score normalization)
# ---------------------------------------------------------------------------

class TestMultiCosine:
    def test_single_row_weighted_profile_score_at_most_one(self):
        """Normalized score should not exceed 1.0 even for weight=1.5 rows."""
        from dailydigest.rank.ranker import _multi_cosine
        vec = np.ones((1, 3), dtype=np.float32) / np.sqrt(3)
        # weight=1.5 × unit_vec: without normalization dot product = 1.5
        profile_row = np.ones((1, 3), dtype=np.float32) * 1.5 / np.sqrt(3)
        scores = _multi_cosine(vec, profile_row)
        assert float(scores[0]) <= 1.0 + 1e-5, f"Score {scores[0]} should be <= 1.0"
        assert float(scores[0]) == pytest.approx(1.0, abs=1e-4)

    def test_unit_profile_unchanged(self):
        """When profile row norms <= 1.0, scores are unaffected."""
        from dailydigest.rank.ranker import _multi_cosine
        vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        profile = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # norm=1
        scores = _multi_cosine(vec, profile)
        assert float(scores[0]) == pytest.approx(1.0, abs=1e-5)

    def test_multi_row_uses_top1_top3_blend(self):
        """With 4 unit-norm profile rows, result is 0.7*max + 0.3*top3-mean."""
        from dailydigest.rank.ranker import _multi_cosine
        vecs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        profile = np.array([
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(1 - 0.25), 0.0],
            [0.3, np.sqrt(1 - 0.09), 0.0],
            [0.1, np.sqrt(1 - 0.01), 0.0],
        ], dtype=np.float32)
        scores = _multi_cosine(vecs, profile)
        top1 = 1.0
        top3_mean = np.mean([1.0, 0.5, 0.3])
        expected = 0.7 * top1 + 0.3 * top3_mean
        assert float(scores[0]) == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# _freshness_penalty
# ---------------------------------------------------------------------------

class TestFreshnessPenalty:
    """Tests for _freshness_penalty across all four section curves."""

    def _row(self, section: str, days_ago: float):
        from unittest.mock import MagicMock
        from datetime import datetime, timedelta, timezone
        row = MagicMock()
        row.section = section
        row.published_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
        return row

    def test_research_fresh_paper_gets_bonus(self):
        from dailydigest.rank.ranker import _freshness_penalty
        pen = _freshness_penalty(self._row("research", 1))
        assert pen < 0, f"Fresh research paper should get a bonus (negative penalty), got {pen}"
        assert pen == pytest.approx(-0.05, abs=0.01)

    def test_research_old_paper_gets_penalty(self):
        from dailydigest.rank.ranker import _freshness_penalty
        pen = _freshness_penalty(self._row("research", 40))
        assert pen > 0, "Old research paper should get a positive penalty"
        assert pen <= 0.09  # capped

    def test_research_no_date_returns_zero(self):
        from unittest.mock import MagicMock
        from dailydigest.rank.ranker import _freshness_penalty
        row = MagicMock()
        row.section = "research"
        row.published_at = None
        assert _freshness_penalty(row) == 0.0

    def test_world_breaking_news_gets_bonus(self):
        from dailydigest.rank.ranker import _freshness_penalty
        pen = _freshness_penalty(self._row("world", 0.1))  # ~2.4 hours ago
        assert pen == pytest.approx(-0.04, abs=1e-6), (
            f"World items <6h old should get -0.04 bonus, got {pen}"
        )

    def test_world_fresh_item_between_6h_and_3d_no_negative(self):
        from dailydigest.rank.ranker import _freshness_penalty
        pen = _freshness_penalty(self._row("world", 0.5))  # 12 hours ago
        assert pen >= 0, f"World item > 6h should not have a bonus, got {pen}"

    def test_world_stale_item_capped_at_0_20(self):
        from dailydigest.rank.ranker import _freshness_penalty
        pen = _freshness_penalty(self._row("world", 14))
        assert pen == pytest.approx(0.20, abs=0.01)

    def test_industry_follows_same_curve_as_world(self):
        from dailydigest.rank.ranker import _freshness_penalty
        pen_w = _freshness_penalty(self._row("world", 3))
        pen_i = _freshness_penalty(self._row("industry", 3))
        assert pen_w == pytest.approx(pen_i, abs=0.001)

    def test_regulatory_ramps_slowly(self):
        from dailydigest.rank.ranker import _freshness_penalty
        pen_2d = _freshness_penalty(self._row("regulatory", 2))
        pen_90d = _freshness_penalty(self._row("regulatory", 100))
        assert 0 <= pen_2d < pen_90d
        assert pen_90d == pytest.approx(0.10, abs=0.01)


# ---------------------------------------------------------------------------
# _mmr_select
# ---------------------------------------------------------------------------

class TestMMRSelect:
    """Tests for _mmr_select greedy Maximal Marginal Relevance selection."""

    def test_mmr_prefers_diverse_over_redundant(self, monkeypatch):
        from unittest.mock import MagicMock
        import numpy as np
        from dailydigest.rank.ranker import _mmr_select

        # Three items: A and B are near-identical, C is diverse
        vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.99, 0.14, 0.0], dtype=np.float32)  # near A
        vec_c = np.array([0.0, 1.0, 0.0], dtype=np.float32)   # diverse

        rows = [MagicMock(), MagicMock(), MagicMock()]
        # A has highest score, C has second-highest, B (near-duplicate) is lowest
        # This ensures C has a high enough normalized score to win MMR over B
        candidates = [(rows[0], 0.9), (rows[1], 0.5), (rows[2], 0.8)]

        # Monkeypatch embed_item_rows to return our known vectors
        monkeypatch.setattr(
            "dailydigest.rank.ranker.embed_item_rows",
            lambda rs: np.stack([vec_a, vec_b, vec_c])
        )

        # Select 2 of 3 — MMR should pick A (highest score) and C (diverse), not A+B (redundant)
        result = _mmr_select(candidates, cap=2, lambda_=0.7)
        selected_rows = [r for r, _ in result]
        assert rows[0] in selected_rows, "Highest-score item should always be selected first"
        assert rows[2] in selected_rows, "Diverse item C should be preferred over near-duplicate B"
        assert rows[1] not in selected_rows, "Near-duplicate B should be excluded"

    def test_mmr_returns_all_when_fewer_than_cap(self, monkeypatch):
        from unittest.mock import MagicMock
        import numpy as np
        from dailydigest.rank.ranker import _mmr_select

        monkeypatch.setattr(
            "dailydigest.rank.ranker.embed_item_rows",
            lambda rs: np.eye(len(rs), 3, dtype=np.float32)
        )
        rows = [MagicMock(), MagicMock()]
        candidates = [(rows[0], 0.9), (rows[1], 0.5)]
        result = _mmr_select(candidates, cap=5)
        assert len(result) == 2
