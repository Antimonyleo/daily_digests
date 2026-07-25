"""Tests for facet attribution (P1) + topic-priority axis (P3).

Covers:
- attribute_items primary/secondary selection over a synthetic vecs/matrix
- empty keywords -> build_attribution_context returns None
- priority normalization (max -> 1.0, unlisted -> 0.5 default)
- priority_bonus == scale * priority
- the 4 new feature-payload keys are ALWAYS present
- the priority bonus is recorded for later selection ordering, never ranker score/gate
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from dailydigest.models import Profile
from dailydigest.rank import profile as profile_mod
from dailydigest.rank.profile import (
    AttributionContext,
    ItemAttribution,
    attribute_items,
    build_attribution_context,
)


# --------------------------------------------------------------------------- #
# Fake embedder: each keyword maps to a distinct one-hot-ish unit vector so we
# can construct exact cosine similarities in tests.
# --------------------------------------------------------------------------- #

# A deterministic mapping from a small vocabulary to unit basis vectors.
_VOCAB = {
    "alpha": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "beta": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "gamma": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "delta": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    def _fake_embed(texts, is_query: bool = False):
        vecs = []
        for t in texts:
            v = _VOCAB.get(t.strip().lower())
            if v is None:
                # Fallback: deterministic hash vector.
                h = hash(t)
                v = np.array([h % 5, h % 7, h % 11, h % 13], dtype=np.float32)
                n = np.linalg.norm(v)
                v = v / n if n else v
            vecs.append(v.astype(np.float32))
        return np.stack(vecs) if vecs else np.zeros((0, 4), dtype=np.float32)

    monkeypatch.setattr(profile_mod, "embed_texts", _fake_embed)
    # Clear the module-level facet-matrix cache so each test re-embeds.
    profile_mod._CORE_FACET_CACHE.clear()


# --------------------------------------------------------------------------- #
# attribute_items primary / secondary
# --------------------------------------------------------------------------- #

def _ctx(labels, priorities=None) -> AttributionContext:
    matrix = np.stack([_VOCAB[label] for label in labels]).astype(np.float32)
    return AttributionContext(matrix=matrix, labels=list(labels), priorities=priorities or {})


class TestAttributeItems:
    def test_primary_is_argmax_facet(self):
        ctx = _ctx(["alpha", "beta", "gamma"])
        # Item aligned entirely to "beta".
        vecs = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        attrs = attribute_items(vecs, ctx)
        assert attrs[0].primary == "beta"

    def test_weak_match_has_empty_primary(self):
        ctx = _ctx(["alpha", "beta"])
        # Item roughly equidistant to two orthogonal facets -> each sim ~0.707,
        # but point it mostly at delta (not in matrix): sim to alpha/beta = 0.31.
        v = np.array([0.31, 0.31, 0.0, 0.897], dtype=np.float32)
        v = v / np.linalg.norm(v)
        attrs = attribute_items(v.reshape(1, -1), ctx)
        # Both facet sims (~0.31) are below the 0.32 primary threshold.
        assert attrs[0].primary == ""
        assert attrs[0].secondaries == []

    def test_secondaries_within_margin(self):
        ctx = _ctx(["alpha", "beta", "gamma"])
        # Item close to alpha, slightly less to beta, far from gamma.
        v = np.array([0.80, 0.76, 0.0, 0.0], dtype=np.float32)
        v = v / np.linalg.norm(v)
        attrs = attribute_items(v.reshape(1, -1), ctx)
        a = attrs[0]
        assert a.primary == "alpha"
        # beta sim within 0.06 of primary and above floor -> included; gamma not.
        assert a.secondaries == ["beta"]

    def test_secondaries_capped_at_two(self):
        ctx = _ctx(["alpha", "beta", "gamma", "delta"])
        # Nearly-equal on all four -> primary + at most 2 secondaries.
        v = np.array([0.51, 0.50, 0.49, 0.50], dtype=np.float32)
        v = v / np.linalg.norm(v)
        attrs = attribute_items(v.reshape(1, -1), ctx)
        assert attrs[0].primary != ""
        assert len(attrs[0].secondaries) <= 2

    def test_empty_ctx_matrix_returns_defaults(self):
        empty = AttributionContext(matrix=np.zeros((0, 0), dtype=np.float32), labels=[], priorities={})
        vecs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        attrs = attribute_items(vecs, empty)
        assert attrs == [ItemAttribution()]


# --------------------------------------------------------------------------- #
# build_attribution_context: None on no keywords
# --------------------------------------------------------------------------- #

class TestBuildContext:
    def test_none_when_no_keywords(self):
        p = Profile(bio="x", keywords=[])
        assert build_attribution_context(p) is None

    def test_context_built_with_keywords(self):
        p = Profile(bio="x", keywords=["alpha", "beta"])
        ctx = build_attribution_context(p)
        assert ctx is not None
        assert ctx.labels == ["alpha", "beta"]
        assert ctx.matrix.shape == (2, 4)
        # Rows are unit vectors.
        norms = np.linalg.norm(ctx.matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_canonical_facets_replace_keyword_attribution_only(self):
        p = Profile(
            bio="x",
            # These stay available to retrieval, but are not attribution labels.
            keywords=["alpha", "beta", "gamma"],
            canonical_facets={
                "structural nucleic-acid nanotechnology": {
                    "anchors": ["alpha"],
                    "aliases": ["beta"],
                    "priority": 18,
                },
                "programmable colloidal assembly": {
                    "anchors": ["gamma"],
                    "priority": 9,
                },
            },
        )
        ctx = build_attribution_context(p)
        assert ctx is not None
        assert ctx.labels == [
            "structural nucleic-acid nanotechnology",
            "programmable colloidal assembly",
        ]
        # An item aligned to gamma receives the canonical (not raw keyword) name.
        attrs = attribute_items(_VOCAB["gamma"].reshape(1, -1), ctx)
        assert attrs[0].primary == "programmable colloidal assembly"
        assert attrs[0].primary_score == pytest.approx(1.0)
        assert attrs[0].primary_facet_score == pytest.approx(1.0)

    def test_canonical_priority_overrides_legacy_keyword_priorities(self):
        p = Profile(
            bio="x",
            keywords=["alpha", "beta"],
            topic_priorities={"alpha": 99, "beta": 1},
            canonical_facets={
                "first": {"anchors": ["alpha"], "priority": 4},
                "second": {"anchors": ["beta"], "priority": 2},
            },
        )
        ctx = build_attribution_context(p)
        assert ctx is not None
        assert ctx.priorities == {"first": pytest.approx(1.0), "second": pytest.approx(0.5)}


# --------------------------------------------------------------------------- #
# Priority normalization + bonus
# --------------------------------------------------------------------------- #

class TestPriorityNormalization:
    def test_max_maps_to_one(self):
        p = Profile(
            bio="x",
            keywords=["alpha", "beta", "gamma"],
            topic_priorities={"alpha": 18.0, "beta": 9.0},
        )
        ctx = build_attribution_context(p)
        assert ctx is not None
        assert ctx.priorities["alpha"] == pytest.approx(1.0)
        assert ctx.priorities["beta"] == pytest.approx(0.5)

    def test_unlisted_keyword_gets_default(self):
        p = Profile(bio="x", keywords=["alpha", "gamma"], topic_priorities={"alpha": 10.0})
        ctx = build_attribution_context(p)
        assert ctx is not None
        # gamma is not listed -> attribute_items uses the moderate default 0.5.
        v = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)  # aligned to gamma
        attrs = attribute_items(v, ctx)
        assert attrs[0].primary == "gamma"
        assert attrs[0].priority == pytest.approx(0.5)

    def test_empty_priorities_all_default(self):
        p = Profile(bio="x", keywords=["alpha"], topic_priorities={})
        ctx = build_attribution_context(p)
        assert ctx is not None
        assert ctx.priorities == {}
        v = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        attrs = attribute_items(v, ctx)
        assert attrs[0].priority == pytest.approx(0.5)

    def test_priority_bonus_equals_scale_times_priority(self, monkeypatch):
        # Pin the scale so the assertion is exact regardless of env/config.
        monkeypatch.setattr(profile_mod, "_topic_priority_bonus_scale", lambda: 0.06)
        ctx = _ctx(["alpha", "beta"], priorities={"alpha": 1.0, "beta": 0.5})
        v_alpha = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        v_beta = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        a = attribute_items(v_alpha, ctx)[0]
        b = attribute_items(v_beta, ctx)[0]
        assert a.priority_bonus == pytest.approx(0.06 * 1.0)
        assert b.priority_bonus == pytest.approx(0.06 * 0.5)


# --------------------------------------------------------------------------- #
# Feature-payload contract + gate invariance
# --------------------------------------------------------------------------- #

class TestFeaturePayloadContract:
    def _fake_ranker_embed(self, monkeypatch):
        """Patch the ranker/embedding-cache embedders with a keyword-aware stub."""
        from dailydigest.rank import embedding_cache as cache_mod

        def _fake(texts, is_query: bool = False):
            vecs = []
            for t in texts:
                key = t.strip().lower()
                v = None
                for name, basis in _VOCAB.items():
                    if name in key:
                        v = basis
                        break
                if v is None:
                    h = hash(t)
                    v = np.array([h % 5, h % 7, h % 11, h % 13], dtype=np.float32)
                    n = np.linalg.norm(v)
                    v = v / n if n else v
                vecs.append(v.astype(np.float32))
            return np.stack(vecs) if vecs else np.zeros((0, 4), dtype=np.float32)

        monkeypatch.setattr(cache_mod, "embed_texts", _fake)
        return _fake

    def _row(self, title, section="research"):
        row = MagicMock()
        row.title = title
        row.abstract = ""
        row.section = section
        row.source = ""
        row.id = None
        return row

    def test_four_keys_always_present_without_attribution(self, monkeypatch):
        self._fake_ranker_embed(monkeypatch)
        from dailydigest.rank.ranker import _cosine_score_items_with_features

        item = self._row("alpha topic paper")
        profile_vec = _VOCAB["alpha"].reshape(1, -1)
        _scored, feats = _cosine_score_items_with_features(
            [item], profile_vec, [], None, attribution=None
        )
        (payload,) = feats.values()
        assert payload["primary_facet"] == ""
        assert payload["secondary_facets"] == []
        assert payload["topic_priority"] == 0.0
        assert payload["topic_priority_bonus"] == 0.0
        assert isinstance(payload["primary_facet"], str)
        assert isinstance(payload["secondary_facets"], list)
        assert isinstance(payload["topic_priority"], float)
        assert isinstance(payload["topic_priority_bonus"], float)

    def test_four_keys_populated_with_attribution(self, monkeypatch):
        self._fake_ranker_embed(monkeypatch)
        monkeypatch.setattr(profile_mod, "_topic_priority_bonus_scale", lambda: 0.06)
        from dailydigest.rank.ranker import _cosine_score_items_with_features

        item = self._row("alpha topic paper")
        profile_vec = _VOCAB["alpha"].reshape(1, -1)
        ctx = _ctx(["alpha", "beta"], priorities={"alpha": 1.0, "beta": 0.5})
        _scored, feats = _cosine_score_items_with_features(
            [item], profile_vec, [], None, attribution=ctx
        )
        (payload,) = feats.values()
        assert payload["primary_facet"] == "alpha"
        assert payload["topic_priority"] == pytest.approx(1.0)
        assert payload["topic_priority_bonus"] == pytest.approx(0.06)

    def test_priority_bonus_does_not_change_ranker_score_or_topic_gate(self, monkeypatch):
        """Priority is persisted for the selection layer, not folded into ranker scores."""
        self._fake_ranker_embed(monkeypatch)
        monkeypatch.setattr(profile_mod, "_topic_priority_bonus_scale", lambda: 0.06)
        from dailydigest.rank.ranker import _apply_quality_adjustments_with_features

        item_hi = self._row("alpha topic paper")
        item_lo = self._row("beta topic paper")
        items = [item_hi, item_lo]
        # Equal base cosine for both.
        base = np.array([0.80, 0.80], dtype=np.float32)
        attrs = [
            ItemAttribution(primary="alpha", secondaries=[], priority=1.0, priority_bonus=0.06),
            ItemAttribution(primary="beta", secondaries=[], priority=0.5, priority_bonus=0.03),
        ]
        finals, feats = _apply_quality_adjustments_with_features(
            items,
            base,
            [],
            None,
            learned_scores=np.zeros(2, dtype=np.float32),
            hybrid_scores=base,
            scoring_mode="cosine",
            facet_attr=attrs,
        )
        f_hi = feats[id(item_hi)]
        f_lo = feats[id(item_lo)]
        # Gate value (topic_score) is the raw base cosine — untouched by the bonus.
        assert f_hi["topic_score"] == pytest.approx(0.80)
        assert f_lo["topic_score"] == pytest.approx(0.80)
        assert f_hi["topic_score"] == pytest.approx(f_lo["topic_score"])
        assert f_hi["topic_priority_bonus"] == pytest.approx(0.06)
        assert f_lo["topic_priority_bonus"] == pytest.approx(0.03)
        # The selection layer may use that metadata later; ranker scores are
        # intentionally equal, so priority cannot cross a quality/final cutoff.
        assert f_hi["final_score"] == pytest.approx(f_lo["final_score"])
        assert finals[0] == pytest.approx(finals[1])

    def test_bonus_absent_leaves_final_equal_to_no_attribution(self, monkeypatch):
        """With attribution=None, final_score == quality-adjusted score (no nudge)."""
        self._fake_ranker_embed(monkeypatch)
        from dailydigest.rank.ranker import _apply_quality_adjustments_with_features

        item = self._row("alpha topic paper")
        base = np.array([0.75], dtype=np.float32)
        finals_no, feats_no = _apply_quality_adjustments_with_features(
            [item], base, [], None,
            learned_scores=np.zeros(1, dtype=np.float32),
            hybrid_scores=base, scoring_mode="cosine", facet_attr=None,
        )
        (p,) = feats_no.values()
        assert p["topic_priority_bonus"] == 0.0
        assert p["primary_facet"] == ""
