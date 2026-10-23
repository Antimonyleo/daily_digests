from __future__ import annotations

import numpy as np


def test_build_profile_matrix_applies_interest_weights(monkeypatch):
    from dailydigest.models import Profile
    from dailydigest.rank import profile as profile_mod

    def _fake_embed(texts: list[str], is_query: bool = False) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr(profile_mod, "embed_texts", _fake_embed)

    profile = Profile(
        bio="I study RNA medicines and clinical translation.",
        keywords=["RNA therapeutics"],
        interest_weights={"RNA therapeutics": 2.0, "arXiv CS methods": 0.5},
    )

    matrix = profile_mod.build_profile_matrix(profile)

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix[1], np.asarray([2.0, 2.0, 2.0], dtype=np.float32))
    assert np.allclose(matrix[2], np.asarray([0.5, 0.5, 0.5], dtype=np.float32))


def test_core_facet_cache_uses_actual_backend_after_fallback(monkeypatch):
    from dailydigest.models import Profile
    from dailydigest.rank import profile as module
    identity = ["preferred"]
    calls, pins = [], []
    def encode(texts, **kwargs):
        calls.append(texts)
        identity[0] = "actual-fallback"
        return np.tile([1., 0.], (len(texts), 1)).astype(np.float32)
    monkeypatch.setattr(module, "active_embedding_signature", lambda **kw: identity[0])
    monkeypatch.setattr(module, "embed_texts", encode)
    monkeypatch.setattr(module, "pin_cached_embedding_backend", lambda: pins.append(identity[0]))
    monkeypatch.setattr(module, "_CORE_FACET_CACHE", {})
    profile = Profile(bio="", keywords=["engineering"])
    matrix, _ = module.build_core_facet_matrix(profile)
    assert {key[0] for key in module._CORE_FACET_CACHE} == {"actual-fallback"}
    repeated, _ = module.build_core_facet_matrix(profile)
    np.testing.assert_array_equal(repeated, matrix)
    assert len(calls) == 1 and pins == ["actual-fallback"]


def test_profile_anchor_modes_keep_unproven_replacement_out_of_default(monkeypatch):
    import pytest
    from dailydigest.models import Profile
    from dailydigest.rank import profile as profile_mod

    calls = []
    def embed(texts, is_query=False):
        calls.append(list(texts))
        return np.tile([1.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)
    monkeypatch.setattr(profile_mod, "embed_texts", embed)
    monkeypatch.setattr(profile_mod, "active_embedding_signature", lambda **kw: "test-anchor-mode")
    profile_mod._CORE_FACET_CACHE.clear()
    profile = Profile(bio="", keywords=["RNA"], canonical_facets={
        "RNA engineering": {"anchors": ["Precise RNA structural design and folding"]}
    })
    legacy = profile_mod.build_profile_matrix(profile)
    assert calls == [["RNA"]]
    detailed = profile_mod.build_profile_matrix(profile, mode="detailed")
    assert detailed.shape[0] == 2
    assert calls[-1] == ["RNA", "RNA engineering; Precise RNA structural design and folding"]
    canonical = profile_mod.build_profile_matrix(profile, mode="canonical")
    assert calls[-1] == ["Precise RNA structural design and folding"]
    additive = profile_mod.build_profile_matrix(profile, mode="additive")
    np.testing.assert_array_equal(additive, np.vstack([legacy, canonical]))
    with pytest.raises(ValueError, match="mode"):
        profile_mod.build_profile_matrix(profile, mode="typo")


def test_append_learned_profile_freezes_cold_start_activation():
    from dailydigest.rank.profile import append_learned_profile

    static = np.tile([1.0, 0.0], (19, 1)).astype(np.float32)
    learned = np.array([0.0, 2.0], dtype=np.float32)
    for count in (0, 1):
        np.testing.assert_array_equal(append_learned_profile(static, learned, count), static)
    for count in (2, 3, 29, 30, 100):
        result = append_learned_profile(static, learned, count)
        gamma = min(0.25, count / 60.0)
        assert result.shape == (20, 2)
        np.testing.assert_array_equal(result[:-1], static)
        np.testing.assert_allclose(result[-1], [0.0, gamma * 19 / (1 - gamma)])
    np.testing.assert_array_equal(static[:, 1], 0.0)


def test_append_learned_profile_invalid_vector_is_static_fallback():
    from dailydigest.rank.profile import append_learned_profile

    static = np.eye(3, dtype=np.float32)
    for learned in (None, np.zeros(3), np.ones(2), np.ones((1, 3)),
                    np.array([np.nan, 0, 1]), np.array([np.inf, 0, 1])):
        np.testing.assert_array_equal(append_learned_profile(static, learned, 30), static)


def test_research_context_seeds_and_background_are_real_embedding_inputs():
    from dailydigest.models import Profile
    from dailydigest.rank.profile import CONTEXT_FACET_WEIGHT, _profile_parts_with_weights
    profile = Profile(bio='I develop molecular sensing methods.', keywords=['Sensors'],
                      seed_works=['A method for programmable molecular sensing'],
                      context_keywords=['Climate policy'])
    parts, weights = _profile_parts_with_weights(profile)
    inputs = dict(zip(parts, weights, strict=True))
    assert inputs[profile.bio] == 1.5
    assert inputs['Sensors'] == 1.0
    assert inputs[profile.seed_works[0]] == 2.0
    assert inputs['Climate policy'] == CONTEXT_FACET_WEIGHT


def test_explicit_anchor_reaches_relevance_in_one_row_per_topic():
    from dailydigest.models import Profile
    from dailydigest.rank.profile import _profile_parts_with_weights

    goal = 'Recruit E3 ligases to degrade target proteins'
    profile = Profile(bio='', keywords=['induced proximity'], canonical_facets={
        'induced proximity': {'anchors': ['induced proximity', goal, goal],
                             'aliases': ['induced proximity', 'molecular glues']}})
    parts, weights = _profile_parts_with_weights(profile, include_anchors=True)
    assert parts == [f'induced proximity; {goal}; molecular glues']
    assert weights == [1.0]


def test_distinct_canonical_anchor_is_used_but_label_placeholder_is_not_extra_vote():
    from dailydigest.models import Profile
    from dailydigest.rank.profile import _profile_parts_with_weights

    profile = Profile(bio='', keywords=['RNA'], canonical_facets={
        'RNA engineering': {'anchors': ['RNA engineering']}})
    assert _profile_parts_with_weights(profile)[0] == ['RNA']
    profile.canonical_facets['RNA engineering'].anchors.append('Design RNA structures')
    assert _profile_parts_with_weights(profile, include_anchors=True)[0] == [
        'RNA', 'RNA engineering; Design RNA structures']


def test_short_meaningful_reader_text_is_not_dropped_by_character_count():
    from dailydigest.models import Profile
    from dailydigest.rank.profile import _profile_parts_with_weights

    profile = Profile(bio='研究蛋白质的定向设计', keywords=['protein design'],
                      seed_works=['蛋白质结构的从头设计', 'RNA design', '...'])
    parts, _ = _profile_parts_with_weights(profile)
    assert profile.bio in parts
    assert profile.seed_works[0] in parts
    assert 'RNA design' in parts
    assert '...' not in parts
