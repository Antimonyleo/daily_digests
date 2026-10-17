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


def test_profile_anchor_modes_preserve_legacy_and_embed_detailed_intent(monkeypatch):
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
