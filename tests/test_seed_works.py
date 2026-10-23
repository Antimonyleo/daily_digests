"""Tests for seed_works profile anchors."""

from __future__ import annotations

import numpy as np

from dailydigest.models import Profile
from dailydigest.rank import profile as profile_mod


def _fake_embed(texts, is_query=False):
    # One row per input text; content irrelevant, count is what we assert on.
    return np.ones((len(texts), 4), dtype=np.float32)


def test_seed_works_add_high_weight_rows(monkeypatch):
    monkeypatch.setattr(profile_mod, "embed_texts", _fake_embed)

    base = Profile(bio="Short bio sentence about biology.", keywords=["CRISPR"])
    with_seed = Profile(
        bio="Short bio sentence about biology.",
        keywords=["CRISPR"],
        seed_works=[
            "A long representative paper abstract about base editing in vivo.",
        ],
    )

    base_mat = profile_mod.build_profile_matrix(base)
    seed_mat = profile_mod.build_profile_matrix(with_seed)

    # Exactly one extra anchor row for the single seed work.
    assert seed_mat.shape[0] == base_mat.shape[0] + 1
    # The seed row carries the high (2.0) weight → largest row norm.
    norms = np.linalg.norm(seed_mat, axis=1)
    assert float(norms.max()) >= float(np.linalg.norm(base_mat, axis=1).max())


def test_seed_works_keeps_short_titles_but_ignores_nonlinguistic_input(monkeypatch):
    encoded = []

    def embed(texts, is_query=False):
        encoded.extend(texts)
        return _fake_embed(texts, is_query=is_query)

    monkeypatch.setattr(profile_mod, "embed_texts", embed)
    base = Profile(bio="Short bio sentence about biology.", keywords=["CRISPR"])
    short_seed = Profile(
        bio="Short bio sentence about biology.",
        keywords=["CRISPR"],
        seed_works=["RNA design", "蛋白质从头设计", "", "  ", "...", "12345"],
    )
    base_rows = profile_mod.build_profile_matrix(base).shape[0]
    encoded.clear()
    assert (
        profile_mod.build_profile_matrix(short_seed).shape[0]
        == base_rows + 2
    )
    assert encoded[-2:] == ["RNA design", "蛋白质从头设计"]
