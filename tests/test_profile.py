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
