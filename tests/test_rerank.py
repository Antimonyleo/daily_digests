"""Tests for the optional cross-encoder reranker."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from dailydigest import config as config_mod
from dailydigest.models import Profile
from dailydigest.rank import rerank as rerank_mod
from dailydigest.store import ItemRow


def _item(title: str) -> ItemRow:
    return ItemRow(
        id=abs(hash(title)) % 100000,
        source="Test",
        section="research",
        external_id=title,
        url=f"https://example.com/{title}",
        title=title,
        abstract="abstract",
        published_at=datetime.now(timezone.utc),
    )


def _scored():
    return [
        (_item("A"), 0.90),
        (_item("B"), 0.80),
        (_item("C"), 0.70),
    ]


def _settings(**overrides):
    return config_mod.load_settings().model_copy(update=overrides)


def test_disabled_is_noop():
    scored = _scored()
    out = rerank_mod.rerank_scored(
        Profile(bio="x", keywords=["y"]), scored, settings=_settings(rerank_enabled=False)
    )
    assert out == scored


def test_rerank_reorders_within_band(monkeypatch):
    # Fake cross-encoder that prefers the reverse of the input order.
    class _FakeCE:
        def predict(self, pairs):
            # Highest score to the last pair, lowest to the first.
            return np.linspace(0.0, 1.0, num=len(pairs), dtype=np.float32)

    monkeypatch.setattr(rerank_mod, "_get_model", lambda name, device: _FakeCE())

    scored = _scored()
    out = rerank_mod.rerank_scored(
        Profile(bio="bio text here", keywords=["topic"]),
        scored,
        settings=_settings(rerank_enabled=True, rerank_weight=1.0, rerank_top_n=60),
    )

    titles = [row.title for row, _ in out]
    # Order reversed by the fake reranker: C, B, A
    assert titles == ["C", "B", "A"]
    # Scores stay within the original band [0.70, 0.90].
    scores = [s for _, s in out]
    assert max(scores) <= 0.9 + 1e-6
    assert min(scores) >= 0.7 - 1e-6


def test_unavailable_model_is_noop(monkeypatch):
    monkeypatch.setattr(rerank_mod, "_get_model", lambda name, device: None)
    scored = _scored()
    out = rerank_mod.rerank_scored(
        Profile(bio="x", keywords=["y"]), scored, settings=_settings(rerank_enabled=True)
    )
    assert out == scored
