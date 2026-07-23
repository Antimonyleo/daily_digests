"""Tests for configurable embedding model / device / prefixes."""

from __future__ import annotations

import numpy as np

from dailydigest import config as config_mod
from dailydigest.rank import embed as embed_mod


def _reset_settings(monkeypatch, **overrides):
    base = config_mod.load_settings().model_copy(update=overrides)
    monkeypatch.setattr(config_mod, "get_settings", lambda: base)
    return base


def test_active_model_name_follows_settings(monkeypatch):
    _reset_settings(monkeypatch, embed_model="allenai/specter2_base")
    assert embed_mod.active_model_name() == "allenai/specter2_base"


def test_default_device_is_cpu(monkeypatch):
    _reset_settings(monkeypatch)
    _model, device, _qp, _dp, _backend = embed_mod._embed_config()
    assert device == "cpu"


def test_query_prefix_applied_only_for_queries(monkeypatch):
    _reset_settings(
        monkeypatch,
        embed_query_prefix="QRY: ",
        embed_doc_prefix="DOC: ",
    )

    captured: list[list[str]] = []

    def _fake_encoder(texts):
        captured.append(list(texts))
        return np.ones((len(texts), 3), dtype=np.float32)

    # _get_encoder returns a callable: list[str] -> np.ndarray.
    monkeypatch.setattr(embed_mod, "_get_encoder", lambda: _fake_encoder)

    embed_mod.embed_texts(["alpha"], is_query=True)
    embed_mod.embed_texts(["beta"], is_query=False)

    assert captured[0] == ["QRY: alpha"]
    assert captured[1] == ["DOC: beta"]


def test_empty_input_returns_empty(monkeypatch):
    _reset_settings(monkeypatch)
    out = embed_mod.embed_texts([])
    assert out.shape == (0, 0)
