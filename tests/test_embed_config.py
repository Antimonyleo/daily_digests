"""Tests for configurable embedding model / device / prefixes."""

from __future__ import annotations

import sys
from types import SimpleNamespace

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


def test_embedding_signature_tracks_document_and_query_prefixes(monkeypatch):
    _reset_settings(
        monkeypatch,
        embed_model="test-model",
        embed_query_prefix="query-a: ",
        embed_doc_prefix="doc-a: ",
        embed_backend="fastembed",
    )
    document_a = embed_mod.active_embedding_signature()
    query_a = embed_mod.active_embedding_signature(is_query=True)

    _reset_settings(
        monkeypatch,
        embed_model="test-model",
        embed_query_prefix="query-b: ",
        embed_doc_prefix="doc-b: ",
        embed_backend="fastembed",
    )

    assert embed_mod.active_embedding_signature() != document_a
    assert embed_mod.active_embedding_signature(is_query=True) != query_a


def test_default_device_is_cpu(monkeypatch):
    _reset_settings(monkeypatch)
    _model, device, _qp, _dp, _backend = embed_mod._embed_config()
    assert device == "cpu"


def test_embedding_runtime_defaults_are_bounded():
    settings = config_mod.Settings(_env_file=None)

    assert settings.embed_batch_size == 8
    assert settings.embed_threads == 4


def test_fastembed_uses_bounded_cpu_runtime_and_restores_input_order(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def embed(self, texts, *, batch_size):
            captured["texts"] = list(texts)
            captured["batch_size"] = batch_size
            return (np.array([len(text)], dtype=np.float32) for text in texts)

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )

    encode = embed_mod._load_fastembed(
        "test-model", device="cpu", batch_size=8, threads=4
    )
    out = encode(["longest text", "x", "medium"])

    assert captured["init"] == {
        "model_name": "test-model",
        "threads": 4,
        "cuda": False,
        "enable_cpu_mem_arena": False,
    }
    assert captured["batch_size"] == 8
    assert captured["texts"] == ["x", "medium", "longest text"]
    assert out[:, 0].tolist() == [12.0, 1.0, 6.0]


def test_fastembed_honors_explicit_cuda_device(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def embed(self, texts, *, batch_size):
            del batch_size
            return (np.ones(2, dtype=np.float32) for _ in texts)

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )

    embed_mod._load_fastembed("test-model", device="cuda", batch_size=8, threads=4)

    assert captured["cuda"] is True


def test_sentence_transformers_uses_bounded_batch_and_restores_order(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["init"] = kwargs

        def encode(self, texts, **kwargs):
            captured["texts"] = list(texts)
            captured["encode"] = kwargs
            return np.asarray([[len(text)] for text in texts], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    encode = embed_mod._load_sentence_transformers(
        "test-model", "cpu", batch_size=8
    )
    out = encode(["longest text", "x", "medium"])

    assert captured["texts"] == ["x", "medium", "longest text"]
    assert captured["encode"] == {
        "batch_size": 8,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    assert out[:, 0].tolist() == [12.0, 1.0, 6.0]


def test_release_encoder_clears_singleton_and_reclaims_process_memory(monkeypatch):
    collected: list[str] = []
    embed_mod._ENCODER = object()
    embed_mod._LOADED_KEY = ("fastembed", "model", "cpu", 8, 4)
    monkeypatch.setattr(embed_mod.gc, "collect", lambda: collected.append("gc"))
    monkeypatch.setattr(
        embed_mod, "_trim_process_memory", lambda: collected.append("trim")
    )

    embed_mod.release_encoder()

    assert embed_mod._ENCODER is None
    assert embed_mod._LOADED_KEY is None
    assert collected == ["gc", "trim"]


def test_release_encoder_clears_loaded_torch_cuda_cache(monkeypatch):
    calls: list[str] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append("empty_cache"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(embed_mod.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(embed_mod, "_trim_process_memory", lambda: calls.append("trim"))

    embed_mod.release_encoder()

    assert calls == ["gc", "empty_cache", "trim"]


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
