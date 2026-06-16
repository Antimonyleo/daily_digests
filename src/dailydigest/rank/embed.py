"""Sentence-transformer embeddings with a singleton model load.

The embedding model, the compute device, and the asymmetric retrieval prefixes
are all configurable via :class:`dailydigest.config.Settings` so a deployment can
swap ``bge-small`` for a stronger scientific encoder (SPECTER2, MedCPT,
``bge-large``, E5, …) without code changes. The item embedding cache keys on
:func:`active_model_name`, so changing the model transparently re-embeds.

The device defaults to CPU. That matches the design goal (run locally / in a
CPU-only GitHub Actions runner) and avoids crashes on machines whose GPU is not
supported by the installed PyTorch build.
"""

from __future__ import annotations

from threading import Lock

import numpy as np

# Backwards-compatible defaults. These are the values used when settings cannot
# be loaded (e.g. very early import) and the historical baseline for A/B tests.
_DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE asymmetric retrieval: queries need this prefix; documents/passages do not.
_DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Legacy module constant kept for any external references; the live value is
# resolved through ``active_model_name()``.
_MODEL_NAME = _DEFAULT_MODEL_NAME
_QUERY_PREFIX = _DEFAULT_QUERY_PREFIX

_MODEL = None
_MODEL_LOCK = Lock()
_LOADED_KEY: tuple[str, str] | None = None


def _embed_config() -> tuple[str, str, str, str]:
    """Return ``(model_name, device, query_prefix, doc_prefix)`` from settings."""
    try:
        from ..config import get_settings

        s = get_settings()
        model = (getattr(s, "embed_model", "") or _DEFAULT_MODEL_NAME).strip()
        device = (getattr(s, "embed_device", "") or "cpu").strip() or "cpu"
        query_prefix = getattr(s, "embed_query_prefix", _DEFAULT_QUERY_PREFIX)
        doc_prefix = getattr(s, "embed_doc_prefix", "") or ""
        return model, device, query_prefix, doc_prefix
    except Exception:
        return _DEFAULT_MODEL_NAME, "cpu", _DEFAULT_QUERY_PREFIX, ""


def active_model_name() -> str:
    """Return the configured embedding model name (used as the cache key)."""
    return _embed_config()[0]


def _construct(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    try:
        # Prefer offline load when the HF cache is already populated. This
        # avoids ~25 HEAD requests per CLI invocation.
        return SentenceTransformer(model_name, device=device, local_files_only=True)
    except TypeError:
        # Older sentence-transformers without local_files_only kwarg: toggle the
        # env var and retry, falling back to network mode.
        import os

        prev = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return SentenceTransformer(model_name, device=device)
        except Exception:
            if prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev
            return SentenceTransformer(model_name, device=device)
    except Exception:
        # Cache miss / first run: pull from network.
        return SentenceTransformer(model_name, device=device)


def _get_model():
    global _MODEL, _LOADED_KEY
    model_name, device, _qp, _dp = _embed_config()
    key = (model_name, device)
    if _MODEL is not None and _LOADED_KEY == key:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None and _LOADED_KEY == key:
            return _MODEL
        _MODEL = _construct(model_name, device)
        _LOADED_KEY = key
        return _MODEL


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Return L2-normalized embeddings of shape [N, D].

    Empty input yields an empty array of shape [0, 0].
    Pass ``is_query=True`` for profile / search vectors — the configured query
    prefix (BGE's retrieval instruction by default) is prepended, which
    measurably improves asymmetric retrieval quality. Documents (item
    title+abstract) get the configured document prefix, empty by default.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    _model_name, _device, query_prefix, doc_prefix = _embed_config()
    prefix = query_prefix if is_query else doc_prefix
    if prefix:
        texts = [prefix + t for t in texts]
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32, copy=False)
