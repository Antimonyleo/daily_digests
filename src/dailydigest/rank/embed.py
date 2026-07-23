"""Local text embeddings with a singleton model load.

Two backends, both returning L2-normalized float32 ``[N, D]``:

- ``fastembed`` (default): ONNX runtime, no PyTorch — a small install (~400 MB
  vs ~5 GB for the CUDA torch stack) and fast on CPU. Its catalog covers the
  bge / e5 / MiniLM families, including the default ``bge-small-en-v1.5``.
- ``sentence-transformers`` (optional ``[hf]`` extra): any HuggingFace encoder.
  Used when ``EMBED_BACKEND=sentence-transformers`` is set, or automatically when
  the configured model is not in fastembed's catalog (e.g. SPECTER2, MedCPT).

The compute device and the asymmetric retrieval prefixes are configurable via
:class:`dailydigest.config.Settings`. Device defaults to CPU (the design goal:
run locally / in a CPU-only CI runner). fastembed's ONNX bge-small matches the
sentence-transformers output to within ~5e-4 cosine, so the two are ranking-
equivalent and their cached vectors interoperate.
"""

from __future__ import annotations

from threading import Lock

import numpy as np

# Backwards-compatible defaults, used when settings cannot be loaded (very early
# import) and as the historical baseline for A/B tests.
_DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE asymmetric retrieval: queries need this prefix; documents/passages do not.
_DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_ENCODER = None  # a callable: list[str] -> np.ndarray [N, D]
_ENCODER_LOCK = Lock()
_LOADED_KEY: tuple[str, str, str] | None = None  # (backend_pref, model, device)


def _embed_config() -> tuple[str, str, str, str, str]:
    """Return ``(model, device, query_prefix, doc_prefix, backend_pref)``."""
    try:
        from ..config import get_settings

        s = get_settings()
        model = (getattr(s, "embed_model", "") or _DEFAULT_MODEL_NAME).strip()
        device = (getattr(s, "embed_device", "") or "cpu").strip() or "cpu"
        query_prefix = getattr(s, "embed_query_prefix", _DEFAULT_QUERY_PREFIX)
        doc_prefix = getattr(s, "embed_doc_prefix", "") or ""
        backend = (getattr(s, "embed_backend", "") or "").strip().lower()
        return model, device, query_prefix, doc_prefix, backend
    except Exception:
        return _DEFAULT_MODEL_NAME, "cpu", _DEFAULT_QUERY_PREFIX, "", ""


def active_model_name() -> str:
    """Return the configured embedding model name (used as the cache key)."""
    return _embed_config()[0]


def _backend_pref() -> str:
    """Which backend to prefer: ``fastembed``, ``st``, or ``auto`` (fastembed
    first, falling back to sentence-transformers)."""
    backend = _embed_config()[4]
    if backend in ("fastembed", "onnx"):
        return "fastembed"
    if backend in ("sentence-transformers", "sentence_transformers", "st", "hf", "torch"):
        return "st"
    return "auto"


def _load_fastembed(model_name: str):
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name)

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(list(model.embed(texts)), dtype=np.float32)

    return encode


def _load_sentence_transformers(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    try:
        # Prefer offline load when the HF cache is populated (avoids ~25 HEAD
        # requests per CLI invocation).
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception:
        model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    return encode


def _build_encoder(pref: str, model_name: str, device: str):
    """Build an encoder callable, trying backends in preference order."""
    if pref == "fastembed":
        order = ["fastembed"]
    elif pref == "st":
        order = ["st"]
    else:  # auto: fastembed first (no torch), then sentence-transformers
        order = ["fastembed", "st"]

    errors: list[str] = []
    for backend in order:
        try:
            if backend == "fastembed":
                return _load_fastembed(model_name)
            return _load_sentence_transformers(model_name, device)
        except Exception as exc:  # noqa: BLE001 — try the next backend
            errors.append(f"{backend}: {exc}")
    raise RuntimeError(
        f"no embedding backend could load {model_name!r} "
        f"(install the 'hf' extra for arbitrary HF models). Tried: {'; '.join(errors)}"
    )


def _get_encoder():
    global _ENCODER, _LOADED_KEY
    model_name, device, _qp, _dp, _b = _embed_config()
    key = (_backend_pref(), model_name, device)
    if _ENCODER is not None and _LOADED_KEY == key:
        return _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is not None and _LOADED_KEY == key:
            return _ENCODER
        _ENCODER = _build_encoder(key[0], model_name, device)
        _LOADED_KEY = key
        return _ENCODER


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Return L2-normalized embeddings of shape ``[N, D]``.

    Empty input yields an empty array of shape ``[0, 0]``. Pass ``is_query=True``
    for profile / search vectors — the configured query prefix (BGE's retrieval
    instruction by default) is prepended, which measurably improves asymmetric
    retrieval. Documents (item title+abstract) get the configured document
    prefix, empty by default. Normalization is applied uniformly here so both
    backends satisfy the same unit-vector (cosine-ready) contract.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    _model, _device, query_prefix, doc_prefix, _b = _embed_config()
    prefix = query_prefix if is_query else doc_prefix
    if prefix:
        texts = [prefix + t for t in texts]
    vecs = np.asarray(_get_encoder()(texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vecs / norms).astype(np.float32, copy=False)
