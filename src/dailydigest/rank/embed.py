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

import ctypes
import gc
import hashlib
import importlib.util
import os
import sys
from threading import Lock

import numpy as np

# Backwards-compatible defaults, used when settings cannot be loaded (very early
# import) and as the historical baseline for A/B tests.
_DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE asymmetric retrieval: queries need this prefix; documents/passages do not.
_DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_ENCODER = None  # a callable: list[str] -> np.ndarray [N, D]
_ENCODER_LOCK = Lock()
_LOADED_KEY: tuple[str, str, str, int, int] | None = None


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


def _resolved_backend_name() -> str:
    preference = _backend_pref()
    if preference != "auto":
        return preference
    if importlib.util.find_spec("fastembed") is not None:
        return "fastembed"
    return "sentence_transformers"


def active_embedding_signature(*, is_query: bool = False) -> str:
    """Stable cache key for vector semantics, not just the model name."""
    model_name, _device, query_prefix, document_prefix, _backend = _embed_config()
    prefix = query_prefix if is_query else document_prefix
    payload = "\0".join(
        ("embedding-v2", _resolved_backend_name(), model_name, prefix)
    )
    return f"{model_name}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _backend_pref() -> str:
    """Which backend to prefer: ``fastembed``, ``st``, or ``auto`` (fastembed
    first, falling back to sentence-transformers)."""
    backend = _embed_config()[4]
    if backend in ("fastembed", "onnx"):
        return "fastembed"
    if backend in ("sentence-transformers", "sentence_transformers", "st", "hf", "torch"):
        return "st"
    return "auto"


def _runtime_limits() -> tuple[int, int]:
    """Return the bounded FastEmbed batch and thread counts."""
    try:
        from ..config import get_settings

        settings = get_settings()
        batch_size = int(getattr(settings, "embed_batch_size", 8))
        requested_threads = int(getattr(settings, "embed_threads", 4))
    except Exception:
        batch_size, requested_threads = 8, 4
    available_cpus = os.cpu_count() or requested_threads
    return (
        max(1, min(batch_size, 256)),
        max(1, min(requested_threads, available_cpus, 64)),
    )


def _load_fastembed(
    model_name: str, *, device: str, batch_size: int, threads: int
):
    from fastembed import TextEmbedding

    model = TextEmbedding(
        model_name=model_name,
        threads=threads,
        cuda=device.strip().lower() in {"cuda", "gpu"},
        # ONNX's CPU arena retains its largest workspaces for the entire server
        # lifetime. DailyDigest embeds occasionally, so returning that memory is
        # more important than shaving a little allocator overhead off later runs.
        enable_cpu_mem_arena=False,
    )

    def encode(texts: list[str]) -> np.ndarray:
        # FastEmbed pads a batch toward its longest text. Grouping similarly
        # sized abstracts reduces both padding work and peak memory; restore the
        # caller's order before returning so every stored vector stays aligned.
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        sorted_texts = [texts[index] for index in order]
        sorted_vecs = np.asarray(
            list(model.embed(sorted_texts, batch_size=batch_size)),
            dtype=np.float32,
        )
        vecs = np.empty_like(sorted_vecs)
        vecs[order] = sorted_vecs
        return vecs

    return encode


def _load_sentence_transformers(model_name: str, device: str, *, batch_size: int):
    from sentence_transformers import SentenceTransformer

    try:
        # Prefer offline load when the HF cache is populated (avoids ~25 HEAD
        # requests per CLI invocation).
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception:
        model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        sorted_texts = [texts[index] for index in order]
        sorted_vecs = np.asarray(
            model.encode(
                sorted_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        vecs = np.empty_like(sorted_vecs)
        vecs[order] = sorted_vecs
        return vecs

    return encode


def _build_encoder(
    pref: str, model_name: str, device: str, batch_size: int, threads: int
):
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
                return _load_fastembed(
                    model_name,
                    device=device,
                    batch_size=batch_size,
                    threads=threads,
                )
            return _load_sentence_transformers(model_name, device, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001 — try the next backend
            errors.append(f"{backend}: {exc}")
    raise RuntimeError(
        f"no embedding backend could load {model_name!r} "
        f"(install the 'hf' extra for arbitrary HF models). Tried: {'; '.join(errors)}"
    )


def _get_encoder():
    global _ENCODER, _LOADED_KEY
    model_name, device, _qp, _dp, _b = _embed_config()
    batch_size, threads = _runtime_limits()
    key = (_backend_pref(), model_name, device, batch_size, threads)
    if _ENCODER is not None and _LOADED_KEY == key:
        return _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is not None and _LOADED_KEY == key:
            return _ENCODER
        _ENCODER = _build_encoder(
            key[0], model_name, device, batch_size, threads
        )
        _LOADED_KEY = key
        return _ENCODER


def _trim_process_memory() -> None:
    """Best-effort release of free glibc pages after an encoder is unloaded."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        # Non-glibc Linux runtimes safely fall back to normal allocator behavior.
        return


def release_encoder() -> None:
    """Drop the loaded model after a long-lived web compute job."""
    global _ENCODER, _LOADED_KEY
    with _ENCODER_LOCK:
        _ENCODER = None
        _LOADED_KEY = None
    gc.collect()
    # Do not import PyTorch into the lightweight FastEmbed process just for
    # cleanup. If the optional sentence-transformers backend loaded it, release
    # cached CUDA blocks after dropping the model so GPU memory also returns.
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (AttributeError, RuntimeError):
        pass
    _trim_process_memory()


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
