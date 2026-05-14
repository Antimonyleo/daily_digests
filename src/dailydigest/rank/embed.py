"""Sentence-transformer embeddings with a singleton model load."""

from __future__ import annotations

from threading import Lock

import numpy as np

_MODEL = None
_MODEL_LOCK = Lock()
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE asymmetric retrieval: queries need this prefix; documents/passages do not.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer

                try:
                    # Prefer offline load when the HF cache is already populated.
                    # This avoids ~25 HEAD requests per CLI invocation.
                    _MODEL = SentenceTransformer(_MODEL_NAME, local_files_only=True)
                except TypeError:
                    # Older sentence-transformers without local_files_only kwarg:
                    # toggle the env var and retry, falling back to network mode.
                    import os

                    prev = os.environ.get("HF_HUB_OFFLINE")
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    try:
                        _MODEL = SentenceTransformer(_MODEL_NAME)
                    except Exception:
                        if prev is None:
                            os.environ.pop("HF_HUB_OFFLINE", None)
                        else:
                            os.environ["HF_HUB_OFFLINE"] = prev
                        _MODEL = SentenceTransformer(_MODEL_NAME)
                except Exception:
                    # Cache miss / first run: pull from network.
                    _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Return L2-normalized embeddings of shape [N, D].

    Empty input yields an empty array of shape [0, 0].
    Pass ``is_query=True`` for profile / search vectors — BGE prepends a
    retrieval instruction that measurably improves asymmetric retrieval quality.
    Documents (item title+abstract) do not need the prefix.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if is_query:
        texts = [_QUERY_PREFIX + t for t in texts]
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32, copy=False)
