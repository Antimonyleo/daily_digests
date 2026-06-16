"""Optional cross-encoder reranking of the top candidates.

A bi-encoder (cosine to the profile) is a cheap first-stage retriever; a
cross-encoder that jointly reads (profile, paper) is far more accurate but too
expensive to run over everything. The standard pattern is to rerank only the
top-N. This module is off by default and degrades gracefully to a no-op when the
model is unavailable (offline, not installed), so the daily run never breaks.

To preserve compatibility with the downstream per-section selection — which uses
score *magnitude* for diversity thresholds — the reranker only reorders the
head; the reordered items are mapped back into the head's original score band so
they stay above the un-reranked tail and keep meaningful magnitudes.
"""

from __future__ import annotations

import logging
from threading import Lock

import numpy as np

from ..store import ItemRow
from .embedding_cache import item_text

logger = logging.getLogger(__name__)

_MODEL = None
_LOADED_NAME: str | None = None
_UNAVAILABLE = False
_LOCK = Lock()


def _profile_query(profile: object) -> str:
    parts: list[str] = []
    bio = (getattr(profile, "bio", "") or "").strip()
    if bio:
        parts.append(bio)
    keywords = getattr(profile, "keywords", None) or []
    if keywords:
        parts.append(", ".join(str(k) for k in keywords[:30]))
    return " ".join(parts).strip()[:2000]


def _get_model(name: str, device: str):
    global _MODEL, _LOADED_NAME, _UNAVAILABLE
    if _UNAVAILABLE:
        return None
    if _MODEL is not None and _LOADED_NAME == name:
        return _MODEL
    with _LOCK:
        if _MODEL is not None and _LOADED_NAME == name:
            return _MODEL
        try:
            from sentence_transformers import CrossEncoder

            try:
                _MODEL = CrossEncoder(name, device=device, local_files_only=True)
            except TypeError:
                _MODEL = CrossEncoder(name, device=device)
            _LOADED_NAME = name
            return _MODEL
        except Exception as e:  # noqa: BLE001
            logger.warning("reranker '%s' unavailable (%s); skipping rerank", name, e)
            _UNAVAILABLE = True
            return None


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        return (x - lo) / (hi - lo + 1e-6)
    return np.full_like(x, 0.5)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rerank_scored(
    profile: object,
    scored: list[tuple[ItemRow, float]],
    *,
    settings: object | None = None,
) -> list[tuple[ItemRow, float]]:
    """Rerank the top candidates with a cross-encoder when enabled.

    Returns ``scored`` unchanged when disabled, when no model is available, or on
    any prediction error.
    """
    if not scored:
        return scored
    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    if not getattr(settings, "rerank_enabled", False):
        return scored

    top_n = max(1, int(getattr(settings, "rerank_top_n", 60)))
    weight = float(getattr(settings, "rerank_weight", 1.0))
    model = _get_model(
        str(getattr(settings, "rerank_model", "") or ""),
        str(getattr(settings, "embed_device", "cpu") or "cpu"),
    )
    if model is None:
        return scored
    query = _profile_query(profile)
    if not query:
        return scored

    head = scored[:top_n]
    tail = scored[top_n:]
    pairs = [[query, item_text(row)] for row, _ in head]
    try:
        raw = np.asarray(model.predict(pairs), dtype=np.float32).reshape(-1)
    except Exception as e:  # noqa: BLE001
        logger.warning("rerank predict failed (%s); keeping original order", e)
        return scored
    if raw.shape[0] != len(head):
        logger.warning("rerank returned %d scores for %d items; skipping", raw.shape[0], len(head))
        return scored

    # Cross-encoders often emit logits; squash to [0,1] when out of range.
    ce = _sigmoid(raw) if (raw.min() < 0.0 or raw.max() > 1.0) else raw
    orig = np.asarray([s for _, s in head], dtype=np.float32)
    blended = weight * ce + (1.0 - weight) * _normalize(orig)

    # Map the reordered head back into its original score band so it stays above
    # the tail and keeps magnitudes the section picker can threshold on.
    band_lo, band_hi = float(orig.min()), float(orig.max())
    mapped = band_lo + _normalize(blended) * (band_hi - band_lo)

    new_head = sorted(
        ((head[i][0], float(mapped[i])) for i in range(len(head))),
        key=lambda t: t[1],
        reverse=True,
    )
    return new_head + tail
