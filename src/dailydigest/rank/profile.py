"""Build profile vectors from a Profile."""

from __future__ import annotations

import logging
import re

import numpy as np

from ..models import Profile
from .embed import embed_texts

logger = logging.getLogger(__name__)

_SENT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def _merged_interest_weights(profile: Profile) -> dict[str, float]:
    weights: dict[str, float] = {}
    for source in (profile.facet_weights, profile.interest_weights):
        for key, value in (source or {}).items():
            text = str(key).strip()
            if not text:
                continue
            try:
                weights[text] = float(value)
            except (TypeError, ValueError):
                logger.warning("Ignoring non-numeric profile interest weight for %r", text)
    return weights


def _clip_weight(value: float) -> float:
    # Keep user weights useful without letting one typo dominate the embedding
    # matrix or flip signs.
    return max(0.05, min(3.0, float(value)))


def _profile_parts_with_weights(profile: Profile) -> tuple[list[str], list[float]]:
    """Collect bio sentences, keywords, and weighted facets as query inputs."""
    parts: list[str] = []
    weights: list[float] = []
    interest_weights = _merged_interest_weights(profile)
    interest_lc = {key.lower(): _clip_weight(value) for key, value in interest_weights.items()}

    if profile.bio.strip():
        sentences = _SENT_RE.split(profile.bio.strip())
        for sentence in sentences:
            text = sentence.strip()
            if text and len(text) > 10:
                parts.append(text)
                weights.append(1.5)  # bio sentences carry more context than individual keywords

    seen = {p.lower() for p in parts}
    for keyword in profile.keywords:
        text = keyword.strip() if keyword else ""
        if not text:
            continue
        parts.append(text)
        weights.append(interest_lc.get(text.lower(), 1.0))
        seen.add(text.lower())

    for text, weight in interest_weights.items():
        if text.lower() in seen:
            continue
        parts.append(text)
        weights.append(_clip_weight(weight))
    return parts, weights


def _profile_parts(profile: Profile) -> list[str]:
    """Collect bio sentences and keywords as separate embedding inputs."""
    parts, _weights = _profile_parts_with_weights(profile)
    return parts


def build_profile_matrix(profile: Profile) -> np.ndarray:
    """Return [N, D] matrix: one L2-normalized query embedding per bio sentence / keyword.

    Enables OR-style cosine scoring (top-k-mean) so a user with diverse interests
    (e.g. drug discovery AND climate science) matches either topic well instead of
    getting a blended centroid that misses both.
    """
    parts, weights = _profile_parts_with_weights(profile)
    if not parts:
        logger.warning(
            "Profile has no bio sentences or keywords; "
            "ranking will use prestige-only. "
            "Add bio text or keywords to your profile.yaml."
        )
        return np.zeros((1, 384), dtype=np.float32)
    vecs = embed_texts(parts, is_query=True)  # [N, 384], L2-normalized
    return (vecs * np.asarray(weights, dtype=np.float32).reshape(-1, 1)).astype(
        np.float32,
        copy=False,
    )


def build_negative_centroid(profile: "Profile") -> np.ndarray | None:
    """Return a normalized negative-interest centroid, or None if not configured.

    Items whose embeddings are similar to this centroid will receive a penalty
    proportional to the similarity magnitude.
    """
    neg_interests = getattr(profile, "negative_interests", None) or {}
    neg_parts = [text.strip() for text in neg_interests.keys() if text.strip()]
    if not neg_parts:
        return None
    try:
        vecs = embed_texts(neg_parts, is_query=True)  # [M, 384]
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-6:
            return None
        return (centroid / norm).astype(np.float32)
    except Exception:
        logger.warning("build_negative_centroid: embedding failed")
        return None


def build_profile_vector(profile: Profile) -> np.ndarray:
    """Mean of L2-normalized embeddings of [bio, *keywords], re-normalized.

    Kept for backward compatibility; new pipeline code uses build_profile_matrix.
    """
    parts = _profile_parts(profile)
    if not parts:
        logger.warning(
            "Profile has no bio sentences or keywords; "
            "ranking will use prestige-only. "
            "Add bio text or keywords to your profile.yaml."
        )
        return np.zeros(384, dtype=np.float32)
    vecs = embed_texts(parts, is_query=True)  # already L2-normalized, shape [N, D]
    mean = vecs.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean.astype(np.float32, copy=False)


def build_profile_matrix_with_rocchio(profile: "Profile", vote_count: int = 0) -> np.ndarray:
    """Build profile matrix, blending in the Rocchio-learned vector when available.

    After a user has voted on items, we accumulate a learned direction vector
    (updated per vote in votes.py). This function blends it into the profile
    matrix as an additional high-weight row so it naturally influences the
    top-k cosine scoring.
    """
    static_mat = build_profile_matrix(profile)
    if vote_count <= 0:
        return static_mat
    try:
        from ..config import get_settings
        from pathlib import Path
        learned_path = Path(get_settings().db_path).parent / "learned_profile.npz"
        if not learned_path.exists():
            return static_mat
        data = np.load(learned_path)
        learned = data["profile"].astype(np.float32)
        norm = np.linalg.norm(learned)
        if norm < 1e-6:
            return static_mat
        learned_normalized = learned / norm
        # gamma ramps from 0 → 0.45 as votes go from 0 → 30
        gamma = min(0.45, vote_count / 67.0)
        if gamma < 0.02:
            return static_mat
        # Add learned vector as an extra row scaled to represent gamma fraction
        # of the total profile mass (equivalent rows = gamma * n_rows / (1-gamma))
        n = static_mat.shape[0]
        learned_weight = gamma * n / max(1.0 - gamma, 0.01)
        learned_row = (learned_normalized * learned_weight).reshape(1, -1).astype(np.float32)
        return np.vstack([static_mat, learned_row])
    except Exception as e:
        logger.warning("build_profile_matrix_with_rocchio: failed: %s", e)
        return static_mat
