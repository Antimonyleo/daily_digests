"""Build profile vectors from a Profile."""

from __future__ import annotations

import logging
import re

import numpy as np

from ..models import Profile
from .embed import embed_texts

logger = logging.getLogger(__name__)

_SENT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def _profile_parts(profile: Profile) -> list[str]:
    """Collect bio sentences and keywords as separate embedding inputs."""
    parts: list[str] = []
    if profile.bio.strip():
        sentences = _SENT_RE.split(profile.bio.strip())
        parts.extend(s.strip() for s in sentences if s.strip() and len(s.strip()) > 10)
    parts.extend(k for k in profile.keywords if k and k.strip())
    return parts


def build_profile_matrix(profile: Profile) -> np.ndarray:
    """Return [N, D] matrix: one L2-normalized query embedding per bio sentence / keyword.

    Enables OR-style cosine scoring (top-k-mean) so a user with diverse interests
    (e.g. drug discovery AND climate science) matches either topic well instead of
    getting a blended centroid that misses both.
    """
    parts = _profile_parts(profile)
    if not parts:
        logger.warning(
            "Profile has no bio sentences or keywords; "
            "ranking will use prestige-only. "
            "Add bio text or keywords to your profile.yaml."
        )
        return np.zeros((1, 384), dtype=np.float32)
    return embed_texts(parts, is_query=True)  # [N, 384], L2-normalized


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
