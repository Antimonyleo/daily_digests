"""Build a single profile vector from a Profile."""

from __future__ import annotations

import numpy as np

from ..models import Profile
from .embed import embed_texts


def build_profile_vector(profile: Profile) -> np.ndarray:
    """Mean of L2-normalized embeddings of [bio, *keywords], re-normalized."""
    parts: list[str] = []
    if profile.bio.strip():
        parts.append(profile.bio.strip())
    parts.extend(k for k in profile.keywords if k and k.strip())
    if not parts:
        # zero vector with a sane default dimension; will yield 0 cosine.
        return np.zeros(384, dtype=np.float32)
    vecs = embed_texts(parts)  # already L2-normalized, shape [N, D]
    mean = vecs.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean.astype(np.float32, copy=False)
