"""Build profile vectors from a Profile."""

from __future__ import annotations

import logging
import re

import numpy as np

from ..models import Profile
from .embed import embed_texts

logger = logging.getLogger(__name__)

_SENT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")

# Down-weight applied to context ("keep-me-informed") interest facets relative to
# core research keywords (weight 1.0). At 0.45 a context-only cosine match of
# ~0.9 becomes ~0.40 — below the research relevance floor — so peripheral topics
# no longer take research slots, while still contributing weakly and driving
# retrieval + the news sections.
CONTEXT_FACET_WEIGHT = 0.45


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

    # Context ("keep-me-informed") interests are embedded as LOW-weight facets so
    # they contribute to relevance only weakly. _multi_cosine applies the row
    # weight (clamped to <=1) post-cosine, so a match on a context-only term can't
    # win the top-1/max the way a core-keyword match can. They still drive
    # retrieval (see profile_search_terms) and the news/industry sections.
    for keyword in getattr(profile, "context_keywords", None) or []:
        text = keyword.strip() if keyword else ""
        if not text or text.lower() in seen:
            continue
        parts.append(text)
        weights.append(interest_lc.get(text.lower(), CONTEXT_FACET_WEIGHT))
        seen.add(text.lower())

    for text, weight in interest_weights.items():
        if text.lower() in seen:
            continue
        parts.append(text)
        weights.append(_clip_weight(weight))

    # Seed works (the user's own / exemplar papers) are the strongest interest
    # signal: real title+abstract text rather than isolated keywords. Embed each
    # as a high-weight anchor row so top-k cosine matches the user's actual line
    # of work, not just topical vocabulary.
    for work in getattr(profile, "seed_works", None) or []:
        text = str(work).strip()
        if len(text) > 10:
            parts.append(text)
            weights.append(2.0)
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
    result = (vecs * np.asarray(weights, dtype=np.float32).reshape(-1, 1)).astype(
        np.float32,
        copy=False,
    )
    if np.allclose(result, 0):
        import logging
        logging.getLogger(__name__).error(
            "Profile matrix is all zeros — check that bio and keywords are non-empty. "
            "All items will score equally and ranking will be arbitrary."
        )
    return result


def query_aware_cosine(vecs: np.ndarray, profile_mat: np.ndarray) -> np.ndarray:
    """Attention-weighted cosine: weight profile rows by their similarity to each item.

    Instead of treating all profile rows equally, this weights each profile row by
    how similar the item is to that row. This means 'for a CRISPR paper, weight the
    CRISPR keyword row more.' Captures the NRMS/attention insight cheaply.

    Falls back to uniform weighting when profile_mat has 1 row.
    """
    if profile_mat.ndim == 1 or profile_mat.shape[0] == 1:
        # Single row: standard cosine
        flat = profile_mat.reshape(-1).astype(np.float32)
        norms_v = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms_p = float(np.linalg.norm(flat)) + 1e-9
        return ((vecs / (norms_v + 1e-9)) @ (flat / norms_p)).astype(np.float32)

    # Compute similarity matrix: (n_items, n_profile_rows)
    vecs_n = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    prof_n = profile_mat / (np.linalg.norm(profile_mat, axis=1, keepdims=True) + 1e-9)
    sims = (vecs_n @ prof_n.T).astype(np.float32)  # (n_items, n_rows)

    # Softmax attention weights per item over profile rows
    exp_sims = np.exp(sims * 3.0)  # temperature=3 sharpens attention
    attn = exp_sims / (exp_sims.sum(axis=1, keepdims=True) + 1e-9)

    # Weighted profile vector per item
    weighted_profiles = attn @ prof_n  # (n_items, embed_dim)
    # Final cosine between item and its attention-weighted profile
    cos = (vecs_n * weighted_profiles).sum(axis=1).astype(np.float32)
    return cos


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


def build_negative_vectors(profile: "Profile") -> list[np.ndarray]:
    """Return individual embedding vectors for each negative interest.

    Unlike build_negative_centroid (which averages), this returns per-topic vectors
    so the penalty can use max(cos(item, neg_i)) instead of cos(item, mean(neg)).
    """
    neg = getattr(profile, "negative_interests", None) or {}
    if not neg:
        return []
    texts = list(neg.keys())
    if not texts:
        return []
    try:
        vecs = embed_texts(texts, is_query=True)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalized = vecs / (norms + 1e-9)
        return [normalized[i] for i in range(len(normalized))]
    except Exception:
        return []


def get_negative_interest_weights(profile: "Profile") -> list[float]:
    """Return weights for each negative interest vector, in same order as build_negative_vectors."""
    neg = getattr(profile, "negative_interests", None) or {}
    return [float(v) for v in neg.values()]


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
        data = np.load(learned_path, allow_pickle=False)
        learned = data["profile"].astype(np.float32)
        norm = np.linalg.norm(learned)
        if norm < 1e-6:
            return static_mat
        learned_normalized = learned / norm
        # gamma ramps linearly from 0 to its 0.25 cap as votes go 0 → 15 (0.25*60)
        gamma = min(0.25, vote_count / 60.0)
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
