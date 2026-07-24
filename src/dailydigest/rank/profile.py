"""Build profile vectors from a Profile."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from ..models import Profile
from .embed import embed_texts

logger = logging.getLogger(__name__)

_SENT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")

# --- Facet attribution / topic-priority (P1 + P3) -------------------------- #
# A matched core interest must reach this cosine to be named the "primary" facet;
# below it the item has no clear facet (primary="").
_PRIMARY_FACET_MIN_SIM = 0.32
# Secondaries must be within this margin of the primary sim, and above this floor.
_SECONDARY_MARGIN = 0.06
_SECONDARY_MIN_SIM = 0.30
_MAX_SECONDARIES = 2
# Default normalized priority for keywords absent from profile.topic_priorities.
_DEFAULT_PRIORITY = 0.5


def _topic_priority_bonus_scale() -> float:
    """Read the topic-priority bonus scale from settings (safe fallback).

    Mirrors ``source_quality._research_quality_weight`` — a small nudge scale
    added to the FINAL ordering score only, never the relevance gate.
    """
    try:
        from ..config import get_settings

        return float(get_settings().topic_priority_bonus_scale)
    except Exception:  # noqa: BLE001
        return 0.06


# Convenience constant mirroring the settings default; the live value is read via
# _topic_priority_bonus_scale() so env overrides take effect.
TOPIC_PRIORITY_BONUS_SCALE = 0.06

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


def build_negative_centroid(profile: Profile) -> np.ndarray | None:
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


def build_negative_vectors(profile: Profile) -> list[np.ndarray]:
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


def get_negative_interest_weights(profile: Profile) -> list[float]:
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


# --------------------------------------------------------------------------- #
# Facet attribution (P1) + topic-priority axis (P3)
# --------------------------------------------------------------------------- #

# Cache the core-facet matrix keyed by the keyword tuple so we don't re-embed on
# every scoring call. Bounded to the most recent profile's keywords.
_CORE_FACET_CACHE: dict[tuple[str, ...], tuple[np.ndarray, list[str]]] = {}


def build_core_facet_matrix(profile: Profile) -> tuple[np.ndarray, list[str]]:
    """Embed each core interest (``profile.keywords``) as a unit query row.

    Returns ``(matrix [K, D], labels)`` where each row is L2-normalized and
    ``labels[i]`` is the originating keyword string. Cached by the keyword tuple
    so repeated calls (e.g. per-run scoring) don't re-embed.
    """
    labels = [str(k).strip() for k in (profile.keywords or []) if k and str(k).strip()]
    if not labels:
        return np.zeros((0, 0), dtype=np.float32), []
    key = tuple(labels)
    cached = _CORE_FACET_CACHE.get(key)
    if cached is not None:
        return cached
    vecs = embed_texts(labels, is_query=True)  # already L2-normalized [K, D]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    matrix = (vecs / np.clip(norms, 1e-9, None)).astype(np.float32, copy=False)
    result = (matrix, labels)
    _CORE_FACET_CACHE[key] = result
    return result


def _normalized_priorities(profile: Profile) -> dict[str, float]:
    """Normalize ``profile.topic_priorities`` so the top raw value maps to 1.0.

    Empty/degenerate input yields an empty dict (callers then fall back to the
    moderate default for every keyword, making the bonus a harmless constant).
    """
    raw: dict[str, float] = {}
    for key, value in (getattr(profile, "topic_priorities", None) or {}).items():
        text = str(key).strip()
        if not text:
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric topic_priority for %r", text)
            continue
        raw[text] = fval
    if not raw:
        return {}
    max_raw = max(raw.values())
    if max_raw <= 0:
        return {}
    return {label: val / max_raw for label, val in raw.items()}


@dataclass(frozen=True)
class AttributionContext:
    """Prebuilt inputs for facet attribution, constructed once per run."""

    matrix: np.ndarray  # [K, D], unit rows
    labels: list[str]
    priorities: dict[str, float]  # normalized 0..1 (top interest = 1.0)


@dataclass(frozen=True)
class ItemAttribution:
    """Per-item facet attribution result."""

    primary: str = ""
    secondaries: list[str] = field(default_factory=list)
    priority: float = 0.0
    priority_bonus: float = 0.0


def build_attribution_context(profile: Profile) -> AttributionContext | None:
    """Build an :class:`AttributionContext`, or ``None`` when no core keywords."""
    matrix, labels = build_core_facet_matrix(profile)
    if not labels or matrix.size == 0:
        return None
    return AttributionContext(
        matrix=matrix,
        labels=labels,
        priorities=_normalized_priorities(profile),
    )


def attribute_items(
    item_vecs: np.ndarray, ctx: AttributionContext
) -> list[ItemAttribution]:
    """Attribute each item vector to the core interest(s) it best matches.

    ``item_vecs`` is ``[N, D]`` (rows need not be normalized — cosine is computed
    against the unit facet rows). Returns one :class:`ItemAttribution` per row,
    aligned to the input order.
    """
    n = int(item_vecs.shape[0]) if item_vecs.ndim == 2 else 0
    if n == 0 or ctx is None or ctx.matrix.size == 0:
        return [ItemAttribution() for _ in range(max(0, n))]

    scale = _topic_priority_bonus_scale()
    vecs = np.asarray(item_vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.clip(norms, 1e-9, None)
    sims = (unit @ ctx.matrix.T.astype(np.float32)).astype(np.float32)  # [N, K]

    out: list[ItemAttribution] = []
    labels = ctx.labels
    for row in sims:
        primary_idx = int(np.argmax(row))
        primary_sim = float(row[primary_idx])
        if primary_sim < _PRIMARY_FACET_MIN_SIM:
            out.append(ItemAttribution())
            continue
        primary = labels[primary_idx]
        sec_floor = max(primary_sim - _SECONDARY_MARGIN, _SECONDARY_MIN_SIM)
        # Candidate secondaries: other facets at/above the floor, highest first.
        cand = [
            (float(row[i]), i)
            for i in range(len(labels))
            if i != primary_idx and float(row[i]) >= sec_floor
        ]
        cand.sort(key=lambda t: t[0], reverse=True)
        secondaries = [labels[i] for _sim, i in cand[:_MAX_SECONDARIES]]
        priority = float(ctx.priorities.get(primary, _DEFAULT_PRIORITY))
        priority_bonus = float(scale * priority)
        out.append(
            ItemAttribution(
                primary=primary,
                secondaries=secondaries,
                priority=priority,
                priority_bonus=priority_bonus,
            )
        )
    return out


def build_profile_matrix_with_rocchio(profile: Profile, vote_count: int = 0) -> np.ndarray:
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
        from pathlib import Path

        from ..config import get_settings
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
