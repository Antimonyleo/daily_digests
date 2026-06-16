"""Cosine + LR ranker with downweight penalties and per-section caps.

The public callable :func:`score_items` is the single entry point used by
the pipeline. Internally it routes between the cosine baseline and the
hybrid cosine+LR scorer when an :class:`LRRanker` and >=30 votes are
available.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import math
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from ..store import ItemRow
from .embedding_cache import embed_item_rows, item_text
from .source_quality import (
    RANKER_VERSION,
    is_arxiv_cs_source,
    is_high_quality_journal_source,
    is_low_impact_research,
    is_preprint_source,
    is_published_journal_source,
    quality_adjusted_score,
    source_bucket,
    should_skip_item,
)

logger = logging.getLogger(__name__)

DOWNWEIGHT_PENALTY = 0.20
HYBRID_COSINE_W = 0.5
HYBRID_LR_W = 0.5
# Reciprocal Rank Fusion constant. Larger = flatter weighting of rank position.
RRF_K = 60

ScoreFeatureMap = dict[int, dict[str, Any]]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        return (x - lo) / (hi - lo + 1e-6)
    return np.full_like(x, 0.5)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """Return 0-based ranks (0 = highest value), ties broken by original order."""
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return ranks


def _fuse_scores(qa: np.ndarray, lr_prob: np.ndarray, mode: str | None = None) -> np.ndarray:
    """Combine quality-adjusted topic scores with LR probability into a rank.

    ``rrf`` (default) fuses the two *rankings* via Reciprocal Rank Fusion, which
    is insensitive to score scale/outliers, then min-maxes the fused result back
    to [0, 1] so downstream magnitude thresholds (e.g. exceptional-preprint
    cutoff) keep working. ``minmax`` is the legacy per-batch normalized blend.
    """
    qa = np.asarray(qa, dtype=np.float32)
    lr_prob = np.asarray(lr_prob, dtype=np.float32)
    if mode is None:
        try:
            from ..config import get_settings
            mode = (get_settings().rank_fusion or "rrf").lower()
        except Exception:  # noqa: BLE001
            mode = "rrf"
    if mode == "minmax":
        return (HYBRID_COSINE_W * _minmax(qa) + HYBRID_LR_W * _minmax(lr_prob)).astype(
            np.float32
        )
    r_qa = _rank_desc(qa)
    r_lr = _rank_desc(lr_prob)
    fused = 1.0 / (RRF_K + r_qa + 1) + 1.0 / (RRF_K + r_lr + 1)
    return _minmax(fused.astype(np.float32)).astype(np.float32)


def _lr_weights_path() -> Path:
    from ..config import get_settings
    return Path(get_settings().db_path).parent / "lr_ranker.npz"


def _current_feature_schema() -> tuple[str, int]:
    from ..votes import LR_FEATURE_DIM, LR_FEATURE_SCHEMA_VERSION

    return LR_FEATURE_SCHEMA_VERSION, LR_FEATURE_DIM


def _npz_scalar(data: Any, key: str) -> Any:
    value = np.asarray(data[key])
    if value.size == 0:
        raise ValueError(f"{key} is empty")
    return value.reshape(-1)[0]


def _schema_version_from_npz(data: Any) -> str:
    version = _npz_scalar(data, "feature_schema_version")
    if isinstance(version, bytes):
        return version.decode("utf-8")
    return str(version)


def _item_text(row: ItemRow) -> str:
    return item_text(row)


# --------------------------------------------------------------------------- #
# LR ranker
# --------------------------------------------------------------------------- #


class LRRanker:
    """Logistic-regression scorer over title+abstract embeddings.

    Trained on votes (+1 / -1). ``score`` returns ``predict_proba(...)[:, 1]``
    (probability of an "up" vote).
    """

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self._classes: np.ndarray | None = None
        self._sk_model = None  # cached fitted sklearn model (when available)

    # ---- training -----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int32)
        # sklearn requires at least 2 classes.
        if np.unique(y).size < 2:
            raise ValueError("LRRanker.fit needs both +1 and -1 examples")

        feature_schema_version, feature_dim = _current_feature_schema()
        actual_dim = int(X.shape[1]) if X.ndim == 2 else -1
        if actual_dim != feature_dim:
            raise ValueError(
                "LRRanker.fit expected "
                f"{feature_dim} features for schema {feature_schema_version}, "
                f"got {actual_dim}"
            )

        C = 0.5 if len(X) < 50 else 1.0
        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
        model = clf
        model.fit(X, y)
        self._sk_model = model
        self._classes = model.classes_.astype(np.int32)
        self.coef_ = model.coef_.astype(np.float32, copy=False)
        self.intercept_ = float(model.intercept_[0])

        target = _lr_weights_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".npz.tmp")
        with tmp.open("wb") as f:
            np.savez(
                f,
                coef=self.coef_,
                intercept=np.asarray([self.intercept_], dtype=np.float32),
                classes=self._classes,
                feature_dim=np.asarray([feature_dim], dtype=np.int32),
                feature_schema_version=np.asarray([feature_schema_version]),
            )
        tmp.replace(target)
        logger.info("LRRanker: saved weights to %s", target)

    # ---- persistence --------------------------------------------------------

    def load(self) -> bool:
        path = _lr_weights_path()
        if not path.exists():
            return False
        try:
            feature_schema_version, feature_dim = _current_feature_schema()
            with np.load(path) as data:
                coef = data["coef"].astype(np.float32, copy=False)
                intercept = float(data["intercept"][0])
                classes = (
                    data["classes"].astype(np.int32)
                    if "classes" in data.files
                    else np.asarray([-1, 1], dtype=np.int32)
                )

                if "feature_schema_version" not in data.files:
                    logger.warning(
                        "LRRanker: stale weights missing feature_schema_version; "
                        "will retrain"
                    )
                    return False
                saved_schema_version = _schema_version_from_npz(data)
                if saved_schema_version != feature_schema_version:
                    logger.warning(
                        "LRRanker: stale weights (feature_schema_version=%r, "
                        "expected=%r); will retrain",
                        saved_schema_version,
                        feature_schema_version,
                    )
                    return False

                if "feature_dim" not in data.files:
                    logger.warning("LRRanker: stale weights missing feature_dim; will retrain")
                    return False
                saved_dim = int(_npz_scalar(data, "feature_dim"))
                actual_dim = int(coef.shape[1] if coef.ndim > 1 else coef.shape[0])
                if saved_dim != feature_dim or actual_dim != feature_dim:
                    logger.warning(
                        "LRRanker: stale weights (dim=%d, coef_dim=%d, expected=%d); "
                        "will retrain",
                        saved_dim,
                        actual_dim,
                        feature_dim,
                    )
                    return False

            self.coef_ = coef
            self.intercept_ = intercept
            self._classes = classes
            self._sk_model = None
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("LRRanker: failed to load weights: %s", e)
            return False

    # ---- inference ----------------------------------------------------------

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if self._sk_model is not None:
            probs = self._sk_model.predict_proba(embeddings)
            classes = self._sk_model.classes_
            pos_indices = np.where(classes == 1)[0]
            pos_idx = int(pos_indices[0]) if len(pos_indices) > 0 else 1
            return probs[:, pos_idx].astype(np.float32, copy=False)

        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LRRanker is not fitted; call fit() or load() first")

        # Manual sigmoid using the persisted weights. sklearn's binary LR
        # stores coef/intercept oriented for class_[1] (the positive class).
        z = embeddings.astype(np.float32, copy=False) @ self.coef_.reshape(-1)
        z = z + np.float32(self.intercept_)
        prob_pos = 1.0 / (1.0 + np.exp(-z))
        return prob_pos.astype(np.float32, copy=False)


# Module-level singleton + load cache. ``None`` = not yet attempted; ``False``
# = attempted and unavailable; otherwise the loaded ranker instance.
_LR_SINGLETON: LRRanker | None | bool = None
_LR_LOCK = Lock()


def get_lr_ranker() -> LRRanker | None:
    """Return a loaded LRRanker singleton, or ``None`` if weights missing."""
    global _LR_SINGLETON
    if _LR_SINGLETON is False:
        if not _lr_weights_path().exists():
            return None
        _LR_SINGLETON = None
    if isinstance(_LR_SINGLETON, LRRanker):
        return _LR_SINGLETON
    with _LR_LOCK:
        if _LR_SINGLETON is False:
            return None
        if isinstance(_LR_SINGLETON, LRRanker):
            return _LR_SINGLETON
        ranker = LRRanker()
        if ranker.load():
            _LR_SINGLETON = ranker
            return ranker
        _LR_SINGLETON = False
        return None


def reset_lr_cache() -> None:
    """Drop the cached LR ranker (used after retraining)."""
    global _LR_SINGLETON
    with _LR_LOCK:
        _LR_SINGLETON = None


def _vote_count() -> int:
    try:
        from ..votes import signed_vote_count
        return signed_vote_count()
    except Exception as e:  # noqa: BLE001
        logger.warning("ranker: failed to count votes: %s", e)
        return 0


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _multi_cosine(vecs: np.ndarray, profile_mat: np.ndarray, k: int | None = None) -> np.ndarray:
    """Score each item by 0.7*max_similarity + 0.3*top3_mean.

    profile_mat rows are weighted (weight × unit_vec), so raw dot products can
    exceed 1.0.  We divide by the maximum row norm so output stays in [-1, 1] —
    a perfect match to the highest-weight row gives exactly 1.0 instead of
    ``max_weight``.  Rewards specialist match over generalist match.
    """
    sims = vecs @ profile_mat.T.astype(np.float32)
    # Normalize so the maximum achievable score is 1.0
    row_norms = np.linalg.norm(profile_mat, axis=1)
    max_norm = float(row_norms.max()) if len(row_norms) > 0 else 1.0
    if max_norm > 1.0:
        sims = sims / max_norm
    n = sims.shape[1]
    if n <= 1:
        return sims.max(axis=1).astype(np.float32)
    top1 = sims.max(axis=1)
    k3 = min(3, n)
    top3 = np.sort(sims, axis=1)[:, -k3:].mean(axis=1)
    return (0.7 * top1 + 0.3 * top3).astype(np.float32)


def _cosine_sim(vecs: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Dispatch to multi-vector or single-vector cosine based on profile shape."""
    if profile.ndim == 1:
        return (vecs @ profile.astype(np.float32, copy=False)).astype(np.float32)
    return _multi_cosine(vecs, profile)


def _reason_penalty_for(row: ItemRow, reason_penalty_map: Mapping[Any, float] | None) -> float:
    """Return an optional per-item feedback penalty supplied by the caller.

    The map is intentionally plain data so qualitative reason-chip feedback can
    influence ranking without coupling this module to the feedback store.
    Primary keys are item ids; stringified ids are also accepted for JSON-loaded
    maps. ``external_id`` and ``url`` are fallback keys for tests or callers
    that rank rows before database ids exist.
    """
    if not reason_penalty_map:
        return 0.0

    candidates: list[Any] = []
    for attr in ("id", "external_id", "url"):
        value = getattr(row, attr, None)
        if isinstance(value, (int, str)) and value not in candidates:
            candidates.append(value)
            if isinstance(value, int):
                candidates.append(str(value))

    for key in candidates:
        if key not in reason_penalty_map:
            continue
        try:
            return max(0.0, min(0.30, float(reason_penalty_map[key])))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _row_feature_key(row: ItemRow) -> int:
    row_id = getattr(row, "id", None)
    return int(row_id) if isinstance(row_id, int) else id(row)


def _downweight_hit(row: ItemRow, downweight_terms: list[str]) -> bool:
    txt = _item_text(row).lower()
    return any(term.lower() in txt for term in downweight_terms if term and term.strip())


def _feature_payload(
    row: ItemRow,
    *,
    topic_score: float,
    learned_score: float,
    hybrid_score: float,
    final_score: float,
    reason_penalty: float,
    downweight_penalty: float,
    scoring_mode: str,
) -> dict[str, Any]:
    return {
        "ranker_version": RANKER_VERSION,
        "topic_score": float(topic_score),
        "learned_score": float(learned_score),
        "hybrid_score": float(hybrid_score),
        "final_score": float(final_score),
        "rank_score": float(final_score),
        "confidence_score": float(final_score),
        "reason_penalty": float(reason_penalty),
        "downweight_penalty": float(downweight_penalty),
        "source_bucket": source_bucket(row),
        "scoring_mode": scoring_mode,
    }


def _cosine_score_items(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
) -> list[tuple[ItemRow, float]]:
    """Cosine baseline: embed title+abstract, score against profile.

    ``profile_vec`` may be a 1-D centroid (shape [D]) or a 2-D matrix (shape
    [N, D]) from :func:`build_profile_matrix`.  The 2-D path uses top-k-mean
    cosine for OR-style multi-interest matching.
    """
    items = [item for item in items if not should_skip_item(item)]
    if not items:
        return []

    vecs = embed_item_rows(items)
    if profile_vec.size == 0 or vecs.size == 0:
        sims = np.zeros(len(items), dtype=np.float32)
    else:
        sims = _cosine_sim(vecs, profile_vec)

    final, _features = _apply_quality_adjustments_with_features(
        items,
        sims,
        downweight_terms,
        reason_penalty_map,
        learned_scores=np.zeros(len(items), dtype=np.float32),
        hybrid_scores=sims,
        scoring_mode="cosine",
    )
    scored = list(zip(items, final, strict=True))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def _freshness_penalty(row: Any) -> float:
    """Return a score penalty based on item age, with per-section decay curves."""
    from datetime import datetime, timezone
    published = getattr(row, "published_at", None)
    if not isinstance(published, datetime):
        return 0.0  # unknown date — don't guess from fetched_at
    ref = published if published.tzinfo is not None else published.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    age_days = max(0.0, (now - ref).total_seconds() / 86400)
    age_hours = age_days * 24
    section = str(getattr(row, "section", "") or "").lower()

    if section == "research":
        # Academic papers stay fresh for weeks; bonus for very new papers
        if age_days < 2.0:
            return -0.05  # freshness bonus
        return min(0.08, max(0.0, (age_days - 2.0) / 28.0) * 0.08)
    elif section in ("world", "industry"):
        # Breaking-news bonus for items published < 6 hours ago
        if age_hours < 6.0:
            return -0.04
        # News decays fast — HN-style gravity
        return min(0.20, ((age_hours / 72.0) ** 1.5) * 0.20)
    elif section == "regulatory":
        # Regulatory updates stay relevant longer than news
        return min(0.10, max(0.0, (age_days - 1.0) / 90.0) * 0.10)
    else:
        # Default: gentle linear ramp
        return min(0.10, max(0.0, (age_days - 1.5) * 0.010))


def _apply_quality_adjustments_with_features(
    items: list[ItemRow],
    base_scores: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
    *,
    learned_scores: np.ndarray,
    hybrid_scores: np.ndarray,
    scoring_mode: str,
) -> tuple[list[float], ScoreFeatureMap]:
    texts = [_item_text(r) for r in items]
    terms_lc = [t.lower() for t in downweight_terms if t and t.strip()]
    result: list[float] = []
    features: ScoreFeatureMap = {}
    for row, base, learned, hybrid, txt in zip(
        items,
        base_scores,
        learned_scores,
        hybrid_scores,
        texts,
        strict=True,
    ):
        reason_penalty = _reason_penalty_for(row, reason_penalty_map)
        score = quality_adjusted_score(row, float(base)) - reason_penalty
        freshness_pen = _freshness_penalty(row)
        score -= freshness_pen  # negative value = bonus (score increases)
        downweight_penalty = (
            DOWNWEIGHT_PENALTY
            if terms_lc and any(term in txt.lower() for term in terms_lc)
            else 0.0
        )
        if downweight_penalty:
            score -= downweight_penalty
        score_float = float(score)
        result.append(score_float)
        features[_row_feature_key(row)] = _feature_payload(
            row,
            topic_score=float(base),
            learned_score=float(learned),
            hybrid_score=float(hybrid),
            final_score=score_float,
            reason_penalty=float(reason_penalty),
            downweight_penalty=float(downweight_penalty),
            scoring_mode=scoring_mode,
        )
    return result, features


def _cosine_score_items_with_features(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
) -> tuple[list[tuple[ItemRow, float]], ScoreFeatureMap]:
    items = [item for item in items if not should_skip_item(item)]
    if not items:
        return [], {}

    vecs = embed_item_rows(items)
    if profile_vec.size == 0 or vecs.size == 0:
        sims = np.zeros(len(items), dtype=np.float32)
    else:
        sims = _cosine_sim(vecs, profile_vec)

    final, features = _apply_quality_adjustments_with_features(
        items,
        sims,
        downweight_terms,
        reason_penalty_map,
        learned_scores=np.zeros(len(items), dtype=np.float32),
        hybrid_scores=sims,
        scoring_mode="cosine",
    )
    scored = list(zip(items, final, strict=True))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored, features


def score_items_lr(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
) -> list[tuple[ItemRow, float]]:
    """Hybrid cosine + LR scorer with downweight penalty.

    Falls back to :func:`_cosine_score_items` when the LR ranker cannot be
    loaded or fewer than 30 votes are available.
    """
    items = [item for item in items if not should_skip_item(item)]
    if not items:
        return []

    from ..votes import MIN_VOTES_FOR_LR

    lr = get_lr_ranker()
    if lr is None or _vote_count() < MIN_VOTES_FOR_LR:
        scored, _features = _cosine_score_items_with_features(
            items,
            profile_vec,
            downweight_terms,
            reason_penalty_map,
        )
        return scored

    vecs = embed_item_rows(items)
    if profile_vec.size == 0 or vecs.size == 0:
        cosine = np.zeros(len(items), dtype=np.float32)
    else:
        cosine = _cosine_sim(vecs, profile_vec)

    try:
        from ..votes import _build_item_features
        features = _build_item_features(
            items,
            profile_vec if profile_vec.ndim == 2 else profile_vec.reshape(1, -1),
        )
        lr_prob = lr.score(features)
    except Exception as e:  # noqa: BLE001
        logger.warning("LRRanker.score failed (%s); falling back to cosine", e)
        scored, _features = _cosine_score_items_with_features(
            items,
            profile_vec,
            downweight_terms,
            reason_penalty_map,
        )
        return scored

    # Apply quality adjustments on raw cosine so calibrated thresholds remain meaningful
    final, _features = _apply_quality_adjustments_with_features(
        items,
        cosine,  # raw cosine, not normalized blend
        downweight_terms,
        reason_penalty_map,
        learned_scores=lr_prob,
        hybrid_scores=cosine,
        scoring_mode="hybrid_lr",
    )
    # Fuse quality-adjusted score with LR probability for final ranking
    blended_rank = _fuse_scores(np.array(final, dtype=np.float32), lr_prob).tolist()
    scored = list(zip(items, blended_rank, strict=True))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def score_items_with_features(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
) -> tuple[list[tuple[ItemRow, float]], ScoreFeatureMap]:
    """Score items and return per-row feature snapshots keyed by item id/id(row)."""
    items = [item for item in items if not should_skip_item(item)]
    if not items:
        return [], {}

    from ..votes import MIN_VOTES_FOR_LR

    lr = get_lr_ranker()
    if lr is None or _vote_count() < MIN_VOTES_FOR_LR:
        return _cosine_score_items_with_features(
            items,
            profile_vec,
            downweight_terms,
            reason_penalty_map,
        )

    vecs = embed_item_rows(items)
    if profile_vec.size == 0 or vecs.size == 0:
        cosine = np.zeros(len(items), dtype=np.float32)
    else:
        cosine = _cosine_sim(vecs, profile_vec)

    try:
        from ..votes import _build_item_features
        features_matrix = _build_item_features(
            items,
            profile_vec if profile_vec.ndim == 2 else profile_vec.reshape(1, -1),
        )
        lr_prob = lr.score(features_matrix)
    except Exception as e:  # noqa: BLE001
        logger.warning("LRRanker.score failed (%s); falling back to cosine", e)
        return _cosine_score_items_with_features(
            items,
            profile_vec,
            downweight_terms,
            reason_penalty_map,
        )

    # Apply quality adjustments on raw cosine so calibrated thresholds remain meaningful
    final, features = _apply_quality_adjustments_with_features(
        items,
        cosine,  # raw cosine, not normalized blend
        downweight_terms,
        reason_penalty_map,
        learned_scores=lr_prob,
        hybrid_scores=cosine,
        scoring_mode="hybrid_lr",
    )
    # Fuse quality-adjusted score with LR probability for final ranking
    blended_rank = _fuse_scores(np.array(final, dtype=np.float32), lr_prob).tolist()
    # Update features to record the hybrid blend
    for i, (row, _) in enumerate(zip(items, blended_rank)):
        key = _row_feature_key(row)
        if key in features:
            features[key]["confidence_score"] = float(final[i])
            features[key]["rank_score"] = float(blended_rank[i])
            features[key]["hybrid_score"] = float(blended_rank[i])
            features[key]["final_score"] = float(blended_rank[i])
            features[key]["scoring_mode"] = "hybrid_lr"
    scored = list(zip(items, blended_rank, strict=True))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored, features


def score_items(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
) -> list[tuple[ItemRow, float]]:
    """Public scorer. Routes through hybrid LR when available, else cosine."""
    return score_items_lr(items, profile_vec, downweight_terms, reason_penalty_map)


def pick_top_per_section(
    scored: list[tuple[ItemRow, float]],
    caps: dict[str, int],
) -> list[tuple[ItemRow, float]]:
    """Take up to caps[section] while protecting research source diversity.

    Research needs editorial balance, not just the top cosine/LR scores. arXiv
    CS and other preprints are allowed through when they are strong matches, but
    they cannot consume most of the research section when high-quality journal
    articles are available.
    """
    out: list[tuple[ItemRow, float]] = []
    for section, cap in caps.items():
        if cap <= 0:
            continue
        section_scored = [(row, score) for row, score in scored if (row.section or "") == section]
        if section == "research":
            out.extend(_pick_research_balanced(section_scored, cap))
        else:
            out.extend(section_scored[:cap])
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _mmr_select(
    candidates: list[tuple[ItemRow, float]],
    cap: int,
    lambda_: float = 0.7,
) -> list[tuple[ItemRow, float]]:
    """Greedy Maximal Marginal Relevance selection for diversity within section.

    Selects `cap` items from candidates, balancing relevance (score) and
    diversity (low similarity to already-selected items). lambda_=0.7 means
    70% weight on relevance, 30% on diversity.
    """
    if len(candidates) <= cap:
        return list(candidates)

    try:
        vecs = embed_item_rows([row for row, _ in candidates])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        emb = vecs / (norms + 1e-9)
    except Exception:
        return candidates[:cap]

    scores = np.array([s for _, s in candidates], dtype=np.float32)
    # Normalize scores to [0,1]
    lo, hi = scores.min(), scores.max()
    if hi > lo:
        scores_norm = (scores - lo) / (hi - lo)
    else:
        scores_norm = np.ones_like(scores) * 0.5

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while len(selected) < cap and remaining:
        if not selected:
            best = max(remaining, key=lambda i: scores_norm[i])
        else:
            sel_emb = emb[selected]
            best_score = -np.inf
            best = remaining[0]
            for idx in remaining:
                max_sim = float((emb[idx] @ sel_emb.T).max())
                mmr = lambda_ * float(scores_norm[idx]) - (1.0 - lambda_) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best = idx
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]


def _pick_research_balanced(
    scored: list[tuple[ItemRow, float]],
    cap: int,
) -> list[tuple[ItemRow, float]]:
    if not scored or cap <= 0:
        return []

    max_arxiv_cs = max(1, min(3, math.ceil(cap * 0.10)))
    max_preprints = max(max_arxiv_cs, math.ceil(cap * 0.20))
    max_aggregators = max(1, math.ceil(cap * 0.10))
    max_per_source = max(2, math.ceil(cap * 0.20)) if cap >= 5 else None
    min_high_quality = min(math.ceil(cap * 0.20), _available(scored, is_high_quality_journal_source))
    min_published = min(math.ceil(cap * 0.30), _available(scored, is_published_journal_source))

    # Low-impact venues should appear only rarely and only when strongly on-topic.
    try:
        from ..config import get_settings

        _s = get_settings()
        max_low_impact = int(cap * float(_s.max_low_impact_research_frac))
        low_impact_floor = float(_s.low_impact_relevance_floor)
    except Exception:  # noqa: BLE001
        max_low_impact = cap // 6
        low_impact_floor = 0.58

    selected: list[tuple[ItemRow, float]] = []
    selected_ids: set[int] = set()
    bucket_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    def add(
        row: ItemRow,
        score: float,
        *,
        enforce_source_cap: bool = True,
        allow_low_impact_override: bool = False,
    ) -> bool:
        row_id = getattr(row, "id", None)
        key = int(row_id) if isinstance(row_id, int) else id(row)
        if key in selected_ids or len(selected) >= cap:
            return False
        if not allow_low_impact_override and is_low_impact_research(row):
            # Frequency cap + relevance floor: low-impact work is gated unless we
            # are in the last-resort fill (override) to avoid a short section.
            if float(score) < low_impact_floor:
                return False
            if bucket_counts.get("low_impact_journal", 0) >= max_low_impact:
                return False
        source = str(getattr(row, "source", "") or "").strip().lower()
        if (
            enforce_source_cap
            and max_per_source is not None
            and source
            and source_counts.get(source, 0) >= max_per_source
        ):
            return False
        selected.append((row, score))
        selected_ids.add(key)
        bucket = source_bucket(row)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
        return True

    for predicate, target in (
        (is_high_quality_journal_source, min_high_quality),
        (is_published_journal_source, min_published),
    ):
        if target <= 0:
            continue
        while sum(1 for row, _score in selected if predicate(row)) < target:
            added = False
            for row, score in scored:
                if predicate(row) and add(row, score):
                    added = True
                    break
            if not added:
                break

    exceptional_preprint_slots = max(1, min(max_preprints, math.ceil(cap * 0.10)))
    # Compute dynamic exceptional threshold from all section scores
    _all_scores = [s for _, s in scored]
    _exceptional_threshold = max(0.55, float(np.percentile(_all_scores, 90)) if len(_all_scores) >= 5 else 0.75)
    for row, score in scored:
        if len(selected) >= cap:
            break
        if not is_preprint_source(row):
            continue
        preprint_count = sum(
            count
            for name, count in bucket_counts.items()
            if name in {"arxiv_cs", "arxiv_other", "bio_med_preprint", "preprint_other"}
        )
        if preprint_count >= exceptional_preprint_slots:
            break
        if score < _exceptional_threshold:
            continue
        if is_arxiv_cs_source(row) and bucket_counts.get(source_bucket(row), 0) >= max_arxiv_cs:
            continue
        add(row, score)

    for row, score in scored:
        if len(selected) >= cap:
            break
        bucket = source_bucket(row)
        if is_arxiv_cs_source(row) and bucket_counts.get(bucket, 0) >= max_arxiv_cs:
            continue
        if is_preprint_source(row):
            preprint_count = sum(
                count
                for name, count in bucket_counts.items()
                if name in {"arxiv_cs", "arxiv_other", "bio_med_preprint", "preprint_other"}
            )
            if preprint_count >= max_preprints:
                continue
        if bucket == "aggregator" and bucket_counts.get(bucket, 0) >= max_aggregators:
            continue
        add(row, score)

    if len(selected) < cap:
        for row, score in scored:
            if len(selected) >= cap:
                break
            if is_preprint_source(row) or source_bucket(row) == "aggregator":
                continue
            add(row, score)

    # Safety valve: only use preprints/aggregators as final fill when the section
    # has no non-preprint items at all (e.g., a purely arXiv research feed).
    if len(selected) < cap:
        all_limited = all(
            is_preprint_source(row) or source_bucket(row) == "aggregator"
            for row, _ in scored
        )
        if all_limited:
            for row, score in scored:
                if len(selected) >= cap:
                    break
                add(row, score)

    # Last resort: rather than ship a short section, allow low-impact items past
    # their frequency cap / relevance floor.
    if len(selected) < cap:
        for row, score in scored:
            if len(selected) >= cap:
                break
            add(row, score, allow_low_impact_override=True)

    # Apply MMR diversity re-ranking over the selected pool
    if len(selected) > 1:
        selected = _mmr_select(selected, cap=len(selected), lambda_=0.7)

    return selected


def _available(scored: list[tuple[ItemRow, float]], predicate) -> int:
    return sum(1 for row, _score in scored if predicate(row))
