"""Cosine + LR ranker with downweight penalties and per-section caps.

The public callable :func:`score_items` is the single entry point used by
the pipeline. Internally it routes between the cosine baseline and the
hybrid cosine+LR scorer when an :class:`LRRanker` and >=30 votes are
available.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import UTC
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
    should_skip_item,
    source_bucket,
)

logger = logging.getLogger(__name__)

DOWNWEIGHT_PENALTY = 0.20
# Reciprocal Rank Fusion constant. Larger = flatter weighting of rank position.
RRF_K = 60

ScoreFeatureMap = dict[int, dict[str, Any]]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        # No epsilon in the denominator: the branch guarantees hi > lo, and a
        # trailing 1e-6 would stop the max element from ever reaching 1.0, subtly
        # shifting how many items clear absolute gates like adaptive_size_bar.
        return (x - lo) / (hi - lo)
    return np.full_like(x, 0.5)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """Return 0-based ranks (0 = highest value), ties broken by original order."""
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return ranks


def _fuse_scores(qa: np.ndarray, lr_score: np.ndarray) -> np.ndarray:
    """Fuse the quality-adjusted topic ranking with the LR ranking via RRF.

    Reciprocal Rank Fusion combines the two *rankings* (not their raw values), so
    it is insensitive to score scale/outliers — which matters because the LR
    margin and the topic score live on different scales. The fused result is
    min-maxed back to [0, 1] so downstream magnitude thresholds (e.g. the
    exceptional-preprint cutoff) keep working.
    """
    r_qa = _rank_desc(np.asarray(qa, dtype=np.float32))
    r_lr = _rank_desc(np.asarray(lr_score, dtype=np.float32))
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
        # Per-feature standardization (z-score) fitted on the training set and
        # reapplied at inference, so every engineered feature enters the model on
        # a common scale — L2 regularization stays even and no single raw-unit
        # feature can dominate / blow up its coefficient.
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._sk_model = None  # cached fitted sklearn model (when available)

    # ---- training -----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, *, persist: bool = True) -> None:
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

        # Standardize features (z-score) before fitting. Guard near-constant
        # columns (std≈0) with scale=1 to avoid amplifying inference noise.
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
        mean = mean.astype(np.float32)
        Xs = (X - mean) / scale

        # Stronger L2 than before (C=0.3, was 0.5–1.0). With the de-confounded,
        # standardized feature set this keeps coefficients bounded (~1.5) and
        # prevents the sigmoid saturation / sign-flips seen with the old v4 set.
        C = 0.3
        clf = LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced", random_state=0
        )
        clf.fit(Xs, y)
        # Serve exclusively through the manual standardized path so train==serve
        # regardless of whether we scored in-process or from reloaded weights.
        self._sk_model = None
        self._classes = clf.classes_.astype(np.int32)
        self.coef_ = clf.coef_.astype(np.float32, copy=False)
        self.intercept_ = float(clf.intercept_[0])
        self.mean_ = mean
        self.scale_ = scale

        if not persist:
            # In-memory fit only (e.g. benchmarks/experiments). Never touch the
            # production model artifact.
            return

        target = _lr_weights_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".npz.tmp")
        with tmp.open("wb") as f:
            np.savez(
                f,
                coef=self.coef_,
                intercept=np.asarray([self.intercept_], dtype=np.float32),
                feature_mean=self.mean_,
                feature_scale=self.scale_,
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
            with np.load(path, allow_pickle=False) as data:
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

                if "feature_mean" not in data.files or "feature_scale" not in data.files:
                    logger.warning(
                        "LRRanker: stale weights missing standardization params; "
                        "will retrain"
                    )
                    return False
                mean = data["feature_mean"].astype(np.float32, copy=False)
                scale = data["feature_scale"].astype(np.float32, copy=False)

            self.coef_ = coef
            self.intercept_ = intercept
            self.mean_ = mean
            self.scale_ = scale
            self._classes = classes
            self._sk_model = None
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("LRRanker: failed to load weights: %s", e)
            return False

    # ---- inference ----------------------------------------------------------

    def decision_function(self, embeddings: np.ndarray) -> np.ndarray:
        """Raw standardized margin (logit z) for the positive class.

        Use this — NOT :meth:`score` — for *ranking*. The retrieved pool is
        pre-filtered to be relevant, so the sigmoid saturates: hundreds of items
        map to prob==1.0 (float precision) and become tied, destroying the LR's
        discrimination exactly at the top of the list. The logit is monotone with
        the probability but keeps every item distinct, so RRF can rank on it.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LRRanker is not fitted; call fit() or load() first")
        # Standardize with the persisted train-set mean/scale. sklearn's binary LR
        # stores coef/intercept oriented for class_[1] (the positive class). Using
        # the standardized manual path for BOTH freshly-fit and reloaded models
        # guarantees train==serve.
        x = embeddings.astype(np.float32, copy=False)
        if self.mean_ is not None and self.scale_ is not None:
            x = (x - self.mean_) / self.scale_
        return (x @ self.coef_.reshape(-1) + np.float32(self.intercept_)).astype(
            np.float32, copy=False
        )

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Positive-class probability (sigmoid of the margin). For DISPLAY/thresholds;
        rank with :meth:`decision_function` instead (see its docstring)."""
        z = self.decision_function(embeddings)
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
    """Score each item by 0.7*max_similarity + 0.3*top3_mean over profile facets.

    Each profile row is L2-normalized to a unit vector first, so every facet
    yields a *true* cosine in [-1, 1] regardless of its construction weight, and
    the best-matching facet (top1) rewards a specialist match over a generalist.

    Why unit-normalize rather than divide by the max row norm (the old approach):
    rows are built as ``weight × unit_vec``, and dividing every similarity by the
    single largest row norm scales down *all the other* facets. A single
    high-weight row — e.g. the Rocchio learned vector, whose weight grows to ~8.7
    as votes accumulate — would then collapse the whole score range (observed:
    a 0.47–0.71 spread crushed to 0.08–0.12), swamping topic separation under the
    venue-prestige bonus. Unit-normalizing makes each facet's weight irrelevant to
    the cosine magnitude, keeping a stable scale that downstream absolute
    thresholds (relevance floor, adaptive size bar) depend on.
    """
    row_norms = np.linalg.norm(profile_mat, axis=1, keepdims=True)
    unit = profile_mat / np.clip(row_norms, 1e-9, None)
    sims = (vecs @ unit.T.astype(np.float32)).astype(np.float32)
    # Bounded per-facet importance applied POST-cosine. Row construction weight
    # (row_norm) is clamped to <=1.0 and used as a multiplier so a *context* facet
    # (a peripheral "keep-me-informed" keyword, built at weight < 1) cannot let an
    # item win on the top-1/max purely by matching that one peripheral term —
    # while core facets (weight >= 1) keep full cosine. Clamping at 1.0 (never
    # boosting above the true cosine) preserves the absolute score scale the
    # relevance floor depends on; a high-weight row (bio, seed, Rocchio) therefore
    # behaves exactly as before (unit cosine), so this is inert for profiles with
    # no context facets.
    facet_w = np.clip(row_norms.reshape(1, -1), 0.0, 1.0).astype(np.float32)
    sims = sims * facet_w
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
    primary_facet: str = "",
    secondary_facets: list[str] | None = None,
    primary_facet_score: float = 0.0,
    topic_priority: float = 0.0,
    topic_priority_bonus: float = 0.0,
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
        # Facet attribution (P1) — CONTRACT keys, always present.
        "primary_facet": str(primary_facet or ""),
        "secondary_facets": list(secondary_facets or []),
        "primary_facet_score": float(primary_facet_score),
        "topic_priority": float(topic_priority),
        "topic_priority_bonus": float(topic_priority_bonus),
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
    from datetime import datetime
    published = getattr(row, "published_at", None)
    if not isinstance(published, datetime):
        return 0.0  # unknown date — don't guess from fetched_at
    ref = published if published.tzinfo is not None else published.replace(tzinfo=UTC)
    now = datetime.now(UTC)
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


def _attribute_or_none(
    vecs: np.ndarray, attribution: Any | None, n_items: int
) -> list[Any] | None:
    """Compute per-item facet attributions aligned to ``items``, or None.

    Returns None (fully backward-compatible: no bonus, default feature keys) when
    no attribution context is supplied or item vectors are unavailable.
    """
    if attribution is None:
        return None
    if vecs is None or getattr(vecs, "size", 0) == 0 or vecs.shape[0] != n_items:
        return None
    try:
        from .profile import attribute_items

        return attribute_items(vecs, attribution)
    except Exception as e:  # noqa: BLE001
        logger.warning("facet attribution failed: %s", e)
        return None


def _apply_quality_adjustments_with_features(
    items: list[ItemRow],
    base_scores: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
    *,
    learned_scores: np.ndarray,
    hybrid_scores: np.ndarray,
    scoring_mode: str,
    facet_attr: list[Any] | None = None,
) -> tuple[list[float], ScoreFeatureMap]:
    texts = [_item_text(r) for r in items]
    terms_lc = [t.lower() for t in downweight_terms if t and t.strip()]
    result: list[float] = []
    features: ScoreFeatureMap = {}
    for idx, (row, base, learned, hybrid, txt) in enumerate(zip(
        items,
        base_scores,
        learned_scores,
        hybrid_scores,
        texts,
        strict=True,
    )):
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
        # Topic priority is persisted here, but applied later as a selection-order
        # preference in the pipeline. Keeping it out of this score ensures it
        # cannot change low-impact eligibility or the final-score cutoff.
        attr = facet_attr[idx] if facet_attr is not None and idx < len(facet_attr) else None
        priority_bonus = float(getattr(attr, "priority_bonus", 0.0)) if attr is not None else 0.0
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
            primary_facet=str(getattr(attr, "primary", "") or "") if attr is not None else "",
            secondary_facets=list(getattr(attr, "secondaries", []) or []) if attr is not None else [],
            primary_facet_score=float(getattr(attr, "primary_score", 0.0)) if attr is not None else 0.0,
            topic_priority=float(getattr(attr, "priority", 0.0)) if attr is not None else 0.0,
            topic_priority_bonus=priority_bonus,
        )
    return result, features


def _cosine_score_items_with_features(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
    attribution: Any | None = None,
) -> tuple[list[tuple[ItemRow, float]], ScoreFeatureMap]:
    items = [item for item in items if not should_skip_item(item)]
    if not items:
        return [], {}

    vecs = embed_item_rows(items)
    if profile_vec.size == 0 or vecs.size == 0:
        sims = np.zeros(len(items), dtype=np.float32)
    else:
        sims = _cosine_sim(vecs, profile_vec)

    facet_attr = _attribute_or_none(vecs, attribution, len(items))
    final, features = _apply_quality_adjustments_with_features(
        items,
        sims,
        downweight_terms,
        reason_penalty_map,
        learned_scores=np.zeros(len(items), dtype=np.float32),
        hybrid_scores=sims,
        scoring_mode="cosine",
        facet_attr=facet_attr,
    )
    scored = list(zip(items, final, strict=True))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored, features


def score_items_lr(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
    attribution: Any | None = None,
) -> list[tuple[ItemRow, float]]:
    """Hybrid scorer without the feature snapshots.

    Delegates to :func:`score_items_with_features` rather than repeating the
    fusion. The two used to carry separate copies of the same blend, so a change
    to one could silently diverge from the other and only one was under test.
    """
    scored, _features = score_items_with_features(
        items,
        profile_vec,
        downweight_terms,
        reason_penalty_map,
        attribution=attribution,
    )
    return scored


def score_items_with_features(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
    attribution: Any | None = None,
) -> tuple[list[tuple[ItemRow, float]], ScoreFeatureMap]:
    """Score items and return per-row feature snapshots keyed by item id/id(row)."""
    items = [item for item in items if not should_skip_item(item)]
    if not items:
        return [], {}

    from ..config import get_settings, resolve_scoring_mode
    from ..votes import MIN_VOTES_FOR_LR

    mode = resolve_scoring_mode(get_settings())
    if mode == "cosine" or _vote_count() < MIN_VOTES_FOR_LR:
        return _cosine_score_items_with_features(
            items,
            profile_vec,
            downweight_terms,
            reason_penalty_map,
            attribution=attribution,
        )

    vecs = embed_item_rows(items)
    if profile_vec.size == 0 or vecs.size == 0:
        cosine = np.zeros(len(items), dtype=np.float32)
    else:
        cosine = _cosine_sim(vecs, profile_vec)

    try:
        if mode == "hybrid_lr":
            lr = get_lr_ranker()
            if lr is None:
                raise RuntimeError("SCORING_MODE=hybrid_lr but no trained weights")
            from ..votes import _build_item_features

            matrix = _build_item_features(
                items,
                profile_vec if profile_vec.ndim == 2 else profile_vec.reshape(1, -1),
            )
            fusion_signal = lr.decision_function(matrix)
            display = lr.score(matrix)
        else:
            from ..votes import knn_preference_scores

            fusion_signal = knn_preference_scores(items)
            # Display/uncertainty scale: [-1, 1] -> [0, 1]. NOTE this is NOT a
            # calibrated probability and its median sits near 0.33, so 0.5 is not
            # a "no signal" midpoint (see the exploration-slot note in pipeline).
            display = ((fusion_signal + 1.0) / 2.0).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s scoring failed (%s); falling back to cosine", mode, e)
        return _cosine_score_items_with_features(
            items,
            profile_vec,
            downweight_terms,
            reason_penalty_map,
            attribution=attribution,
        )

    facet_attr = _attribute_or_none(vecs, attribution, len(items))
    # Apply quality adjustments on raw cosine so calibrated thresholds remain meaningful
    final, features = _apply_quality_adjustments_with_features(
        items,
        cosine,  # raw cosine, not normalized blend
        downweight_terms,
        reason_penalty_map,
        # Display/uncertainty scale: [-1, 1] -> [0, 1], 0.5 = no vote signal.
        learned_scores=display,
        hybrid_scores=cosine,
        scoring_mode=mode,
        facet_attr=facet_attr,
    )
    # Fuse the quality-adjusted ranking with the learned ranking. Compare modes
    # with scripts/benchmark_ranker.py, which reports head metrics (nDCG, P@K,
    # must-read recall) for every configuration — full-list pairwise accuracy
    # alone once hid a head-of-list regression.
    blended_rank = _fuse_scores(
        np.array(final, dtype=np.float32), fusion_signal
    ).tolist()
    # Update features to record the hybrid blend
    for i, (row, _) in enumerate(zip(items, blended_rank, strict=True)):
        key = _row_feature_key(row)
        if key in features:
            features[key]["confidence_score"] = float(final[i])
            features[key]["rank_score"] = float(blended_rank[i])
            features[key]["hybrid_score"] = float(blended_rank[i])
            features[key]["final_score"] = float(blended_rank[i])
            features[key]["scoring_mode"] = mode
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


def _pick_news_balanced(
    section_scored: list[tuple[ItemRow, float]],
    cap: int,
) -> list[tuple[ItemRow, float]]:
    """Fill a news section by score but cap any single source's share.

    A prolific feed (e.g. STAT) would otherwise take most of the section with
    near-duplicate items. Limit each source to ~1/3 of the cap (>=2) on a first
    pass, then relax to fill any remaining slots so the section is never short.
    """
    if cap <= 0 or not section_scored:
        return []
    ranked = sorted(section_scored, key=lambda t: t[1], reverse=True)
    max_per_source = max(2, math.ceil(cap / 3))
    counts: dict[str, int] = {}
    picked: list[tuple[ItemRow, float]] = []
    picked_ids: set[int] = set()

    def _key(row: ItemRow) -> str:
        return str(getattr(row, "source", "") or "").strip().lower()

    for row, score in ranked:
        if len(picked) >= cap:
            break
        src = _key(row)
        if counts.get(src, 0) >= max_per_source:
            continue
        picked.append((row, score))
        picked_ids.add(id(row))
        counts[src] = counts.get(src, 0) + 1
    # Relax the per-source cap to backfill if diversity left the section short.
    if len(picked) < cap:
        for row, score in ranked:
            if len(picked) >= cap:
                break
            if id(row) not in picked_ids:
                picked.append((row, score))
    return picked


def pick_top_per_section(
    scored: list[tuple[ItemRow, float]],
    caps: dict[str, int],
    catch_up: bool = False,
) -> list[tuple[ItemRow, float]]:
    """Take up to caps[section] while protecting research source diversity.

    Research needs editorial balance, not just the top cosine/LR scores. arXiv
    CS and other preprints are allowed through when they are strong matches, but
    they cannot consume most of the research section when high-quality journal
    articles are available. ``catch_up`` relaxes that balance so a post-gap
    backlog (which the date-backfilling preprint/aggregator sources dominate)
    can fill the expanded section.
    """
    out: list[tuple[ItemRow, float]] = []
    for section, cap in caps.items():
        if cap <= 0:
            continue
        section_scored = [(row, score) for row, score in scored if (row.section or "") == section]
        if section == "research":
            out.extend(_pick_research_balanced(section_scored, cap, catch_up=catch_up))
        else:
            out.extend(_pick_news_balanced(section_scored, cap))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _pick_research_balanced(
    scored: list[tuple[ItemRow, float]],
    cap: int,
    catch_up: bool = False,
) -> list[tuple[ItemRow, float]]:
    if not scored or cap <= 0:
        return []

    if catch_up:
        # After a usage gap the backlog is dominated by the sources that CAN
        # backfill by date (bioRxiv, arXiv, OpenAlex) — RSS journals only expose
        # their current window. Relax the source-diversity caps so on-topic
        # backlog papers actually fill the (expanded) section; relevance beats
        # editorial balance when catching up.
        max_arxiv_cs = math.ceil(cap * 0.35)
        max_preprints = math.ceil(cap * 0.75)
        max_aggregators = math.ceil(cap * 0.60)
        max_per_source = max(3, math.ceil(cap * 0.60))
    else:
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
        if getattr(_s, "adaptive_relevance_floor", False):
            from .calibrate import adaptive_relevance_floor as _adaptive_floor

            low_impact_floor = _adaptive_floor(low_impact_floor)
    except Exception:  # noqa: BLE001
        max_low_impact = cap // 6
        low_impact_floor = 0.58

    selected: list[tuple[ItemRow, float]] = []
    selected_ids: set[int] = set()
    bucket_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    preprint_source_counts: dict[str, int] = {}

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
        if is_preprint_source(row):
            key = str(getattr(row, "source", "") or "").strip().lower()
            preprint_source_counts[key] = preprint_source_counts.get(key, 0) + 1
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
            # Keep one preprint slot open for a SECOND server. bioRxiv posts an
            # order of magnitude more than ChemRxiv or cond-mat.soft, so it
            # filled the entire preprint quota every day and equally relevant
            # work from the other repositories never appeared — measured: their
            # best items scored 0.752/0.753 against the day's best 0.792, all
            # well clear of the relevance floor, and neither was selected.
            source_name = str(getattr(row, "source", "") or "").strip().lower()
            if (
                max_preprints >= 2
                and preprint_source_counts.get(source_name, 0)
                >= max(1, max_preprints - 1)
            ):
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

    # Last resort: fill only up to a small HARD MINIMUM (not the full cap) by
    # overriding the low-impact frequency cap / relevance floor. A short section of
    # genuinely strong items beats padding every slot with weak low-impact work —
    # historically the padded tail slots had far lower positive-feedback rates. A
    # quiet day should simply yield fewer papers, not fifteen mediocre ones.
    hard_min = min(3, cap)
    if len(selected) < hard_min:
        for row, score in scored:
            if len(selected) >= hard_min:
                break
            add(row, score, allow_low_impact_override=True)

    return _apply_final_score_cutoff(selected)


def _apply_final_score_cutoff(
    selected: list[tuple[ItemRow, float]],
) -> list[tuple[ItemRow, float]]:
    """Drop research picks the learned model scored near the bottom.

    Section sizing gates on TOPIC cosine, but slots are then filled by FINAL
    fused score — so a high-topic, high-prestige paper the model learned the
    reader dislikes (near-zero final score) could still pad the section. This
    enforces a floor on the FINAL score: keep items scoring at least
    ``frac`` of the section's top pick, dropping the rest.

    RELATIVE (fraction of the top) rather than absolute: the fused score is RRF
    min-maxed to [0,1] PER RUN, so the top is ~1.0 and the bottom ~0.0 every
    run — a fixed absolute cut would drift day to day. A hard-minimum keeps the
    strongest N picks regardless so the gate never empties the section.
    """
    if not selected:
        return selected
    try:
        from ..config import get_settings

        s = get_settings()
        frac = float(getattr(s, "research_final_score_floor_frac", 0.0))
        min_keep = int(getattr(s, "research_final_score_min_keep", 3))
    except Exception:  # noqa: BLE001
        return selected
    if frac <= 0.0:
        return selected

    ranked = sorted(selected, key=lambda t: t[1], reverse=True)
    top = float(ranked[0][1])
    if top <= 0.0:
        # Degenerate run (all scores collapsed to 0): keep the hard-minimum so the
        # digest is never empty, rather than dropping everything against a 0 floor.
        return ranked[: max(0, min_keep)] if min_keep else ranked
    floor = frac * top
    kept = [(row, score) for row, score in ranked if float(score) >= floor]
    if len(kept) < min_keep:
        # Never drop below the hard-minimum of top-ranked items.
        kept = ranked[: min(min_keep, len(ranked))]
    return kept


def _available(scored: list[tuple[ItemRow, float]], predicate) -> int:
    return sum(1 for row, _score in scored if predicate(row))


def apply_exploration(
    picked: list[tuple[ItemRow, float]],
    candidates: list[tuple[ItemRow, float]],
    uncertainty: Mapping[int, float],
    *,
    slots: int,
    eligible,
) -> list[tuple[ItemRow, float]]:
    """Swap the lowest-scored picked research items for high-uncertainty ones.

    Active learning: surfacing the items the learned ranker is least sure about
    (uncertainty near 1.0) yields the most informative votes. Only research items
    are touched, and only ``eligible`` (high-quality) unselected candidates are
    eligible — exploration never introduces low-impact or preprint work.
    """
    if slots <= 0:
        return picked

    picked_keys = {_row_feature_key(row) for row, _ in picked}
    pool = [
        (row, score)
        for row, score in candidates
        if (row.section or "") == "research"
        and _row_feature_key(row) not in picked_keys
        and eligible(row)
        and _row_feature_key(row) in uncertainty
    ]
    if not pool:
        return picked
    # Most-uncertain first.
    pool.sort(key=lambda t: uncertainty[_row_feature_key(t[0])], reverse=True)

    # Lowest-scored research picks are the replacement candidates.
    research_idx = [i for i, (row, _) in enumerate(picked) if (row.section or "") == "research"]
    research_idx.sort(key=lambda i: picked[i][1])  # ascending score

    out = list(picked)
    n = min(slots, len(pool), len(research_idx))
    for j in range(n):
        out[research_idx[j]] = pool[j]
    return out
