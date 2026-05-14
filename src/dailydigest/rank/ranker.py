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
    is_arxiv_cs_source,
    is_high_quality_journal_source,
    is_preprint_source,
    is_published_journal_source,
    quality_adjusted_score,
    source_bucket,
    should_skip_item,
)

logger = logging.getLogger(__name__)

DOWNWEIGHT_PENALTY = 0.05
HYBRID_COSINE_W = 0.5
HYBRID_LR_W = 0.5
_MULTI_COSINE_K = 2  # top-k sub-profiles to average for OR semantics


def _lr_weights_path() -> Path:
    from ..config import get_settings
    return Path(get_settings().db_path).parent / "lr_ranker.npz"


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

        # C=0.05 gives strong L2 regularization — with O(30) training points and
        # 384 features, the default C=1.0 wildly overfits.
        model = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.05)
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
            )
        tmp.replace(target)
        logger.info("LRRanker: saved weights to %s", target)

    # ---- persistence --------------------------------------------------------

    def load(self) -> bool:
        path = _lr_weights_path()
        if not path.exists():
            return False
        try:
            data = np.load(path)
            self.coef_ = data["coef"].astype(np.float32, copy=False)
            self.intercept_ = float(data["intercept"][0])
            if "classes" in data.files:
                self._classes = data["classes"].astype(np.int32)
            else:
                self._classes = np.asarray([-1, 1], dtype=np.int32)
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
            pos_idx = int(np.argmax(classes))  # +1 column
            return probs[:, pos_idx].astype(np.float32, copy=False)

        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LRRanker is not fitted; call fit() or load() first")

        # Manual sigmoid using the persisted weights. sklearn's binary LR
        # stores coef/intercept oriented for class_[1] (the positive class).
        z = embeddings.astype(np.float32, copy=False) @ self.coef_.reshape(-1)
        z = z + np.float32(self.intercept_)
        prob_pos = 1.0 / (1.0 + np.exp(-z))

        if self._classes is not None and self._classes[-1] != 1:
            # Defensive: if class ordering is reversed, flip.
            prob_pos = 1.0 - prob_pos
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
    # Local import to avoid a circular import (votes -> store + embed).
    try:
        from ..store import VoteRow, init_db, session_scope
        from sqlalchemy import select
    except Exception:  # noqa: BLE001
        return 0
    try:
        init_db()
        with session_scope() as s:
            rows = s.execute(
                select(VoteRow.item_id, VoteRow.value)
                .where(VoteRow.value.in_((-1, 1)))
                .order_by(VoteRow.item_id, VoteRow.created_at.desc(), VoteRow.id.desc())
            ).all()
        seen: set[int] = set()
        signed = 0
        for item_id, _value in rows:
            iid = int(item_id)
            if iid in seen:
                continue
            seen.add(iid)
            signed += 1
        return signed
    except Exception as e:  # noqa: BLE001
        logger.warning("ranker: failed to count votes: %s", e)
        return 0


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _multi_cosine(vecs: np.ndarray, profile_mat: np.ndarray, k: int = _MULTI_COSINE_K) -> np.ndarray:
    """Top-k-mean cosine across profile sub-vectors (OR semantics).

    Each row of ``profile_mat`` is a separate profile component (e.g., one bio
    sentence or one keyword).  Scoring uses the mean of the k highest cosine
    similarities so that a niche-interest item can rank well if it is strongly
    relevant to *any* sub-profile, not just the centroid.
    """
    sims = vecs @ profile_mat.T.astype(np.float32)  # [N_items, N_profile]
    k = min(k, sims.shape[1])
    if k <= 1:
        return sims.max(axis=1).astype(np.float32)
    top_k = np.sort(sims, axis=1)[:, -k:]
    return top_k.mean(axis=1).astype(np.float32)


def _cosine_sim(vecs: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Dispatch to multi-vector or single-vector cosine based on profile shape."""
    if profile.ndim == 1:
        return (vecs @ profile.astype(np.float32, copy=False)).astype(np.float32)
    return _multi_cosine(vecs, profile)


def _apply_downweight(
    base_scores: np.ndarray,
    texts: list[str],
    downweight_terms: list[str],
) -> list[float]:
    terms_lc = [t.lower() for t in downweight_terms if t and t.strip()]
    out: list[float] = []
    for s, txt in zip(base_scores, texts, strict=True):
        score = float(s)
        if terms_lc and any(term in txt.lower() for term in terms_lc):
            score -= DOWNWEIGHT_PENALTY
        out.append(score)
    return out


def _apply_quality_adjustments(
    items: list[ItemRow],
    base_scores: np.ndarray,
    downweight_terms: list[str],
    reason_penalty_map: Mapping[Any, float] | None = None,
) -> list[float]:
    """Apply quality adjustments and then user downweight / reason penalties.

    Quality adjustment (prestige, novelty, promo) is computed from the raw
    cosine score so it is not distorted by the downweight penalty.  User
    editorial penalties (downweight terms, reason chips) are applied after so
    that the net effect on the final score is always exactly their stated value.
    """
    texts = [_item_text(r) for r in items]
    terms_lc = [t.lower() for t in downweight_terms if t and t.strip()]
    result: list[float] = []
    for row, base, txt in zip(items, base_scores, texts, strict=True):
        score = quality_adjusted_score(row, float(base)) - _reason_penalty_for(row, reason_penalty_map)
        if terms_lc and any(term in txt.lower() for term in terms_lc):
            score -= DOWNWEIGHT_PENALTY
        result.append(score)
    return result


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
            return max(0.0, float(reason_penalty_map[key]))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


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

    final = _apply_quality_adjustments(items, sims, downweight_terms, reason_penalty_map)
    scored = list(zip(items, final, strict=True))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


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
        return _cosine_score_items(items, profile_vec, downweight_terms, reason_penalty_map)

    vecs = embed_item_rows(items)
    if profile_vec.size == 0 or vecs.size == 0:
        cosine = np.zeros(len(items), dtype=np.float32)
    else:
        cosine = _cosine_sim(vecs, profile_vec)

    try:
        lr_prob = lr.score(vecs)
    except Exception as e:  # noqa: BLE001
        logger.warning("LRRanker.score failed (%s); falling back to cosine", e)
        return _cosine_score_items(items, profile_vec, downweight_terms, reason_penalty_map)

    blended = HYBRID_COSINE_W * cosine + HYBRID_LR_W * lr_prob
    final = _apply_quality_adjustments(items, blended, downweight_terms, reason_penalty_map)
    scored = list(zip(items, final, strict=True))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


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


def _pick_research_balanced(
    scored: list[tuple[ItemRow, float]],
    cap: int,
) -> list[tuple[ItemRow, float]]:
    if not scored or cap <= 0:
        return []

    max_arxiv_cs = max(1, min(3, math.ceil(cap * 0.10)))
    max_preprints = max(max_arxiv_cs, math.ceil(cap * 0.20))
    max_aggregators = max(1, math.ceil(cap * 0.10))
    min_high_quality = min(math.ceil(cap * 0.20), _available(scored, is_high_quality_journal_source))
    min_published = min(math.ceil(cap * 0.55), _available(scored, is_published_journal_source))

    selected: list[tuple[ItemRow, float]] = []
    selected_ids: set[int] = set()
    bucket_counts: dict[str, int] = {}

    def add(row: ItemRow, score: float) -> bool:
        key = id(row)
        if key in selected_ids or len(selected) >= cap:
            return False
        selected.append((row, score))
        selected_ids.add(key)
        bucket = source_bucket(row)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
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
                if name in {"arxiv_cs", "arxiv_other", "bio_med_preprint"}
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

    selected.sort(key=lambda t: t[1], reverse=True)
    return selected


def _available(scored: list[tuple[ItemRow, float]], predicate) -> int:
    return sum(1 for row, _score in scored if predicate(row))
