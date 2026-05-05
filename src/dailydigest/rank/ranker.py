"""Cosine + LR ranker with downweight penalties and per-section caps.

The public callable :func:`score_items` is the single entry point used by
the pipeline. Internally it routes between the cosine baseline and the
hybrid cosine+LR scorer when an :class:`LRRanker` and >=30 votes are
available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

import numpy as np

from ..store import ItemRow
from .embed import embed_texts

logger = logging.getLogger(__name__)

DOWNWEIGHT_PENALTY = 0.05
HYBRID_COSINE_W = 0.5
HYBRID_LR_W = 0.5


def _lr_weights_path() -> Path:
    from ..config import get_settings
    return Path(get_settings().db_path).parent / "lr_ranker.npz"


def _item_text(row: ItemRow) -> str:
    title = (row.title or "").strip()
    abstract = (row.abstract or "").strip()
    if abstract:
        return f"{title}. {abstract}"
    return title


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

        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X, y)
        self._sk_model = model
        self._classes = model.classes_.astype(np.int32)
        self.coef_ = model.coef_.astype(np.float32, copy=False)
        self.intercept_ = float(model.intercept_[0])

        target = _lr_weights_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".npz.tmp")
        np.savez(
            tmp,
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
        return None
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
        from sqlalchemy import func, select
    except Exception:  # noqa: BLE001
        return 0
    try:
        init_db()
        with session_scope() as s:
            n = s.execute(select(func.count()).select_from(VoteRow)).scalar_one()
            return int(n or 0)
    except Exception as e:  # noqa: BLE001
        logger.warning("ranker: failed to count votes: %s", e)
        return 0


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


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


def _cosine_score_items(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
) -> list[tuple[ItemRow, float]]:
    """Cosine baseline: embed title+abstract, dot-product with profile vec."""
    if not items:
        return []

    texts = [_item_text(r) for r in items]
    vecs = embed_texts(texts)
    if profile_vec.size == 0 or vecs.size == 0:
        sims = np.zeros(len(items), dtype=np.float32)
    else:
        sims = vecs @ profile_vec.astype(np.float32, copy=False)

    final = _apply_downweight(sims, texts, downweight_terms)
    scored = list(zip(items, final, strict=True))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def score_items_lr(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
) -> list[tuple[ItemRow, float]]:
    """Hybrid cosine + LR scorer with downweight penalty.

    Falls back to :func:`_cosine_score_items` when the LR ranker cannot be
    loaded or fewer than 30 votes are available.
    """
    if not items:
        return []

    from ..votes import MIN_VOTES_FOR_LR

    lr = get_lr_ranker()
    if lr is None or _vote_count() < MIN_VOTES_FOR_LR:
        return _cosine_score_items(items, profile_vec, downweight_terms)

    texts = [_item_text(r) for r in items]
    vecs = embed_texts(texts)
    if profile_vec.size == 0 or vecs.size == 0:
        cosine = np.zeros(len(items), dtype=np.float32)
    else:
        cosine = vecs @ profile_vec.astype(np.float32, copy=False)

    try:
        lr_prob = lr.score(vecs)
    except Exception as e:  # noqa: BLE001
        logger.warning("LRRanker.score failed (%s); falling back to cosine", e)
        return _cosine_score_items(items, profile_vec, downweight_terms)

    blended = HYBRID_COSINE_W * cosine + HYBRID_LR_W * lr_prob
    final = _apply_downweight(blended, texts, downweight_terms)
    scored = list(zip(items, final, strict=True))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def score_items(
    items: list[ItemRow],
    profile_vec: np.ndarray,
    downweight_terms: list[str],
) -> list[tuple[ItemRow, float]]:
    """Public scorer. Routes through hybrid LR when available, else cosine."""
    return score_items_lr(items, profile_vec, downweight_terms)


def pick_top_per_section(
    scored: list[tuple[ItemRow, float]],
    caps: dict[str, int],
) -> list[tuple[ItemRow, float]]:
    """Walk scored desc, take up to caps[section] per section. Skip unknown sections."""
    counts: dict[str, int] = {k: 0 for k in caps}
    out: list[tuple[ItemRow, float]] = []
    for row, score in scored:
        section = row.section or ""
        cap = caps.get(section)
        if cap is None:
            continue
        if counts[section] >= cap:
            continue
        counts[section] += 1
        out.append((row, score))
    return out
