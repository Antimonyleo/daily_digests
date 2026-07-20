"""Platt calibration of ranking scores to P(relevant) from vote history.

The quality-adjusted / fused ranking scores are not probabilities, so absolute
thresholds (e.g. the low-impact relevance floor) are heuristics. This module
fits a logistic map ``P(relevant) = sigmoid(a*score + b)`` from the persisted
(final_score, vote) pairs, which gives two things:

* an interpretable confidence for display/eval, and
* a self-tuning relevance floor: the score at which P(relevant) crosses a
  target, so the floor adapts to the user's feedback instead of staying a
  hardcoded constant.

A monotonic (a > 0) calibrator does not reorder items, so this never changes
ranking order — only the meaning of the score scale and the derived threshold.
Everything degrades to the configured default when no calibrator exists.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from ..store import DigestItemFeatureRow, init_db, session_scope
from ..votes import _latest_vote_values

logger = logging.getLogger(__name__)

MIN_VOTES_FOR_CALIBRATION = 12


def _calibrator_path() -> Path:
    from ..config import get_settings

    return Path(get_settings().db_path).parent / "calibrator.json"


def _calibration_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) from persisted digest features joined to votes."""
    init_db()
    with session_scope() as s:
        rows = s.query(
            DigestItemFeatureRow.item_id, DigestItemFeatureRow.final_score
        ).all()
    by_item: dict[int, float] = {}
    for item_id, score in rows:
        if item_id is None or score is None:
            continue
        by_item[int(item_id)] = float(score)  # last write wins; one score per item
    if not by_item:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    votes = _latest_vote_values(list(by_item.keys()))
    scores: list[float] = []
    labels: list[int] = []
    for item_id, score in by_item.items():
        v = votes.get(item_id)
        if v in (1, -1):
            scores.append(score)
            labels.append(1 if v > 0 else 0)
    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def fit_calibrator() -> dict | None:
    """Fit and persist a Platt calibrator; return its params or None."""
    X, y = _calibration_dataset()
    if len(X) < MIN_VOTES_FOR_CALIBRATION or np.unique(y).size < 2:
        return None
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X.reshape(-1, 1), y)
    params = {
        "a": float(clf.coef_[0][0]),
        "b": float(clf.intercept_[0]),
        "n": int(len(X)),
    }
    path = _calibrator_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a crash mid-write must not leave a truncated calibrator that
    # load_calibrator would then parse into garbage.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(params))
    tmp.replace(path)
    logger.info("calibrator: fit on %d votes (a=%.3f b=%.3f)", params["n"], params["a"], params["b"])
    return params


def load_calibrator() -> dict | None:
    path = _calibrator_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if "a" in data and "b" in data:
            return data
    except Exception as e:  # noqa: BLE001
        logger.warning("calibrator: failed to load: %s", e)
    return None


def calibrated_probability(score: float, calib: dict | None = None) -> float | None:
    calib = calib or load_calibrator()
    if not calib:
        return None
    z = calib["a"] * float(score) + calib["b"]
    return 1.0 / (1.0 + math.exp(-z))


def adaptive_relevance_floor(
    default: float,
    target_prob: float = 0.5,
    calib: dict | None = None,
) -> float:
    """Return the score at which P(relevant) == target, clamped near ``default``.

    Falls back to ``default`` when there is no calibrator or it is not usefully
    monotonic. The clamp keeps sparse-data calibrators from moving the floor far.
    """
    calib = calib or load_calibrator()
    if not calib:
        return default
    a = float(calib["a"])
    b = float(calib["b"])
    if a <= 1e-6:  # non-informative or inverted — don't trust it
        return default
    logit = math.log(target_prob / (1.0 - target_prob))
    floor = (logit - b) / a
    return max(default - 0.10, min(default + 0.20, floor))
