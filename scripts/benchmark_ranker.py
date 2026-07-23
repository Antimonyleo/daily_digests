#!/usr/bin/env python
"""Leakage-free chronological held-out benchmark of the preference ranker.

Prior status claimed "0.79 vs 0.74 pairwise held-out" with no committed,
reproducible script. This is that script.

What it does, from the REAL votes DB:

  1. Take the latest signed (+1/-1) vote per item and order the items
     CHRONOLOGICALLY by that vote's timestamp.
  2. Split TRAIN = older 75%, TEST = newer 25%.
  3. Build pos/neg exemplar arrays from TRAIN votes ONLY. This is the crux of
     "leakage-free": the test items and their votes must never appear in the
     exemplar construction that produces the `pos_affinity`/`neg_affinity`
     memory features, otherwise a test item can "recognize itself".
     NOTE: `votes._build_item_features` is NOT reused here because it calls
     `_load_vote_exemplars`, which loads ALL votes (train + test) from the DB —
     that would leak the test set into the affinity features. We therefore build
     the affinity columns manually from the train-only exemplar arrays via
     `votes._affinity`, and compute the remaining columns with the same helpers
     the production feature builder uses.
  4. Train a LogisticRegression (fixed random_state) on the TRAIN feature matrix.
  5. Score TEST items with (a) topic-cosine only and (b) the full model, and
     report pairwise accuracy: over every (liked, disliked) test pair, the
     fraction the scorer orders correctly (liked scored above disliked).

The real-DB number depends entirely on the live vote history and is NOT a fixed
claim. Run it to get a current, reproducible number:

    uv run python scripts/benchmark_ranker.py

Exit code is 0 on success (even if the model does not beat the baseline — this
is a measurement tool, not a gate).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np


def _as_utc(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _build_features_train_only(
    rows: list,
    labels: list[int],
    profile_mat: np.ndarray,
    pos_ex,
    neg_ex,
) -> np.ndarray:
    """Recompute the v6 LR feature matrix, sourcing affinity from GIVEN exemplars.

    Mirrors `votes._build_item_features` column-for-column, but takes the pos/neg
    exemplar arrays as arguments instead of loading them from the whole DB. That
    is what makes the held-out evaluation leakage-free: pass TRAIN-only exemplars
    when featurizing the test set.
    """
    from dailydigest.rank.embedding_cache import embed_item_rows
    from dailydigest.rank.ranker import _multi_cosine
    from dailydigest.rank.source_quality import (
        access_friction_score as _friction,
        novelty_score as _nov,
        promotional_score as _promo,
    )
    from dailydigest.votes import _affinity

    try:
        from dailydigest.config import load_profile
        from dailydigest.rank.authors import author_match_score, load_watchlist

        watchlist = load_watchlist(load_profile())
    except Exception:  # noqa: BLE001
        watchlist = []
        author_match_score = None  # type: ignore[assignment]

    vecs = embed_item_rows(rows)
    if vecs.size == 0:
        cos = np.zeros(len(rows), dtype=np.float32)
        cand_unit = np.zeros((len(rows), 1), dtype=np.float32)
    else:
        cos = _multi_cosine(
            vecs, profile_mat if profile_mat.ndim == 2 else profile_mat.reshape(1, -1)
        )
        cand_unit = (vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)).astype(
            np.float32
        )

    cand_ids = [
        int(rid) if isinstance((rid := getattr(r, "id", None)), int) else None for r in rows
    ]
    pos_aff = _affinity(cand_unit, cand_ids, pos_ex)
    neg_aff = _affinity(cand_unit, cand_ids, neg_ex)

    now = datetime.now(timezone.utc)
    features: list[list[float]] = []
    for i, row in enumerate(rows):
        published = getattr(row, "published_at", None)
        if isinstance(published, datetime):
            ref = published if published.tzinfo else published.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - ref).total_seconds() / 86400)
            age_norm = min(1.0, age_days / 14.0)
        else:
            age_norm = 0.5

        if watchlist and author_match_score is not None:
            author_match = author_match_score(
                str(getattr(row, "authors", "") or ""), watchlist
            )
        else:
            author_match = 0.0

        cos_val = float(cos[i])
        features.append(
            [
                cos_val,
                float(_nov(row)),
                float(_promo(row)),
                float(_friction(row)),
                cos_val * (1.0 - age_norm),
                float(author_match),
                float(pos_aff[i]),
                float(neg_aff[i]),
            ]
        )
    if not features:
        from dailydigest.votes import LR_FEATURE_DIM

        return np.zeros((0, LR_FEATURE_DIM), dtype=np.float32)
    return np.asarray(features, dtype=np.float32)


def _pack_exemplars(rows: list, weights: list[float]):
    """Build a (ids, unit_vecs, weights) exemplar tuple from item rows."""
    from dailydigest.rank.embedding_cache import embed_item_rows

    if not rows:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 1), np.float32),
            np.zeros(0, np.float32),
        )
    ids = np.array(
        [int(getattr(r, "id", -1)) for r in rows], dtype=np.int64
    )
    vecs = embed_item_rows(rows).astype(np.float32)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    ws = np.array(weights, dtype=np.float32)
    return ids, vecs, ws


def _pairwise_accuracy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """Fraction of (liked, disliked) pairs the scorer orders correctly.

    Ties (equal score) count as 0.5. Returns (accuracy, n_pairs); accuracy is
    float('nan') when there is no valid pair.
    """
    pos_idx = np.where(labels > 0)[0]
    neg_idx = np.where(labels < 0)[0]
    n_pairs = len(pos_idx) * len(neg_idx)
    if n_pairs == 0:
        return float("nan"), 0
    correct = 0.0
    for pi in pos_idx:
        for ni in neg_idx:
            if scores[pi] > scores[ni]:
                correct += 1.0
            elif scores[pi] == scores[ni]:
                correct += 0.5
    return correct / n_pairs, n_pairs


def run_benchmark(
    rows: list,
    labels: list[int],
    timestamps: list[datetime],
    grades: list[int] | None = None,
    profile_mat: np.ndarray | None = None,
    train_frac: float = 0.75,
    random_state: int = 0,
) -> dict:
    """Core benchmark logic — reusable by the synthetic test.

    ``rows`` are item-like objects (real ItemRow or SimpleNamespace with ``id``,
    ``title``, ``abstract``, ``published_at``), ``labels`` are +1/-1 per row,
    ``timestamps`` are the signing vote's time (chronological order key).

    Returns a dict with keys: n_train, n_test, topic_acc, full_acc, n_test_pairs,
    train_exemplar_ids, test_ids.
    """
    from sklearn.linear_model import LogisticRegression

    from dailydigest.votes import VOTE_GRADE_NEUTRAL, value_to_grade

    n = len(rows)
    if grades is None:
        grades = [value_to_grade(v) for v in labels]

    order = sorted(range(n), key=lambda i: (timestamps[i], i))
    n_train = max(1, int(round(n * train_frac)))
    n_train = min(n_train, n - 1) if n > 1 else n
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    if profile_mat is None:
        from pathlib import Path

        import yaml

        from dailydigest.config import load_settings
        from dailydigest.models import Profile
        from dailydigest.rank.profile import build_profile_matrix_with_rocchio

        settings = load_settings()
        profile_data = yaml.safe_load(
            Path(settings.profile_path).read_text(encoding="utf-8")
        )
        profile = Profile(**profile_data)
        # Only TRAIN votes exist "so far" chronologically -> use their count for
        # the Rocchio ramp so the profile matches what the model would have seen.
        profile_mat = build_profile_matrix_with_rocchio(profile, len(train_idx))

    # Build TRAIN-only exemplars for the affinity features (leakage-free).
    pos_rows, pos_w, neg_rows, neg_w = [], [], [], []
    for i in train_idx:
        w = max(0.05, abs(grades[i] - VOTE_GRADE_NEUTRAL) / 50.0)
        if labels[i] > 0:
            pos_rows.append(rows[i])
            pos_w.append(w)
        else:
            neg_rows.append(rows[i])
            neg_w.append(w)
    pos_ex = _pack_exemplars(pos_rows, pos_w)
    neg_ex = _pack_exemplars(neg_rows, neg_w)

    train_exemplar_ids = np.concatenate([pos_ex[0], neg_ex[0]]).tolist()
    test_ids = [
        int(rid)
        for i in test_idx
        if isinstance((rid := getattr(rows[i], "id", None)), int)
    ]

    train_rows = [rows[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    test_labels = np.array([labels[i] for i in test_idx], dtype=np.int32)

    X_train = _build_features_train_only(
        train_rows, train_labels, profile_mat, pos_ex, neg_ex
    )
    X_test = _build_features_train_only(
        test_rows, test_labels.tolist(), profile_mat, pos_ex, neg_ex
    )
    y_train = np.array(train_labels, dtype=np.int32)

    # Full model: standardized LR over the v6 feature set (fixed seed).
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    Xtr = (X_train - mean) / scale
    Xte = (X_test - mean) / scale

    if np.unique(y_train).size < 2:
        raise ValueError("train split has only one class; cannot fit LR")

    clf = LogisticRegression(
        C=0.3, max_iter=2000, class_weight="balanced", random_state=random_state
    )
    clf.fit(Xtr, y_train)
    pos_col = int(np.where(clf.classes_ == 1)[0][0])
    full_scores = clf.predict_proba(Xte)[:, pos_col]

    # Topic-cosine-only baseline is column 0 of the (unstandardized) features.
    topic_scores = X_test[:, 0]

    topic_acc, n_pairs = _pairwise_accuracy(topic_scores, test_labels)
    full_acc, _ = _pairwise_accuracy(full_scores, test_labels)

    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "topic_acc": topic_acc,
        "full_acc": full_acc,
        "n_test_pairs": n_pairs,
        "train_exemplar_ids": train_exemplar_ids,
        "test_ids": test_ids,
    }


def _load_signed_votes_chronological():
    """Return (rows, labels, timestamps, grades) for the latest signed vote/item.

    Ordered so the newest-vote items are last; chronological split key is the
    vote's ``created_at``.
    """
    from sqlalchemy import select

    from dailydigest.store import ItemRow, VoteRow, init_db, session_scope
    from dailydigest.votes import value_to_grade

    init_db()
    with session_scope() as s:
        raw = s.execute(
            select(VoteRow.item_id, VoteRow.value, VoteRow.grade, VoteRow.created_at, ItemRow)
            .join(ItemRow, VoteRow.item_id == ItemRow.id)
            .order_by(VoteRow.item_id, VoteRow.created_at.desc(), VoteRow.id.desc())
        ).all()
        seen: set[int] = set()
        rows: list = []
        labels: list[int] = []
        timestamps: list[datetime] = []
        grades: list[int] = []
        for item_id, value, grade, created_at, row in raw:
            iid = int(item_id)
            if iid in seen:
                continue
            seen.add(iid)
            v = int(value)
            if v not in (-1, 1):
                continue
            s.expunge(row)
            ts = _as_utc(created_at) or datetime.now(timezone.utc)
            rows.append(row)
            labels.append(v)
            timestamps.append(ts)
            grades.append(int(grade) if grade is not None else value_to_grade(v))
    return rows, labels, timestamps, grades


def main() -> int:
    rows, labels, timestamps, grades = _load_signed_votes_chronological()
    n = len(rows)
    n_pos = sum(1 for v in labels if v > 0)
    n_neg = n - n_pos

    print("=" * 64)
    print("  DailyDigest ranker — leakage-free chronological benchmark")
    print("=" * 64)
    print(f"  signed votes: {n}  (+{n_pos} / -{n_neg})")

    if n < 8 or n_pos < 2 or n_neg < 2:
        print("  INSUFFICIENT DATA: need >=8 signed votes with >=2 of each sign.")
        print("  (real-DB result depends on live votes; nothing to report yet)")
        print("=" * 64)
        return 0

    try:
        result = run_benchmark(rows, labels, timestamps, grades=grades)
    except ValueError as e:
        print(f"  could not run benchmark: {e}")
        print("=" * 64)
        return 0

    # Leakage self-check: no test item may appear among train exemplars.
    leaked = set(result["test_ids"]) & set(result["train_exemplar_ids"])
    topic = result["topic_acc"]
    full = result["full_acc"]
    delta = full - topic

    print(f"  train / test split: {result['n_train']} / {result['n_test']} "
          f"({result['n_test_pairs']} held-out pairs)")
    print(f"  leakage check: {'FAIL' if leaked else 'OK'} "
          f"({len(leaked)} test ids in train exemplars)")
    print("-" * 64)
    print(f"  topic-cosine only  pairwise acc : {topic:.3f}")
    print(f"  full model         pairwise acc : {full:.3f}")
    print(f"  delta (full - topic)            : {delta:+.3f} "
          f"({'model helps' if delta > 0 else 'model does NOT beat baseline'})")
    print("=" * 64)
    print("  NOTE: this number reflects the CURRENT live vote history and is not")
    print("  a fixed guarantee. Re-run after voting to re-measure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
