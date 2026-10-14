#!/usr/bin/env python
"""Exploratory chronological component diagnostics on current rated-item metadata.

Split at whole UTC vote-day boundaries; purge paper families crossing the split.
Training exemplars and learned profile use TRAIN only. Freeze feature time at the
split boundary. Current metadata/profile and overwritten latest votes cannot
reconstruct historical decisions. These are rated-pool component probes, not a
full serving replay, unbiased candidate recall, or evidence for promotion.

A: pointwise affinity-feature probe. B: static topic, current fusion, kNN-only
(diagnostic only), and fusion with an actual TRAIN-only Rocchio profile. Run on a
copied DB/profile with cached embeddings for isolated offline experiments.
"""
from __future__ import annotations

import math
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
    *,
    reference_time: datetime,
) -> np.ndarray:
    """Recompute the v6 LR feature matrix, sourcing affinity from GIVEN exemplars.

    Mirrors `votes._build_item_features` column-for-column, but takes the pos/neg
    exemplar arrays as arguments instead of loading them from the whole DB. That
    prevents exemplar self-recognition: pass TRAIN-only exemplars
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

    now = _as_utc(reference_time)
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


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """nDCG@k for a single ranked list; label +1 -> gain 1, else 0.

    Ranks items by ``scores`` (desc, stable) and compares the resulting gain
    sequence to the ideal ordering. Returns float('nan') when no positive label
    exists (nDCG undefined). Mirrors ``rank.evaluate._ndcg_at_k`` gains.
    """
    n = len(labels)
    if n == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ranked_gains = [1.0 if labels[i] > 0 else 0.0 for i in order[:k]]
    ideal = sorted((1.0 if lbl > 0 else 0.0 for lbl in labels), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg <= 0:
        return float("nan")
    return _dcg(ranked_gains) / idcg


def _precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of the top-k that the reader actually liked.

    The digest shows ~``top_research`` items, so this — not full-list pairwise
    accuracy — is the metric that matches what the reader sees. Reporting only
    pairwise accuracy is how a head-of-list regression stayed invisible.
    """
    n = len(labels)
    if n == 0 or k <= 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    top = order[: min(k, n)]
    return float(sum(1 for i in top if labels[i] > 0) / len(top))


def _recall_at_k(scores: np.ndarray, wanted: np.ndarray, k: int) -> float:
    """Fraction of the ``wanted`` items that land in the top-k.

    Used for grade-100 ("Must read") recall: the reader's complaint was that
    important papers were missed, which is a recall-at-the-head question that
    pairwise accuracy cannot answer.
    """
    wanted = np.asarray(wanted, dtype=bool)
    total = int(wanted.sum())
    if total == 0 or k <= 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    top = order[: min(k, len(wanted))]
    return float(sum(1 for i in top if wanted[i]) / total)


def _paper_family_ids(rows: list) -> list[int]:
    """Transitive identities shared by split purging and component diagnostics."""
    from dailydigest.dedupe import candidate_identity_keys, _normalized_title_key

    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tokens = {}
    for i, row in enumerate(rows):
        title = _normalized_title_key(str(getattr(row, "title", "") or ""))
        # Include metadata-only DOI, PubMed IDs and arXiv versions. Retain
        # date-agnostic title identity across differently dated paper versions.
        keys = candidate_identity_keys(row) + (["title:" + title] if title else [])
        for key in keys:
            if key in tokens:
                parent[root(i)] = root(tokens[key])
            else:
                tokens[key] = i
    return [root(i) for i in range(len(rows))]


def _split_chronological(
    n: int, timestamps: list[datetime], train_frac: float, *, rows: list | None = None,
) -> tuple[list[int], list[int]]:
    """Nearest whole UTC-day boundary; purge TEST families also seen in TRAIN."""
    if len(timestamps) != n or not 0 < train_frac < 1:
        raise ValueError("timestamps must align and train_frac must be between 0 and 1")
    stamps = [_as_utc(stamp).astimezone(timezone.utc) for stamp in timestamps]
    order = sorted(range(n), key=lambda i: (stamps[i], i))
    cuts = [i for i in range(1, n) if stamps[order[i-1]].date() != stamps[order[i]].date()]
    if not cuts:
        raise ValueError("chronological benchmark needs two distinct UTC vote days")
    cut = min(cuts, key=lambda i: (abs(i - n * train_frac), i))
    train, test = order[:cut], order[cut:]
    if rows is not None:
        families = _paper_family_ids(rows)
        train_families = {families[i] for i in train}
        test = [i for i in test if families[i] not in train_families]
    if not test:
        raise ValueError("no independent held-out families after purging")
    return train, test


def _train_learned_profile(rows, grades, timestamps, train_idx):
    """Pure mirror of production grade-weighted, newest-first Rocchio rebuild."""
    from dailydigest.rank.embedding_cache import embed_item_rows
    from dailydigest.votes import _ROCCHIO_ALPHA, _ROCCHIO_BETA, _ROCCHIO_DECAY, grade_to_weight

    indices = sorted(train_idx, key=lambda i: (timestamps[i], i), reverse=True)
    vecs = embed_item_rows([rows[i] for i in indices]).astype(np.float32)
    learned = np.zeros(vecs.shape[1], dtype=np.float32)
    for rank, i in enumerate(indices):
        weight = grade_to_weight(grades[i])
        rate = _ROCCHIO_ALPHA if weight >= 0 else _ROCCHIO_BETA
        learned += rate * weight * _ROCCHIO_DECAY ** rank * vecs[rank]
    return learned


def _build_train_exemplars(
    rows: list, labels: list[int], grades: list[int], train_idx: list[int]
):
    """Pack TRAIN-only (pos, neg) grade-weighted exemplar tuples (leakage-free)."""
    from dailydigest.votes import VOTE_GRADE_NEUTRAL

    pos_rows, pos_w, neg_rows, neg_w = [], [], [], []
    for i in train_idx:
        w = max(0.05, abs(grades[i] - VOTE_GRADE_NEUTRAL) / 50.0)
        if labels[i] > 0:
            pos_rows.append(rows[i])
            pos_w.append(w)
        else:
            neg_rows.append(rows[i])
            neg_w.append(w)
    return _pack_exemplars(pos_rows, pos_w), _pack_exemplars(neg_rows, neg_w)


def _load_default_static_profile() -> np.ndarray:
    """Load the STATIC config profile matrix (no Rocchio, no learned_profile.npz).

    Using the static profile avoids a persisted feedback-vector leak: ``build_profile_matrix`` is a pure function of the config
    profile file, so it cannot encode the held-out test votes the way the
    vote-derived Rocchio vector (rebuilt from ALL votes) would.
    """
    from pathlib import Path

    import yaml

    from dailydigest.config import load_settings
    from dailydigest.models import Profile
    from dailydigest.rank.profile import build_profile_matrix  # STATIC — no Rocchio

    settings = load_settings()
    profile_data = yaml.safe_load(
        Path(settings.profile_path).read_text(encoding="utf-8")
    )
    profile = Profile(**profile_data)
    return build_profile_matrix(profile)


def run_benchmark(
    rows: list,
    labels: list[int],
    timestamps: list[datetime],
    grades: list[int] | None = None,
    profile_mat: np.ndarray | None = None,
    train_frac: float = 0.75,
    random_state: int = 0,
) -> dict:
    """PREFERENCE-FEATURE PROBE (mode A) — NOT the deployed ranker.

    Trains a POINTWISE LogisticRegression on the v6 features and evaluates its
    probability directly. This measures whether the pos/neg affinity features
    help a simple pointwise model separate held-out liked/disliked items — a
    feature sanity probe. The deployed default (``hybrid_knn``) fuses the graded
    kNN preference score with topic-cosine via RRF, and pairwise LR is only the
    retired/optional ``hybrid_lr`` mode; for the deployed-style measurement, use
    :func:`run_production_benchmark`.

    ``rows`` are item-like objects (real ItemRow or SimpleNamespace with ``id``,
    ``title``, ``abstract``, ``published_at``), ``labels`` are +1/-1 per row,
    ``timestamps`` are the signing vote's time (chronological order key).

    The default (``profile_mat is None``) path uses the STATIC config profile
    matrix (``build_profile_matrix``), never the Rocchio ``learned_profile.npz``,
    so the matrix excludes learned held-out labels (current YAML is not historical).

    Returns a dict with keys: n_train, n_test, topic_acc, full_acc, n_test_pairs,
    train_exemplar_ids, test_ids.
    """
    from sklearn.linear_model import LogisticRegression

    from dailydigest.votes import value_to_grade

    n = len(rows)
    if grades is None:
        grades = [value_to_grade(v) for v in labels]

    train_idx, test_idx = _split_chronological(n, timestamps, train_frac, rows=rows)
    reference_time = min(timestamps[i] for i in test_idx)

    if profile_mat is None:
        profile_mat = _load_default_static_profile()

    # Build TRAIN-only exemplars for the affinity features (leakage-free).
    pos_ex, neg_ex = _build_train_exemplars(rows, labels, grades, train_idx)

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
        train_rows, train_labels, profile_mat, pos_ex, neg_ex, reference_time=reference_time
    )
    X_test = _build_features_train_only(
        test_rows, test_labels.tolist(), profile_mat, pos_ex, neg_ex, reference_time=reference_time
    )
    y_train = np.array(train_labels, dtype=np.int32)

    # Full model: standardized POINTWISE LR over the v6 feature set (fixed seed).
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


def _pairwise_training_matrix(
    X: np.ndarray, y: np.ndarray, max_pairs: int = 300, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate ``votes.vote_dataset`` pairwise construction on a given (X, y).

    For each sampled (up, down) index pair, emit the feature difference
    ``X[up] - X[down]`` with label +1 and its negation with label -1, so the LR
    learns the gradient from disliked to liked. Sampling and the ± balancing
    mirror ``_sample_training_pairs`` / ``vote_dataset`` exactly (fixed seed).
    """
    import itertools

    up_indices = [i for i, lbl in enumerate(y) if lbl == 1]
    down_indices = [i for i, lbl in enumerate(y) if lbl == -1]
    n_up, n_down = len(up_indices), len(down_indices)
    total = n_up * n_down
    if total == 0:
        raise ValueError("pairwise construction needs both +1 and -1 examples")

    if total <= max_pairs:
        pairs = list(itertools.product(up_indices, down_indices))
    else:
        rng = np.random.default_rng(seed)  # fixed seed → reproducible (matches vote_dataset)
        chosen = rng.choice(total, size=max_pairs, replace=False)
        pairs = [
            (up_indices[int(i) // n_down], down_indices[int(i) % n_down]) for i in chosen
        ]

    pairs_X: list[np.ndarray] = []
    pairs_y: list[float] = []
    for ui, di in pairs:
        diff = X[ui] - X[di]
        pairs_X.append(diff)
        pairs_y.append(1.0)
        pairs_X.append(-diff)
        pairs_y.append(-1.0)
    return np.array(pairs_X, dtype=np.float32), np.array(pairs_y, dtype=np.float32)


def run_production_benchmark(
    rows: list,
    labels: list[int],
    timestamps: list[datetime],
    grades: list[int] | None = None,
    profile_mat: np.ndarray | None = None,
    train_frac: float = 0.75,
) -> dict:
    """Rated-pool component probe with train-only feedback and learned profile.

    TRAIN may include all sections (as deployed exemplars do); TEST is research.
    This omits eligibility, quality adjustments, enrichment, source caps and
    reading budgets, so results do not estimate the quality of a served digest.
    The default static matrix uses current YAML, never a persisted learned file.
    """
    from dailydigest.rank.ranker import LRRanker, _fuse_scores
    from dailydigest.votes import value_to_grade

    n = len(rows)
    if grades is None:
        grades = [value_to_grade(v) for v in labels]

    train_idx, test_idx = _split_chronological(n, timestamps, train_frac, rows=rows)
    family_purged = n - len(train_idx) - len(test_idx)
    reference_time = min(timestamps[i] for i in test_idx)
    test_idx = [i for i in test_idx if getattr(rows[i], "section", "research") == "research"]
    if not test_idx:
        raise ValueError("no research items in held-out whole-day split")

    if profile_mat is None:
        profile_mat = _load_default_static_profile()

    pos_ex, neg_ex = _build_train_exemplars(rows, labels, grades, train_idx)
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
        train_rows, train_labels, profile_mat, pos_ex, neg_ex, reference_time=reference_time
    )
    X_test = _build_features_train_only(
        test_rows, test_labels.tolist(), profile_mat, pos_ex, neg_ex, reference_time=reference_time
    )
    y_train = np.array(train_labels, dtype=np.int32)

    if np.unique(y_train).size < 2:
        raise ValueError("train split has only one class; cannot fit pairwise LR")

    # PAIRWISE training as the retired/optional ``hybrid_lr`` mode does
    # (votes.vote_dataset), then fit the same standardized LRRanker it uses. This
    # feeds only the comparison line below; the deployed default is kNN fusion.
    pair_X, pair_y = _pairwise_training_matrix(X_train, y_train)
    ranker = LRRanker()
    ranker.fit(pair_X, pair_y, persist=False)

    # DEPLOYED ranking (2026-08-18-knn-preference-v7): RRF-fuse the topic-cosine
    # ranking (feature col 0) with the graded kNN preference score computed from
    # TRAIN-only votes (leakage-free). The retired pairwise-LR fusion is kept as
    # a comparison line — on live votes it had drifted BELOW topic-only.
    # Production fuses quality-adjusted scores. This component probe uses raw
    # semantic scores; omitted source/quality adjustments can change item order.
    from dailydigest.rank.embedding_cache import embed_item_rows
    from dailydigest.votes import KNN_PREFERENCE_K, _knn_scores

    ex_ids = np.concatenate([pos_ex[0], neg_ex[0]])
    ex_vecs = np.concatenate(
        [v for v in (pos_ex[1], neg_ex[1]) if v.shape[0]], axis=0
    )
    gw = np.concatenate([pos_ex[2], -neg_ex[2]]).astype(np.float32)
    test_vecs = embed_item_rows(test_rows).astype(np.float32)
    test_unit = test_vecs / (np.linalg.norm(test_vecs, axis=1, keepdims=True) + 1e-9)
    knn_scores = _knn_scores(
        test_unit,
        [getattr(r, "id", None) for r in test_rows],
        ex_ids,
        ex_vecs,
        gw,
        KNN_PREFERENCE_K,
    )

    lr_margin = ranker.decision_function(X_test)
    topic_scores = X_test[:, 0].astype(np.float32)
    fused_scores = _fuse_scores(topic_scores, knn_scores)
    lr_fused_scores = _fuse_scores(topic_scores, lr_margin)

    # HEAD metrics are reported for EVERY configuration. The digest serves about
    # `top_research` items, so a config can win full-list pairwise accuracy while
    # losing the head — which is exactly what happened when only `fused_ndcg` was
    # reported and the LR's nDCG@10 silently dropped out of the comparison.
    try:
        from dailydigest.config import get_settings

        head_k = int(getattr(get_settings(), "top_research", 10)) or 10
    except Exception:  # noqa: BLE001
        head_k = 10
    test_grades = np.array([grades[i] for i in test_idx], dtype=np.float32)
    must_read = test_grades >= 100

    topic_acc, n_pairs = _pairwise_accuracy(topic_scores, test_labels)
    fused_acc, _ = _pairwise_accuracy(fused_scores, test_labels)
    lr_fused_acc, _ = _pairwise_accuracy(lr_fused_scores, test_labels)

    def _head(scores: np.ndarray) -> dict[str, float]:
        return {
            "ndcg": _ndcg_at_k(scores, test_labels, k=10),
            "p_at_k": _precision_at_k(scores, test_labels, head_k),
            "must_read_recall": _recall_at_k(scores, must_read, head_k),
        }

    topic_head = _head(topic_scores)
    fused_head = _head(fused_scores)
    lr_head = _head(lr_fused_scores)
    from dailydigest.rank.profile import append_learned_profile
    from dailydigest.rank.ranker import _multi_cosine

    learned = _train_learned_profile(rows, grades, timestamps, train_idx)
    learned_mat = append_learned_profile(profile_mat, learned, len(train_idx))
    learned_scores = _multi_cosine(test_vecs, learned_mat)
    learned_fused = _fuse_scores(learned_scores, knn_scores)
    diagnostic = {}
    for name, scores in (("knn_only", knn_scores), ("learned_fused", learned_fused)):
        diagnostic[name + "_acc"] = _pairwise_accuracy(scores, test_labels)[0]
        diagnostic.update({name + "_" + key: value for key, value in _head(scores).items()})

    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_pairs": n_pairs,
        "head_k": head_k,
        "n_must_read": int(must_read.sum()),
        **diagnostic,
        "learned_train_ids": train_exemplar_ids,
        "decision_time_proxy": reference_time.isoformat(),
        "split_basis": "whole UTC latest-vote days; crossing title/DOI/URL families purged",
        "n_family_purged": family_purged,
        "limitations": "current metadata/profile and latest votes; not historical serving replay; rated-pool conditional metrics",
        "head_k_basis": "configured diagnostic K, not historical actual served K",
        "must_read_basis": "observed held-out must-reads only; not retrieval recall",
        "topic_acc": topic_acc,
        "fused_acc": fused_acc,
        "lr_fused_acc": lr_fused_acc,
        "topic_ndcg": topic_head["ndcg"],
        "fused_ndcg": fused_head["ndcg"],
        "lr_fused_ndcg": lr_head["ndcg"],
        "topic_p_at_k": topic_head["p_at_k"],
        "fused_p_at_k": fused_head["p_at_k"],
        "lr_fused_p_at_k": lr_head["p_at_k"],
        "topic_must_read_recall": topic_head["must_read_recall"],
        "fused_must_read_recall": fused_head["must_read_recall"],
        "lr_fused_must_read_recall": lr_head["must_read_recall"],
        "train_exemplar_ids": train_exemplar_ids,
        "test_ids": test_ids,
    }


def _load_signed_votes_chronological(research_only: bool = False):
    """Return (rows, labels, timestamps, grades) for the latest signed vote/item.

    Ordered so the newest-vote items are last; chronological split key is the
    vote's ``created_at``. When ``research_only`` is True, only items whose
    ``section == "research"`` are returned (the section production's LR path
    ranks).
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
            if research_only and str(getattr(row, "section", "") or "").lower() != "research":
                continue
            s.expunge(row)
            ts = _as_utc(created_at)
            if ts is None:
                continue  # Missing time cannot support a chronological claim.
            rows.append(row)
            labels.append(v)
            timestamps.append(ts)
            grades.append(int(grade) if grade is not None else value_to_grade(v))
    return rows, labels, timestamps, grades


def _fmt(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main() -> int:
    print("=" * 68)
    print("  DailyDigest ranker — chronological component diagnostics")
    print("  (whole UTC vote days; train-only feedback; current metadata)")
    print("=" * 68)

    # ------------------------------------------------------------------ #
    # Mode A: preference-FEATURE PROBE (pointwise LR on v6 features).
    # ------------------------------------------------------------------ #
    rows, labels, timestamps, grades = _load_signed_votes_chronological()
    n = len(rows)
    n_pos = sum(1 for v in labels if v > 0)
    n_neg = n - n_pos
    print("  [A] preference-feature probe  (all sections)")
    print(f"      signed votes: {n}  (+{n_pos} / -{n_neg})")

    if n < 8 or n_pos < 2 or n_neg < 2:
        print("      INSUFFICIENT DATA: need >=8 signed votes with >=2 of each sign.")
    else:
        try:
            a = run_benchmark(rows, labels, timestamps, grades=grades)
            leaked_a = set(a["test_ids"]) & set(a["train_exemplar_ids"])
            print(
                f"      split {a['n_train']}/{a['n_test']} "
                f"({a['n_test_pairs']} held-out pairs)  "
                f"item-ID overlap: {'FAIL' if leaked_a else 'none'}"
            )
            print(f"      topic-cosine only   pairwise acc : {_fmt(a['topic_acc'])}")
            print(f"      pointwise+affinity  pairwise acc : {_fmt(a['full_acc'])}")
            print(
                f"      delta                            : "
                f"{a['full_acc'] - a['topic_acc']:+.3f}"
            )
            print("      NOTE: probe of the affinity FEATURES, NOT the deployed ranker.")
        except ValueError as e:
            print(f"      could not run probe: {e}")

    print("-" * 68)

    # ------------------------------------------------------------------ #
    # Mode B: deployed-scorer probe (graded kNN preference + RRF), research only.
    # ------------------------------------------------------------------ #
    r_rows, r_labels, r_ts, r_grades = _load_signed_votes_chronological(research_only=True)
    rn = len(r_rows)
    rn_pos = sum(1 for v in r_labels if v > 0)
    rn_neg = rn - rn_pos
    print("  [B] component probe (all-section TRAIN, research TEST)")
    print(f"      signed research votes: {rn}  (+{rn_pos} / -{rn_neg})")

    if rn < 8 or rn_pos < 2 or rn_neg < 2:
        print("      INSUFFICIENT DATA: need >=8 signed research votes, >=2 each sign.")
    else:
        try:
            b = run_production_benchmark(rows, labels, timestamps, grades=grades)
            leaked_b = set(b["test_ids"]) & set(b["train_exemplar_ids"])
            print(
                f"      split {b['n_train']}/{b['n_test']} "
                f"({b['n_test_pairs']} held-out pairs)  "
                f"item-ID overlap: {'FAIL' if leaked_b else 'none'}"
            )
            k = b["head_k"]
            print(f"      {'config':<14s} {'pw acc':>7s} {'nDCG@10':>8s} "
                  f"{'P@' + str(k):>7s} {'mustR@' + str(k):>8s}")
            for label, pre in (
                ("topic-only", "topic"),
                ("kNN fuse", "fused"),
                ("LR fuse", "lr_fused"),
                ("kNN diagnostic", "knn_only"),
                ("learned fuse", "learned_fused"),
            ):
                print(f"      {label:<14s} {_fmt(b[pre + '_acc']):>7s} "
                      f"{_fmt(b[pre + '_ndcg']):>8s} {_fmt(b[pre + '_p_at_k']):>7s} "
                      f"{_fmt(b[pre + '_must_read_recall']):>8s}")
            print(f"      (head metrics at K={k}; {b['n_must_read']} must-read items held out)")
            print("      Pairwise acc scores the WHOLE list; nDCG/P@K/mustR@K score the")
            print("      configured diagnostic K, not historical actual served K. Unknown")
            print("      candidate labels/old vote revisions prevent exact historical replay.")
            print(f"      Family overlap purged: {b['n_family_purged']}; freeze: {b['decision_time_proxy']}")
        except ValueError as e:
            print(f"      could not run production benchmark: {e}")

    print("=" * 68)
    print("  These diagnostics reflect CURRENT latest votes and are not fixed")
    print("  guarantees. Current metadata/profile; latest labels cannot recover old votes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
