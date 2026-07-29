#!/usr/bin/env python
"""Offline experiment (Phase 2): does a PER-FACET ranker beat the deployed ranker?

Compares THREE rankers on a leakage-free, production-faithful, chronological,
RESEARCH-ONLY held-out evaluation, then applies a deployment gate:

  1. topic-only (baseline)           — rank by ``_multi_cosine`` topic score.
  2. current deployed v6 ranker      — the 8-feature pairwise LR
     (``votes.LR_FEATURE_NAMES``) fused with topic-cosine via RRF
     (``ranker._fuse_scores``). Exactly what ships.
  3. CANDIDATE per-facet ranker      — a regularized LR whose feature set is
     the item's cosine similarity to EACH core-keyword facet of the profile
     (one feature per ``profile.keywords`` entry, each facet row taken from
     ``build_profile_matrix`` and unit-normalized), OPTIONALLY plus the same
     pos/neg affinity features the deployed ranker uses (TRAIN-only exemplars).
     Trained the SAME way as production — PAIRWISE feature differences on the
     TRAIN split, standardized LogisticRegression (C=0.3, balanced, seed 0) —
     then ranked by RRF-fusing the LR margin with topic-cosine. This is an
     apples-to-apples swap of the LR's FEATURE SET only.

Methodology is inherited verbatim from ``scripts/benchmark_ranker.py`` (studied
and reused): latest signed vote per item, chronological train=older-75% /
test=newer-25%, RESEARCH items only, TRAIN-only affinity/profile exemplars, the
STATIC ``build_profile_matrix`` (never the Rocchio ``learned_profile.npz`` —
that leaks), pairwise construction identical to ``votes.vote_dataset``, and RRF
fusion identical to ``ranker._fuse_scores``. Deterministic (fixed seed).

GATE: recommend adopting the per-facet ranker ONLY if its held-out nDCG@10
exceeds the CURRENT DEPLOYED ranker's nDCG@10 by at least ``GATE_DELTA`` (0.08).
Otherwise: STOP / keep the current ranker. This script only REPORTS the decision;
it does NOT touch production.

    uv run python scripts/experiment_per_facet.py

Exit code is 0 on success regardless of the decision (measurement tool, not CI).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Reuse the audited, leakage-free helpers from the existing benchmark rather than
# re-deriving the split / pairwise / metric logic (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_ranker import (  # noqa: E402
    _build_train_exemplars,
    _load_default_static_profile,
    _load_signed_votes_chronological,
    _ndcg_at_k,
    _pairwise_accuracy,
    _pairwise_training_matrix,
    _split_chronological,
    _fmt,
)

GATE_DELTA = 0.08


# --------------------------------------------------------------------------- #
# Per-facet feature construction
# --------------------------------------------------------------------------- #
def build_facet_matrix() -> tuple[np.ndarray, list[str]]:
    """Return (facets [K, D] unit-normalized, keyword_names) — one row per keyword.

    The facet rows are the SAME rows ``build_profile_matrix`` produces for the
    profile's ``keywords`` list (identical text, identical BGE query embedding),
    unit-normalized so each column of the per-facet feature matrix is a true
    cosine in [-1, 1] independent of the profile's construction weight. Equivalent
    to slicing the keyword block out of the static profile matrix and dividing by
    each row norm; we embed the keyword texts directly (via the same
    ``embed_texts`` that backs ``build_profile_matrix``) so the mapping
    keyword -> column is unambiguous.
    """
    from pathlib import Path as _P

    import yaml

    from dailydigest.config import load_settings
    from dailydigest.models import Profile
    from dailydigest.rank.embed import embed_texts

    settings = load_settings()
    profile_data = yaml.safe_load(_P(settings.profile_path).read_text(encoding="utf-8"))
    profile = Profile(**profile_data)

    keywords = [k.strip() for k in (profile.keywords or []) if k and k.strip()]
    if not keywords:
        return np.zeros((0, 0), dtype=np.float32), []

    vecs = embed_texts(keywords, is_query=True).astype(np.float32)  # [K, D], L2-normed
    # embed_texts already L2-normalizes, but re-normalize defensively.
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    return vecs.astype(np.float32), keywords


def build_perfacet_features(
    rows: list,
    facets: np.ndarray,
    pos_ex,
    neg_ex,
    *,
    with_affinity: bool = True,
) -> np.ndarray:
    """Per-facet feature matrix: item·facet cosine per keyword (+ optional affinity).

    Columns 0..K-1  : cosine(item, keyword_facet_j) for each profile keyword.
    Columns K, K+1  : pos_affinity, neg_affinity (only if ``with_affinity``), built
                      from the GIVEN TRAIN-only exemplars — leakage-free exactly as
                      ``benchmark_ranker._build_features_train_only`` does.
    """
    from dailydigest.rank.embedding_cache import embed_item_rows
    from dailydigest.votes import _affinity

    n = len(rows)
    k = int(facets.shape[0]) if facets.ndim == 2 else 0

    vecs = embed_item_rows(rows)
    if vecs.size == 0 or k == 0:
        base = np.zeros((n, k), dtype=np.float32)
        cand_unit = np.zeros((n, 1), dtype=np.float32)
    else:
        cand_unit = (vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)).astype(
            np.float32
        )
        base = (cand_unit @ facets.T).astype(np.float32)  # [n, K] cosine per facet

    if not with_affinity:
        return base

    cand_ids = [
        int(rid) if isinstance((rid := getattr(r, "id", None)), int) else None for r in rows
    ]
    pos_aff = _affinity(cand_unit, cand_ids, pos_ex).reshape(-1, 1).astype(np.float32)
    neg_aff = _affinity(cand_unit, cand_ids, neg_ex).reshape(-1, 1).astype(np.float32)
    return np.hstack([base, pos_aff, neg_aff]).astype(np.float32)


def _fit_standardized_lr(X: np.ndarray, y: np.ndarray, random_state: int = 0):
    """Fit a standardized LR the SAME way production does (mirrors LRRanker.fit).

    Production's ``LRRanker.fit`` hard-checks ``X.shape[1] == LR_FEATURE_DIM`` (8),
    so it cannot be used for the per-facet feature count. This replicates its exact
    training recipe — z-score standardization with std<1e-6 guarded to 1.0, then
    ``LogisticRegression(C=0.3, class_weight='balanced', random_state=0)`` — minus
    the 8-feature dimensionality assertion. Returns (coef, intercept, mean, scale).
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int32)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    mean = mean.astype(np.float32)
    Xs = (X - mean) / scale
    clf = LogisticRegression(
        C=0.3, max_iter=2000, class_weight="balanced", random_state=random_state
    )
    clf.fit(Xs, y)
    coef = clf.coef_.reshape(-1).astype(np.float32)
    intercept = float(clf.intercept_[0])
    return coef, intercept, mean, scale


def _lr_margin(X: np.ndarray, coef, intercept, mean, scale) -> np.ndarray:
    x = np.asarray(X, dtype=np.float32)
    x = (x - mean) / scale
    return (x @ coef + np.float32(intercept)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Deployed (v6) production-faithful ranker — mirrors benchmark_ranker mode B
# --------------------------------------------------------------------------- #
def eval_deployed(
    rows, labels, timestamps, grades, profile_mat, train_frac=0.75
) -> dict:
    """CURRENT DEPLOYED v6 ranker: 8-feature pairwise LR fused with topic via RRF.

    This is exactly ``benchmark_ranker.run_production_benchmark`` inlined so the
    train/test split (and thus the held-out set) is IDENTICAL to the per-facet
    evaluation — a fair apples-to-apples comparison on the same rows.
    """
    from benchmark_ranker import _build_features_train_only
    from dailydigest.rank.ranker import LRRanker, _fuse_scores

    n = len(rows)
    train_idx, test_idx = _split_chronological(n, timestamps, train_frac)
    pos_ex, neg_ex = _build_train_exemplars(rows, labels, grades, train_idx)
    train_exemplar_ids = np.concatenate([pos_ex[0], neg_ex[0]]).tolist()
    test_ids = [
        int(rid) for i in test_idx if isinstance((rid := getattr(rows[i], "id", None)), int)
    ]

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = np.array([labels[i] for i in test_idx], dtype=np.int32)

    X_train = _build_features_train_only(train_rows, train_labels, profile_mat, pos_ex, neg_ex)
    X_test = _build_features_train_only(
        test_rows, test_labels.tolist(), profile_mat, pos_ex, neg_ex
    )
    y_train = np.array(train_labels, dtype=np.int32)
    if np.unique(y_train).size < 2:
        raise ValueError("train split has only one class; cannot fit deployed LR")

    pair_X, pair_y = _pairwise_training_matrix(X_train, y_train)
    ranker = LRRanker()
    ranker.fit(pair_X, pair_y, persist=False)
    lr_margin = ranker.decision_function(X_test)
    topic_scores = X_test[:, 0].astype(np.float32)
    fused = _fuse_scores(topic_scores, lr_margin)

    topic_acc, n_pairs = _pairwise_accuracy(topic_scores, test_labels)
    dep_acc, _ = _pairwise_accuracy(fused, test_labels)
    topic_ndcg = _ndcg_at_k(topic_scores, test_labels, k=10)
    dep_ndcg = _ndcg_at_k(fused, test_labels, k=10)
    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_pairs": n_pairs,
        "topic_acc": topic_acc,
        "topic_ndcg": topic_ndcg,
        "deployed_acc": dep_acc,
        "deployed_ndcg": dep_ndcg,
        "train_exemplar_ids": train_exemplar_ids,
        "test_ids": test_ids,
    }


# --------------------------------------------------------------------------- #
# CANDIDATE per-facet ranker
# --------------------------------------------------------------------------- #
def eval_perfacet(
    rows,
    labels,
    timestamps,
    grades,
    facets: np.ndarray,
    *,
    with_affinity: bool = True,
    train_frac: float = 0.75,
    random_state: int = 0,
) -> dict:
    """CANDIDATE ranker: per-facet-cosine LR (pairwise) fused with topic via RRF.

    Same split, same TRAIN-only exemplars, same pairwise construction, same RRF
    fusion as the deployed ranker — only the LR's feature set changes (per-keyword
    facet cosines instead of the 8 engineered v6 features). The topic-cosine that
    is fused in is the SAME ``_multi_cosine`` topic score the other two rankers use,
    so the fusion baseline is identical across all three.
    """
    from dailydigest.rank.ranker import _fuse_scores, _multi_cosine
    from dailydigest.rank.embedding_cache import embed_item_rows

    n = len(rows)
    train_idx, test_idx = _split_chronological(n, timestamps, train_frac)
    pos_ex, neg_ex = _build_train_exemplars(rows, labels, grades, train_idx)
    train_exemplar_ids = np.concatenate([pos_ex[0], neg_ex[0]]).tolist()
    test_ids = [
        int(rid) for i in test_idx if isinstance((rid := getattr(rows[i], "id", None)), int)
    ]

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = np.array([labels[i] for i in test_idx], dtype=np.int32)

    X_train = build_perfacet_features(
        train_rows, facets, pos_ex, neg_ex, with_affinity=with_affinity
    )
    X_test = build_perfacet_features(
        test_rows, facets, pos_ex, neg_ex, with_affinity=with_affinity
    )
    y_train = np.array(train_labels, dtype=np.int32)
    if np.unique(y_train).size < 2:
        raise ValueError("train split has only one class; cannot fit per-facet LR")

    pair_X, pair_y = _pairwise_training_matrix(X_train, y_train)
    coef, intercept, mean, scale = _fit_standardized_lr(pair_X, pair_y, random_state)
    lr_margin = _lr_margin(X_test, coef, intercept, mean, scale)

    # Fuse with the SAME topic-cosine (_multi_cosine) baseline the others use.
    prof_mat = _load_default_static_profile()
    test_vecs = embed_item_rows(test_rows)
    if test_vecs.size == 0:
        topic_scores = np.zeros(len(test_rows), dtype=np.float32)
    else:
        topic_scores = _multi_cosine(
            test_vecs, prof_mat if prof_mat.ndim == 2 else prof_mat.reshape(1, -1)
        ).astype(np.float32)

    fused = _fuse_scores(topic_scores, lr_margin)

    pf_acc, n_pairs = _pairwise_accuracy(fused, test_labels)
    pf_ndcg = _ndcg_at_k(fused, test_labels, k=10)
    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_pairs": n_pairs,
        "n_facets": int(facets.shape[0]) if facets.ndim == 2 else 0,
        "with_affinity": with_affinity,
        "perfacet_acc": pf_acc,
        "perfacet_ndcg": pf_ndcg,
        "train_exemplar_ids": train_exemplar_ids,
        "test_ids": test_ids,
    }


def main() -> int:
    print("=" * 70)
    print("  DailyDigest — PER-FACET ranker experiment (Phase 2)")
    print("  leakage-free, chronological, RESEARCH-ONLY, static profile")
    print("=" * 70)

    rows, labels, timestamps, grades = _load_signed_votes_chronological(research_only=True)
    n = len(rows)
    n_pos = sum(1 for v in labels if v > 0)
    n_neg = n - n_pos
    print(f"  signed research votes: {n}  (+{n_pos} / -{n_neg})")

    if n < 8 or n_pos < 2 or n_neg < 2:
        print("  INSUFFICIENT DATA: need >=8 signed research votes with >=2 of each sign.")
        print("  DECISION: STOP — not enough held-out data to evaluate. Keep current ranker.")
        return 0

    profile_mat = _load_default_static_profile()
    facets, keyword_names = build_facet_matrix()
    print(f"  profile keyword facets: {len(keyword_names)}")

    try:
        dep = eval_deployed(rows, labels, timestamps, grades, profile_mat)
    except ValueError as e:
        print(f"  could not evaluate deployed ranker: {e}")
        return 0

    try:
        pf = eval_perfacet(rows, labels, timestamps, grades, facets, with_affinity=True)
        pf_noaff = eval_perfacet(rows, labels, timestamps, grades, facets, with_affinity=False)
    except ValueError as e:
        print(f"  could not evaluate per-facet ranker: {e}")
        return 0

    leaked = set(dep["test_ids"]) & set(dep["train_exemplar_ids"])
    print(
        f"  split {dep['n_train']}/{dep['n_test']}  "
        f"({dep['n_test_pairs']} held-out pairs)  "
        f"leakage: {'FAIL' if leaked else 'OK'}"
    )
    print("-" * 70)
    print("  ranker                     pairwise-acc    nDCG@10")
    print("  " + "-" * 66)
    print(f"  1. topic-only (baseline)   {_fmt(dep['topic_acc']):>10}    {_fmt(dep['topic_ndcg']):>7}")
    print(f"  2. deployed v6 (LR+RRF)    {_fmt(dep['deployed_acc']):>10}    {_fmt(dep['deployed_ndcg']):>7}")
    print(f"  3a. per-facet + affinity   {_fmt(pf['perfacet_acc']):>10}    {_fmt(pf['perfacet_ndcg']):>7}"
          f"   ({pf['n_facets']} facets + 2 aff)")
    print(f"  3b. per-facet (facets only){_fmt(pf_noaff['perfacet_acc']):>10}    {_fmt(pf_noaff['perfacet_ndcg']):>7}"
          f"   ({pf_noaff['n_facets']} facets)")
    print("-" * 70)

    dep_ndcg = dep["deployed_ndcg"]
    pf_ndcg = pf["perfacet_ndcg"]

    def _delta(a: float, b: float) -> str:
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (
            isinstance(b, float) and math.isnan(b)
        ):
            return "n/a"
        return f"{a - b:+.3f}"

    print(f"  per-facet(+aff) nDCG@10 - deployed nDCG@10 : {_delta(pf_ndcg, dep_ndcg)}"
          f"   (gate needs >= +{GATE_DELTA:.2f})  [split {int(round(0.75 * 100))}/25]")

    # ---- ROBUSTNESS: is the gate outcome stable to the split point? --------- #
    # nDCG@10 on a positive-sparse held-out slice is dominated by whether a few
    # liked items happen to fall in the top 10, so a single mandated split can be
    # misleading. Re-run across nearby train fractions and report how often the
    # +GATE_DELTA gate would pass. A robust ADOPT should pass at MOST fractions.
    print("-" * 70)
    print("  robustness sweep (per-facet+aff nDCG@10 - deployed nDCG@10):")
    fracs = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
    passes = 0
    considered = 0
    for frac in fracs:
        try:
            d_f = eval_deployed(rows, labels, timestamps, grades, profile_mat, train_frac=frac)
            p_f = eval_perfacet(
                rows, labels, timestamps, grades, facets, with_affinity=True, train_frac=frac
            )
        except ValueError:
            continue
        dv, pv = d_f["deployed_ndcg"], p_f["perfacet_ndcg"]
        if (
            dv is None or pv is None
            or (isinstance(dv, float) and math.isnan(dv))
            or (isinstance(pv, float) and math.isnan(pv))
        ):
            continue
        considered += 1
        delta = pv - dv
        ok = delta >= GATE_DELTA
        passes += 1 if ok else 0
        _, test_idx_f = _split_chronological(len(rows), timestamps, frac)
        npos_f = sum(1 for i in test_idx_f if labels[i] > 0)
        print(f"    frac {frac:.2f}  n_test={d_f['n_test']:>3} (+{npos_f:>2})  "
              f"deployed={_fmt(dv)}  per-facet={_fmt(pv)}  "
              f"delta={delta:+.3f}  {'PASS' if ok else 'fail'}")
    print(f"  gate PASSES at {passes}/{considered} split fractions "
          f"(sign flips = NOT robust if < all).")
    print("-" * 70)

    # Gate decision.
    gate_pass = (
        dep_ndcg is not None
        and pf_ndcg is not None
        and not (isinstance(dep_ndcg, float) and math.isnan(dep_ndcg))
        and not (isinstance(pf_ndcg, float) and math.isnan(pf_ndcg))
        and (pf_ndcg - dep_ndcg) >= GATE_DELTA
    )
    robust = considered > 0 and passes == considered
    print("=" * 70)
    if gate_pass and robust:
        print(f"  DECISION: ADOPT per-facet ranker — mandated 75/25 split passes "
              f"(nDCG@10 {_fmt(pf_ndcg)} vs deployed {_fmt(dep_ndcg)}, +{pf_ndcg - dep_ndcg:.3f})")
        print(f"  and the gate holds at ALL {considered} nearby split fractions (robust).")
    elif gate_pass and not robust:
        print("  DECISION: STOP (keep current ranker) — DO NOT trust the pass.")
        print("  The mandated 75/25 split passes the gate "
              f"(per-facet nDCG@10 {_fmt(pf_ndcg)} vs deployed {_fmt(dep_ndcg)}, "
              f"+{pf_ndcg - dep_ndcg:.3f}),")
        print("  BUT the sign FLIPS across split fractions (gate passes only "
              f"{passes}/{considered}), so the")
        print("  advantage is a small-sample nDCG artifact, not a durable signal. "
              "Also note the")
        print(f"  per-facet ranker's pairwise acc ({_fmt(pf['perfacet_acc'])}) is WORSE than "
              f"deployed ({_fmt(dep['deployed_acc'])})")
        print(f"  / topic-only ({_fmt(dep['topic_acc'])}) on {dep['n_test_pairs']} pairs — "
              f"the stabler metric disagrees.")
    else:
        print(f"  DECISION: STOP — keep the current ranker "
              f"(per-facet nDCG@10 {_fmt(pf_ndcg)} does NOT beat deployed "
              f"{_fmt(dep_ndcg)} by >= {GATE_DELTA} on the mandated 75/25 split).")
    print("=" * 70)
    print(f"  CAVEAT: test slice = {dep['n_test']} research items, "
          f"{dep['n_test_pairs']} pairs, positive-sparse "
          f"(only {sum(1 for i in _split_chronological(len(rows), timestamps, 0.75)[1] if labels[i] > 0)} "
          f"positives in the 75/25 test set).")
    print("  nDCG@10 is unstable on a slice this small; pairwise accuracy (483 pairs) is")
    print("  the more reliable held-out metric here. Treat the decision as directional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
