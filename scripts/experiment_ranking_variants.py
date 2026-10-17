#!/usr/bin/env python
"""Predeclared offline component contrasts; never selects a production winner.

Requires an explicit copied DB and profile. Model loading is offline-only. The
nine arms isolate representation, learned-profile composition and fusion weight;
this is not a combinatorial parameter search. Current metadata/latest votes are
not historical snapshots. Quality/freshness/downweight/reason adjustments and
serving gates are OMITTED, so no arm is a deployable serving recommendation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np


VARIANTS = (
    {"name": "legacy_appended_equal", "representation": "legacy", "learning": "appended", "preference_weight": 0.5},
    {"name": "legacy_disabled_equal", "representation": "legacy", "learning": "disabled", "preference_weight": 0.5},
    {"name": "canonical_appended_equal", "representation": "canonical", "learning": "appended", "preference_weight": 0.5},
    {"name": "additive_appended_equal", "representation": "additive", "learning": "appended", "preference_weight": 0.5},
    {"name": "additive_disabled_equal", "representation": "additive", "learning": "disabled", "preference_weight": 0.5},
    {"name": "additive_bounded_equal", "representation": "additive", "learning": "bounded", "preference_weight": 0.5},
    {"name": "legacy_appended_preference_075", "representation": "legacy", "learning": "appended", "preference_weight": 0.75},
    {"name": "legacy_appended_semantic_only", "representation": "legacy", "learning": "appended", "preference_weight": 0.0},
    {"name": "knn_only_diagnostic", "representation": "legacy", "learning": "disabled", "preference_weight": 1.0},
)


def _benchmark():
    path = Path(__file__).with_name("benchmark_ranker.py")
    spec = importlib.util.spec_from_file_location("component_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite(value):
    return float(value) if math.isfinite(value) else None


def run_experiments(rows, labels, timestamps, grades, profile_matrices,
                    *, low_impact_floor: float = 0.72, head_ks=(5, 11, 15)) -> dict:
    """Run two disjoint TEST vote-day windows with expanding TRAIN-only learning."""
    from dailydigest.rank.embedding_cache import embed_item_rows
    from dailydigest.rank.profile import append_learned_profile
    from dailydigest.rank.ranker import _fuse_scores, _multi_cosine, profile_relevance_scores
    from dailydigest.rank.source_quality import is_low_impact_research
    from dailydigest.votes import KNN_PREFERENCE_K, _knn_scores

    bench = _benchmark()
    family_ids = bench._paper_family_ids(rows)
    splits = [bench._split_chronological(len(rows), timestamps, fraction, rows=rows)
              for fraction in (0.6, 0.8)]
    later_boundary = min(bench._as_utc(timestamps[i]).astimezone(timezone.utc).date()
                         for i in splits[1][1])
    folds = []
    for fold_index, (train, test) in enumerate(splits):
        if fold_index == 0:
            test = [i for i in test if bench._as_utc(timestamps[i]).astimezone(timezone.utc).date() < later_boundary]
        test = [i for i in test if getattr(rows[i], "section", "research") == "research"]
        if not test:
            folds.append({"fold": fold_index, "status": "unavailable", "reason": "no independent research test families"})
            continue
        train_ids = [int(rows[i].id) for i in train]
        family_counts = Counter(family_ids[i] for i in test)
        test_rows = [rows[i] for i in test]
        test_ids = [int(row.id) for row in test_rows]
        y = np.asarray([labels[i] for i in test])
        g = np.asarray([grades[i] for i in test])
        must = g >= 100
        vecs = embed_item_rows(test_rows).astype(np.float32)
        unit = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        pos, neg = bench._build_train_exemplars(rows, labels, grades, train)
        ex_vecs = np.concatenate([ex[1] for ex in (pos, neg) if len(ex[0])], axis=0)
        knn = _knn_scores(unit, test_ids, np.concatenate([pos[0], neg[0]]), ex_vecs,
                          np.concatenate([pos[2], -neg[2]]), KNN_PREFERENCE_K)
        learned = bench._train_learned_profile(rows, grades, timestamps, train)
        low_impact = np.asarray([is_low_impact_research(row) for row in test_rows])
        results = []
        for variant in VARIANTS:
            mat = profile_matrices[variant["representation"]]
            if variant["learning"] == "appended":
                topic = _multi_cosine(vecs, append_learned_profile(mat, learned, len(train)))
            elif variant["learning"] == "bounded":
                topic = profile_relevance_scores(vecs, mat, learned_vector=learned,
                                                 learned_weight=0.25)
            else:
                topic = _multi_cosine(vecs, mat)
            # Pure preference scores retain ties; the diagnostic never treats
            # absent feedback contrast as evidence of confidence.
            scores = knn if variant["preference_weight"] == 1 else _fuse_scores(
                topic, knn, preference_weight=variant["preference_weight"])
            order = sorted(range(len(test)), key=lambda i: (-float(scores[i]), test_ids[i]))
            below_floor = low_impact & (topic < low_impact_floor)
            head = {}
            for k in head_ks:
                top = order[:k]
                gains = [3 if g[i] >= 100 else 1 if y[i] > 0 else 0 for i in top]
                recovered = sum(bool(must[i]) for i in top)
                head[str(k)] = {
                    "evaluated_k": len(top), "positive_count": sum(y[i] > 0 for i in top).item(),
                    "precision": sum(y[i] > 0 for i in top).item() / len(top),
                    "must_read_recovered": recovered, "must_read_available": int(must.sum()),
                    "must_read_recall": recovered / int(must.sum()) if must.any() else None,
                    "graded_utility": sum(gains),
                    "strong_rejection_count": int(sum(g[i] <= 10 for i in top)),
                    "below_low_impact_topic_floor": int(sum(below_floor[i] for i in top)),
                }
            accuracy, pairs = bench._pairwise_accuracy(scores, y)
            # Every head metric must describe the declared item-ID tie order,
            # rather than the original chronological order of TEST rows.
            ranked_labels = y[order]
            ordered_ndcg = bench._ndcg_at_k(
                np.arange(len(order), 0, -1), ranked_labels, 10,
            )
            results.append({**variant, "head": head,
                            "observed_ndcg_at_10": _finite(ordered_ndcg),
                            "pairwise_accuracy": _finite(accuracy), "pairs": pairs,
                            "pool_below_low_impact_topic_floor": int(below_floor.sum()),
                            "ranking_ids": [test_ids[i] for i in order],
                            "score_signature": hashlib.sha256(np.asarray(scores, dtype=np.float32).tobytes()).hexdigest()})
        folds.append({"fold": fold_index, "status": "complete", "train_ids": train_ids,
                      "test_ids": test_ids, "test_family_count": len(family_counts),
                      "test_repeated_family_count": sum(count > 1 for count in family_counts.values()),
                      "test_extra_family_versions": len(test) - len(family_counts),
                      "train_last_vote_at": max(timestamps[i] for i in train).isoformat(),
                      "test_first_vote_at": min(timestamps[i] for i in test).isoformat(),
                      "test_last_vote_at": max(timestamps[i] for i in test).isoformat(),
                      "variants": results})
    return {
        "scope": "exploratory rated-pool component contrasts, not full serving replay",
        "split": "whole UTC latest-vote days; family-purged; disjoint 60-80% and 80-100% TEST windows",
        "limitations": ["current profile and metadata; overwritten vote history cannot reconstruct historical decisions",
                        "all TEST items already rated; labels are exposure-selected",
                        "quality/freshness/downweight/reason adjustments, eligibility and source/reading caps omitted",
                        "quality-floor counts are diagnostics on an unchanged floor, not enforced eligibility",
                        "must-read recall conditional on observed TEST items; not retrieval recall",
                        "crossfold families are blocked; multiple versions within TEST can remain separate rated items",
                        "5/11/15 are diagnostic depths; 11 approximates ordinary research load, not every historical run",
                        "no winner selected; these data are exploratory, not a fresh promotion holdout"],
        "bounded_weight": 0.25, "low_impact_floor": low_impact_floor,
        "profile_matrix_signatures": {mode: hashlib.sha256(matrix.tobytes()).hexdigest()
                                      for mode, matrix in sorted(profile_matrices.items())},
        "folds": folds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="Explicit disposable copied database")
    parser.add_argument("--profile", required=True, type=Path, help="Explicit frozen profile YAML")
    parser.add_argument("--output", required=True, type=Path, help="Local JSON report")
    args = parser.parse_args()
    # Fail closed to cached models, before importing any application embedding code.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DB_PATH"] = str(args.db.resolve())
    os.environ["PROFILE_PATH"] = str(args.profile.resolve())
    from dailydigest.config import load_profile, get_settings
    from dailydigest.rank.profile import build_profile_matrix

    profile = load_profile()
    matrices = {mode: build_profile_matrix(profile, mode=mode) for mode in ("legacy", "canonical", "additive")}
    rows, labels, timestamps, grades = _benchmark()._load_signed_votes_chronological()
    result = run_experiments(rows, labels, timestamps, grades, matrices,
                             low_impact_floor=float(get_settings().low_impact_relevance_floor))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Offline component report: {args.output}")
    print("No winner or production configuration selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
