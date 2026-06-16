"""Tests for the offline ranking evaluation harness."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dailydigest import store as store_mod
from dailydigest import votes as votes_mod
from dailydigest.rank import evaluate as eval_mod
from dailydigest.rank.evaluate import (
    _average_precision,
    _ndcg_at_k,
    _pairwise_accuracy,
    _precision_at_k,
    evaluate_history,
)


def _insert_item(title: str, section: str = "research") -> int:
    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Test",
            section=section,
            external_id=title,
            url=f"https://example.com/{title}",
            title=title,
            abstract="abstract",
            published_at=datetime.now(timezone.utc),
        )
        s.add(row)
        s.flush()
        return int(row.id)


# --- pure metric units -------------------------------------------------------


def test_ndcg_perfect_order_is_one():
    assert _ndcg_at_k([1, 1, 0, 0], k=10) == pytest.approx(1.0)


def test_ndcg_undefined_without_relevant():
    assert _ndcg_at_k([0, 0, 0], k=10) is None


def test_precision_at_k():
    assert _precision_at_k([1, 0, 1, 0], k=4) == pytest.approx(0.5)
    assert _precision_at_k([1, 1, 0, 0], k=2) == pytest.approx(1.0)


def test_average_precision():
    # relevant at ranks 1 and 3 → (1/1 + 2/3) / 2
    assert _average_precision([1, 0, 1, 0]) == pytest.approx((1.0 + 2 / 3) / 2)


def test_pairwise_accuracy():
    # labels in rank order: +,-,+,-  → 3 of 4 (up,down) pairs correct
    ranked = [(1, 0), (-1, 1), (1, 2), (-1, 3)]
    assert _pairwise_accuracy(ranked) == pytest.approx(0.75)


def test_pairwise_undefined_without_both_signs():
    assert _pairwise_accuracy([(1, 0), (1, 1)]) is None


# --- end-to-end replay over a persisted digest -------------------------------


def test_evaluate_history_scores_persisted_digest():
    a = _insert_item("A")
    b = _insert_item("B")
    c = _insert_item("C")
    d = _insert_item("D")

    # Persisted ranking order by descending final_score: A > B > C > D
    store_mod.write_digest_features(
        "2026-01-01",
        [
            ("R1", a, 0.9, {}),
            ("R2", b, 0.8, {}),
            ("R3", c, 0.7, {}),
            ("R4", d, 0.6, {}),
        ],
    )
    # Feedback: A,C liked; B,D disliked
    votes_mod.record_vote_by_id(a, 1)
    votes_mod.record_vote_by_id(c, 1)
    votes_mod.record_vote_by_id(b, -1)
    votes_mod.record_vote_by_id(d, -1)

    report = evaluate_history(k=10)

    assert report.n_digests_scored == 1
    assert report.n_votes == 4
    assert report.precision_at_k == pytest.approx(0.5)
    assert report.pairwise_accuracy == pytest.approx(0.75)
    assert report.map_score == pytest.approx((1.0 + 2 / 3) / 2)
    # nDCG for gains [1,0,1,0] vs ideal [1,1,0,0]
    assert report.ndcg_at_k == pytest.approx(0.9197, abs=1e-3)


def test_evaluate_history_skips_unvoted_digests():
    a = _insert_item("A")
    store_mod.write_digest_features("2026-01-02", [("R1", a, 0.9, {})])

    report = evaluate_history(k=10)
    assert report.n_digests_total == 1
    assert report.n_digests_scored == 0
    assert report.ndcg_at_k is None


def test_evaluate_history_empty_store():
    report = evaluate_history(k=10)
    assert report.n_digests_scored == 0
    assert report.as_dict()["ndcg_at_k"] is None
