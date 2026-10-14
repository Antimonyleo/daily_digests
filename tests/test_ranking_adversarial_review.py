"""Independent serving-budget and temporal-feedback evaluation regressions."""

from copy import deepcopy

import pytest

from dailydigest.rank.replay import evaluate_replay, feedback_as_of, frozen_selection, validate_replay


def _snapshot(mode="usual"):
    candidates = []
    for iid in range(1, 20):
        section = "research" if iid <= 15 else ("industry", "ai", "world", "regulatory")[iid - 16]
        candidates.append({
            "item_id": iid, "family_id": str(iid), "section": section,
            "eligible": True, "selected": iid <= 11 or iid >= 16,
            "position": iid if section == "research" else 1,
        })
    return {
        "schema_version": 1, "run_id": "review", "digest_id": "2026-10-02",
        "decision_at": "2026-10-02T12:00:00Z", "reading_mode": mode,
        "complete_candidate_pool": True,
        "section_budgets": {"research": 15, "industry": 1, "ai": 1,
                            "world": 1, "regulatory": 1},
        "candidates": candidates,
    }


def test_usual_challenger_cannot_buy_gains_by_exceeding_reading_budget():
    snap = _snapshot()
    slate = {"research": list(range(1, 16)), "industry": [16], "ai": [17],
             "world": [18], "regulatory": [19]}
    with pytest.raises(ValueError, match="reading.*budget"):
        evaluate_replay(snap, [], slate=slate, outcome_as_of="2026-10-03T00:00:00Z")


def test_minimal_challenger_obeys_research_limit():
    with pytest.raises(ValueError, match="minimal.*research"):
        evaluate_replay(_snapshot("minimal"), [], slate={"research": [1, 2, 3, 4, 5, 6]},
                        outcome_as_of="2026-10-03T00:00:00Z")


def test_full_mode_and_legitimate_underfill_improvement_keep_section_budgets():
    slate = {"research": list(range(1, 16)), "industry": [16], "ai": [17],
             "world": [18], "regulatory": [19]}
    full = evaluate_replay(_snapshot("full"), [], slate=slate,
                           outcome_as_of="2026-10-03T00:00:00Z")
    assert full["sections"]["research"]["served_k"] == 15
    usual = evaluate_replay(_snapshot(), [], slate={"research": list(range(1, 13))},
                            outcome_as_of="2026-10-03T00:00:00Z")
    assert usual["sections"]["research"]["served_k"] == 12


def test_minimal_challenger_obeys_one_pick_per_news_section():
    snap = _snapshot("minimal")
    snap["candidates"][16]["section"] = "industry"
    snap["section_budgets"]["industry"] = 2
    with pytest.raises(ValueError, match="minimal news"):
        evaluate_replay(snap, [], slate={"industry": [16, 17]},
                        outcome_as_of="2026-10-03T00:00:00Z")


@pytest.mark.parametrize("mode,limit", [("usual", 6), ("minimal", 3)])
def test_deadline_picks_keep_their_separate_bounded_budget(mode, limit):
    snap = _snapshot(mode)
    snap["section_budgets"] = {"opportunities": 10}
    for candidate in snap["candidates"]:
        candidate["section"] = "opportunities"
    with pytest.raises(ValueError, match="deadline.*budget"):
        evaluate_replay(snap, [], slate={"opportunities": list(range(1, limit + 2))},
                        outcome_as_of="2026-10-03T00:00:00Z")


def test_empty_and_unrated_replay_keep_unknowns_explicit():
    snap = _snapshot()
    empty = evaluate_replay(snap, [], slate={"research": []}, outcome_as_of="2026-10-03T00:00:00Z")
    assert empty["sections"]["research"]["served_k"] == 0
    assert empty["sections"]["research"]["precision_lower"] is None
    assert empty["sections"]["research"]["must_read_recall_observed_pool"] is None
    unrated = evaluate_replay(snap, [], outcome_as_of="2026-10-03T00:00:00Z")
    assert unrated["sections"]["research"]["unknown"] == 11
    assert unrated["sections"]["research"]["precision_lower"] == 0
    assert unrated["sections"]["research"]["precision_upper"] == 1


def test_feedback_same_timestamp_uses_event_order_without_future_grade_leak():
    first = {"id": 1, "item_id": 1, "value": 1, "grade": 70, "recorded_at": "2026-10-02T10:00:00Z"}
    revision = {**first, "id": 2, "grade": 100}
    neutral = {**first, "id": 3, "value": 0, "grade": 50, "recorded_at": "2026-10-02T11:00:00Z"}
    events = [neutral, revision, first]
    assert feedback_as_of(events, "2026-10-02T10:00:00Z") == {}
    assert feedback_as_of(events, "2026-10-02T11:00:00Z")[1]["grade"] == 100
    assert feedback_as_of(events, "2026-10-02T12:00:00Z") == {}
    assert events == [neutral, revision, first]


def test_declared_incompatible_scorer_does_not_certify_replay():
    snap = deepcopy(_snapshot())
    snap["scorer_version"] = "incompatible-future-policy"
    result = validate_replay(snap, frozen_selection)
    assert result["status"] == "incomplete"
    assert "version" in result["reason"]
