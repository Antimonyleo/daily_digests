"""Captured-pool evaluation must not infer missing labels or temporal history."""
from copy import deepcopy

import pytest

from dailydigest.rank.replay import (
    evaluate_replay, feedback_as_of, paired_run_metrics, recorded_slate, validate_replay,
)


def snapshot():
    return {
        "schema_version": 1, "run_id": "run-1", "digest_id": "2026-10-01",
        "decision_at": "2026-10-01T12:00:00+00:00", "reading_mode": "usual",
        "complete_candidate_pool": True, "section_budgets": {"research": 2},
        "candidates": [
            {"item_id": 1, "family_id": "a", "section": "research", "eligible": True,
             "selected": True, "position": 1, "metadata": {"title": "frozen"}},
            {"item_id": 2, "family_id": "b", "section": "research", "eligible": True,
             "selected": True, "position": 2},
            {"item_id": 3, "family_id": "c", "section": "research", "eligible": True,
             "selected": False, "position": None},
        ],
    }


def events():
    return [
        {"id": 1, "item_id": 1, "value": 1, "grade": 100, "recorded_at": "2026-10-01T13:00:00Z"},
        {"id": 2, "item_id": 3, "value": -1, "grade": 10, "recorded_at": "2026-10-01T13:00:00Z"},
        {"id": 3, "item_id": 1, "value": -1, "grade": 10, "recorded_at": "2026-10-02T13:00:00Z"},
    ]


def test_feedback_cutoff_ignores_future_revisions_and_neutral_clears():
    assert feedback_as_of(events(), "2026-10-01T12:00:00Z") == {}
    assert feedback_as_of(events(), "2026-10-02T00:00:00Z")[1]["grade"] == 100
    neutral = {"id": 4, "item_id": 1, "value": 0, "grade": 50, "recorded_at": "2026-10-03T00:00:00Z"}
    assert 1 not in feedback_as_of(events() + [neutral], "2026-10-04T00:00:00Z")


def test_recorded_slate_and_replay_require_independent_selector():
    snap = snapshot()
    assert recorded_slate(snap) == {"research": [1, 2]}
    assert validate_replay(snap)["status"] == "incomplete"
    assert validate_replay(snap, lambda frozen, feedback: {"research": [1, 2]}, events())["status"] == "matched"
    def independent(frozen, feedback):
        assert feedback == {}
        frozen["candidates"][0]["metadata"]["title"] = "mutated"
        return {"research": [2, 1]}
    assert validate_replay(snap, independent, events())["status"] == "mismatch"
    assert snap["candidates"][0]["metadata"]["title"] == "frozen"


def test_evaluate_full_pool_reports_actual_k_unknown_and_conditional_recall():
    report = evaluate_replay(snapshot(), events(), outcome_as_of="2026-10-02T00:00:00Z")
    research = report["sections"]["research"]
    assert research["served_k"] == 2
    assert research["precision_lower"] == 0.5
    assert research["precision_upper"] == 1
    assert research["judged_coverage"] == 0.5
    assert research["graded_utility"] == 3
    assert research["must_read_available"] == 1
    assert research["must_read_recovered"] == 1
    assert report["reading_mode"] == "usual"


def test_replay_rejects_ineligible_unknown_duplicate_family_and_over_budget():
    snap = snapshot()
    for slate in ([99], [1, 1], [1, 2, 3]):
        with pytest.raises(ValueError):
            evaluate_replay(snap, events(), slate={"research": slate}, outcome_as_of="2026-10-02T00:00:00Z")
    snap["candidates"][1]["family_id"] = "a"
    with pytest.raises(ValueError, match="family"):
        evaluate_replay(snap, events(), outcome_as_of="2026-10-02T00:00:00Z")


def test_paired_reports_require_same_run_and_expose_unknowns():
    baseline = evaluate_replay(snapshot(), events(), outcome_as_of="2026-10-02T00:00:00Z")
    challenger = evaluate_replay(snapshot(), events(), slate={"research": [3, 2]}, outcome_as_of="2026-10-02T00:00:00Z")
    result = paired_run_metrics([baseline], [challenger])
    assert result["n_days"] == 1
    assert result["mean_positive_count_delta"] == -1
    assert result["day_bootstrap_interval"] is None
    changed = deepcopy(challenger)
    changed["run_id"] = "other"
    with pytest.raises(ValueError, match="same runs"):
        paired_run_metrics([baseline], [changed])


def test_must_read_denominator_counts_available_families_once():
    snap = snapshot()
    snap["candidates"][2]["family_id"] = "a"
    labels = events()[:1] + [{"id": 9, "item_id": 3, "value": 1, "grade": 100,
                             "recorded_at": "2026-10-01T13:00:00Z"}]
    metrics = evaluate_replay(snap, labels, outcome_as_of="2026-10-02T00:00:00Z")["sections"]["research"]
    assert metrics["must_read_available"] == 1
    assert metrics["must_read_recovered"] == 1
    assert metrics["graded_ndcg_observed_pool"] == 1


def selection_snapshot():
    candidates = []
    for iid in range(1, 25):
        section = "research" if iid <= 20 else ("industry", "ai", "world", "regulatory")[iid - 21]
        chosen = iid <= 11 or iid >= 21
        candidates.append({
            "item_id": iid, "family_id": str(iid), "section": section, "eligible": True,
            "selected": chosen, "position": iid if section == "research" else 1,
            "input_position": iid - 1, "ranking_score": 1 - iid / 100,
            "metadata": {"source": f"Frozen Journal {iid % 5}", "title": f"paper {iid}",
                         "abstract": "", "metadata_json": "{}"},
            "features": {"topic_score": 0.9, "learned_score": 0.8, "scoring_mode": "hybrid_knn"},
            "selection_facts": {"source_bucket": "published_journal", "is_arxiv_cs": False,
                                "is_preprint": False, "is_published": True,
                                "is_high_quality": True, "is_low_impact": False},
        })
    return {**snapshot(), "candidates": candidates,
            "section_budgets": {"research": 15, "industry": 1, "ai": 1, "world": 1, "regulatory": 1},
            "selection_settings": {"max_preprint_research_frac": 0.55,
                                   "max_low_impact_research_frac": 0.2,
                                   "low_impact_relevance_floor": 0.7,
                                   "research_final_score_floor_frac": 0.0,
                                   "research_final_score_min_keep": 3,
                                   "adaptive_relevance_floor": False},
            "decision_local_date": "2026-10-01", "catch_up": False, "exploration_slots": 0}


def test_frozen_selection_replays_real_policy_without_current_settings(monkeypatch):
    from dailydigest import config
    from dailydigest.rank import source_quality
    from dailydigest.rank.replay import frozen_selection
    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not consult current settings or sources")
    monkeypatch.setattr(config, "get_settings", forbidden)
    monkeypatch.setattr(source_quality, "_configured_sources_cached", forbidden)
    snap = selection_snapshot()
    assert validate_replay(snap, frozen_selection)["status"] == "matched"
    snap["candidates"].reverse()
    assert validate_replay(snap, frozen_selection)["status"] == "matched"


def test_v14_replay_keeps_historical_source_facts_after_journal_identity_fix():
    from dailydigest.rank.replay import frozen_selection
    from dailydigest.rank.source_quality import source_bucket
    from types import SimpleNamespace

    snap = selection_snapshot()
    snap["reading_mode"] = "full"
    first = snap["candidates"][0]
    first["metadata"].update(
        source="PubMed (your topics)",
        metadata_json='{"venue":"Angewandte Chemie (International ed. in English)"}',
    )
    # New work recognizes the journal; historical replay must use the recorded
    # unknown-source floor and score, not silently rescore old evidence.
    assert source_bucket(SimpleNamespace(**first["metadata"], section="research")) == "published_journal"
    first["features"]["topic_score"] = .69
    first["selection_facts"].update(
        source_bucket="unknown_journal", is_published=False, is_high_quality=False,
    )
    expected = frozen_selection(snap)
    assert 1 not in expected["research"]
    snap["scorer_version"] = "2026-10-15-local-topic-verification-v14"
    assert frozen_selection(snap) == expected
    snap["scorer_version"] = "unknown-future-selector"
    with pytest.raises(ValueError, match="incompatible"):
        frozen_selection(snap)


def test_frozen_selection_uses_raw_topic_floor_and_source_caps():
    from dailydigest.rank.replay import frozen_selection
    snap = selection_snapshot()
    snap["reading_mode"] = "full"
    first = snap["candidates"][0]
    first["selection_facts"]["is_low_impact"] = True
    first["selection_facts"]["source_bucket"] = "low_impact_journal"
    first["features"]["topic_score"] = 0.1
    assert 1 not in frozen_selection(snap)["research"]
    for candidate in snap["candidates"]:
        candidate["metadata"]["source"] = "One journal"
    assert len(frozen_selection(snap)["research"]) == 3
    assert frozen_selection(snap)["industry"] == [21]


def test_frozen_selection_incomplete_inputs_do_not_claim_parity():
    from dailydigest.rank.replay import frozen_selection
    assert validate_replay(snapshot(), frozen_selection)["status"] == "incomplete"
    snap = selection_snapshot()
    del snap["candidates"][0]["selection_facts"]
    assert validate_replay(snap, frozen_selection)["status"] == "incomplete"


def test_frozen_selection_replays_final_floor_and_exploration():
    from dailydigest.rank.replay import frozen_selection
    snap = selection_snapshot()
    snap["reading_mode"] = "full"
    snap["selection_settings"]["research_final_score_floor_frac"] = 0.99
    snap["selection_settings"]["research_final_score_min_keep"] = 2
    assert frozen_selection(snap)["research"] == [1, 2]
    snap["exploration_slots"] = 1
    snap["candidates"][19]["features"]["learned_score"] = 0.5
    assert frozen_selection(snap)["research"] == [1, 20]


def test_replay_cli_reports_real_selection_parity_and_missing_run(monkeypatch):
    from typer.testing import CliRunner
    from dailydigest import store
    from dailydigest.cli import app
    monkeypatch.setattr(store, "load_ranking_snapshot", lambda run_id: selection_snapshot() if run_id == "run-1" else None)
    result = CliRunner().invoke(app, ["replay-run", "run-1"])
    assert result.exit_code == 0, result.output
    assert '"status": "matched"' in result.output
    missing = CliRunner().invoke(app, ["replay-run", "old-run"])
    assert missing.exit_code == 2
    assert "no immutable snapshot" in missing.output
