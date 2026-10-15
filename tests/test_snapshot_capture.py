"""Adversarial contracts for frozen selection evidence, without model/network IO."""
from datetime import date, datetime, timezone
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dailydigest.rank import snapshot as snapshot_mod


def _row(i, *, title=None, doi=None):
    return SimpleNamespace(
        id=i, section="research", source="Nature", external_id=f"paper-{i}",
        url=f"https://example.org/paper-{i}", title=title or f"Distinct research result {i}",
        abstract=f"Unique evidence for method {i}", authors="A. Researcher",
        published_at=datetime(2026, 10, 1, 12, tzinfo=timezone.utc),
        fetched_at=datetime(2026, 10, 2, 8, tzinfo=timezone.utc),
        metadata_json=json.dumps({"doi": doi} if doi else {}),
    )


@pytest.fixture
def capture_args(monkeypatch):
    from dailydigest.config import get_settings
    from dailydigest.models import Profile

    monkeypatch.setattr(snapshot_mod, "active_embedding_signature",
                        lambda **kwargs: "frozen-query" if kwargs.get("is_query") else "frozen-doc")
    rows = [_row(i) for i in range(1, 132)]
    scored = [(row, 0.9 - i / 1000) for i, row in enumerate(rows)]
    # Keep rejected and selected-tail evidence beyond the old impression limit.
    eligible = scored[1:]
    selected = eligible[-1:]
    settings = get_settings().model_copy(update={
        "llm_api_key": "DO-NOT-ARCHIVE-THIS-KEY",
        "db_path": "/private/path/must-not-appear.db", "exploration_slots": 0,
    })
    return dict(
        run_id="capture-run", digest_id="2026-10-02",
        decision_at=datetime(2026, 10, 2, 12, tzinfo=timezone.utc),
        decision_local_date=date(2026, 10, 2), reading_mode="full",
        candidates=rows, scored=scored, eligible=eligible,
        labeled=[(row, score, "R1") for row, score in selected],
        features={row.id: {"topic_score": 0.8, "confidence_score": 0.7,
                           "rank_score": score, "scoring_mode": "cosine"}
                  for row, score in scored},
        section_budgets={"research": 1}, settings=settings,
        profile=Profile(bio="I research molecular structures.", keywords=["RNA"]),
        profile_matrix=np.array([[1.0, 0.0]], dtype=np.float32), catch_up=False,
        ingest_stats=[], feedback_before={"max_event_id": 0, "votes": []},
        feedback_after={"max_event_id": 0, "votes": []},
        stages={"after_reviewed": rows[1:], "after_quality_gate": rows[1:]},
    )


def test_full_pool_frozen_inputs_do_not_truncate_or_alias_mutable_state(capture_args):
    frozen = snapshot_mod.build_ranking_snapshot(**capture_args)
    encoded = json.dumps(frozen)
    assert len(frozen["candidates"]) == 131
    assert frozen["complete_candidate_pool"] is True
    assert frozen["candidates"][0]["first_rejected_stage"] == "after_reviewed"
    assert frozen["candidates"][0]["eligible"] is False
    assert frozen["candidates"][-1]["selected"] is True
    assert frozen["candidates"][-1]["position"] == 0
    assert "DO-NOT-ARCHIVE" not in encoded and "/private/path" not in encoded
    assert frozen["embedding_signature"] == "frozen-doc"
    assert frozen["query_embedding_signature"] == "frozen-query"
    original_title = frozen["candidates"][0]["metadata"]["title"]
    capture_args["candidates"][0].title = "changed after decision"
    capture_args["features"][1]["topic_score"] = -1.0
    capture_args["profile"].keywords.append("new interest")
    capture_args["profile_matrix"][0, 0] = 2
    capture_args["section_budgets"]["research"] = 99
    capture_args["feedback_before"]["max_event_id"] = 50
    assert frozen["candidates"][0]["metadata"]["title"] == original_title
    assert frozen["candidates"][0]["features"]["topic_score"] == 0.8
    assert frozen["profile"]["keywords"] == ["RNA"]
    assert frozen["profile_matrix"] == [[1.0, 0.0]]
    assert frozen["section_budgets"] == {"research": 1}
    assert frozen["feedback_at_start"]["max_event_id"] == 0
    assert json.loads(encoded) == frozen


def test_rejected_candidate_retains_its_latest_semantic_score(capture_args):
    iid = capture_args['candidates'][0].id
    capture_args['features'][iid].update(final_score=.41, rank_score=.41,
                                        contribution_rerank={'rank_signal': .2})
    frozen = snapshot_mod.build_ranking_snapshot(**capture_args)
    rejected = next(c for c in frozen['candidates'] if c['item_id'] == iid)
    assert rejected['eligible'] is False
    assert rejected['ranking_score'] == .41
    assert rejected['features']['final_score'] == rejected['ranking_score']


def test_paper_families_union_transitive_doi_and_title_aliases():
    from itertools import permutations

    a = _row(1, title="Distinct first title", doi="10.1234/first")
    b = _row(2, title="Shared alias title", doi="10.1234/first")
    c = _row(3, title="Shared alias title", doi="10.1234/second")
    d = _row(4, title="Unrelated last title", doi="10.1234/unrelated")
    expected = snapshot_mod._families([a, b, c, d])
    assert expected[1] == expected[2] == expected[3]
    assert expected[4] != expected[1]
    for order in permutations([a, b, c, d]):
        assert snapshot_mod._families(order) == expected


def test_feedback_changes_are_explicit_not_silently_asof(capture_args):
    capture_args["feedback_after"] = {"max_event_id": 1, "votes": [
        {"item_id": 999, "value": 1, "grade": 100,
         "created_at": "2026-10-02T12:00:01+00:00"},
    ]}
    frozen = snapshot_mod.build_ranking_snapshot(**capture_args)
    assert frozen["feedback_consistent"] is False
    assert frozen["feedback_cutoff"] == frozen["decision_at"]
    assert frozen["feedback_at_start"]["max_event_id"] == 0
    assert frozen["feedback_at_end"]["max_event_id"] == 1


def test_captured_selection_inputs_replay_without_current_profile_or_settings(capture_args, monkeypatch):
    from dailydigest.rank.replay import frozen_selection, validate_replay
    from dailydigest.rank.ranker import pick_top_per_section
    from dailydigest.pipeline import apply_reading_mode

    picked = pick_top_per_section(
        capture_args["eligible"], capture_args["section_budgets"],
        score_features=capture_args["features"], settings=capture_args["settings"],
    )
    picked = apply_reading_mode(picked, "full", as_of_date=capture_args["decision_local_date"])
    capture_args["labeled"] = [(row, score, f"R{i}") for i, (row, score) in enumerate(picked, 1)]
    frozen = snapshot_mod.build_ranking_snapshot(**capture_args)
    roundtrip = json.loads(json.dumps(frozen))
    # No runtime source knowledge or settings should be necessary after capture.
    from dailydigest import config
    from dailydigest.rank import ranker
    monkeypatch.setattr(config, "get_settings", lambda: pytest.fail("consulted current settings"))
    for name in ("source_bucket", "is_preprint_source", "is_arxiv_cs_source",
                 "is_published_journal_source", "is_high_quality_journal_source",
                 "is_low_impact_research"):
        monkeypatch.setattr(snapshot_mod.quality, name, lambda _row: pytest.fail("consulted source facts"))
        # Selection no longer has a field-specific CS preprint dependency.
        if name != "is_arxiv_cs_source":
            monkeypatch.setattr(ranker, name, lambda _row: pytest.fail("consulted source facts"))
    assert validate_replay(roundtrip, frozen_selection)["status"] == "matched"


def test_capture_rejects_nonfinite_evidence(capture_args):
    capture_args["features"][1]["topic_score"] = float("nan")
    with pytest.raises(ValueError):
        snapshot_mod.build_ranking_snapshot(**capture_args)


def test_capture_freezes_exact_enrichment_inputs_and_adjustments(capture_args):
    evidence = {"doi": "10.1234/one", "cited_by_count": 12,
                "venue_impact": 9.0, "venue": "Small", "citation_boost": .08,
                "venue_adjustment": 0.0, "quality_resolved_before_scoring": True}
    capture_args["candidates"][0].enrichment_evidence = evidence
    frozen = snapshot_mod.build_ranking_snapshot(**capture_args)
    assert frozen["ranking_settings"]["venue_quality_weight"] == capture_args["settings"].venue_quality_weight
    assert frozen["candidates"][0]["features"]["enrichment"] == evidence
    evidence["cited_by_count"] = 100
    assert frozen["candidates"][0]["features"]["enrichment"]["cited_by_count"] == 12


def test_previous_upstream_scorer_version_retains_frozen_selection_replay(capture_args):
    from dailydigest.rank.replay import frozen_selection, validate_replay

    frozen = snapshot_mod.build_ranking_snapshot(**capture_args)
    selected = frozen_selection(frozen)
    for candidate in frozen["candidates"]:
        ids = selected.get(candidate["section"], [])
        candidate["selected"] = candidate["item_id"] in ids
        candidate["position"] = ids.index(candidate["item_id"]) if candidate["selected"] else None
    frozen["scorer_version"] = "2026-10-02-quality-guards-v10"
    assert validate_replay(frozen, frozen_selection)["status"] == "matched"


def test_supplied_selection_facts_are_frozen_without_reconsulting_sources(capture_args, monkeypatch):
    facts = snapshot_mod.capture_selection_facts(capture_args["candidates"])
    expected = dict(facts[1])
    capture_args["selection_facts"] = facts
    monkeypatch.setattr(snapshot_mod, "capture_selection_facts",
                        lambda _rows: pytest.fail("recomputed captured source facts"))
    for name in ("source_bucket", "is_preprint_source", "is_arxiv_cs_source",
                 "is_published_journal_source", "is_high_quality_journal_source",
                 "is_low_impact_research"):
        monkeypatch.setattr(snapshot_mod.quality, name,
                            lambda _row: pytest.fail("consulted current source facts"))
    snapshot = snapshot_mod.build_ranking_snapshot(**capture_args)
    assert snapshot["candidates"][0]["selection_facts"] == expected
    facts[1]["source_bucket"] = "changed-after-capture"
    assert snapshot["candidates"][0]["selection_facts"] == expected
