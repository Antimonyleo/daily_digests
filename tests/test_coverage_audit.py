"""Offline coverage audits separate observable losses from retrieval uncertainty."""

import copy

from dailydigest.rank.coverage_audit import audit_coverage


def _candidate(i, **changes):
    row = {"item_id": i, "section": "research", "eligible": True,
           "selected": False, "first_rejected_stage": None,
           "identity_keys": [f"doi:10.9999/{i}"],
           "metadata": {"source": "Topics", "title": "Engineering proteins",
                        "abstract": "a" * 120, "authors": "A Author",
                        "published_at": "2026-10-01", "metadata_json": "{}"}}
    row.update(changes)
    return row


def test_metadata_audit_does_not_call_short_abstracts_truncated():
    row = _candidate(1)
    row["metadata"]["abstract"] = "Short result."
    report = audit_coverage({"candidates": [row], "source_health": []})
    assert report["research_metadata"]["short_nonempty_abstract"] == 1
    assert report["source_health"]["status"] == "unavailable"
    assert "truncated" not in report["research_metadata"]


def test_known_papers_are_attributed_to_observed_stage_or_local_absence():
    rows = [_candidate(1, selected=True),
            _candidate(2, eligible=False, first_rejected_stage="after_topic_gate"),
            _candidate(3)]
    report = audit_coverage({"candidates": rows}, known_papers=[
        {"item_id": 1}, {"item_id": 2}, {"item_id": 3}, {"item_id": 9},
        {"item_id": 10, "identity_keys": ["doi:10.9999/1"]},
    ])
    statuses = [r["status"] for r in report["known_papers"]["papers"]]
    assert statuses == ["selected", "rejected", "eligible_unselected",
                        "absent_from_local_pool", "selected"]
    assert report["known_papers"]["papers"][1]["first_rejected_stage"] == "after_topic_gate"
    assert report["known_papers"]["papers"][3]["retrieval_miss_confirmed"] is False


def test_missing_metadata_is_counted_by_source_without_mutating_inputs():
    row = _candidate(1)
    row["metadata"].update(abstract="", published_at=None, metadata_json="invalid")
    snapshot = {"candidates": [row, _candidate(2, section="world")]}
    before = copy.deepcopy(snapshot)
    report = audit_coverage(snapshot)
    assert report["research_metadata"]["missing_abstract"] == 1
    assert report["research_metadata"]["missing_date"] == 1
    assert report["research_metadata"]["invalid_metadata_json"] == 1
    assert report["by_source"]["Topics"]["count"] == 1
    assert snapshot == before


def test_health_audit_separates_failures_truncation_and_unmeasured_delay():
    snapshot = {"candidates": [], "source_health": [
        {"source": "Topics", "ok": True, "diagnostics": {"queries": [
            {"attempted": True, "truncated": True, "stop_reason": "item_cap",
             "indexing_delay": None},
            {"attempted": False, "truncated": True, "stop_reason": "item_budget"},
        ]}},
        {"source": "Broken", "ok": False, "diagnostics": {"queries": [
            {"attempted": True, "failure": "TimeoutError", "truncated": True},
        ]}},
        {"source": "Legacy", "ok": True},
    ]}
    health = audit_coverage(snapshot)["source_health"]
    assert health["source_failures"] == 1
    assert health["query_failures"] == 1
    assert health["unattempted_queries"] == 1
    assert health["truncated_queries"] == 3
    assert health["sources_without_query_diagnostics"] == ["Legacy"]
    assert health["indexing_delay_measured_queries"] == 0


def test_duplicate_known_versions_use_best_observed_state():
    rows = [_candidate(1, eligible=False, first_rejected_stage="after_identity_dedupe"),
            _candidate(2, selected=True, identity_keys=["doi:10.9999/1"])]
    report = audit_coverage({"candidates": rows}, known_papers=[{"item_id": 1}])
    assert report["known_papers"]["papers"][0]["status"] == "selected"
    assert report["known_papers"]["papers"][0]["matched_item_ids"] == [1, 2]


def test_facet_input_completeness_distinguishes_pool_eligibility_and_selection():
    rows = [_candidate(1, selected=True, features={"primary_facet": "Protein design", "secondary_facets": ["RNA", "Protein design"]}),
            _candidate(2, eligible=False, features={"primary_facet": "RNA"}),
            _candidate(3, features={})]
    rows[1]["metadata"]["abstract"] = ""
    snapshot = {"profile": {"canonical_facets": {"Protein design": {}, "RNA": {}, "Colloids": {}}},
                "candidates": rows}
    before = copy.deepcopy(snapshot)
    facets = audit_coverage(snapshot)["by_facet"]
    assert facets["RNA"]["raw"]["count"] == 2
    assert facets["RNA"]["raw"]["missing_abstract"] == 1
    assert facets["RNA"]["eligible"]["count"] == 1
    assert facets["RNA"]["selected"]["count"] == 1
    assert facets["Colloids"]["raw"]["count"] == 0
    assert facets["unknown"]["raw"]["count"] == 1
    assert facets["Protein design"]["raw"]["count"] == 1
    assert "not" in audit_coverage(snapshot)["facet_limit"]
    assert snapshot == before


def test_unknown_facet_does_not_invent_match_from_topic_words():
    report = audit_coverage({"candidates": [_candidate(1)], "profile": {"keywords": ["Protein design"]}})
    assert report["by_facet"]["Protein design"]["raw"]["count"] == 0
    assert report["by_facet"]["unknown"]["raw"]["count"] == 1
