"""Retrieval coverage requires targets independent of the captured candidates."""
from copy import deepcopy

import pytest

from dailydigest.rank.retrieval_evaluation import evaluate_retrieval, prepare_openalex_manifest


def manifest():
    return {
        "window_start": "2026-10-01", "window_end": "2026-10-22",
        "topic_ids": ["catalysis", "ecology"],
        "targets": [{
            "id": "a", "doi": "https://doi.org/10.1234/ABC", "topic_ids": ["catalysis"],
            "published_at": "2026-10-01",
            "provenance": {"method": "independent_reader", "reference": "reader-list-1"},
        }],
        "captures": [{
            "id": "query-1", "source": "index", "topic_ids": ["catalysis"],
            "status": "complete", "truncated": False, "official": True,
            "items": [{"url": "https://doi.org/10.1234/abc", "abstract": "Some evidence."}],
        }],
    }


def test_doi_identity_normalization_and_unknown_topics_are_separate():
    report = evaluate_retrieval(manifest())
    assert report["target_coverage"] == {"found": 1, "total": 1, "fraction": 1.0}
    assert report["topics"]["ecology"]["request_health"] == "unknown"
    assert report["request_health"] == "unknown"
    assert report["targets"][0]["matched_capture_ids"] == ["query-1"]
    assert report["release_approved"] is False


def test_explicit_aliases_match_but_equal_titles_do_not():
    data = manifest()
    data["targets"][0]["aliases"] = ["https://example.org/paper?a=1"]
    data["captures"][0]["items"] = [{
        "url": "https://EXAMPLE.org/paper?a=1&utm_source=mail#abstract", "title": "Shared title",
    }]
    assert evaluate_retrieval(data)["target_coverage"]["found"] == 1
    data["targets"][0].pop("aliases")
    data["targets"][0]["title"] = "Shared title"
    assert evaluate_retrieval(data)["target_coverage"]["found"] == 0


def test_date_window_is_inclusive_and_unknown_dates_not_silently_excluded():
    data = manifest()
    for key, date in [("b", "2026-10-22"), ("c", "2026-09-30"), ("d", None)]:
        item = deepcopy(data["targets"][0])
        item.update(id=key, doi=f"10.1234/{key}", published_at=date)
        data["targets"].append(item)
    report = evaluate_retrieval(data)
    assert report["target_coverage"] == {"found": 1, "total": 2, "fraction": .5}
    assert report["excluded_outside_window"] == ["c"]
    assert report["unresolved_date"] == ["d"]
    assert report["missing_target_ids"] == ["b"]


def test_multi_topic_target_is_counted_once_and_channel_attribution_is_explicit():
    data = manifest()
    data["targets"][0]["topic_ids"].append("ecology")
    report = evaluate_retrieval(data)
    assert report["target_coverage"]["total"] == 1
    assert report["topics"]["ecology"]["target_coverage"]["found"] == 1
    assert report["topics"]["ecology"]["found_via_own_queries"] == 0
    assert report["topics"]["catalysis"]["found_via_own_queries"] == 1


@pytest.mark.parametrize("status,truncated,expected", [
    (None, None, "unknown"), ("complete", None, "unknown"),
    ("partial", True, "degraded"), ("failed", False, "degraded"),
])
def test_missing_health_or_truncation_metadata_cannot_become_success(status, truncated, expected):
    data = manifest()
    data["captures"][0].update(status=status, truncated=truncated)
    assert evaluate_retrieval(data)["captures"][0]["health"] == expected


def test_official_empty_results_and_abstract_completeness_are_reported_honestly():
    data = manifest()
    data["captures"][0]["items"] = []
    report = evaluate_retrieval(data)
    assert report["captures"][0]["official_empty_result"] is True
    assert report["captures"][0]["abstract_completeness"]["fraction"] is None
    assert report["missing_target_ids"] == ["a"]
    data["captures"][0]["status"] = "failed"
    assert evaluate_retrieval(data)["captures"][0]["official_empty_result"] is False


def test_no_targets_no_claim_of_recall_and_unknown_capture_health():
    data = manifest()
    data.update(targets=[], captures=[])
    report = evaluate_retrieval(data)
    assert report["target_coverage"]["fraction"] is None
    assert report["request_health"] == "unknown"


@pytest.mark.parametrize("mutation,match", [
    (lambda data: data["targets"][0]["provenance"].update(method="captured_pool"), "independent"),
    (lambda data: data["targets"].append(deepcopy(data["targets"][0])), "Duplicate"),
    (lambda data: data["targets"][0].update(topic_ids=["unknown"]), "topic"),
    (lambda data: data["captures"][0].update(topic_ids=["unknown"]), "topic"),
    (lambda data: data["captures"][0].update(truncated="false"), "truncated"),
    (lambda data: data["targets"][0].update(published_at="not-a-date"), "date"),
])
def test_invalid_or_nonindependent_inputs_fail_closed(mutation, match):
    data = manifest()
    mutation(data)
    with pytest.raises(ValueError, match=match):
        evaluate_retrieval(data)


def test_paper_cannot_inflate_target_denominator_under_another_id():
    data = manifest()
    duplicate = deepcopy(data["targets"][0])
    duplicate["id"] = "another-id"
    data["targets"].append(duplicate)
    with pytest.raises(ValueError, match="identity"):
        evaluate_retrieval(data)


def test_conflicting_item_dates_are_visible_without_erasing_verified_target_date():
    data = manifest()
    data["captures"][0]["items"][0]["published_at"] = "2026-09-01"
    report = evaluate_retrieval(data)
    assert report["target_coverage"]["found"] == 1
    assert report["targets"][0]["publication_date_conflicts"] == ["query-1"]


@pytest.mark.parametrize("evidence", [{"total_available": 2}, {"next_cursor": "next-page"}])
def test_truncation_evidence_cannot_be_overridden_by_success(evidence):
    data = manifest()
    data["captures"][0].update(evidence)
    with pytest.raises(ValueError, match="conflicts"):
        evaluate_retrieval(data)
    data["captures"][0].pop("truncated")
    capture = evaluate_retrieval(data)["captures"][0]
    assert capture["truncated"] is True
    assert capture["health"] == "degraded"


def test_utc_window_and_valid_but_encoded_path_are_not_conflated():
    data = manifest()
    data["targets"][0].update(published_at="2026-10-01T00:30:00+02:00")
    assert evaluate_retrieval(data)["excluded_outside_window"] == ["a"]
    data = manifest()
    data["targets"][0].update(doi="", url="https://example.org/a%2Fb")
    data["captures"][0]["items"] = [{"url": "https://example.org/a/b"}]
    assert evaluate_retrieval(data)["target_coverage"]["found"] == 0


def test_null_reference_is_not_evidence_of_independent_target_selection():
    data = manifest()
    data["targets"][0]["provenance"]["reference"] = None
    with pytest.raises(ValueError, match="independent"):
        evaluate_retrieval(data)


def test_alias_query_uses_explicit_parent_topic_without_semantic_guessing():
    data = manifest()
    data["captures"][0]["query"] = "organocatalyst"
    report = evaluate_retrieval(data)
    assert report["captures"][0]["query"] == "organocatalyst"
    assert report["topics"]["catalysis"]["found_via_own_queries"] == 1
    assert report["topics"]["ecology"]["found_via_own_queries"] == 0


def health_rows():
    return [{"source": "OpenAlex topics", "diagnostics": {"queries": [{
        "interest": "catalysis", "query": "catalysis OR organocatalyst",
        "filter": "from_publication_date:2026-10-01,to_publication_date:2026-10-22,type:article",
        "attempted": True, "request_count": 1, "fetched_count": 1, "returned_count": 1,
        "total_available": 1, "exhausted": True, "truncated": False,
        "stop_reason": "exhausted", "failure": None,
    }]}}]


def test_health_only_producer_preserves_missing_identity_evidence_instead_of_empty_success():
    result = prepare_openalex_manifest(health_rows(), topic_map={"catalysis": ["chem"]})
    assert result["manifest"] is None
    assert result["prepared_captures"][0]["items"] is None
    assert result["gaps"] == [{"capture_id": "OpenAlex topics:0",
                               "reason": "per_query_item_identities_unavailable"}]


def test_producer_maps_real_query_items_and_original_window_without_guessing_alias_topics():
    target = manifest()["targets"][0]
    target["topic_ids"] = ["chem"]
    result = prepare_openalex_manifest(
        health_rows(), topic_map={"catalysis": ["chem"]}, targets=[target],
        query_items={"OpenAlex topics:0": [{"doi": "10.1234/abc", "abstract": "Evidence"}]})
    assert result["status"] == "ready"
    assert result["manifest"]["window_start"] == "2026-10-01"
    assert evaluate_retrieval(result["manifest"])["target_coverage"]["found"] == 1


@pytest.mark.parametrize("failed", [False, True])
def test_recorded_zero_can_be_reconstructed_but_failed_zero_is_not_official_empty(failed):
    rows = health_rows()
    rows[0]["diagnostics"]["queries"][0].update(
        fetched_count=0, returned_count=0, total_available=0,
        failure="Timeout" if failed else None, truncated=failed)
    result = prepare_openalex_manifest(rows, topic_map={"catalysis": ["chem"]})
    report = evaluate_retrieval(result["manifest"])
    assert report["captures"][0]["official_empty_result"] is (not failed)


def test_missing_original_window_is_not_reconstructed_from_candidate_publication_dates():
    rows = health_rows()
    rows[0]["diagnostics"]["queries"][0].pop("filter")
    result = prepare_openalex_manifest(
        rows, topic_map={"catalysis": ["chem"]},
        query_items={"OpenAlex topics:0": [{"doi": "10.1234/abc", "published_at": "2026-10-12"}]})
    assert result["manifest"] is None
    assert any(gap["reason"] == "original_request_window_unavailable" for gap in result["gaps"])


def test_mixed_original_windows_and_unmapped_interests_block_preparation():
    rows = health_rows()
    other = deepcopy(rows[0])
    other["source"] = "OpenAlex other"
    other["diagnostics"]["queries"][0].update(
        interest="unmapped", fetched_count=0, returned_count=0,
        filter="from_publication_date:2026-09-01,to_publication_date:2026-09-22")
    result = prepare_openalex_manifest(rows + [other], topic_map={"catalysis": ["chem"]})
    assert result["manifest"] is None
    reasons = {gap["reason"] for gap in result["gaps"]}
    assert {"parent_topic_unmapped", "mixed_or_unknown_original_request_windows"} <= reasons


def test_producer_rejects_post_deduplication_item_counts_and_extra_capture_ids():
    with pytest.raises(ValueError, match="pre-deduplication"):
        prepare_openalex_manifest(health_rows(), topic_map={"catalysis": ["chem"]},
                                  query_items={"OpenAlex topics:0": []})
    with pytest.raises(ValueError, match="absent"):
        prepare_openalex_manifest(health_rows(), topic_map={"catalysis": ["chem"]},
                                  query_items={"unrelated-query": []})


def test_actual_adapter_diagnostics_feed_preparation_with_parse_exclusions(monkeypatch):
    from dailydigest.ingest import openalex
    from dailydigest.models import SourceSpec

    # Two fetched works, one discarded for lacking a title. Provider count is
    # not the normalized Item count and must not imply another unseen page.
    payload = {"meta": {"count": 2}, "results": [
        {"id": "https://openalex.org/W1", "title": "Catalysis result", "doi": "https://doi.org/10.1234/a",
         "publication_date": "2026-10-01"},
        {"id": "https://openalex.org/W2", "title": None},
    ]}
    monkeypatch.setattr(openalex, "_get_json", lambda *_args: payload)
    source = openalex.OpenAlexSource()
    items = source.fetch(SourceSpec(name="OpenAlex", kind="openalex", section="research", query="catalysis"))
    result = prepare_openalex_manifest(
        [{"source": "OpenAlex", "diagnostics": source.last_fetch_diagnostics}],
        topic_map={"catalysis": ["chem"]},
        query_items={"OpenAlex:0": [item.model_dump(mode="json") for item in items]})
    assert result["status"] == "ready"
    capture = evaluate_retrieval(result["manifest"])["captures"][0]
    assert capture["returned_count"] == 1
    assert capture["truncated"] is False
    assert capture["health"] == "complete"
    assert result["prepared_captures"][0]["provider_total_available"] == 2


@pytest.mark.parametrize("target_url,capture_url", [
    ("https://arxiv.org/abs/2401.12345v1", "https://arxiv.org/abs/2401.12345"),
    ("https://arxiv.org/pdf/2401.12345v2.pdf", "https://export.arxiv.org/abs/2401.12345v1"),
    ("https://arxiv.org/abs/cond-mat/9901234v1", "https://arxiv.org/pdf/cond-mat/9901234.pdf"),
    ("https://arxiv.org/abs/math.GT/0301234v2", "https://arxiv.org/abs/math.GT/0301234"),
    ("https://doi.org/10.48550/arXiv.2401.12345", "https://arxiv.org/pdf/2401.12345v3"),
    ("https://doi.org/10.48550/arXiv.cond-mat/9901234", "https://arxiv.org/abs/cond-mat/9901234v2"),
])
def test_arxiv_native_links_and_explicit_dois_share_one_version_family(target_url, capture_url):
    data = manifest()
    data["targets"][0].update(doi="", url=target_url)
    data["captures"][0]["items"] = [{"url": capture_url}]
    assert evaluate_retrieval(data)["target_coverage"]["found"] == 1


@pytest.mark.parametrize("host", ["arxiv.org.evil.example", "notarxiv.org", "arxiv.org@evil.example"])
def test_lookalike_arxiv_hosts_cannot_match_a_native_family(host):
    data = manifest()
    data["targets"][0].update(doi="", url="https://arxiv.org/abs/2401.12345v1")
    data["captures"][0]["items"] = [{"url": f"https://{host}/abs/2401.12345v2"}]
    assert evaluate_retrieval(data)["target_coverage"]["found"] == 0


def test_arxiv_family_duplicates_cannot_inflate_targets_across_versions_or_doi():
    data = manifest()
    data["targets"][0].update(doi="", url="https://arxiv.org/abs/2401.12345v1")
    duplicate = deepcopy(data["targets"][0])
    duplicate.update(id="other-version", url="https://doi.org/10.48550/arXiv.2401.12345v2")
    data["targets"].append(duplicate)
    with pytest.raises(ValueError, match="identity"):
        evaluate_retrieval(data)
