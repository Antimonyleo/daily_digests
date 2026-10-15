"""Shared retrieval regressions backported independently of browser launch."""
from types import SimpleNamespace
import pytest
from dailydigest.ingest import openalex, pubmed
from dailydigest.models import SourceSpec

def _spec(kind="openalex", **kwargs):
    return SourceSpec(name="Topics", kind=kind, section="research", **kwargs)


def _profile(monkeypatch, core, context=()):
    monkeypatch.setattr(
        "dailydigest.config.load_profile",
        lambda: SimpleNamespace(keywords=core, context_keywords=context),
    )


def _work(key):
    return {"id": f"https://openalex.org/{key}", "title": key,
            "publication_date": "2026-10-01"}


def test_openalex_saturated_queries_cover_every_core_before_context(monkeypatch):
    core = [f"topic-{i:02d}" for i in range(10)]
    _profile(monkeypatch, core, ["context-a", "context-b"])
    requests = []

    def get_json(url, params, headers):
        requests.append(dict(params))
        return {"results": [_work(f"{params['search']}-{i}")
                            for i in range(int(params["per-page"]))],
                "meta": {"count": 999, "next_cursor": "more"}}

    monkeypatch.setattr(openalex, "_get_json", get_json)
    source = openalex.OpenAlexSource()
    items = source.fetch(_spec(profile_driven=True))
    assert [r["search"] for r in requests] == core
    assert len(items) == 180
    assert len(requests) == 10
    assert sum(int(r["per-page"]) for r in requests) == 180
    diagnostics = source.last_fetch_diagnostics
    assert all(q["truncated"] for q in diagnostics["queries"] if q["attempted"])
    assert all(not q["attempted"] for q in diagnostics["queries"][-2:])


def test_openalex_query_budget_and_results_are_profile_order_invariant(monkeypatch):
    core = [f"topic-{i:02d}" for i in range(11)]
    requests = []

    def get_json(url, params, headers):
        requests.append((params["search"], int(params["per-page"])))
        return {"results": [_work(f"{params['search']}-{i}")
                            for i in range(int(params["per-page"]))],
                "meta": {"count": 999, "next_cursor": "more"}}

    monkeypatch.setattr(openalex, "_get_json", get_json)
    _profile(monkeypatch, core)
    first = openalex.OpenAlexSource().fetch(_spec(profile_driven=True))
    first_requests = requests[:]
    requests.clear()
    _profile(monkeypatch, list(reversed(core)))
    second = openalex.OpenAlexSource().fetch(_spec(profile_driven=True))
    assert requests == first_requests
    assert {i.external_id for i in first} == {i.external_id for i in second}
    assert len(first) == 180
    assert max(n for _, n in requests) - min(n for _, n in requests) <= 1


def test_openalex_duplicate_heavy_queries_cannot_expand_request_budget(monkeypatch):
    _profile(monkeypatch, [f"topic-{i:02d}" for i in range(10)])
    requests = []

    def get_json(url, params, headers):
        requests.append(params)
        return {"results": [_work("same")] * int(params["per-page"]),
                "meta": {"count": 999, "next_cursor": str(len(requests))}}

    monkeypatch.setattr(openalex, "_get_json", get_json)
    source = openalex.OpenAlexSource()
    items = source.fetch(_spec(profile_driven=True))
    assert len(items) == 1
    assert len(requests) == 10
    assert sum(int(r["per-page"]) for r in requests) <= 180
    assert sum(q["unique_yield"] for q in source.last_fetch_diagnostics["queries"]) == 1


def test_openalex_empty_core_queries_release_budget_to_context(monkeypatch):
    _profile(monkeypatch, ["z-core", "a-core"], ["context"])
    requests = []

    def get_json(url, params, headers):
        term = params["search"]
        requests.append(term)
        return {"results": [] if term.endswith("core") else [_work("context-hit")],
                "meta": {"next_cursor": None}}

    monkeypatch.setattr(openalex, "_get_json", get_json)
    source = openalex.OpenAlexSource()
    items = source.fetch(_spec(profile_driven=True))
    assert requests == ["a-core", "z-core", "context"]
    assert [item.title for item in items] == ["context-hit"]
    assert all(q["exhausted"] for q in source.last_fetch_diagnostics["queries"])


def test_openalex_more_than_soft_limit_of_core_terms_keeps_all_cores(monkeypatch):
    _profile(monkeypatch, [f"topic-{i:02d}" for i in range(13)], ["context"])
    requests = []

    def get_json(url, params, headers):
        requests.append(dict(params))
        return {"results": [_work(f"{params['search']}-{i}")
                            for i in range(int(params["per-page"]))],
                "meta": {"next_cursor": "more"}}

    monkeypatch.setattr(openalex, "_get_json", get_json)
    source = openalex.OpenAlexSource()
    assert len(source.fetch(_spec(profile_driven=True))) == 180
    assert len(requests) == 13
    assert source.last_fetch_diagnostics["request_budget"] == 13
    assert all(r["search"] != "context" for r in requests)


def test_openalex_impossible_core_budget_fails_before_fetch(monkeypatch):
    _profile(monkeypatch, [f"topic-{i}" for i in range(181)])
    requests = []
    monkeypatch.setattr(openalex, "_get_json", lambda *args: requests.append(args))
    with pytest.raises(RuntimeError, match="core queries exceed"):
        openalex.OpenAlexSource().fetch(_spec(profile_driven=True))
    assert requests == []


def test_openalex_later_page_failure_keeps_failure_coverage(monkeypatch):
    calls = 0

    def get_json(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("upstream unavailable")
        return {"results": [_work("one")], "meta": {"count": 2, "next_cursor": "more"}}

    monkeypatch.setattr(openalex, "_get_json", get_json)
    source = openalex.OpenAlexSource()
    with pytest.raises(RuntimeError, match="fetch failed"):
        source.fetch(_spec(query="RNA"))
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["request_count"] == 2
    assert query["fetched_count"] == 1
    assert query["stop_reason"] == "failure"
    assert query["truncated"] and not query["exhausted"]


def test_openalex_repeated_cursor_is_truncation_not_exhaustion(monkeypatch):
    monkeypatch.setattr(openalex, "_get_json", lambda *args: {
        "results": [_work("one")], "meta": {"count": 200, "next_cursor": "same"},
    })
    source = openalex.OpenAlexSource()
    source.fetch(_spec(query="RNA"))
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["request_count"] == 2
    assert query["stop_reason"] == "repeated_cursor"
    assert query["truncated"] and not query["exhausted"]


def test_openalex_missing_cursor_with_more_reported_results_is_truncated(monkeypatch):
    monkeypatch.setattr(openalex, "_get_json", lambda *args: {
        "results": [_work("one")], "meta": {"count": 200},
    })
    source = openalex.OpenAlexSource()
    source.fetch(_spec(query="RNA"))
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["stop_reason"] == "missing_cursor"
    assert query["truncated"] and not query["exhausted"]


def test_pubmed_profile_search_does_not_independently_admit_clinical_trials(monkeypatch):
    _profile(monkeypatch, ["RNA design"])
    queries = []
    monkeypatch.setattr(pubmed.PubMedSource, "_esearch", lambda self, client, query: queries.append(query) or [])
    source = pubmed.PubMedSource()
    assert source.fetch(_spec("pubmed", profile_driven=True, query="clinical trial[pt]")) == []
    assert queries == [
        '("RNA design"[tiab])',
        '(rna[tiab] AND design*[tiab]) NOT ("RNA design"[tiab])',
    ]


def test_pubmed_explicit_nonprofile_query_is_unchanged(monkeypatch):
    queries = []
    monkeypatch.setattr(pubmed.PubMedSource, "_esearch", lambda self, client, query: queries.append(query) or [])
    query = '("RNA"[tiab]) AND clinical trial[pt]'
    assert pubmed.PubMedSource().fetch(_spec("pubmed", query=query)) == []
    assert queries == [query]


def test_pubmed_no_profile_terms_uses_configured_fallback(monkeypatch):
    _profile(monkeypatch, [])
    queries = []
    monkeypatch.setattr(pubmed.PubMedSource, "_esearch", lambda self, client, query: queries.append(query) or [])
    source = pubmed.PubMedSource()
    source.fetch(_spec("pubmed", profile_driven=True, query="RNA[tiab]"))
    assert queries == ["RNA[tiab]"]
    assert source.last_fetch_diagnostics["queries"][0]["exhausted"]
