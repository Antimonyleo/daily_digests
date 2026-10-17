"""Offline retrieval budgets and truthful query coverage diagnostics."""

from types import SimpleNamespace

import pytest

from dailydigest.ingest import biorxiv, openalex, pubmed
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


def _preprint(key):
    return {"doi": f"10.1101/{key}", "title": "Engineered molecular assembly",
            "date": "2026-10-01", "abstract": "We build a programmable device."}


def test_biorxiv_cap_preserves_query_coverage_diagnostics(monkeypatch):
    monkeypatch.setattr(biorxiv, "_MAX_ROWS", 2)
    monkeypatch.setattr(biorxiv, "_get_json", lambda *args: {
        "collection": [_preprint("one"), _preprint("two")],
        "messages": [{"total": 10}],
    })
    source = biorxiv.BiorxivSource()
    assert len(source.fetch(_spec("biorxiv"))) == 2
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["fetched_count"] == 2
    assert query["total_available"] == 10
    assert query["truncated"] and not query["exhausted"]
    assert query["stop_reason"] == "item_cap"


def test_biorxiv_unknown_total_cannot_claim_complete_coverage(monkeypatch):
    monkeypatch.setattr(biorxiv, "_get_json", lambda *args: {
        "collection": [_preprint("one")], "messages": [{"total": 0}],
    })
    source = biorxiv.BiorxivSource()
    assert len(source.fetch(_spec("biorxiv"))) == 1
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["stop_reason"] == "unknown_total"
    assert query["truncated"] and not query["exhausted"]


def test_biorxiv_unexpected_empty_page_is_not_success(monkeypatch):
    pages = iter([
        {"collection": [_preprint("one")], "messages": [{"total": 2}]},
        {"collection": [], "messages": [{"total": 2}]},
    ])
    monkeypatch.setattr(biorxiv, "_get_json", lambda *args: next(pages))
    source = biorxiv.BiorxivSource()
    with pytest.raises(RuntimeError, match="unexpectedly empty"):
        source.fetch(_spec("biorxiv"))
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["request_count"] == 2
    assert query["fetched_count"] == 1
    assert query["failure"] == "inconsistent_empty_page"


def test_biorxiv_later_transport_failure_retains_progress(monkeypatch):
    def fetch(client, url):
        if url.endswith("/1"):
            raise RuntimeError("upstream failed")
        return {"collection": [_preprint("one")], "messages": [{"total": 2}]}

    monkeypatch.setattr(biorxiv, "_get_json", fetch)
    source = biorxiv.BiorxivSource()
    with pytest.raises(RuntimeError, match="fetch failed"):
        source.fetch(_spec("biorxiv"))
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["fetched_count"] == 1
    assert query["stop_reason"] == "failure"
    assert query["truncated"]


def test_biorxiv_complete_window_is_distinct_from_withdrawn_filtering(monkeypatch):
    withdrawn = {**_preprint("two"), "type": "withdrawn"}
    monkeypatch.setattr(biorxiv, "_get_json", lambda *args: {
        "collection": [_preprint("one"), withdrawn], "messages": [{"total": 2}],
    })
    source = biorxiv.BiorxivSource()
    assert len(source.fetch(_spec("biorxiv"))) == 1
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["exhausted"] and not query["truncated"]
    assert query["skipped_withdrawn"] == 1
    assert query["fetched_count"] == 2 and query["returned_count"] == 1


@pytest.mark.parametrize("valid_record", [False, True])
def test_pubmed_unusable_records_do_not_claim_exhaustive_coverage(monkeypatch, valid_record):
    invalid = "<PubmedArticle><MedlineCitation><PMID>1</PMID></MedlineCitation></PubmedArticle>"
    valid = "<PubmedArticle><MedlineCitation><PMID>2</PMID><Article><ArticleTitle>Protein design</ArticleTitle></Article></MedlineCitation></PubmedArticle>"
    articles = invalid + (valid if valid_record else "")
    monkeypatch.setattr(pubmed.PubMedSource, "_esearch", lambda *args: ["1", "2"] if valid_record else ["1"])
    monkeypatch.setattr(pubmed.PubMedSource, "_efetch", lambda *args: f"<PubmedArticleSet>{articles}</PubmedArticleSet>")
    monkeypatch.setattr(pubmed.time, "sleep", lambda *args: None)
    source = pubmed.PubMedSource()
    if valid_record:
        assert len(source.fetch(_spec("pubmed", query="protein design"))) == 1
    else:
        with pytest.raises(RuntimeError, match="no usable PubMed articles"):
            source.fetch(_spec("pubmed", query="protein design"))
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["skipped_invalid"] == 1
    assert query["truncated"] and not query["exhausted"]
    assert query["stop_reason"] == ("parsing_loss" if valid_record else "failure")


def test_openalex_saturated_queries_cover_every_core_before_context(monkeypatch):
    core = [f"topic-{i:02d}" for i in range(10)]
    _profile(monkeypatch, core, ["context-a", "context-b"])
    requests = []

    def get_json(url, params, headers):
        requests.append(dict(params))
        return {"results": [_work(f"{params['search.title_abstract_keywords']}-{i}")
                            for i in range(int(params["per-page"]))],
                "meta": {"count": 999, "next_cursor": "more"}}

    monkeypatch.setattr(openalex, "_get_json", get_json)
    source = openalex.OpenAlexSource()
    items = source.fetch(_spec(profile_driven=True))
    assert [r["search.title_abstract_keywords"] for r in requests] == core
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
        requests.append((params["search.title_abstract_keywords"], int(params["per-page"])))
        return {"results": [_work(f"{params['search.title_abstract_keywords']}-{i}")
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
        term = params["search.title_abstract_keywords"]
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
        return {"results": [_work(f"{params['search.title_abstract_keywords']}-{i}")
                            for i in range(int(params["per-page"]))],
                "meta": {"next_cursor": "more"}}

    monkeypatch.setattr(openalex, "_get_json", get_json)
    source = openalex.OpenAlexSource()
    assert len(source.fetch(_spec(profile_driven=True))) == 180
    assert len(requests) == 13
    assert source.last_fetch_diagnostics["request_budget"] == 13
    assert all(r["search.title_abstract_keywords"] != "context" for r in requests)


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
    assert queries == ['("RNA design"[tiab])', '(rna[tiab] AND design*[tiab]) NOT ("RNA design"[tiab])']
    assert all('clinical trial' not in query for query in queries)


def test_pubmed_explicit_nonprofile_query_is_unchanged(monkeypatch):
    queries = []
    monkeypatch.setattr(pubmed.PubMedSource, "_esearch", lambda self, client, query: queries.append(query) or [])
    query = '("RNA"[tiab]) AND clinical trial[pt]'
    assert pubmed.PubMedSource().fetch(_spec("pubmed", query=query)) == []
    assert queries == [query]


def test_pubmed_requests_supported_publication_date_sort(monkeypatch):
    requests = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"esearchresult": {"idlist": [], "count": "0"}}

    def get(client, url, *, params):
        requests.append(params)
        return Response()

    monkeypatch.setattr(pubmed.httpx.Client, "get", get)
    assert pubmed.PubMedSource().fetch(_spec("pubmed", query='"gene therapy"[tiab]')) == []
    assert requests[0]["sort"] == "pub date"
    assert requests[0]["datetype"] == "pdat"


def test_pubmed_no_profile_terms_uses_configured_fallback(monkeypatch):
    _profile(monkeypatch, [])
    queries = []
    monkeypatch.setattr(pubmed.PubMedSource, "_esearch", lambda self, client, query: queries.append(query) or [])
    source = pubmed.PubMedSource()
    source.fetch(_spec("pubmed", profile_driven=True, query="RNA[tiab]"))
    assert queries == ["RNA[tiab]"]
    assert source.last_fetch_diagnostics["queries"][0]["exhausted"]


@pytest.mark.parametrize("payload", [{}, {"esearchresult": {"idlist": [], "errorlist": {"fieldnotfound": ["bad"]}}}])
def test_pubmed_invalid_search_is_a_failure(monkeypatch, payload):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(pubmed.httpx.Client, "get", lambda *args, **kwargs: Response())
    source = pubmed.PubMedSource()
    with pytest.raises(RuntimeError, match="fetch failed"):
        source.fetch(_spec("pubmed", query="RNA"))
    assert source.last_fetch_diagnostics["queries"][0]["stop_reason"] == "failure"


@pytest.mark.parametrize("xml", ["", "<ERROR>bad</ERROR>", "invalid XML", "<PubmedArticleSet/>"])
def test_pubmed_nonempty_search_cannot_succeed_with_invalid_fetch(monkeypatch, xml):
    monkeypatch.setattr(pubmed.PubMedSource, "_esearch", lambda *args: ["123"])
    monkeypatch.setattr(pubmed.PubMedSource, "_efetch", lambda *args: xml)
    monkeypatch.setattr(pubmed.time, "sleep", lambda *args: None)
    source = pubmed.PubMedSource()
    with pytest.raises(RuntimeError, match="PubMed"):
        source.fetch(_spec("pubmed", query="RNA"))
    assert source.last_fetch_diagnostics["queries"][0]["stop_reason"] == "failure"


def test_pubmed_search_total_and_item_cap_are_recorded(monkeypatch):
    class Response:
        text = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
        <PMID>123</PMID><Article><ArticleTitle>RNA design</ArticleTitle>
        </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""

        def raise_for_status(self):
            pass

        def json(self):
            return {"esearchresult": {"idlist": ["123"], "count": "999"}}

    monkeypatch.setattr(pubmed.httpx.Client, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(pubmed.time, "sleep", lambda *args: None)
    source = pubmed.PubMedSource()
    source.MAX_ITEMS = 1
    assert len(source.fetch(_spec("pubmed", query="RNA"))) == 1
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["total_available"] == 999
    assert query["request_count"] == 2
    assert query["stop_reason"] == "item_cap"
    assert query["truncated"] and not query["exhausted"]


@pytest.mark.parametrize("ids", [[], ["123"]])
@pytest.mark.parametrize("warnings", [
    {"outputmessages": ["Wildcard search used only the first available variations."]},
    {"phrasesignored": ["protein*"]},
    {"quotedphrasesnotfound": ["protein engineering"]},
    {"outputmessages": ["No items found.", "Search expansion was limited."]},
])
def test_pubmed_search_warnings_preserve_results_without_claiming_exhaustion(
    monkeypatch, caplog, ids, warnings,
):
    from dailydigest import health

    class Response:
        text = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
        <PMID>123</PMID><Article><ArticleTitle>Protein engineering</ArticleTitle>
        </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""

        def raise_for_status(self):
            pass

        def json(self):
            return {"esearchresult": {
                "idlist": ids, "count": str(len(ids)), "warninglist": warnings,
            }}

    monkeypatch.setattr(pubmed.httpx.Client, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(pubmed.time, "sleep", lambda *args: None)
    source = pubmed.PubMedSource()
    items = source.fetch(_spec("pubmed", query="protein*[tiab]"))
    assert [item.external_id for item in items] == ids
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["warnings"] == warnings
    assert query["truncated"] and not query["exhausted"]
    assert query["stop_reason"] == "search_warning"
    assert query["failure"] is None
    assert query["request_count"] == (2 if ids else 1)
    assert "PubMed search warnings" in caplog.text
    monkeypatch.setattr(health, "_load_existing", lambda path: {"latest": [{
        "source": "PubMed", "ok": True, "items": len(items),
        "diagnostics": source.last_fetch_diagnostics,
    }]})
    assert health.coverage_summary()["limited"] == 1


@pytest.mark.parametrize("warnings", [
    {},
    {"phrasesignored": [], "quotedphrasesnotfound": [], "outputmessages": []},
    {"outputmessages": ["No items found."]},
])
def test_pubmed_empty_result_notice_is_not_a_coverage_failure(monkeypatch, caplog, warnings):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"esearchresult": {"idlist": [], "count": "0", "warninglist": warnings}}

    monkeypatch.setattr(pubmed.httpx.Client, "get", lambda *args, **kwargs: Response())
    source = pubmed.PubMedSource()
    assert source.fetch(_spec("pubmed", query="protein*[tiab]")) == []
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["exhausted"] and not query["truncated"]
    assert query["stop_reason"] == "exhausted"
    assert "PubMed search warnings" not in caplog.text


@pytest.mark.parametrize("payload,reason", [
    ({"results": [], "meta": {"count": 0}}, "exhausted"),
    ({"results": [_work("one")], "meta": {"count": 2, "next_cursor": "more"}}, "item_cap"),
])
def test_openalex_empty_and_capped_coverage_are_distinct(monkeypatch, payload, reason):
    monkeypatch.setattr(openalex, "_get_json", lambda *args: payload)
    source = openalex.OpenAlexSource()
    source.MAX_ITEMS = 1
    source.fetch(_spec(query="RNA"))
    query = source.last_fetch_diagnostics["queries"][0]
    assert query["stop_reason"] == reason
    assert query["exhausted"] == (reason == "exhausted")
    assert query["truncated"] == (reason != "exhausted")


def test_openalex_diagnostics_reset_between_fetches(monkeypatch):
    monkeypatch.setattr(openalex, "_get_json", lambda *args: {
        "results": [], "meta": {"count": 0},
    })
    source = openalex.OpenAlexSource()
    source.fetch(_spec(query="old"))
    source.fetch(_spec(query="new"))
    assert [q["query"] for q in source.last_fetch_diagnostics["queries"]] == ["new"]


@pytest.mark.parametrize("kind", ["openalex_venues", "openalex_authors"])
def test_openalex_wrappers_expose_work_query_diagnostics(monkeypatch, kind):
    monkeypatch.setattr(openalex, "_get_json", lambda *args: {
        "results": [], "meta": {"count": 0},
    })
    if kind == "openalex_venues":
        source = openalex.OpenAlexVenuesSource()
        spec = _spec(kind, venue_ids=["S123"])
    else:
        source = openalex.OpenAlexAuthorsSource()
        monkeypatch.setattr(openalex, "watched_author_names", lambda *args: ["Author"])
        monkeypatch.setattr(source, "_resolve_author_id", lambda *args: "A123")
        spec = _spec(kind)
    assert source.fetch(spec) == []
    query = source.last_fetch_diagnostics["queries"][0]
    assert "123" in query["filter"]
    assert query["exhausted"]


def test_openalex_author_resolution_all_failures_raise_with_diagnostics(monkeypatch):
    monkeypatch.setattr(openalex, "watched_author_names", lambda *args: ["A", "B"])

    def unavailable(*args):
        raise RuntimeError("author endpoint unavailable")

    monkeypatch.setattr(openalex, "_get_json", unavailable)
    source = openalex.OpenAlexAuthorsSource()
    with pytest.raises(RuntimeError, match="author resolution"):
        source.fetch(_spec("openalex_authors"))
    assert not source.last_fetch_diagnostics["author_resolution_complete"]
    assert all(q["failure"] for q in source.last_fetch_diagnostics["author_queries"])
    assert source.last_fetch_diagnostics["queries"] == []


def test_openalex_author_resolution_partial_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(openalex, "watched_author_names", lambda *args: ["A", "B"])

    def get_json(url, params, headers):
        if url == openalex.OpenAlexAuthorsSource.AUTHORS_URL:
            if params["search"] == "B":
                raise RuntimeError("failed")
            return {"results": [{"id": "https://openalex.org/A123", "display_name": "A"}]}
        return {"results": [_work("one")], "meta": {"count": 1}}

    monkeypatch.setattr(openalex, "_get_json", get_json)
    source = openalex.OpenAlexAuthorsSource()
    assert len(source.fetch(_spec("openalex_authors"))) == 1
    diagnostic = source.last_fetch_diagnostics
    assert not diagnostic["author_resolution_complete"]
    assert diagnostic["truncated"]
    assert diagnostic["author_queries"][1]["failure"] == "RuntimeError"
    assert diagnostic["queries"][0]["exhausted"]


def test_openalex_author_resolution_valid_no_matches_remains_empty(monkeypatch):
    monkeypatch.setattr(openalex, "watched_author_names", lambda *args: ["A"])
    monkeypatch.setattr(openalex, "_get_json", lambda *args: {"results": []})
    source = openalex.OpenAlexAuthorsSource()
    assert source.fetch(_spec("openalex_authors")) == []
    diagnostic = source.last_fetch_diagnostics
    assert diagnostic["author_resolution_complete"]
    assert not diagnostic["truncated"]
    assert diagnostic["author_queries"][0]["stop_reason"] == "no_match"
