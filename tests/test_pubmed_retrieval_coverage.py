"""Shared PubMed retrieval regressions, fixture-only."""
from types import SimpleNamespace
import pytest
from dailydigest.ingest import pubmed
from dailydigest.models import SourceSpec

def _spec(kind="openalex", **kwargs):
    return SourceSpec(name="Topics", kind=kind, section="research", **kwargs)


def _profile(monkeypatch, core, context=()):
    monkeypatch.setattr(
        "dailydigest.config.load_profile",
        lambda: SimpleNamespace(keywords=core, context_keywords=context),
    )


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
