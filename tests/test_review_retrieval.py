"""Offline regressions from the October retrieval review."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from xml.etree.ElementTree import fromstring, tostring

import httpx
import pytest
from tenacity import RetryError, wait_none

from dailydigest.ingest import arxiv, openalex, pubmed
from dailydigest.models import SourceSpec


def spec(kind, **kwargs):
    return SourceSpec(name="Fixture source", kind=kind, section="research", **kwargs)


def date_xml(instant):
    return f"<Year>{instant.year}</Year><Month>{instant.month}</Month><Day>{instant.day}</Day>"


def article(print_date="", electronic="", history=""):
    return fromstring(f"""<PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
      <ArticleTitle>Fixture paper</ArticleTitle><Journal><JournalIssue>
      <PubDate>{print_date}</PubDate></JournalIssue></Journal>{electronic}
      </Article></MedlineCitation><PubmedData><History>{history}</History></PubmedData>
      </PubmedArticle>""")


def test_pubmed_prefers_electronic_date_over_future_print():
    now = datetime.now(timezone.utc)
    electronic = f'<ArticleDate DateType="Electronic">{date_xml(now)}</ArticleDate>'
    item = pubmed.PubMedSource()._parse_article(
        article(date_xml(now + timedelta(days=60)), electronic), spec("pubmed"),
    )
    assert item.published_at.date() == now.date()
    assert item.metadata['publication_date_source'] == 'published'
    assert item.metadata['publication_date_precision'] == 'day'


@pytest.mark.parametrize("value", ["<Year>2026</Year>", "<Year>2026</Year><Month>Oct</Month>",
                                  "<MedlineDate>2026 Oct-Dec</MedlineDate>"])
def test_pubmed_incomplete_dates_remain_unknown(value):
    assert pubmed.PubMedSource()._parse_article(article(value), spec("pubmed")).published_at is None


def test_pubmed_history_uses_publication_event_not_indexing_event():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=5)
    history = (f'<PubMedPubDate PubStatus="entrez">{date_xml(now)}</PubMedPubDate>'
               f'<PubMedPubDate PubStatus="epublish">{date_xml(old)}</PubMedPubDate>')
    item = pubmed.PubMedSource()._parse_article(article(history=history), spec("pubmed"))
    assert item.published_at.date() == old.date()
    indexing_only = f'<PubMedPubDate PubStatus="entrez">{date_xml(now)}</PubMedPubDate>'
    assert pubmed.PubMedSource()._parse_article(article(history=indexing_only), spec("pubmed")).published_at is None


def test_pubmed_future_and_invalid_electronic_dates_fall_back_to_print():
    now = datetime.now(timezone.utc)
    electronic = (f'<ArticleDate DateType="Electronic">{date_xml(now + timedelta(days=2))}</ArticleDate>'
                  '<ArticleDate DateType="Electronic"><Year>2026</Year><Month>13</Month><Day>1</Day></ArticleDate>')
    item = pubmed.PubMedSource()._parse_article(article(date_xml(now), electronic), spec("pubmed"))
    assert item.published_at.date() == now.date()
    assert pubmed.PubMedSource()._parse_article(article(date_xml(now + timedelta(days=2))), spec("pubmed")).published_at is None


def atom(entries, total=None):
    count = f'<opensearch:totalResults>{total}</opensearch:totalResults>' if total is not None else ''
    content = ''.join(f'<entry><id>https://arxiv.org/abs/{key}</id><title>Fixture {key}</title>'
                      f'<published>{instant.isoformat()}</published></entry>' for key, instant in entries)
    return ('<feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">' + count + content + '</feed>')


def test_arxiv_paginates_until_window_boundary_and_reports_coverage(monkeypatch):
    now = datetime.now(timezone.utc)
    pages = iter([atom([('2610.00001', now), ('2610.00002', now)], 10),
                  atom([('2610.00003', now - timedelta(days=3))], 10)])
    requests = []
    monkeypatch.setattr(arxiv.ArxivSource, 'MAX_RESULTS', 2)
    monkeypatch.setattr(arxiv, '_http_get_text', lambda url, params: requests.append(dict(params)) or next(pages))
    source = arxiv.ArxivSource()
    assert len(source.fetch(spec('arxiv'), days=2)) == 3
    assert [r['start'] for r in requests] == ['0', '2']
    diagnostic = source.last_fetch_diagnostics['queries'][0]
    assert diagnostic['exhausted'] and not diagnostic['truncated']
    assert diagnostic['stop_reason'] == 'window_exhausted'
    assert diagnostic['request_count'] == 2


def test_arxiv_cap_is_explicit_and_compound_categories_are_scoped(monkeypatch):
    now = datetime.now(timezone.utc)
    requests = []
    monkeypatch.setattr(arxiv.ArxivSource, 'MAX_ITEMS', 2, raising=False)
    monkeypatch.setattr(arxiv, '_http_get_text', lambda url, params: requests.append(dict(params)) or atom([
        ('2610.00001', now), ('2610.00002', now)], 20))
    source = arxiv.ArxivSource()
    source.fetch(spec('arxiv', category='q-bio.GN+OR+q-bio.QM'))
    assert requests[0]['search_query'] == 'cat:q-bio.GN OR cat:q-bio.QM'
    diagnostic = source.last_fetch_diagnostics['queries'][0]
    assert diagnostic['truncated'] and not diagnostic['exhausted']
    assert diagnostic['stop_reason'] == 'item_cap'


def test_arxiv_repeated_page_cannot_claim_window_exhaustion(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(arxiv.ArxivSource, 'MAX_RESULTS', 1)
    monkeypatch.setattr(arxiv, '_http_get_text', lambda *args: atom([('2610.00001', now)], 10))
    source = arxiv.ArxivSource()
    assert len(source.fetch(spec('arxiv'))) == 1
    assert source.last_fetch_diagnostics['queries'][0]['stop_reason'] == 'repeated_page'
    assert source.last_fetch_diagnostics['queries'][0]['truncated']


def test_arxiv_later_failure_retains_failure_diagnostics(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(arxiv.ArxivSource, 'MAX_RESULTS', 1)
    calls = []

    def fetch(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError('fixture unavailable')
        return atom([('2610.00001', now)], 10)

    monkeypatch.setattr(arxiv, '_http_get_text', fetch)
    source = arxiv.ArxivSource()
    with pytest.raises(RuntimeError, match='fetch failed'):
        source.fetch(spec('arxiv'))
    diagnostic = source.last_fetch_diagnostics['queries'][0]
    assert diagnostic['failure'] and diagnostic['truncated']
    assert diagnostic['fetched_count'] == 1


def test_arxiv_request_budget_is_limited_even_below_item_cap(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(arxiv.ArxivSource, 'MAX_RESULTS', 1)
    monkeypatch.setattr(arxiv.ArxivSource, 'MAX_PAGES', 1)
    monkeypatch.setattr(arxiv, '_http_get_text', lambda *args: atom([('2610.00001', now)], 10))
    source = arxiv.ArxivSource()
    source.fetch(spec('arxiv'))
    diagnostic = source.last_fetch_diagnostics['queries'][0]
    assert diagnostic['stop_reason'] == 'request_budget'
    assert diagnostic['truncated'] and not diagnostic['exhausted']


def test_arxiv_valid_empty_window_is_distinct_from_invalid_response(monkeypatch):
    monkeypatch.setattr(arxiv, '_http_get_text', lambda *args: atom([], 0))
    source = arxiv.ArxivSource()
    assert source.fetch(spec('arxiv')) == []
    assert source.last_fetch_diagnostics['queries'][0]['exhausted']
    monkeypatch.setattr(arxiv, '_http_get_text', lambda *args: '<html>provider unavailable</html>')
    with pytest.raises(RuntimeError, match='Invalid arXiv Atom'):
        source.fetch(spec('arxiv'))
    assert source.last_fetch_diagnostics['queries'][0]['failure']


def profile(monkeypatch, core, context=()):
    monkeypatch.setattr('dailydigest.config.load_profile', lambda: SimpleNamespace(
        keywords=core, context_keywords=context, canonical_facets={}))


def test_openalex_failed_term_preserves_other_terms_and_coverage(monkeypatch):
    profile(monkeypatch, ['a-good', 'b-failure', 'c-good'])
    calls = []

    def fetch(url, params, headers):
        term = params['search.title_abstract_keywords']
        calls.append(term)
        if term == 'b-failure':
            raise RuntimeError('fixture unavailable')
        return {'results': [{'id': f'https://openalex.org/{term}', 'title': term}], 'meta': {'count': 1}}

    monkeypatch.setattr(openalex, '_get_json', fetch)
    source = openalex.OpenAlexSource()
    assert [item.title for item in source.fetch(spec('openalex', profile_driven=True))] == ['a-good', 'c-good']
    assert calls == ['a-good', 'b-failure', 'c-good']
    assert source.last_fetch_diagnostics['queries'][1]['failure']
    assert source.last_fetch_diagnostics['queries'][1]['interest'] == 'b-failure'


def test_openalex_all_core_failure_cannot_be_rescued_by_context(monkeypatch):
    profile(monkeypatch, ['core'], ['context'])

    def fetch(url, params, headers):
        if params['search.title_abstract_keywords'] == 'core':
            raise RuntimeError('fixture unavailable')
        return {'results': [{'id': 'https://openalex.org/W1', 'title': 'Context'}], 'meta': {'count': 1}}

    monkeypatch.setattr(openalex, '_get_json', fetch)
    with pytest.raises(RuntimeError, match='all core profile searches failed'):
        openalex.OpenAlexSource().fetch(spec('openalex', profile_driven=True))


@pytest.mark.parametrize('status,attempts', [(400, 1), (403, 1), (404, 1), (429, 3), (503, 3)])
def test_openalex_retries_only_transient_http_failures(monkeypatch, status, attempts):
    calls = []

    def fetch(client, url, **kwargs):
        calls.append(url)
        return httpx.Response(status, request=httpx.Request('GET', url))

    monkeypatch.setattr(httpx.Client, 'get', fetch)
    with pytest.raises((httpx.HTTPStatusError, RetryError)):
        openalex._get_json.retry_with(wait=wait_none())('https://fixture.test', {}, {})
    assert len(calls) == attempts


@pytest.mark.parametrize('kind', ['openalex', 'pubmed'])
def test_unreadable_profile_fails_personalized_retrieval(monkeypatch, kind):
    def unavailable():
        raise ValueError('fixture unreadable profile')

    monkeypatch.setattr('dailydigest.config.load_profile', unavailable)
    # No request may escape the profile-load guard.
    monkeypatch.setattr(httpx.Client, 'get', lambda *args, **kwargs: pytest.fail('unexpected request'))
    source = openalex.OpenAlexSource() if kind == 'openalex' else pubmed.PubMedSource()
    with pytest.raises(RuntimeError, match='profile'):
        source.fetch(spec(kind, profile_driven=True, query='generic fallback'))


def test_unreadable_profile_cannot_look_like_no_watched_authors(monkeypatch):
    def unavailable():
        raise ValueError('fixture unreadable profile')

    monkeypatch.setattr('dailydigest.config.load_profile', unavailable)
    with pytest.raises(RuntimeError, match='profile'):
        openalex.OpenAlexAuthorsSource().fetch(spec('openalex_authors'))


def test_pubmed_contact_parameters_are_sent_on_both_requests(monkeypatch):
    calls = []
    now = datetime.now(timezone.utc)

    def fetch(client, url, *, params):
        calls.append(dict(params))
        if url.endswith('esearch.fcgi'):
            return httpx.Response(200, json={'esearchresult': {'idlist': ['1'], 'count': '1'}},
                                  request=httpx.Request('GET', url))
        xml = '<PubmedArticleSet>' + tostring(article(date_xml(now)), encoding='unicode') + '</PubmedArticleSet>'
        return httpx.Response(200, text=xml, request=httpx.Request('GET', url))

    monkeypatch.setattr(httpx.Client, 'get', fetch)
    monkeypatch.setattr(pubmed.time, 'sleep', lambda *args: None)
    pubmed.PubMedSource().fetch(spec('pubmed', query='fixture', polite_email='developer@example.test'))
    assert len(calls) == 2
    assert all(p['tool'] == 'dailydigest' and p['email'] == 'developer@example.test' for p in calls)


def test_profile_search_uses_relevance_inside_recency_filter(monkeypatch):
    profile(monkeypatch, ['fixture'])
    calls = []
    monkeypatch.setattr(openalex, '_get_json', lambda url, params, headers: calls.append(dict(params)) or {'results': []})
    openalex.OpenAlexSource().fetch(spec('openalex', profile_driven=True))
    assert calls[0]['sort'] == 'relevance_score:desc'
    assert 'from_publication_date:' in calls[0]['filter']
    assert 'to_publication_date:' in calls[0]['filter']


@pytest.mark.parametrize('results,reason,resolved', [
    ([{'id': 'https://openalex.org/A1', 'display_name': 'Another Person'}], 'unverified', None),
    ([{'id': 'https://openalex.org/A1', 'display_name': 'Expected Person'},
      {'id': 'https://openalex.org/A2', 'display_name': 'Expected Person'}], 'ambiguous', None),
    ([{'id': 'https://openalex.org/A1', 'display_name': 'Another Person'},
      {'id': 'https://openalex.org/A2', 'display_name': 'Expected Person'}], 'matched', 'A2'),
])
def test_author_resolution_requires_unique_name_match(monkeypatch, results, reason, resolved):
    monkeypatch.setattr(openalex, '_get_json', lambda *args: {'results': results})
    source = openalex.OpenAlexAuthorsSource()
    assert source._resolve_author_id('Expected Person', {}) == resolved
    assert source.last_fetch_diagnostics['author_queries'][0]['stop_reason'] == reason


def test_ambiguous_author_is_visible_as_incomplete_coverage(monkeypatch):
    monkeypatch.setattr(openalex, 'watched_author_names', lambda *args: ['Expected Person'])
    monkeypatch.setattr(openalex, '_get_json', lambda *args: {'results': [
        {'id': 'https://openalex.org/A1', 'display_name': 'Expected Person'},
        {'id': 'https://openalex.org/A2', 'display_name': 'Expected Person'},
    ]})
    source = openalex.OpenAlexAuthorsSource()
    assert source.fetch(spec('openalex_authors')) == []
    assert not source.last_fetch_diagnostics['author_resolution_complete']
    assert source.last_fetch_diagnostics['truncated']
