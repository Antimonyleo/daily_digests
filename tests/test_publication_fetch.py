import json

import httpx
import pytest

from dailydigest import publication_fetch as fetch

_HTTPX_STREAM = httpx.Client.stream

A = '10.1000/review'
B = '10.1000/research'


def setup_transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(original, 'stream', _HTTPX_STREAM)
    calls = []

    def route(request):
        calls.append(request)
        return handler(request)

    monkeypatch.setattr(fetch.httpx, 'Client', lambda **kw: original(
        transport=httpx.MockTransport(route), **kw))
    return calls


def core(records):
    return {'hitCount': len(records), 'resultList': {'result': records}}


def article(doi, pmid, types, **extra):
    return {'doi': doi, 'id': pmid, 'source': 'MED',
            'pubTypeList': {'pubType': types}, **extra}


def test_exact_batch_types_keep_generic_unknown(monkeypatch):
    records = [article(A, '1', ['Journal Article', 'Review']),
               article(B, '2', ['research-article'])]
    calls = setup_transport(monkeypatch, lambda r: httpx.Response(
        200, json=core(records)) if r.url.host == 'www.ebi.ac.uk' else httpx.Response(404))
    result = fetch.fetch_publication_metadata([A.upper(), B, A, 'bad'])
    assert len(calls) == 2
    assert calls[0].url.params['query'] == f'DOI:"{A}" OR DOI:"{B}"'
    assert result[A]['metadata']['publication_types'] == ['Journal Article', 'Review']
    assert result[B]['metadata']['publication_types'] == ['Research Article']
    assert result[B]['metadata']['publication_type_evidence'][0]['value'] == 'research-article'
    assert result[A]['status'] == 'fresh'


def test_comment_on_resolves_exact_med_id_never_citations(monkeypatch):
    review = article(A, '1', ['Comment'], commentCorrectionList={'commentCorrection': [
        {'source': 'MED', 'id': '2', 'type': 'Comment on'},
        {'source': 'MED', 'id': '3', 'type': 'Preprint in'},
        {'source': 'MED', 'id': '4', 'type': 'Erratum for'},
        {'source': 'PMC', 'id': '5', 'type': 'Comment on'}]})
    def handler(r):
        query = r.url.params['query']
        records = [article(B, '2', ['Journal Article'])] if 'EXT_ID' in query else [review]
        return httpx.Response(200, json=core(records))
    calls = setup_transport(monkeypatch, handler)
    result = fetch.fetch_publication_metadata([A])
    assert len(calls) == 2
    assert calls[1].url.params['query'] == 'SRC:MED AND (EXT_ID:2)'
    assert result[A]['metadata']['companion_of_dois'] == [B]
    assert result[A]['metadata']['companion_relation_evidence'][0]['rel'] == 'is-commentary-on'


def test_comment_in_inverts_only_for_requested_child(monkeypatch):
    parent = article(B, '2', ['Journal Article'], commentCorrectionList={'commentCorrection': [
        {'source': 'MED', 'id': '1', 'type': 'Comment in'}]})
    child = article(A, '1', ['Comment'])
    setup_transport(monkeypatch, lambda r: httpx.Response(200, json=core([parent, child])))
    result = fetch.fetch_publication_metadata([A, B])
    assert result[A]['metadata']['companion_of_dois'] == [B]
    assert 'companion_of_dois' not in result[B]['metadata']


def test_crossref_explicit_relation_only(monkeypatch):
    def handler(r):
        if r.url.host == 'www.ebi.ac.uk':
            return httpx.Response(200, json=core([article(A, '1', ['Review'])]))
        return httpx.Response(200, json={'message': {'DOI': A, 'type': 'journal-article',
            'reference': [{'DOI': '10.1000/not-parent'}], 'relation': {
                'is-review-of': [{'id-type': 'doi', 'id': B}],
                'references': [{'id-type': 'doi', 'id': '10.1000/not-parent'}]}}})
    setup_transport(monkeypatch, handler)
    result = fetch.fetch_publication_metadata([A])
    assert result[A]['metadata']['companion_of_dois'] == [B]
    assert result[A]['metadata']['companion_relation_evidence'][0]['rel'] == 'is-review-of'


@pytest.mark.parametrize('genre', [
    'News & Views', 'News and Views', 'Perspective', 'Research Highlight',
    'Research Highlights', '  nEwS   &   vIeWs  ', 'Systematic Review', 'Meta-Analysis',
])
def test_supported_commentary_genres_resolve_explicit_crossref_parent(monkeypatch, genre):
    def handler(request):
        if request.url.host == 'www.ebi.ac.uk':
            return httpx.Response(200, json=core([article(A, '1', [genre])]))
        return httpx.Response(200, json={'message': {'DOI': A, 'type': 'journal-article',
            'relation': {'is-commentary-on': [{'id-type': 'doi', 'id': B}]},
            'reference': [{'DOI': '10.1000/not-parent'}]}})
    calls = setup_transport(monkeypatch, handler)
    result = fetch.fetch_publication_metadata([A])[A]
    assert len(calls) == 2
    assert result['status'] == 'fresh' and not result.get('incomplete', False)
    assert result['metadata']['companion_of_dois'] == [B]
    assert result['metadata']['companion_relation_evidence'][0]['source'] == 'Crossref'


def test_missing_and_failed_distinct_no_redirect(monkeypatch):
    calls = setup_transport(monkeypatch, lambda r: httpx.Response(302, headers={'Location': 'https://evil.example'}))
    assert fetch.fetch_publication_metadata([A])[A] == {'status': 'failed', 'metadata': {}}
    assert len(calls) == 1


@pytest.mark.parametrize('payload', [{}, {'hitCount': 2, 'resultList': {'result': []}}, []])
def test_malformed_or_truncated_is_failure(monkeypatch, payload):
    setup_transport(monkeypatch, lambda r: httpx.Response(200, json=payload))
    assert fetch.fetch_publication_metadata([A])[A]['status'] == 'failed'


def test_missing_is_empty_success(monkeypatch):
    setup_transport(monkeypatch, lambda r: httpx.Response(200, json=core([])))
    assert fetch.fetch_publication_metadata([A])[A] == {'status': 'missing', 'metadata': {}}


def test_limits_twelve_dois_four_requests(monkeypatch):
    dois = [f'10.1000/{i}' for i in range(20)]
    records = [article(d, str(i), ['Review']) for i, d in enumerate(dois[:12])]
    calls = setup_transport(monkeypatch, lambda r: httpx.Response(200, json=core(records)))
    assert len(fetch.fetch_publication_metadata(dois)) == 12
    assert len(calls) == 4


def test_zero_budget_and_invalid_identifiers_do_not_contact_network(monkeypatch):
    calls = setup_transport(monkeypatch, lambda r: pytest.fail('network called'))
    assert fetch.fetch_publication_metadata(['not-doi']) == {}
    assert fetch.fetch_publication_metadata([A], deadline_seconds=0)[A]['status'] == 'failed'
    assert calls == []


def test_oversized_response_fails(monkeypatch):
    monkeypatch.setattr(fetch, '_MAX_BYTES', 20)
    setup_transport(monkeypatch, lambda r: httpx.Response(200, content=json.dumps(core([])).encode()))
    assert fetch.fetch_publication_metadata([A])[A]['status'] == 'failed'


def test_unrelated_returned_doi_does_not_enrich(monkeypatch):
    setup_transport(monkeypatch, lambda r: httpx.Response(200, json=core([article(B, '2', ['Review'])])))
    assert fetch.fetch_publication_metadata([A])[A] == {'status': 'missing', 'metadata': {}}


def test_stop_exception_propagates(monkeypatch):
    from dailydigest.brew import BrewCancelled
    def handler(r):
        raise BrewCancelled()
    setup_transport(monkeypatch, handler)
    with pytest.raises(BrewCancelled):
        fetch.fetch_publication_metadata([A])


def test_browser_cors_failure_is_unavailable(monkeypatch):
    def handler(request):
        raise RuntimeError('Browser CORS request unavailable')
    setup_transport(monkeypatch, handler)
    assert fetch.fetch_publication_metadata([A])[A]['status'] == 'failed'


def test_mixed_malformed_results_cannot_establish_absence(monkeypatch):
    setup_transport(monkeypatch, lambda r: httpx.Response(200, json=core([{} , 'bad'])))
    assert fetch.fetch_publication_metadata([A])[A]['status'] == 'failed'


def test_shared_deadline_prevents_fallback(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(fetch.time, 'monotonic', lambda: clock[0])
    def handler(request):
        clock[0] = 19.0
        return httpx.Response(200, json=core([article(A, '1', ['Review'])]))
    calls = setup_transport(monkeypatch, handler)
    assert fetch.fetch_publication_metadata([A])[A]['status'] == 'failed'
    assert len(calls) == 1
    assert calls[0].extensions['timeout']['read'] == 2.0


def test_ambiguous_pmid_never_links(monkeypatch):
    record = article(A, '1', ['Comment'], commentCorrectionList={'commentCorrection': [
        {'source': 'MED', 'id': '2', 'type': 'Comment on'}]})
    def handler(request):
        if request.url.host != 'www.ebi.ac.uk':
            return httpx.Response(404)
        query = request.url.params['query']
        records = [article(B, '2', []), article('10.1000/other', '2', [])] if 'EXT_ID' in query else [record]
        return httpx.Response(200, json=core(records))
    setup_transport(monkeypatch, handler)
    assert 'companion_of_dois' not in fetch.fetch_publication_metadata([A])[A]['metadata']


def test_explicit_retraction_preserved(monkeypatch):
    setup_transport(monkeypatch, lambda r: httpx.Response(200, json=core([
        article(A, '1', ['Journal Article', 'Retracted Publication'])])))
    assert fetch.fetch_publication_metadata([A])[A]['metadata']['is_retracted'] is True


@pytest.mark.parametrize('genre', ['Review', 'News & Views', 'Perspective', 'Research Highlight'])
def test_type_success_crossref_failure_is_fresh_but_incomplete(monkeypatch, genre):
    def handler(request):
        if request.url.host == 'www.ebi.ac.uk':
            return httpx.Response(200, json=core([article(A, '1', [genre])]))
        return httpx.Response(503)
    setup_transport(monkeypatch, handler)
    result = fetch.fetch_publication_metadata([A])[A]
    assert result['status'] == 'fresh'
    assert result['metadata']['publication_types'] == [genre]
    assert result['incomplete'] is True


def test_successful_empty_crossref_relation_is_complete(monkeypatch):
    def handler(request):
        if request.url.host == 'www.ebi.ac.uk':
            return httpx.Response(200, json=core([article(A, '1', ['Review'])]))
        return httpx.Response(200, json={'message': {'DOI': A, 'relation': {}}})
    setup_transport(monkeypatch, handler)
    result = fetch.fetch_publication_metadata([A])[A]
    assert result['status'] == 'fresh'
    assert not result.get('incomplete', False)


@pytest.mark.parametrize('genre', ['Review', 'News & Views', 'Perspective', 'Research Highlight'])
def test_remaining_known_commentaries_marked_incomplete_at_request_limit(monkeypatch, genre):
    dois = [f'10.1000/review{i}' for i in range(5)]
    def handler(request):
        if request.url.host == 'www.ebi.ac.uk':
            return httpx.Response(200, json=core([
                article(doi, str(i + 1), [genre]) for i, doi in enumerate(dois)]))
        doi = request.url.path.removeprefix('/works/')
        return httpx.Response(200, json={'message': {'DOI': doi, 'relation': {}}})
    calls = setup_transport(monkeypatch, handler)
    result = fetch.fetch_publication_metadata(dois)
    assert len(calls) == 4
    assert [doi for doi in dois if result[doi].get('incomplete')] == dois[3:]
    assert all(result[doi]['status'] == 'fresh' for doi in dois)


@pytest.mark.parametrize('resolution', ['failed', 'truncated', 'empty'])
def test_pmid_resolution_failure_is_partial_but_successful_absence_is_not(monkeypatch, resolution):
    linked = article(A, '1', ['Journal Article'], commentCorrectionList={'commentCorrection': [
        {'source': 'MED', 'id': '2', 'type': 'Comment on'}]})
    def handler(request):
        if 'EXT_ID' not in request.url.params['query']:
            return httpx.Response(200, json=core([linked]))
        if resolution == 'failed':
            return httpx.Response(503)
        return httpx.Response(200, json={'hitCount': 1 if resolution == 'truncated' else 0,
                                       'resultList': {'result': []}})
    calls = setup_transport(monkeypatch, handler)
    result = fetch.fetch_publication_metadata([A])[A]
    assert result['status'] == 'fresh'
    assert result.get('incomplete', False) is (resolution != 'empty')
    assert len(calls) == 2


@pytest.mark.parametrize('genre', ['Review', 'News & Views', 'Perspective', 'Research Highlight'])
def test_time_budget_defers_commentary_lookup_without_losing_type(monkeypatch, genre):
    clock = [10.0]
    monkeypatch.setattr(fetch.time, 'monotonic', lambda: clock[0])
    original_types = fetch._types
    def types_after_deadline(*args, **kwargs):
        original_types(*args, **kwargs)
        clock[0] = 19.0
    monkeypatch.setattr(fetch, '_types', types_after_deadline)
    calls = setup_transport(monkeypatch, lambda request: httpx.Response(
        200, json=core([article(A, '1', [genre])])))
    result = fetch.fetch_publication_metadata([A])[A]
    assert result['status'] == 'fresh'
    assert result['metadata']['publication_types'] == [genre]
    assert result['incomplete'] is True
    assert len(calls) == 1
