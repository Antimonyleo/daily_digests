from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from dailydigest.publication_enrichment import enrich_publications, merge_publication_metadata

NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)
ON = SimpleNamespace(publication_enrichment=True)


def row(doi='10.1234/example', metadata=None):
    return SimpleNamespace(external_id=doi, url='', metadata_json=json.dumps(metadata or {}))


def metadata(item):
    return json.loads(item.metadata_json)


def fetch(status='fresh', evidence=None, calls=None):
    def run(dois, *, deadline_seconds):
        assert deadline_seconds == 8.0
        if calls is not None:
            calls.append(dois)
        return {doi: {'status': status, 'metadata': evidence or {}} for doi in dois}
    return run


def test_disabled_is_offline_and_leaves_row_untouched():
    item = row()
    def forbidden(*args, **kwargs):
        pytest.fail('disabled enrichment performed IO')
    assert not enrich_publications([item], settings=SimpleNamespace(), fetcher=forbidden)['enabled']
    assert item.metadata_json == '{}'


def test_cap_respects_order_deduplicates_and_updates_same_doi():
    rows = [row(f'10.1234/{i}') for i in range(14)]
    duplicate = row('https://doi.org/10.1234/0')
    calls = []
    report = enrich_publications([*rows, duplicate], settings=ON, now=NOW,
        fetcher=fetch(evidence={'publication_types': ['Perspective']}, calls=calls))
    assert calls == [[f'10.1234/{i}' for i in range(12)]]
    assert report['requested'] == report['fresh'] == 12
    assert report['deferred'] == 2
    assert metadata(duplicate)['publication_types'] == ['Perspective']
    assert metadata(rows[12]) == {}


@pytest.mark.parametrize(('status', 'ttl'), [('fresh', timedelta(days=7)),
    ('missing', timedelta(days=1)), ('failed', timedelta(minutes=15))])
def test_cache_ttl_and_retry(status, ttl):
    item, calls = row(), []
    enrich_publications([item], settings=ON, now=NOW, fetcher=fetch(status, calls=calls))
    report = enrich_publications([item], settings=ON, now=NOW + ttl - timedelta(seconds=1),
                                fetcher=fetch(calls=calls))
    assert report['cached'] == 1
    assert len(calls) == 1
    enrich_publications([item], settings=ON, now=NOW + ttl, fetcher=fetch(calls=calls))
    assert len(calls) == 2


def test_failure_preserves_prior_success_timestamp_and_positive_evidence():
    item = row(metadata={'publication_types': ['News & Views'], 'saved': True})
    enrich_publications([item], settings=ON, now=NOW, fetcher=fetch())
    old = metadata(item)['publication_enrichment']['checked_at']
    def unavailable(*args, **kwargs):
        raise OSError('outage')
    report = enrich_publications([item], settings=ON, now=NOW + timedelta(days=8), fetcher=unavailable)
    actual = metadata(item)
    assert actual['publication_types'] == ['News & Views']
    assert actual['saved'] is True
    assert actual['publication_enrichment']['checked_at'] == old
    assert actual['publication_enrichment']['status'] == 'failed'
    assert report['failed'] == 1 and report['missing'] == 0


def test_duplicate_cached_evidence_applies_without_fetch():
    old = row(metadata={'publication_types': ['Perspective']})
    enrich_publications([old], settings=ON, now=NOW, fetcher=fetch())
    new = row(metadata={'source_field': 'retained'})
    report = enrich_publications([new, old], settings=ON, now=NOW,
        fetcher=lambda *a, **kw: pytest.fail('cache ignored'))
    assert report['cached'] == 1
    assert metadata(new)['publication_types'] == ['Perspective']
    assert metadata(new)['source_field'] == 'retained'


@pytest.mark.parametrize('result', [None, {}, {'10.1234/example': None},
    {'10.1234/example': {'status': 'missing'}},
    {'10.1234/example': {'status': 'unexpected', 'metadata': {}}}])
def test_incomplete_or_malformed_response_is_failure(result):
    item = row()
    report = enrich_publications([item], settings=ON, now=NOW, fetcher=lambda *a, **kw: result)
    assert report['failed'] == 1 and report['missing'] == 0


def test_evidence_merge_preserves_fields_and_retraction():
    old = {'publication_types': ['Review'], 'publication_type_evidence': [{'source': 'rss'}],
           'companion_of_dois': ['10.1234/a'], 'is_retracted': True, 'doi': '10.1234/own'}
    new = {'publication_types': ['Review', 'Perspective'],
           'publication_type_evidence': [{'source': 'Crossref'}],
           'companion_of_dois': ['10.1234/b'], 'is_retracted': False, 'doi': '10.1234/other'}
    actual = merge_publication_metadata(old, new)
    assert actual['publication_types'] == ['Review', 'Perspective']
    assert actual['companion_of_dois'] == ['10.1234/a', '10.1234/b']
    assert actual['is_retracted'] and actual['doi'] == '10.1234/own'
    assert len(actual['publication_type_evidence']) == 2
    assert old['publication_types'] == ['Review']


def test_cancellation_before_fetch_propagates():
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        enrich_publications([row()], settings=ON, now=NOW, check_cancel=cancel,
            fetcher=lambda *a, **kw: pytest.fail('called after cancellation'))


def test_unknown_doi_does_not_trigger_fetch():
    report = enrich_publications([row('not-a-doi')], settings=ON, now=NOW,
        fetcher=lambda *a, **kw: pytest.fail('invalid DOI sent'))
    assert report['without_doi'] == 1 and report['requested'] == 0


def test_fetch_cancellation_is_not_cached_as_provider_failure():
    from dailydigest.brew import BrewCancelled

    item = row()
    def cancelled(*args, **kwargs):
        raise BrewCancelled('user stopped brew')
    with pytest.raises(BrewCancelled, match='user stopped brew'):
        enrich_publications([item], settings=ON, now=NOW, fetcher=cancelled)
    assert metadata(item) == {}


def test_detached_store_row_and_dictionary_are_supported():
    from dailydigest.store import ItemRow

    database_row = ItemRow(external_id='10.1234/example', metadata_json='{}')
    dictionary = {'external_id': '10.1234/example', 'metadata': {'retained': True}}
    enrich_publications([database_row, dictionary], settings=ON, now=NOW,
                        fetcher=fetch(evidence={'publication_types': ['Perspective']}))
    assert metadata(database_row)['publication_types'] == ['Perspective']
    assert dictionary['metadata']['publication_types'] == ['Perspective']
    assert dictionary['metadata']['retained'] is True
    assert json.loads(dictionary['metadata_json']) == dictionary['metadata']



def test_partial_success_retains_type_but_retries_relation_after_fifteen_minutes():
    item, calls = row(), []
    def partial(dois, *, deadline_seconds):
        calls.append(dois)
        return {doi: {'status': 'fresh', 'incomplete': True,
                      'metadata': {'publication_types': ['Review']}} for doi in dois}
    enrich_publications([item], settings=ON, now=NOW, fetcher=partial)
    actual = metadata(item)
    assert actual['publication_types'] == ['Review']
    assert actual['publication_enrichment'] == {
        'version': 1, 'status': 'fresh', 'incomplete': True, 'checked_at': NOW.isoformat()}
    result = enrich_publications([item], settings=ON, now=NOW + timedelta(minutes=14), fetcher=partial)
    assert result['cached'] == 1 and len(calls) == 1
    enrich_publications([item], settings=ON, now=NOW + timedelta(minutes=15), fetcher=partial)
    assert len(calls) == 2


def test_complete_retry_removes_partial_marker_and_restores_seven_day_cache():
    item = row(metadata={'publication_types': ['Review'], 'publication_enrichment': {
        'version': 1, 'status': 'fresh', 'incomplete': True, 'checked_at': NOW.isoformat()}})
    calls = []
    checked = NOW + timedelta(minutes=15)
    enrich_publications([item], settings=ON, now=checked, fetcher=fetch(evidence={
        'companion_of_dois': ['10.1234/parent'],
        'companion_relation_evidence': [{'rel': 'is-commentary-on', 'doi': '10.1234/parent'}],
    }, calls=calls))
    actual = metadata(item)
    assert actual['publication_types'] == ['Review']
    assert actual['companion_of_dois'] == ['10.1234/parent']
    assert actual['publication_enrichment'] == {
        'version': 1, 'status': 'fresh', 'checked_at': checked.isoformat()}
    report = enrich_publications([item], settings=ON, now=checked + timedelta(days=6),
                                 fetcher=fetch(calls=calls))
    assert report['cached'] == 1 and len(calls) == 1
