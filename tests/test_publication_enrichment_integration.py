"""Publication identity lookup through ranking and isolated persistent state."""
import json
from types import SimpleNamespace

from sqlalchemy import select

from dailydigest import pipeline, publication_fetch, store
from dailydigest.models import Item


def settings(**overrides):
    values = dict(publication_enrichment=True, citation_enrichment=False,
                  venue_relevance_credit=0.0, min_topic_relevance=0.5,
                  adaptive_section_sizes=True)
    return SimpleNamespace(**(values | overrides))


def rows(count=1):
    store.upsert_items([Item(source='Test', section='research', external_id=f'paper-{i}',
        title=f'Research {i}', url=f'https://doi.org/10.9999/{i}',
        metadata={'doi': f'10.9999/{i}', 'unrelated': {'preserve': True}})
        for i in range(count)])
    with store.session_scope() as session:
        result = list(session.scalars(select(store.ItemRow).order_by(store.ItemRow.id)))
        session.expunge_all()
        return result


def lookup(calls):
    def fetch(dois, *, deadline_seconds):
        calls.append(dois)
        return {doi: {'status': 'fresh', 'metadata': {
            'publication_types': ['News & Views'],
            'publication_type_evidence': [{'source': 'Crossref', 'value': 'News & Views'}],
            'companion_of_dois': ['10.9999/parent'],
        }} for doi in dois}
    return fetch


def test_disabled_pipeline_does_not_lookup_or_write(monkeypatch):
    monkeypatch.setattr(pipeline, 'get_settings', lambda: settings(publication_enrichment=False))
    def forbidden(*args, **kwargs):
        raise AssertionError('offline enrichment accessed provider or persistence')
    monkeypatch.setattr(publication_fetch, 'fetch_publication_metadata', forbidden)
    monkeypatch.setattr(store, 'persist_publication_metadata', forbidden)
    pipeline._prepare_ranking_enrichment(rows(), [0.9])


def test_pipeline_prioritizes_relevance_and_caps_database_writes(monkeypatch):
    candidates = rows(15)
    calls, writes = [], []
    monkeypatch.setattr(pipeline, 'get_settings', settings)
    monkeypatch.setattr(publication_fetch, 'fetch_publication_metadata', lookup(calls))
    monkeypatch.setattr(store, 'persist_publication_metadata', lambda changed: writes.extend(changed))
    scores = [0.6 + i * 0.01 for i in range(15)]
    pipeline._prepare_ranking_enrichment(candidates, scores)
    assert calls == [[f'10.9999/{i}' for i in range(14, 2, -1)]]
    assert len(writes) == 12
    assert {r.id for r in writes} == {r.id for r in candidates[3:]}


def test_pipeline_excludes_irrelevant_and_nonresearch_rows(monkeypatch):
    candidates = rows(3)
    candidates[2].section = 'world'
    calls = []
    monkeypatch.setattr(pipeline, 'get_settings', settings)
    monkeypatch.setattr(publication_fetch, 'fetch_publication_metadata', lookup(calls))
    pipeline._prepare_ranking_enrichment(candidates, [0.49, 0.5, 0.99])
    assert calls == [['10.9999/1']]


def test_pipeline_persists_cache_and_unchanged_repeat_has_no_writes(monkeypatch):
    candidates, calls = rows(), []
    monkeypatch.setattr(pipeline, 'get_settings', settings)
    monkeypatch.setattr(publication_fetch, 'fetch_publication_metadata', lookup(calls))
    pipeline._prepare_ranking_enrichment(candidates, [0.9])
    with store.session_scope() as session:
        saved = session.get(store.ItemRow, candidates[0].id)
        assert store.item_metadata(saved)['publication_enrichment']['status'] == 'fresh'
        assert store.item_metadata(saved)['publication_types'] == ['News & Views']
        session.expunge(saved)
    writes = []
    monkeypatch.setattr(store, 'persist_publication_metadata', lambda changed: writes.extend(changed))
    pipeline._prepare_ranking_enrichment([saved], [0.9])
    assert len(calls) == 1
    assert writes == []


def test_store_merge_preserves_concurrent_metadata_and_reading_state():
    stale = rows()[0]
    with store.session_scope() as session:
        saved = session.get(store.ItemRow, stale.id)
        saved.title = 'Updated title'
        saved.summary = 'Existing reading summary'
        saved.score = 0.93
        existing = store.item_metadata(saved)
        existing['newer_field'] = {'do_not_erase': True}
        existing['publication_types'] = ['Perspective']
        saved.metadata_json = json.dumps(existing)
    stale.title, stale.summary, stale.score = 'stale', 'stale', 0
    stale.metadata_json = json.dumps({'publication_types': ['News & Views'],
        'publication_enrichment': {'version': 1, 'status': 'fresh',
                                  'checked_at': '2026-10-22T00:00:00+00:00'},
        'unrelated': 'must not replace stored value'})
    store.persist_publication_metadata([stale])
    with store.session_scope() as session:
        saved = session.get(store.ItemRow, stale.id)
        actual = store.item_metadata(saved)
        assert actual['newer_field'] == {'do_not_erase': True}
        assert actual['unrelated'] == {'preserve': True}
        assert actual['publication_types'] == ['Perspective', 'News & Views']
        assert actual['publication_enrichment']['status'] == 'fresh'
        assert (saved.title, saved.summary, saved.score) == ('Updated title', 'Existing reading summary', 0.93)


def test_generic_ingestion_refresh_retains_persisted_cache():
    item = rows()[0]
    item.metadata_json = json.dumps({'publication_types': ['Perspective'],
        'publication_enrichment': {'version': 1, 'status': 'missing',
                                  'checked_at': '2026-10-22T00:00:00+00:00'}})
    store.persist_publication_metadata([item])
    store.upsert_items([Item(source='Test', section='research', external_id='paper-0',
        title='Research 0', url='https://doi.org/10.9999/0', metadata={'work_type': 'article'})])
    with store.session_scope() as session:
        actual = store.item_metadata(session.get(store.ItemRow, item.id))
        assert actual['publication_enrichment']['status'] == 'missing'
        assert actual['publication_types'] == ['Perspective']


def test_new_publication_evidence_is_rejected_by_post_scoring_quality_gate(monkeypatch):
    """Exercise lookup callback then the production quality gate, not a full Brew."""
    candidates = rows(3)
    for candidate in candidates:
        candidate.title = 'Assembly methods for programmable lattice structures'
        candidate.abstract = 'We demonstrate an experimentally verified method for assembling lattices.'
    initially_eligible, _ = pipeline._quality_gate(candidates)
    assert initially_eligible == candidates
    monkeypatch.setattr(pipeline, 'get_settings', settings)
    def fetch(dois, *, deadline_seconds):
        evidence = {
            '10.9999/0': {'publication_types': ['News & Views'],
                         'companion_of_dois': ['10.9999/2']},
            '10.9999/1': {'publication_types': ['Retracted Publication'], 'is_retracted': True},
            '10.9999/2': {'publication_types': ['Research Article']},
        }
        return {doi: {'status': 'fresh', 'metadata': evidence[doi]} for doi in dois}
    monkeypatch.setattr(publication_fetch, 'fetch_publication_metadata', fetch)
    pipeline._prepare_ranking_enrichment(candidates, [0.99, 0.98, 0.90])
    eligible, drops = pipeline._quality_gate(candidates)
    assert eligible == [candidates[2]]
    assert {drop['item_id'] for drop in drops} == {candidates[0].id, candidates[1].id}
