"""Withdrawal observations survive a failed scan without counting as new supply."""
from types import SimpleNamespace

import pytest

from dailydigest import pipeline, store
from dailydigest.models import Item, SourceSpec


@pytest.mark.parametrize('fetch_fails', [False, True])
def test_withdrawal_is_recorded_before_empty_supply_aborts(monkeypatch, fetch_fails):
    notice = Item(source='bioRxiv', section='research', external_id='withdrawn',
                  url='https://doi.org/10.1101/fixture', title='Withdrawn paper',
                  metadata={'is_retracted': True})
    observed = []

    def fetch(*args, **kwargs):
        if fetch_fails:
            raise RuntimeError('A later fixture page failed')
        return []

    source = SimpleNamespace(fetch=fetch, last_fetch_withdrawals=[notice])
    spec = SourceSpec(name='bioRxiv', kind='biorxiv', section='research')
    monkeypatch.setattr(pipeline, 'init_db', lambda: None)
    monkeypatch.setattr(pipeline, 'load_sources', lambda: [spec])
    monkeypatch.setattr(pipeline, 'dispatch_source', lambda spec: source)
    monkeypatch.setattr(pipeline.health, 'record', lambda rows: None)
    monkeypatch.setattr(store, 'mark_research_withdrawn', lambda rows: observed.extend(rows), raising=False)
    stats = []
    with pytest.raises(RuntimeError, match='No items were retrieved'):
        pipeline.ingest_all(stats_out=stats)
    assert observed == [notice]
    assert stats[0].items == 0
    assert stats[0].ok is not fetch_fails


def test_a_new_sibling_copy_of_a_withdrawal_never_counts_as_live_supply(monkeypatch):
    notice = Item(source='bioRxiv', section='research', external_id='withdrawn',
                  url='https://doi.org/10.1101/fixture', title='Withdrawn paper',
                  metadata={'is_retracted': True})
    sibling = notice.model_copy(update={'source': 'Journal feed', 'metadata': {}})
    specs = [SourceSpec(name='bioRxiv', kind='biorxiv', section='research'),
             SourceSpec(name='Journal feed', kind='rss', section='research', url='https://fixture.invalid/rss')]
    adapters = {'bioRxiv': SimpleNamespace(fetch=lambda *a, **k: [], last_fetch_withdrawals=[notice]),
                'Journal feed': SimpleNamespace(fetch=lambda *a, **k: [sibling])}
    observed = []
    monkeypatch.setattr(pipeline, 'init_db', lambda: None)
    monkeypatch.setattr(pipeline, 'load_sources', lambda: specs)
    monkeypatch.setattr(pipeline, 'dispatch_source', lambda spec: adapters[spec.name])
    monkeypatch.setattr(pipeline.health, 'record', lambda rows: None)
    monkeypatch.setattr(store, 'mark_research_withdrawn', lambda rows: observed.extend(rows), raising=False)
    with pytest.raises(RuntimeError, match='No items were retrieved'):
        pipeline.ingest_all()
    assert observed == [notice]
