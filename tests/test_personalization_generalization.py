"""Source cancellation preserves the previous digest."""
from dailydigest.models import SourceSpec


def test_ingest_cancellation_stops_before_next_source(monkeypatch):
    from dailydigest import pipeline

    fetched = []
    monkeypatch.setattr(pipeline, 'init_db', lambda: None)
    monkeypatch.setattr(pipeline, 'load_sources', lambda: [SourceSpec(name=name, kind='rss', url='https://example.test/feed') for name in ['first', 'second']])
    class Source:
        def fetch(self, spec, **kwargs):
            fetched.append(spec.name)
            return []
    monkeypatch.setattr(pipeline, 'dispatch_source', lambda spec: Source())
    import pytest
    with pytest.raises(pipeline.BrewCancelled):
        pipeline.ingest_all(should_cancel=lambda: bool(fetched))
    assert fetched == ['first']


def test_cancellation_inside_final_request_is_reported_as_stopped(monkeypatch):
    from dailydigest import pipeline
    import pytest

    cancelled, writes = [False], []
    class Source:
        def fetch(self, spec, **kwargs):
            cancelled[0] = True
            raise RuntimeError('adapter wrapped cancellation during the final request')
    monkeypatch.setattr(pipeline, 'init_db', lambda: None)
    monkeypatch.setattr(pipeline, 'load_sources', lambda: [SourceSpec(name='last', kind='rss', url='https://example.test/feed')])
    monkeypatch.setattr(pipeline, 'dispatch_source', lambda _: Source())
    monkeypatch.setattr(pipeline.health, 'record', lambda _: writes.append('health'))
    monkeypatch.setattr(pipeline, 'upsert_items', lambda _: writes.append('items'))
    with pytest.raises(pipeline.BrewCancelled, match='cancelled during source collection'):
        pipeline.ingest_all(should_cancel=lambda: cancelled[0])
    assert writes == []
