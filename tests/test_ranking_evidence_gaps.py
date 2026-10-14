"""Ranking must use resolved evidence rather than metadata timing or claim wording."""
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dailydigest.rank import enrich, ranker, source_quality as quality


NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)


def article(i=1, source='Small (Wiley)', **kwargs):
    fields = dict(id=i, source=source, section='research', title='Measured molecular assembly',
                  abstract='We evaluate assembly with independent measurements.',
                  url=f'https://doi.org/10.1234/test-{i}', external_id=str(i),
                  published_at=NOW - timedelta(days=30), metadata_json='{}')
    return SimpleNamespace(**(fields | kwargs))


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.setattr(quality, '_configured_sources_cached', lambda: {})
    monkeypatch.setattr(quality, '_research_quality_weight', lambda: 1.4)
    settings = SimpleNamespace(citation_enrichment=True, venue_quality_weight=.18,
                               citation_polite_email='')
    monkeypatch.setattr(enrich, 'save_enrichment', lambda _entries: None)

    def prepare(rows, entries, fetched=False):
        monkeypatch.setattr(enrich, 'load_cached_enrichment', lambda _ids: {} if fetched else entries)
        def fetch(dois, email):
            assert fetched
            return {enrich.derive_doi(row): entries[row.id] for row in rows if row.id in entries}
        enrich.prepare_enrichment(rows, settings=settings, fetcher=fetch)
        return settings

    return prepare


@pytest.mark.parametrize('fetched', [False, True])
@pytest.mark.parametrize('source,impact,venue', [
    ('Small (Wiley)', .4, 'Small'), ('Unlisted Journal', 9., 'Unlisted Journal'),
    ('OpenAlex (your topics)', 9., 'Small'),
])
def test_cache_and_fetch_evidence_produce_same_quality_score(preparation, fetched, source, impact, venue):
    row = article(source=source)
    settings = preparation([row], {1: {'cited_by_count': 0, 'venue_impact': impact, 'venue': venue}}, fetched)
    score = quality.quality_adjusted_score(row, .76)
    result = enrich.enrich_scored([(row, score)], settings=settings, prepared_only=True, now=NOW)
    expected = article(2, source, venue_name=venue, venue_low_impact=impact == .4,
                       venue_high_impact=impact == 9.)
    expected_score = quality.quality_adjusted_score(expected, .76)
    if source == 'Unlisted Journal':
        expected_score += .09
    assert result[0][1] == pytest.approx(expected_score)


@pytest.mark.parametrize('mode', ['cosine', 'hybrid_knn'])
def test_quality_evidence_hook_runs_before_fusion_without_second_embedding_pass(monkeypatch, preparation, mode):
    from dailydigest.config import get_settings
    from dailydigest import votes
    rows = [article(), article(2, 'Science')]
    calls = []
    def embed(items):
        calls.append(len(items))
        return np.array([[.76, np.sqrt(1-.76**2)]] * len(items), dtype=np.float32)
    monkeypatch.setattr(ranker, 'embed_item_rows', embed)
    monkeypatch.setattr(ranker, '_vote_count', lambda: 100)
    monkeypatch.setattr(votes, 'knn_preference_scores', lambda _rows: np.array([.1, .2], dtype=np.float32))
    monkeypatch.setenv('SCORING_MODE', mode)
    get_settings.cache_clear()
    def hook(items, scores):
        assert scores == pytest.approx([.76, .76])
        preparation(items, {1: {'cited_by_count': 0, 'venue_impact': .4, 'venue': 'Small'}})
    _, features = ranker.score_items_with_features(rows, np.array([[1., 0.]]), [], prepare_quality=hook)
    assert calls == [2]
    assert features[1]['quality_adjustment'] < 0
    assert features[1]['confidence_score'] < features[2]['confidence_score']


def test_claim_words_do_not_waive_unknown_venue_penalty(monkeypatch):
    monkeypatch.setattr(quality, '_configured_sources_cached', lambda: {})
    weak = article(source='Unlisted Journal', abstract='First demonstration of unprecedented assembly methods with orders of magnitude improvements.')
    verified = article(2, 'Unlisted Journal', abstract=weak.abstract, venue_high_impact=True)
    assert quality.novelty_score(weak) >= .5
    assert quality.quality_adjusted_score(weak, .81) < quality.quality_adjusted_score(verified, .81) - .09


@pytest.mark.parametrize('published', [None, NOW + timedelta(days=1)])
def test_citation_velocity_requires_nonfuture_publication_date(published):
    assert enrich.citation_score(30, published, now=NOW) == 0


def test_update_date_is_not_publication_age(preparation):
    row = article(published_at=NOW-timedelta(hours=1), metadata_json=json.dumps({'publication_date_source': 'updated'}))
    settings = preparation([row], {1: {'cited_by_count': 30, 'venue_impact': 9., 'venue': 'Small'}})
    assert enrich.enrich_scored([(row, .6)], settings=settings, prepared_only=True, now=NOW)[0][1] == .6
    assert ranker._freshness_penalty(row) == 0


def test_prepared_enrichment_is_reused_without_fetch_or_cache_read(monkeypatch, preparation):
    row = article()
    settings = preparation([row], {1: {'cited_by_count': 12, 'venue_impact': 9., 'venue': 'Small'}})
    monkeypatch.setattr(enrich, 'load_cached_enrichment', lambda _ids: pytest.fail('Second cache read'))
    monkeypatch.setattr(enrich, 'fetch_openalex', lambda *_args: pytest.fail('Second provider call'))
    result = enrich.enrich_scored([(row, .6)], settings=settings, prepared_only=True, now=NOW)
    assert result[0][1] > .6
    assert row.enrichment_evidence['citation_boost'] > 0
    assert row.enrichment_evidence['venue_adjustment'] == 0


@pytest.mark.parametrize('field,value', [
    ('dc_identifier', 'doi:10.1234/from-field'), ('prism_doi', '10.1234/from-field'),
    ('id', 'https://doi.org/10.1234/from-field'),
])
def test_rss_retains_explicit_doi(monkeypatch, field, value):
    from dailydigest.ingest import rss
    from dailydigest.models import SourceSpec
    entry = dict(title='A measured method', link='https://publisher.test/article', **{field:value})
    monkeypatch.setattr(rss, '_http_get_bytes', lambda _url: b'feed')
    monkeypatch.setattr(rss.feedparser, 'parse', lambda _body: SimpleNamespace(bozo=False, entries=[entry]))
    item = rss.RSSSource().fetch(SourceSpec(name='Nature', kind='rss', section='research', url='https://publisher.test/feed'))[0]
    assert item.metadata['doi'] == '10.1234/from-field'


def test_rss_keeps_update_date_provenance(monkeypatch):
    from dailydigest.ingest import rss
    from dailydigest.models import SourceSpec
    entry = dict(title='A measured method', link='https://publisher.test/article', updated_parsed=NOW.timetuple())
    monkeypatch.setattr(rss, '_http_get_bytes', lambda _url: b'feed')
    monkeypatch.setattr(rss.feedparser, 'parse', lambda _body: SimpleNamespace(bozo=False, entries=[entry]))
    item = rss.RSSSource().fetch(SourceSpec(name='Nature', kind='rss', section='research', url='https://publisher.test/feed'))[0]
    assert item.metadata['publication_date_source'] == 'updated'


def test_rss_does_not_claim_doi_from_related_article_or_invent_identifier():
    from dailydigest.ingest.rss import _entry_metadata
    entry = {'id': 's41586-026-12345', 'link': 'https://publisher.test/article',
             'links': [{'rel': 'related', 'href': 'https://doi.org/10.1234/cited-work'}]}
    assert 'doi' not in _entry_metadata(entry)
    entry['links'][0]['rel'] = 'alternate'
    assert _entry_metadata(entry)['doi'] == '10.1234/cited-work'


def test_actual_rss_xml_identifier_and_date_are_preserved(monkeypatch):
    from dailydigest.ingest import rss
    from dailydigest.models import SourceSpec
    body = b'''<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:prism="http://prismstandard.org/namespaces/basic/2.0/"><channel><title>Research</title>
    <item><title>A measured method</title><link>https://publisher.test/article</link>
    <dc:identifier>doi:10.1234/from-xml</dc:identifier>
    <pubDate>Mon, 05 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>'''
    monkeypatch.setattr(rss, '_http_get_bytes', lambda _url: body)
    item = rss.RSSSource().fetch(SourceSpec(name='Nature', kind='rss', section='research', url='https://publisher.test/feed'))[0]
    assert item.metadata == {'doi': '10.1234/from-xml', 'publication_date_source': 'published'}


def test_ranking_preparation_skips_clearly_off_topic_and_nonresearch(monkeypatch):
    from dailydigest import pipeline
    settings = SimpleNamespace(citation_enrichment=True, venue_relevance_credit=.1,
                               min_topic_relevance=.65, adaptive_section_sizes=True)
    monkeypatch.setattr(pipeline, 'get_settings', lambda: settings)
    prepared = []
    monkeypatch.setattr(enrich, 'prepare_enrichment', lambda rows, **_kwargs: prepared.extend(row.id for row in rows))
    rows = [article(1), article(2), article(3), article(4, section='world')]
    pipeline._prepare_ranking_enrichment(rows, [.76, .61, .2, .9])
    assert prepared == [1, 2]


def test_disabled_preparation_keeps_scoring_local(monkeypatch):
    from dailydigest import pipeline
    monkeypatch.setattr(pipeline, 'get_settings', lambda: SimpleNamespace(citation_enrichment=False))
    monkeypatch.setattr(enrich, 'prepare_enrichment', lambda *_args, **_kwargs: pytest.fail('Enrichment enabled implicitly'))
    pipeline._prepare_ranking_enrichment([article()], [.8])
