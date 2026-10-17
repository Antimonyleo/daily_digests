"""Venue metadata must not add a second bonus to an existing source prior."""
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from dailydigest.rank import enrich, source_quality


NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


def row(i, source, *, venue=None, url=None):
    return SimpleNamespace(
        id=i, section='research', source=source, title='Transferable design method',
        abstract='We evaluate a programmable molecular design method with independent measurements.',
        url=url or f'https://doi.org/10.1234/paper-{i}', external_id=str(i),
        published_at=NOW - timedelta(days=30),
        metadata_json=json.dumps({'venue': venue} if venue else {}),
    )


@pytest.fixture
def run_enrichment(monkeypatch):
    monkeypatch.setattr(source_quality, '_configured_sources_cached', lambda: {})
    settings = SimpleNamespace(citation_enrichment=True, venue_quality_weight=.18,
                               citation_polite_email='')

    def run(scored, entries, *, fetched=False):
        monkeypatch.setattr(enrich, 'load_cached_enrichment', lambda ids: {} if fetched else entries)
        saved = []
        monkeypatch.setattr(enrich, 'save_enrichment', lambda data: saved.append(data))

        def fetch(dois, email):
            assert fetched, 'Cached entries must not be fetched again'
            return {enrich.derive_doi(article): entries[article.id]
                    for article, _ in scored if article.id in entries}

        return enrich.enrich_scored(scored, settings=settings, fetcher=fetch, now=NOW)

    return run


@pytest.mark.parametrize('source,venue', [
    ('Small (Wiley)', None), ('Small', None), ('Nature', None),
    ('OpenAlex (your topics)', 'Small'), ('PubMed (your topics)', 'Nature'),
])
@pytest.mark.parametrize('fetched', [False, True])
def test_curated_journal_with_or_without_metadata_has_same_score(run_enrichment, source, venue, fetched):
    enriched = row(1, source, venue=venue)
    no_lookup = row(2, source, venue=venue, url='https://example.test/no-doi')
    out = run_enrichment([(enriched, .6), (no_lookup, .6)], {
        1: {'cited_by_count': 0, 'venue_impact': 9.057942422913927, 'venue': venue or source},
    }, fetched=fetched)
    assert {article.id: score for article, score in out} == {1: .6, 2: .6}
    assert source_quality.is_high_quality_journal_source(enriched)


def test_curated_journal_keeps_actual_citation_boost(run_enrichment):
    article = row(1, 'Small (Wiley)')
    out = run_enrichment([(article, .6)], {
        1: {'cited_by_count': 12, 'venue_impact': 9.06, 'venue': 'Small'},
    })
    expected = .6 + enrich.CITATION_BOOST * enrich.citation_score(12, article.published_at, now=NOW)
    assert out[0][1] == pytest.approx(expected)


@pytest.mark.parametrize('source,venue', [('OpenAlex (your topics)', 'Small'), ('Unlisted journal', 'Unlisted journal')])
def test_newly_discovered_venue_evidence_is_applied_once(run_enrichment, source, venue):
    article = row(1, source)
    assert not source_quality.is_high_quality_journal_source(article)
    out = run_enrichment([(article, .6)], {
        1: {'cited_by_count': 0, 'venue_impact': 9.06, 'venue': venue},
    })
    assert out[0][1] == pytest.approx(.69)
    assert source_quality.source_bucket(article) == 'published_journal'


def test_curated_journal_still_honors_contradictory_low_impact_evidence(run_enrichment):
    article = row(1, 'Small (Wiley)')
    out = run_enrichment([(article, .6)], {
        1: {'cited_by_count': 0, 'venue_impact': .4, 'venue': 'Small'},
    })
    assert out[0][1] < .6
    assert source_quality.source_bucket(article) == 'low_impact_journal'
    assert not source_quality.is_high_quality_journal_source(article)
