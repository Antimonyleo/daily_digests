"""Topic searches spend their reserved slots on article-level evidence."""
from datetime import datetime, timezone

from dailydigest.models import Profile, SourceSpec
from dailydigest.ingest import openalex


def test_profile_search_avoids_fulltext_citation_only_matches(monkeypatch):
    monkeypatch.setattr('dailydigest.config.load_profile', lambda: Profile(
        bio='', keywords=['asymmetric organocatalysis']))
    calls = []

    def fetch(url, params, headers):
        calls.append(dict(params))
        scoped = params.get('search.title_abstract_keywords') == 'asymmetric organocatalysis'
        return {'meta': {'count': 1}, 'results': [{
            'id': 'https://openalex.org/W123', 'type': 'article',
            'title': ('New chiral organic catalysts for asymmetric synthesis' if scoped
                      else 'Marine polysaccharides for breast cancer therapy'),
            'publication_date': datetime.now(timezone.utc).date().isoformat(),
            'primary_location': {'landing_page_url': 'https://example.org/paper'},
        }]}

    monkeypatch.setattr(openalex, '_get_json', fetch)
    source = openalex.OpenAlexSource()
    items = source.fetch(SourceSpec(name='topics', kind='openalex', section='research',
                                    profile_driven=True), days=21)
    assert len(items) == 1
    assert items[0].title == 'New chiral organic catalysts for asymmetric synthesis'
    assert 'search' not in calls[0]
    assert calls[0]['search.title_abstract_keywords'] == 'asymmetric organocatalysis'
    assert source.last_fetch_diagnostics['queries'][0]['search_scope'] == 'title_abstract_keywords'


def test_static_openalex_search_keeps_its_existing_broad_scope(monkeypatch):
    calls = []
    monkeypatch.setattr(openalex, '_get_json', lambda url, params, headers: (
        calls.append(dict(params)) or {'meta': {'count': 0}, 'results': []}))
    source = openalex.OpenAlexSource()
    source.fetch(SourceSpec(name='broad', kind='openalex', section='research',
                            query='catalysis'), days=21)
    assert calls[0]['search'] == 'catalysis'
