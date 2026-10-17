from types import SimpleNamespace

import pytest

from dailydigest import summarize
from dailydigest.dedupe import filter_english
from dailydigest.models import Item
from dailydigest.rank.ranker import _downweight_hit


TITLES = [
    'Perovskite solar cells', 'DNA origami nanostructures',
    'RNA nanotechnology', 'Targeted protein degradation', 'Induced proximity',
    'Asymmetric catalysis', 'Molecular glues', 'Gene therapy',
    'Lipid nanoparticles for mRNA delivery', 'CRISPR genome editing',
    'Protein engineering', 'Quantum materials', 'Single-cell transcriptomics',
    'Nucleic acid therapeutics', 'Photoredox catalysis',
]


def item(title, abstract=''):
    return Item(source='fixture', section='research', external_id=title,
                url='https://example.org/paper', title=title, abstract=abstract)


@pytest.mark.parametrize('title', TITLES)
def test_short_technical_titles_are_not_language_evidence(title):
    row = item(title)
    assert filter_english([row]) == [row]


def test_language_filter_rejects_clear_long_non_english_text():
    french = item('Une nouvelle méthode', 'Cette étude présente une nouvelle méthode de recherche. '
                  'Les résultats montrent une amélioration importante de la précision. ' * 5)
    english = item('A new method', 'This study presents a new research method. '
                   'The results show an important improvement in accuracy. ' * 5)
    assert filter_english([french, english]) == [english]


@pytest.mark.parametrize(('title', 'expected'), [
    ('A certain mechanism maintains stability', False),
    ('The main result', False), ('AI-driven protein design', True),
    ('ai for catalysis', True), ('An artificial intelligence model', False),
])
def test_short_interest_terms_match_words(title, expected, monkeypatch):
    row = SimpleNamespace(title=title, abstract='')
    assert _downweight_hit(row, ['AI', ' ']) is expected
    monkeypatch.setattr(summarize, '_READER_KEYWORDS', ['AI'])
    assert summarize._matched_interests(row) == (['AI'] if expected else [])


def test_phrase_whitespace_is_normalized():
    assert _downweight_hit(SimpleNamespace(title='RNA\n delivery', abstract=''), [' RNA delivery '])


def test_ingest_reports_language_drops_in_progress_and_source_audit(monkeypatch):
    from dailydigest import pipeline

    rows = [item('DNA origami nanostructures'), item('Une nouvelle méthode',
        'Cette étude présente une nouvelle méthode de recherche et les résultats obtenus. ' * 6)]
    monkeypatch.setattr(pipeline, 'init_db', lambda: None)
    monkeypatch.setattr(pipeline, 'load_sources', lambda: [SimpleNamespace(name='fixture')])
    monkeypatch.setattr(pipeline, 'dispatch_source', lambda _: SimpleNamespace(fetch=lambda *args, **kwargs: rows))
    monkeypatch.setattr(pipeline.health, 'record', lambda _: None)
    monkeypatch.setattr(pipeline, 'dedupe_by_url', lambda items: items)
    monkeypatch.setattr(pipeline, 'upsert_items', lambda items: len(items))
    stats, events = [], []
    assert pipeline.ingest_all(stats_out=stats, progress_callback=lambda *event: events.append(event)) == 1
    assert stats[0].diagnostics['language_filter'] == {'input': 2, 'dropped': 1}
    assert next(payload for stage, payload in events if stage == 'dedupe_done')['language_dropped'] == 1
