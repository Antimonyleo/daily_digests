"""Publisher-declared highlights are reports about research, not new studies."""
from types import SimpleNamespace

import pytest

from dailydigest.dedupe import dedupe_ranking_candidates
from dailydigest.ingest.rss import _entry_metadata
from dailydigest.models import Item
from dailydigest.rank import source_quality as quality
from dailydigest import store


@pytest.mark.parametrize('genre', ['Research Highlight', 'Research Highlights'])
@pytest.mark.parametrize('field', ['dc_type', 'prism_genre', 'prism_section'])
def test_explicit_highlight_genre_survives_rss_and_dedupe(genre, field):
    metadata = _entry_metadata({field: genre, 'prism_doi': '10.9999/highlight'})
    assert metadata['publication_types'] == [genre]
    evidence = Item(source='Publisher', section='research', external_id='typed',
                    url='https://publisher.test/highlight', title='Engineering molecular assemblies',
                    abstract='We discuss a new method and experimental results.', metadata=metadata)
    generic = evidence.model_copy(update={'source': 'OpenAlex', 'external_id': 'generic',
        'url': 'https://doi.org/10.9999/highlight',
        'abstract': 'A detailed method and experimental results. ' * 40,
        'metadata': {'work_type': 'article'}})
    result = dedupe_ranking_candidates([generic, evidence])
    assert len(result) == 1
    assert quality.content_type(result[0]) == 'commentary'
    assert quality.should_skip_item(result[0])
    assert result[0].metadata['publication_type_evidence'] == [{'field': field, 'value': genre}]


@pytest.mark.parametrize('metadata', [
    {'publication_types': ['Journal Article', 'Research Highlight']},
    {'work_type': 'research highlights'},
])
def test_declared_highlight_is_excluded_even_with_method_words(metadata):
    article = SimpleNamespace(title='Molecular assembly', abstract='We develop a method and report results.',
        section='research', source='Nature', metadata=metadata)
    assert quality.should_skip_item(article)
    assert quality.content_type(article) == 'commentary'
    article.section = 'industry'
    assert not quality.should_skip_item(article)


def test_incidental_highlight_words_do_not_exclude_primary_research():
    metadata = _entry_metadata({'dc_subject': 'Research Highlight',
                               'tags': [{'term': 'Research Highlights'}]})
    assert 'publication_types' not in metadata
    article = SimpleNamespace(title='Programmable molecular assemblies',
        abstract='This research highlights a new method. We report experimental measurements.',
        section='research', source='Nature', metadata={'work_type': 'article'})
    assert not quality.should_skip_item(article)
    assert quality.content_type(article) == 'method'


def test_generic_refresh_keeps_publisher_highlight_evidence():
    from sqlalchemy import select

    evidence = {'field': 'prism_section', 'value': 'Research Highlight'}
    item = Item(source='Publisher', section='research', external_id='highlight',
                url='https://publisher.test/highlight', title='Engineering molecular assemblies',
                metadata={'publication_types': ['Research Highlight'],
                          'publication_type_evidence': [evidence]})
    store.upsert_items([item])
    store.upsert_items([item.model_copy(update={'metadata': {'work_type': 'article'}})])
    with store.session_scope() as session:
        saved = session.scalar(select(store.ItemRow))
        assert quality.should_skip_item(saved)
        assert quality.content_type(saved) == 'commentary'
        assert quality.research_exclusion_metadata(saved)['publication_type_evidence'] == [evidence]
