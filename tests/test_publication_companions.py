import json
from types import SimpleNamespace

import pytest

from dailydigest.ingest.rss import _entry_metadata
from dailydigest.publications import (
    companion_entries, companion_metadata, companion_target_dois, is_companion, normalize_doi, publication_label,
)


def row(id=1, genre='Research Article', doi='10.1234/parent', **metadata):
    return SimpleNamespace(id=id, title='A paper', url=f'https://doi.org/{doi}',
        external_id=doi, metadata={'publication_types': [genre], **metadata})


@pytest.mark.parametrize('genre,label', [
    ('News and Views', 'News & Views'), ('Perspective', 'Perspective'),
    ('Research Highlights', 'Research Highlight'), ('Review', 'Review'),
    ('Research Article', 'Research article'), ('Journal Article', 'Publication'),
    ('article', 'Publication'), ('preprint', 'Preprint'),
])
def test_specific_labels_not_generic_article_claims(genre, label):
    assert publication_label(row(genre=genre)) == label


def test_no_type_inference_from_incidental_abstract_or_subject():
    entry = {'dc_subject': 'News & Views', 'summary': 'This perspective reviews articles.'}
    item = row(genre='article', **_entry_metadata(entry))
    item.abstract = entry['summary']
    assert publication_label(item) == 'Publication'
    assert not is_companion(item)


def test_directed_feed_link_keeps_identity_and_specific_genre():
    metadata = _entry_metadata({'prism_genre': 'News & Views',
        'prism_doi': '10.1234/commentary', 'links': [
            {'rel': 'is-commentary-on', 'href': 'https://doi.org/10.1234/PARENT'},
            {'rel': 'related', 'href': 'https://doi.org/10.1234/irrelevant'},
        ]})
    assert metadata['doi'] == '10.1234/commentary'
    assert metadata['publication_types'] == ['News & Views']
    child = row(2, 'News & Views', '10.1234/commentary')
    child.metadata = metadata
    assert companion_target_dois(child) == {'10.1234/parent'}
    assert companion_entries([row()], [child]) == {1: [{
        'title': 'A paper', 'url': child.url, 'publication_label': 'News & Views', 'item_id': 2,
    }]}


def test_explicit_links_survive_json_rows_and_deduplicate():
    child = row(2, 'Research Highlight', '10.1234/commentary', companion_of_dois=['10.1234/parent'])
    stored = SimpleNamespace(**{k: v for k, v in vars(child).items() if k != 'metadata'},
        metadata_json=json.dumps(child.metadata))
    assert len(companion_entries([row()], [stored, child])[1]) == 1
    assert publication_label(row()) == 'Research article'


@pytest.mark.parametrize('metadata', [
    {'references': ['10.1234/parent']},
    {'relation': {'references': [{'id': '10.1234/parent'}]}},
    {'relation': {'related': [{'id': '10.1234/parent'}]}},
])
def test_arbitrary_references_cannot_link(metadata):
    child = row(2, 'News & Views', '10.1234/child', **metadata)
    child.abstract = 'This News & Views discusses https://doi.org/10.1234/parent.'
    assert not companion_target_dois(child)
    assert not companion_entries([row()], [child])


def test_review_requires_explicit_is_review_of_not_just_citation():
    review = row(2, 'Review', '10.1234/review', companion_of_doi='10.1234/parent')
    assert not is_companion(review)
    review.metadata['relation'] = {'is-review-of': [{'id-type': 'doi', 'id': '10.1234/parent'}]}
    assert is_companion(review)
    assert companion_entries([row()], [review])


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'data:text/html,test', 'https://[', '//example.test',
                                'https://user:secret@example.test'])
def test_unsafe_links_not_presented(url):
    child = row(2, 'News & Views', '10.1234/child', companion_of_doi='10.1234/parent')
    child.url = url
    assert not companion_entries([row()], [child])


def test_self_links_primary_articles_and_similarity_do_not_attach():
    primary = row(2, 'Research Article', '10.1234/child', companion_of_doi='10.1234/parent')
    self_link = row(3, 'News & Views', '10.1234/parent', companion_of_doi='10.1234/parent')
    unrelated = row(4, 'News & Views', '10.1234/other', companion_of_doi='10.1234/different')
    assert not companion_entries([row()], [primary, self_link, unrelated])


@pytest.mark.parametrize('value,expected', [
    ('https://doi.org/10.1234/UPPER', '10.1234/upper'),
    ('doi:10.1234/a(b)', '10.1234/a(b)'), ('10.1234/a).', '10.1234/a'),
    ('https://example.test/?doi=10.1234/a', None), ('https://[', None),
])
def test_strict_identity_normalization(value, expected):
    assert normalize_doi(value) == expected


def test_self_only_and_invalid_review_relation_cannot_suppress():
    assert not is_companion(row(2, 'News & Views', '10.1234/child', companion_of_doi='10.1234/child'))
    review = row(2, 'Review', '10.1234/child', companion_of_doi='10.1234/parent',
                 relation={'is-review-of': [{'id': 'not a DOI'}]})
    assert not is_companion(review)


def test_dedupe_keeps_relationship_on_same_work_never_parent():
    from dailydigest.dedupe import dedupe_ranking_candidates
    from dailydigest.models import Item
    parent = Item(source='Journal', section='research', external_id='10.1234/parent',
                  title='An assembly method', url='https://doi.org/10.1234/parent')
    child = Item(source='Journal', section='research', external_id='10.1234/child',
                 title='A view of assembly', url='https://doi.org/10.1234/child',
                 metadata={'publication_types': ['News & Views'], 'companion_of_dois': ['10.1234/parent']})
    generic = child.model_copy(update={'source': 'OpenAlex', 'metadata': {'work_type': 'article'},
                                      'abstract': 'Longer original text. ' * 100})
    result = dedupe_ranking_candidates([generic, child, parent])
    assert len(result) == 2
    saved_child = next(item for item in result if item.external_id == '10.1234/child')
    assert is_companion(saved_child)
    assert companion_target_dois(saved_child) == {'10.1234/parent'}
    assert not is_companion(parent)
    assert not companion_target_dois(parent)


def test_preprint_server_status_is_preserved_without_guessing_from_source_name():
    item = row(genre='article')
    item.url = 'https://arxiv.org/abs/2607.12345'
    assert publication_label(item) == 'Preprint'
    item.url = 'https://arxiv.org.example.test/abs/2607.12345'
    assert publication_label(item) == 'Publication'


@pytest.mark.parametrize('extra', [
    {'is_retracted': True},
    {'publication_types': ['News & Views', 'Retracted Publication']},
    {'publication_types': ['News & Views'], 'work_type': 'retraction notice'},
])
def test_retracted_companion_never_rendered(extra):
    child = row(2, 'News & Views', '10.1234/child', companion_of_doi='10.1234/parent')
    child.metadata.update(extra)
    assert not companion_entries([row()], [child])


def test_explicit_published_doi_of_preprint_parent_accepts_companion():
    parent = row(1, 'preprint', '10.1234/preprint', published_doi='10.1234/published')
    child = row(2, 'News & Views', '10.1234/child', companion_of_doi='10.1234/published')
    assert companion_entries([parent], [child])[1][0]['item_id'] == 2
    assert parent.metadata['published_doi'] == '10.1234/published'
    assert parent.external_id == '10.1234/preprint'


def test_crosssection_duplicate_preserves_directed_relationship_and_genre():
    from dailydigest.dedupe import dedupe_ranking_candidates
    from dailydigest.models import Item
    child = Item(source='Industry', section='industry', external_id='10.1234/child',
                 title='An informed perspective', url='https://doi.org/10.1234/child',
                 metadata={'publication_types': ['Perspective'], 'companion_of_dois': ['10.1234/parent']})
    generic = child.model_copy(update={'section': 'research', 'source': 'OpenAlex',
                                      'metadata': {}, 'abstract': 'Richer abstract. ' * 100})
    result = dedupe_ranking_candidates([generic, child])
    assert len(result) == 1
    assert is_companion(result[0])
    assert publication_label(result[0]) == 'Perspective'


def test_crosssection_generic_copy_cannot_revive_retracted_companion():
    from dailydigest.dedupe import dedupe_ranking_candidates
    from dailydigest.models import Item
    child = Item(source='Industry', section='industry', external_id='10.1234/child',
                 title='An informed perspective', url='https://doi.org/10.1234/child',
                 metadata={'publication_types': ['Perspective'], 'is_retracted': True,
                           'companion_of_dois': ['10.1234/parent']})
    generic = child.model_copy(update={'section': 'research', 'source': 'OpenAlex',
                                      'metadata': {}, 'abstract': 'Richer abstract. ' * 100})
    result = dedupe_ranking_candidates([generic, child])
    assert result[0].metadata['is_retracted'] is True
    assert not companion_entries([row()], result)


@pytest.mark.parametrize('relation_name', ['is-commentary-on', 'is-review-of'])
@pytest.mark.parametrize('representation', ['relation', 'evidence'])
def test_review_with_explicit_directed_relationship_attaches_and_keeps_semantics(relation_name, representation):
    review = row(2, 'Review', '10.1234/review', companion_of_dois=['10.1234/parent'])
    if representation == 'relation':
        review.metadata['relation'] = {relation_name: [{'id-type': 'doi', 'id': '10.1234/parent'}]}
    else:
        review.metadata['companion_relation_evidence'] = [
            {'rel': relation_name, 'doi': '10.1234/parent', 'source': 'Europe PMC'}]
    assert is_companion(review)
    assert companion_entries([row()], [review])[1][0]['item_id'] == 2
    assert companion_metadata(review)['companion_relation_evidence'] == [
        {'rel': relation_name, 'doi': '10.1234/parent'}]


def test_plain_review_targets_cannot_be_promoted_to_directed_evidence_on_copy():
    review = row(2, 'Review', '10.1234/review', companion_of_dois=['10.1234/parent'])
    assert not is_companion(review)
    assert companion_metadata(review) == {}
    review.metadata.update(companion_metadata(review))
    assert not is_companion(review)
    assert companion_entries([row()], [review]) == {}
