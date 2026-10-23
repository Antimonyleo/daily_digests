"""Text evidence must not overstate a semantic assignment or gate eligibility."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dailydigest import pipeline

CORPUS = json.loads((Path(__file__).parent / 'fixtures/cross_field_alignment.json').read_text())


@pytest.mark.parametrize('profile_id,title_fragment', [
    ('asymmetric_organocatalysis', 'P450-catalyzed'),
    ('marine_microbial_ecology', 'elephant megacarcasses'),
    ('perovskite_solar_cells', 'Structural Rigidity Enables'),
    ('topological_photonics', 'Ultrastrong-coupling'),
    ('induced_proximity', 'Superconducting proximity effect'),
])
def test_neighboring_field_is_not_claimed_as_text_supported_from_high_cosine(profile_id, title_fragment):
    profile = next(p for p in CORPUS['profiles'] if p['id'] == profile_id)
    item = next(x for x in profile['pool'] if title_fragment in x['title'])
    row = SimpleNamespace(**{**item, 'id': 1})
    feature = {'topic_score': .78, 'primary_facet': profile['profile']['keywords'][0]}
    from dailydigest.rank.topic_support import topic_support

    assert topic_support(row, feature['primary_facet'])['status'] == 'semantic_only'
    # These are observed false positives, not proof that lack of text excludes
    # all relevant papers. Preserve serving behavior until recall is validated.
    assert pipeline._research_relevant(row, feature, .65)


@pytest.mark.parametrize('topic,title,abstract,aliases', [
    ('marine microbial ecology', 'Marine microbial communities', 'We study the ecology of marine microorganisms.', []),
    ('perovskite solar cells', 'Perovskite solar cell efficiency', '', []),
    ('quantum error correction', 'Quantum error detection', 'We detect errors in logical qubits.', ['quantum error detection']),
    ('causal inference', 'Methods for causal inference in economics', 'An identification strategy.', []),
    ('固态电池', '高性能固态电池材料', '新型固态电池具有更高能量密度。', []),
])
def test_explicit_terms_inflections_and_reader_aliases_remain_available(topic, title, abstract, aliases):
    from dailydigest.rank.topic_support import topic_support
    row = SimpleNamespace(title=title, abstract=abstract)
    result = topic_support(row, topic, aliases)
    assert result['status'] == 'text_supported'
    assert result['quote'] in title or result['quote'] in abstract


def test_stems_do_not_conflate_organic_catalysis_and_organophosphates():
    from dailydigest.rank.topic_support import topic_support
    row = SimpleNamespace(title='Asymmetric organophosphate reactions', abstract='')
    assert topic_support(row, 'asymmetric organocatalysis')['status'] == 'semantic_only'


def test_missing_attribution_preserves_scoring_fallback():
    assert pipeline._research_relevant(SimpleNamespace(title='', abstract=''), {'topic_score': .8}, .65)


def test_negative_interests_still_exclude_a_text_match():
    feature = {'topic_score': .78, 'negative_interest_penalty': .2, 'primary_facet': 'causal inference'}
    assert not pipeline._research_relevant(SimpleNamespace(title='Causal inference', abstract=''), feature, .65)


def test_another_declared_interest_can_supply_text_evidence_without_reassigning_weights():
    from dailydigest.models import Profile
    from dailydigest.rank.topic_support import annotate_topic_support

    row = SimpleNamespace(id=1, section='research', title='RNA delivery', abstract='A new method.')
    profile = Profile(bio="", keywords=['protein structure prediction', 'RNA delivery'])
    feature = {'primary_facet': 'protein structure prediction', 'topic_score': .8, 'topic_priority': .7}
    annotate_topic_support([(row, .8)], {1: feature}, profile)
    assert feature['topic_support']['topic'] == 'RNA delivery'
    assert feature['primary_facet'] == 'protein structure prediction'
    assert feature['topic_priority'] == .7
    assert pipeline._research_relevant(row, feature, .65)


def test_unsupported_synonyms_are_not_excluded_by_display_evidence():
    from dailydigest.models import Profile
    from dailydigest.rank.topic_support import annotate_topic_support

    row = SimpleNamespace(id=1, section='research', title='Logical qubit protection', abstract='Fault-tolerant computation.')
    feature = {'primary_facet': 'quantum error correction', 'topic_score': .8}
    annotate_topic_support([(row, .8)], {1: feature}, Profile(bio="", keywords=['quantum error correction']))
    assert feature['topic_support']['status'] == 'semantic_only'
    assert pipeline._research_relevant(row, feature, .65)


def _render_match(item):
    from jinja2 import Environment

    source = (Path(__file__).parents[1] / 'templates/digest_web.html.j2').read_text()
    fragment = source[source.index('          {% set uncertain_fit'):source.index('          {% if item.reason_line')]
    return Environment(autoescape=True).from_string(fragment).render(item=item, section={'key': 'research'})


def test_display_names_supported_secondary_interest_without_claiming_primary_text_match():
    html = _render_match({'abstract': 'RNA delivery method.', 'ranking': {
        'primary_facet': 'Protein structure prediction',
        'topic_support': {'status': 'text_supported', 'topic': 'RNA delivery'},
    }})
    assert '<strong>RNA delivery</strong>' in html
    assert 'Protein structure prediction' not in html


@pytest.mark.parametrize('keywords', [['RNA'], ['ribonucleic acid']])
def test_secondary_evidence_uses_canonical_interest_not_retrieval_keyword(keywords):
    from types import SimpleNamespace

    from dailydigest.models import Profile
    from dailydigest.rank.topic_support import annotate_topic_support

    profile = Profile(bio='', keywords=keywords, canonical_facets={
        'Protein structure prediction': {},
        'Therapeutic RNA transport': {'aliases': ['RNA delivery']},
    })
    row = SimpleNamespace(id=1, section='research', title='RNA delivery',
                          abstract='RNA delivery by lipid nanoparticles.')
    feature = {'primary_facet': 'Protein structure prediction', 'topic_priority': .7}
    annotate_topic_support([(row, .8)], {1: feature}, profile)
    assert feature['topic_support']['topic'] == 'Therapeutic RNA transport'
    assert feature['topic_support']['term'] == 'RNA delivery'
    assert feature['primary_facet'] == 'Protein structure prediction'
    assert feature['topic_priority'] == .7


@pytest.mark.parametrize('ranking', [{}, {'primary_facet': 'RNA delivery'}])
def test_missing_abstract_is_visible_without_opening_details(ranking):
    assert 'Title only; fit needs checking' in _render_match({'abstract': '', 'ranking': ranking})


def test_disagreeing_topic_models_are_visible_before_details():
    html = _render_match({'abstract': 'Supplied abstract.', 'ranking': {
        'primary_facet': 'RNA delivery', 'topic_alignment': {'uncertain': True, 'complete': True},
    }})
    assert 'Topic fit is uncertain' in html
    assert 'topic-support' in html
