"""Semantic input retains the reader's actual target scope."""
import pytest

from dailydigest.rank.research_policy import RESEARCH_INSTRUCTION, reader_research_query


def test_materials_policy_retains_explicit_photonic_and_framework_targets():
    profile = {'canonical_facets': {'Self-assembled functional materials': {
        'aliases': ['photonic crystals', 'framework materials'],
        'anchors': ['Designed photonic crystals and programmable framework materials'],
    }}}
    text = reader_research_query(profile)
    for target in ('photonic crystals', 'framework materials', profile['canonical_facets']['Self-assembled functional materials']['anchors'][0]):
        assert target in text


def test_all_supplied_targets_retained_without_bio_name_or_venue():
    profile = {'name': 'PRIVATE_NAME', 'bio': 'PRIVATE_BIO', 'canonical_facets': {
        'Protein design': {'aliases': ['enzyme optimization'], 'anchors': ['Assess designed protein function']},
        'RNA devices': {'aliases': ['RNA origami'], 'anchors': ['Programmable RNA assemblies']},
    }}
    text = reader_research_query(profile)
    for facet, evidence in profile['canonical_facets'].items():
        assert facet in text
        assert all(term in text for term in evidence['aliases'] + evidence['anchors'])
    assert 'PRIVATE_NAME' not in text and 'PRIVATE_BIO' not in text
    assert 'transferable' in RESEARCH_INSTRUCTION and 'method evaluation' in RESEARCH_INSTRUCTION


def test_keyword_only_profile_is_explicit_and_empty_profile_cannot_invent_interests():
    assert 'molecular design' in reader_research_query({'keywords': ['molecular design']})
    with pytest.raises(ValueError, match='explicit reader interests'):
        reader_research_query({})
