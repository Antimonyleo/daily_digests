"""Independent adversarial checks; no remote/model/private-data operations."""
import json
from types import SimpleNamespace

import pytest

from dailydigest.models import SourceSpec
from dailydigest.pipeline import _quality_gate
from dailydigest.rank.ranker import apply_exploration, pick_top_per_section
from dailydigest.rank.semantic_rerank import apply_contribution_scores
from dailydigest.rank import source_quality as quality
from tests.test_snapshot_capture import capture_args as capture_args


@pytest.fixture(autouse=True)
def stable_quality_configuration(monkeypatch):
    monkeypatch.setattr(quality, '_configured_sources_cached', lambda: {})
    monkeypatch.setattr(quality, '_research_quality_weight', lambda: 1.)


def row(iid, source='Nature', **attributes):
    return SimpleNamespace(**{
        'id': iid, 'source': source, 'section': 'research',
        'title': 'Programmable molecular structure construction',
        'abstract': 'We assess the designed objects with independently held out measurements.',
        'metadata_json': '{}', 'url': f'https://example.test/{iid}',
        'external_id': str(iid), **attributes,
    })


def settings(**updates):
    return SimpleNamespace(**{
        'max_preprint_research_frac': .55, 'max_low_impact_research_frac': .2,
        'low_impact_relevance_floor': .72, 'adaptive_relevance_floor': False,
        'research_final_score_floor_frac': 0., 'research_final_score_min_keep': 3,
        **updates,
    })


def test_international_edition_aliases_share_final_selection_source_cap():
    pool = [(row(i, 'Angew. Chem. Int. Ed.'), .96 - i / 1000) for i in range(1, 4)]
    pool += [(row(i, 'Angewandte Chemie International Edition'), .95 - i / 1000) for i in range(4, 7)]
    pool += [(row(i, 'Nature'), .8 - i / 1000) for i in range(7, 10)]
    pool += [(row(i, 'Cell'), .7 - i / 1000) for i in range(10, 13)]
    features = {article.id: {'topic_score': .8} for article, _ in pool}
    selected = pick_top_per_section(pool, {'research': 6}, score_features=features, settings=settings())
    count = sum(article.source in {'Angew. Chem. Int. Ed.', 'Angewandte Chemie International Edition'}
                for article, _ in selected)
    assert count <= 2, f'One journal received {count}/6 slots via name aliases; policy ceiling is2'


def test_canonical_source_caps_are_frozen_and_legacy_captures_keep_raw_policy(capture_args, monkeypatch):
    from copy import deepcopy
    from dailydigest.rank import ranker
    from dailydigest.rank.snapshot import build_ranking_snapshot
    from dailydigest.rank.replay import frozen_selection, IncompleteReplay

    rows = capture_args['candidates'][:12]
    sources = ['Angew. Chem. Int. Ed.', 'Angewandte Chemie International Edition', 'Nature', 'Cell']
    for i, article in enumerate(rows):
        article.source = sources[i // 3]
    pool = [(article, .95 - i / 100) for i, article in enumerate(rows)]
    capture_args.update(candidates=rows, scored=pool, eligible=pool,
                        section_budgets={'research': 6}, labeled=[])
    frozen = build_ranking_snapshot(**capture_args)
    legacy = deepcopy(frozen)
    for candidate in legacy['candidates']:
        candidate['selection_facts'].pop('source_diversity_key')
    monkeypatch.setattr(ranker, 'source_diversity_key', lambda _: pytest.fail('consulted mutable source policy'))
    new_ids = frozen_selection(frozen)['research']
    old_ids = frozen_selection(legacy)['research']
    assert len(new_ids) == len(old_ids) == 6
    assert sum(i <= 6 for i in new_ids) == 2
    assert sum(i <= 6 for i in old_ids) == 4
    invalid = deepcopy(frozen)
    invalid['candidates'][0]['selection_facts']['source_diversity_key'] = None
    with pytest.raises(IncompleteReplay, match='diversity key'):
        frozen_selection(invalid)


def test_source_identity_keeps_distinct_journals_and_unknown_feeds_separate():
    assert quality.source_diversity_key(row(1, 'Nature')) != quality.source_diversity_key(row(2, 'Nature Methods'))
    assert quality.source_diversity_key(row(1, 'Angewandte Chemie')) != quality.source_diversity_key(
        row(2, 'Angewandte Chemie International Edition'))
    assert quality.source_diversity_key(row(1, 'Unknown A')) != quality.source_diversity_key(row(2, 'Unknown B'))
    assert quality.source_diversity_key(row(1, 'OpenAlex', metadata_json='{"venue":"Nature"}')) == 'nature'


@pytest.mark.parametrize('kind', ['arxiv', 'biorxiv'])
def test_configured_preprint_kind_vetoes_journal_protection_under_custom_name(monkeypatch, kind):
    name = 'Protein engineering updates'
    monkeypatch.setattr(quality, '_configured_sources_cached', lambda: {
        name.lower(): SourceSpec(name=name, kind=kind, section='research',
                                quality_tier='high', prestige_score=.95),
    })
    article = row(1, name)
    assert quality.is_preprint_source(article)
    assert not quality.is_published_journal_source(article)
    assert not quality.is_high_quality_journal_source(article)
    assert quality.venue_relevance_credit(article) == 0


def test_custom_arxiv_cs_category_obeys_common_source_cap(monkeypatch):
    name = 'Machine learning updates'
    monkeypatch.setattr(quality, '_configured_sources_cached', lambda: {
        name.lower(): SourceSpec(name=name, kind='arxiv', category='cs.LG', section='research',
                                quality_tier='high', prestige_score=.95),
    })
    article = row(1, name)
    assert quality.source_bucket(article) == 'arxiv_cs'
    assert quality.is_arxiv_cs_source(article)
    pool = [(row(i, name), .99) for i in range(1, 4)]
    pool += [(row(i, source), .8) for i, source in enumerate(['Nature', 'Nature', 'Cell', 'Cell', 'Science'], 4)]
    picked = pick_top_per_section(pool, {'research': 6}, settings=settings())
    assert sum(item.source == name for item, _ in picked) <= 2


@pytest.mark.parametrize('metadata', [
    {'work_type': 'preprint'}, {'publication_types': ['Preprint']},
])
def test_explicit_preprint_metadata_vetoes_curated_journal_label(metadata):
    article = row(1, 'Nature', metadata_json=json.dumps(metadata))
    assert quality.is_preprint_source(article)
    assert not quality.is_high_quality_journal_source(article)
    assert quality.venue_relevance_credit(article) == 0


@pytest.mark.parametrize('venue', [
    'Nature Materials Letters', 'Small Ruminant Research', 'Neuronal Signaling',
    'International Journal of Nucleic Acids Research', 'Advanced Materials Research',
])
def test_prestigious_name_substrings_do_not_grant_journal_protection(venue):
    article = row(1, 'OpenAlex (your topics)', metadata_json=json.dumps({'venue': venue}))
    assert quality.recognized_research_venue(venue) is None
    assert not quality.is_high_quality_journal_source(article)
    assert quality.venue_relevance_credit(article) == 0
    assert quality.source_bucket(article) == 'unknown_journal'


@pytest.mark.parametrize('learned_signal', [0., .5, 1.])
def test_semantic_score_cannot_replace_low_impact_raw_topic_floor(learned_signal):
    article = row(1, 'Nature', venue_low_impact=True)
    features = {1: {'topic_score': .70}}
    adjusted = apply_contribution_scores([(article, .99)], features, {1: learned_signal}, weight=.7)
    assert not pick_top_per_section(adjusted, {'research': 10}, score_features=features, settings=settings())


@pytest.mark.parametrize('cap', [1, 2, 3, 6, 10])
def test_minimum_fill_cannot_bypass_low_impact_frequency_cap(cap):
    pool = [(row(i, 'Nature', venue_low_impact=True), .99) for i in range(1, cap + 2)]
    features = {article.id: {'topic_score': .99} for article, _ in pool}
    picked = pick_top_per_section(pool, {'research': cap}, score_features=features, settings=settings())
    assert len(picked) <= max(1, int(cap * .2))


def test_exploration_preserves_low_impact_frequency_gate():
    picked = [(row(i, source), .9) for i, source in enumerate(['Nature', 'Nature', 'Science', 'Science', 'Cell'], 1)]
    candidate = row(6, 'Nature', venue_low_impact=True)
    features = {i: {'topic_score': .99} for i in range(1, 7)}
    result = apply_exploration(picked, [*picked, (candidate, .99)], {6: 1.}, slots=1,
                              eligible=lambda article: True, caps={'research': 5},
                              score_features=features, settings=settings(max_low_impact_research_frac=0))
    assert {article.id for article, _ in result} == {1, 2, 3, 4, 5}


def test_explicit_low_impact_wins_over_contradictory_high_impact_flag():
    article = row(1, 'Nature', venue_low_impact=True, venue_high_impact=True)
    assert quality.source_bucket(article) == 'low_impact_journal'
    assert not quality.is_high_quality_journal_source(article)


def test_real_preprint_low_mean_citation_flag_cannot_become_low_impact_journal():
    article = row(1, 'bioRxiv (recent)', venue_low_impact=True)
    assert quality.source_bucket(article) == 'bio_med_preprint'
    assert not quality.is_high_quality_journal_source(article)


@pytest.mark.parametrize('metadata', [
    {'is_retracted': True}, {'publication_types': ['Retracted Publication']},
    {'work_type': 'editorial'}, {'work_type': 'correction'},
])
def test_protected_venue_does_not_override_publication_exclusion(metadata):
    article = row(1, 'Nature', metadata_json=json.dumps(metadata))
    assert _quality_gate([article])[0] == []
