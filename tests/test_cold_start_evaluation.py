from copy import deepcopy

import pytest

from dailydigest.rank.cold_start_evaluation import (
    compare_slates, evaluate_first_slate, prepare_profile, review_packet,
)


def example():
    corpus = {'profiles': [{'id': 'reader', 'intent': 'Develop reusable methods',
                           'profile': {'bio': '', 'keywords': ['ecology']},
                           'pool': [{'url': str(i), 'title': f'Paper {i}', 'abstract': 'Evidence'}
                                    for i in range(7)]}],
              'cases': [{'profile_id': 'reader', 'url': str(i), 'label': label}
                        for i, label in enumerate(['aligned', 'off_task', 'uncertain',
                                                   'aligned', 'aligned', 'aligned'])]}
    decisions = [{'profile_id': 'reader', 'url': str(i), 'eligible': True,
                  'selected': i < 5, 'selected_position': i + 1 if i < 5 else None}
                 for i in range(7)]
    return corpus, decisions


def test_first_screen_uses_display_positions_and_keeps_unknowns():
    corpus, decisions = example()
    decisions[1]['selected_position'], decisions[3]['selected_position'] = 4, 2
    result = evaluate_first_slate(corpus, list(reversed(decisions)))['profiles']['reader']
    assert [row['url'] for row in result['first_slate']] == ['0', '3', '2', '1', '4']
    assert result['useful_fraction_bounds'] == [3 / 5, 4 / 5]
    assert result['first_three_off_task'] == 0
    assert result['first_three_unknown'] == 1
    assert result['screen_target'] == 'not_demonstrated'
    assert result['aligned_pool_retained'] == {'eligible': 4, 'selected': 3, 'total': 4}


def test_unjudged_is_not_negative_and_short_slates_cannot_pass_by_shrinking():
    corpus, decisions = example()
    for decision in decisions:
        decision.update(selected=decision['url'] in {'0', '6'}, selected_position=None)
    decisions[0]['selected_position'] = 1
    decisions[6]['selected_position'] = 2
    result = evaluate_first_slate(corpus, decisions)['profiles']['reader']
    assert result['counts'] == {'aligned': 1, 'off_task': 0, 'uncertain': 0, 'unjudged': 1}
    assert result['useful_fraction_bounds'] == [.5, 1.]
    assert result['screen_target'] == 'not_demonstrated'


def test_positions_and_population_must_be_complete():
    corpus, decisions = example()
    with pytest.raises(ValueError, match='population'):
        evaluate_first_slate(corpus, decisions[:-1])
    decisions[0]['selected_position'] = 2
    with pytest.raises(ValueError, match='positions'):
        evaluate_first_slate(corpus, decisions)


def test_empty_slate_has_unknown_precision_and_zero_retention():
    corpus, decisions = example()
    for decision in decisions:
        decision.update(selected=False, selected_position=None)
    result = evaluate_first_slate(corpus, decisions)['profiles']['reader']
    assert result['useful_fraction_bounds'] == [None, None]
    assert result['aligned_pool_retained']['selected'] == 0
    assert result['screen_target'] == 'not_demonstrated'


def test_input_variants_use_explicit_intent_without_mutation_or_mined_positive_examples():
    corpus, _ = example()
    entry = corpus['profiles'][0]
    original = deepcopy(entry)
    intent = prepare_profile(entry, 'intent')
    assert intent['canonical_facets']['ecology']['description'] == entry['intent']
    context = prepare_profile(entry, 'context')
    assert context['bio'] == entry['intent']
    assert 'canonical_facets' not in context
    assert prepare_profile(entry, 'anchors')['canonical_facets']['ecology']['anchors'] == [entry['intent']]
    with pytest.raises(ValueError, match='representative'):
        prepare_profile(entry, 'examples')
    entry['example_works'] = ['A separately supplied representative paper']
    assert prepare_profile(entry, 'examples')['seed_works'] == entry['example_works']
    del entry['example_works']
    assert entry == original


def test_multitopic_intent_requires_topic_mapping_instead_of_copying_goal_to_every_topic():
    corpus, _ = example()
    entry = corpus['profiles'][0]
    entry['profile']['keywords'].append('astronomy')
    with pytest.raises(ValueError, match='per-topic'):
        prepare_profile(entry, 'intent')
    entry['topic_intents'] = {'ecology': 'Reusable methods', 'astronomy': 'Galaxy formation'}
    assert prepare_profile(entry, 'intent')['canonical_facets']['astronomy']['description'] == 'Galaxy formation'


def test_review_packet_blinds_scores_labels_and_selection():
    corpus, decisions = example()
    packet = review_packet(corpus)
    assert len(packet['cases']) == len(decisions)
    for case in packet['cases']:
        assert set(case) == {'id', 'profile_id', 'intent', 'topic_intents', 'supplied_profile',
                             'profile_input', 'article', 'label', 'rationale', 'reviewer'}
        assert case['label'] is None and case['reviewer'] is None
        assert case['supplied_profile']['bio'] == ''
        assert case['profile_input'] == 'provided'
        assert set(case['article']) == {'url', 'title', 'abstract'}


def test_comparison_names_lost_positives_not_just_average_precision():
    corpus, baseline = example()
    candidate = deepcopy(baseline)
    candidate[4].update(selected=False, selected_position=None)
    candidate[6].update(selected=True, selected_position=5)
    result = compare_slates(corpus, baseline, candidate)['profiles']['reader']
    assert result['lost_aligned'] == ['4']
    assert result['added_aligned'] == []


@pytest.mark.parametrize('mode', ['provided', 'intent', 'anchors', 'examples'])
def test_seed_overlap_is_checked_for_every_input_mode(mode):
    corpus, _ = example()
    entry = corpus['profiles'][0]
    entry['profile']['seed_works'] = ['Paper   0']
    entry['example_works'] = ['Paper\n0']
    with pytest.raises(ValueError, match='overlap'):
        prepare_profile(entry, mode)


def test_seed_doi_identity_is_normalized_and_short_external_id_is_not_substring_match():
    corpus, _ = example()
    entry = corpus['profiles'][0]
    entry['pool'][0].update(external_id='1', metadata={'doi': '10.1234/ABC'})
    entry['example_works'] = ['A useful paper from 2021']
    assert prepare_profile(entry, 'examples')['seed_works'] == entry['example_works']
    entry['example_works'] = ['https://doi.org/10.1234%2Fabc']
    with pytest.raises(ValueError, match='overlap'):
        prepare_profile(entry, 'examples')


@pytest.mark.parametrize('fifth', ['uncertain', 'unjudged'])
def test_screen_target_requires_all_five_decisive_judgments(fifth):
    corpus, decisions = example()
    corpus['cases'] = [{'profile_id': 'reader', 'url': str(i), 'label': 'aligned'} for i in range(4)]
    if fifth == 'uncertain':
        corpus['cases'].append({'profile_id': 'reader', 'url': '4', 'label': 'uncertain'})
    result = evaluate_first_slate(corpus, decisions)['profiles']['reader']
    assert result['screen_target'] == 'not_demonstrated'
    corpus['cases'] = [{'profile_id': 'reader', 'url': str(i), 'label': 'aligned'} for i in range(5)]
    assert evaluate_first_slate(corpus, decisions)['profiles']['reader']['screen_target'] == 'met_on_provisional_labels'


def test_ordering_only_keeps_baseline_picks_gates_and_stable_ties():
    from dailydigest.rank.cold_start_evaluation import reorder_selected_by_intent
    corpus, baseline = example()
    for row in baseline:
        row['stages'] = {'after_quality_gate': True}
    original = deepcopy(baseline)
    alternative = deepcopy(baseline)
    for row in alternative:
        row['features'] = {'topic_score': 1 if row['url'] in {'3', '4', '6'} else .5}
        row.update(eligible=False, selected=False, selected_position=None)
    result = reorder_selected_by_intent(corpus, baseline, alternative)
    chosen = sorted((r for r in result if r['selected']), key=lambda r: r['selected_position'])
    assert [r['url'] for r in chosen] == ['3', '4', '0', '1', '2']
    assert all(r['eligible'] for r in result)
    assert result[6]['selected'] is False  # highest alternate score cannot add a paper
    assert baseline == original


@pytest.mark.parametrize('defect', ['missing', 'nan', 'quality'])
def test_ordering_study_rejects_missing_scores_or_changed_quality(defect):
    from dailydigest.rank.cold_start_evaluation import reorder_selected_by_intent
    corpus, baseline = example()
    for row in baseline:
        row['stages'] = {'after_quality_gate': True}
        row['features'] = {'topic_score': .5}
    alternative = deepcopy(baseline)
    if defect == 'missing':
        alternative[0]['features'] = {}
    elif defect == 'nan':
        alternative[0]['features']['topic_score'] = float('nan')
    else:
        alternative[0]['stages']['after_quality_gate'] = False
    with pytest.raises(ValueError):
        reorder_selected_by_intent(corpus, baseline, alternative)
