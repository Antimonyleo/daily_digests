from copy import deepcopy
import json

import pytest

from dailydigest.rank.reader_study import analyze_responses, prepare_study, render_study


def fixture():
    corpus = {'profiles': [{'id': 'ecology', 'intent': 'Measure ecosystems',
                            'profile': {'keywords': ['ecology']},
                            'pool': [{'url': f'https://example.org/{i}', 'title': f'Paper {i}',
                                      'abstract': 'Evidence'} for i in range(10)]}], 'cases': []}
    rows = [{'profile_id': 'ecology', 'url': item['url'], 'eligible': True,
             'selected': i < 5, 'selected_position': i + 1 if i < 5 else None}
            for i, item in enumerate(corpus['profiles'][0]['pool'])]
    return corpus, {'baseline': rows}


def response(packet):
    return {'schema_version': 1, 'study_id': packet['study_id'], 'reader': 'reader-one',
            'role': 'graduate', 'independent': True, 'judgments': []}


def test_sampling_blinded_includes_top_five_and_omissions_without_using_labels():
    corpus, arms = fixture()
    original = deepcopy(arms)
    packet, manifest = prepare_study(corpus, arms, missed_per_profile=2)
    corpus['cases'] = [{'SECRET_LABEL': True}]
    assert prepare_study(corpus, arms, missed_per_profile=2)[0] == packet
    assert len(packet['cases']) == 7
    assert set(manifest['arm_slates']['baseline']['ecology']) <= {c['id'] for c in packet['cases']}
    encoded = json.dumps(packet)
    for secret in ('selected', 'eligible', 'SECRET_LABEL', 'baseline', 'rank_score'):
        assert secret not in encoded
    assert arms == original


def test_union_includes_both_arms_and_preserves_selection_positions():
    corpus, arms = fixture()
    candidate = deepcopy(arms['baseline'])
    candidate[0].update(selected=False, selected_position=None)
    candidate[5].update(selected=True, selected_position=1)
    arms['candidate'] = candidate
    packet, manifest = prepare_study(corpus, arms, missed_per_profile=1)
    assert len(packet['cases']) == 7
    assert manifest['arm_slates']['candidate']['ecology'][0] != manifest['arm_slates']['baseline']['ecology'][0]


def test_pending_is_not_human_validation_and_unknowns_not_negative():
    corpus, arms = fixture()
    packet, manifest = prepare_study(corpus, arms)
    assert analyze_responses(packet, manifest, [])['status'] == 'awaiting_readers'
    result = analyze_responses(packet, manifest, [response(packet)])
    stats = result['readers'][0]['profiles']['ecology']['baseline']
    assert stats['direct_usefulness_bounds'] == [0, 1]
    assert stats['first_five_counts']['unjudged'] == 5
    assert stats['screen_target'] == 'not_demonstrated'
    assert not result['release_approved']


def test_reader_results_keep_transferability_disagreement_and_missed_papers():
    corpus, arms = fixture()
    packet, manifest = prepare_study(corpus, arms)
    top = manifest['arm_slates']['baseline']['ecology']
    missed = next(c['id'] for c in packet['cases'] if c['id'] not in top)
    first = response(packet)
    first['judgments'] = [{'id': cid, 'label': 'useful'} for cid in [*top, missed]]
    second = deepcopy(first)
    second['reader'] = 'reader-two'
    second['judgments'][0]['label'] = 'transferable'
    report = analyze_responses(packet, manifest, [first, second])
    assert report['disagreement_count'] == 1
    assert report['readers'][0]['profiles']['ecology']['baseline']['sampled_useful_outside_slate'] == 1
    assert report['readers'][1]['profiles']['ecology']['baseline']['first_five_counts']['transferable'] == 1


@pytest.mark.parametrize('mutation', ['identity', 'duplicate', 'label', 'case', 'role', 'independence'])
def test_import_rejects_invalid_or_mismatched_responses(mutation):
    corpus, arms = fixture()
    packet, manifest = prepare_study(corpus, arms)
    data = response(packet)
    row = {'id': packet['cases'][0]['id'], 'label': 'useful'}
    data['judgments'] = [row]
    if mutation == 'identity':
        data['study_id'] = 'wrong'
    elif mutation == 'duplicate':
        data['judgments'].append(row)
    elif mutation == 'label':
        row['label'] = 'good'
    elif mutation == 'case':
        row['id'] = 'unknown'
    elif mutation == 'role':
        data['role'] = 'unknown'
    else:
        data['independent'] = 'true'
    with pytest.raises(ValueError):
        analyze_responses(packet, manifest, [data])


def test_duplicate_readers_and_tampered_packets_rejected():
    corpus, arms = fixture()
    packet, manifest = prepare_study(corpus, arms)
    with pytest.raises(ValueError):
        analyze_responses(packet, manifest, [response(packet), response(packet)])
    packet['cases'][0]['intent'] = 'Changed goal'
    with pytest.raises(ValueError, match='identity'):
        analyze_responses(packet, manifest, [])


def test_html_contains_no_labels_or_script_injection_from_article():
    corpus, arms = fixture()
    corpus['profiles'][0]['pool'][0]['title'] = '</script><script>EVIL()</script>'
    packet, _ = prepare_study(corpus, arms)
    html = render_study(packet)
    assert '</script><script>EVIL()' not in html
    assert 'connect-src \'none\'' in html
    assert 'localStorage' not in html
    assert 'awaiting_readers' in html


def test_incomplete_fifth_judgment_cannot_certify_first_screen():
    corpus, arms = fixture()
    packet, manifest = prepare_study(corpus, arms)
    top = manifest['arm_slates']['baseline']['ecology']
    data = response(packet)
    data['judgments'] = [{'id': cid, 'label': 'useful'} for cid in top[:4]]
    for extra in ([], [{'id': top[4], 'label': 'unsure'}]):
        attempt = {**data, 'judgments': data['judgments'] + extra}
        stats = analyze_responses(packet, manifest, [attempt])['readers'][0]['profiles']['ecology']['baseline']
        assert stats['screen_target'] == 'not_demonstrated'


@pytest.mark.parametrize('mutation', ['order', 'unknown', 'duplicate', 'population'])
def test_analyst_manifest_tampering_is_detected(mutation):
    corpus, arms = fixture()
    packet, manifest = prepare_study(corpus, arms)
    slate = manifest['arm_slates']['baseline']['ecology']
    if mutation == 'order':
        slate.reverse()
    elif mutation == 'unknown':
        slate[0] = 'unknown'
    elif mutation == 'duplicate':
        slate[1] = slate[0]
    else:
        manifest['populations']['ecology']['candidate_count'] = 999
    with pytest.raises(ValueError, match='manifest identity'):
        analyze_responses(packet, manifest, [response(packet)])


def test_empty_study_rejected_instead_of_rendering_broken_navigation():
    corpus, arms = fixture()
    for row in arms['baseline']:
        row.update(selected=False, selected_position=None)
    with pytest.raises(ValueError, match='at least one'):
        prepare_study(corpus, arms, missed_per_profile=0)


@pytest.mark.parametrize('duplicate', ['profile', 'paper'])
def test_duplicate_study_inputs_are_rejected(duplicate):
    corpus, arms = fixture()
    if duplicate == 'profile':
        corpus['profiles'].append(deepcopy(corpus['profiles'][0]))
    else:
        corpus['profiles'][0]['pool'].append(deepcopy(corpus['profiles'][0]['pool'][0]))
    with pytest.raises(ValueError, match='unique'):
        prepare_study(corpus, arms)



def test_all_candidate_mode_includes_selected_tail_and_complete_omissions():
    corpus, arms = fixture()
    arms['baseline'][5].update(selected=True, selected_position=6)
    packet, manifest = prepare_study(corpus, arms, missed_per_profile=0, include_all=True)
    assert len(packet['cases']) == 10
    assert manifest['populations']['ecology']['omissions_sampled'] == 4
    assert manifest['sample_policy'] == 'Every captured candidate'
    assert manifest['arm_slates']['baseline']['ecology'][5] in {c['id'] for c in packet['cases']}
