from copy import deepcopy
from types import SimpleNamespace

import pytest

from dailydigest.rank.semantic_rerank import apply_contribution_scores, interest_signature


def row(iid, section='research', abstract=None):
    return SimpleNamespace(id=iid, section=section, title='Title', abstract=(
        'A complete abstract describes the object and contribution of this work.'
        if abstract is None else abstract))


def test_semantic_signal_changes_scores_and_display_order_without_changing_topic_or_quality():
    rows = [row(1), row(2), row(3)]
    features = {r.id: {'topic_score': .8, 'quality_score': .9, 'final_score': s}
                for r, s in zip(rows, [1., .9, .75], strict=True)}
    result = apply_contribution_scores(list(zip(rows, [1., .9, .75], strict=True)),
                                       features, {1: .1, 2: .3, 3: .9}, model_fingerprint='model')
    assert [r.id for r, _ in result] == [3, 2, 1]
    assert [s for _, s in result] == pytest.approx([.855, .48, .37])
    for r, score in result:
        assert features[r.id]['final_score'] == score
        assert features[r.id]['rank_score'] == score
        assert features[r.id]['topic_score'] == .8
        assert features[r.id]['quality_score'] == .9
        assert features[r.id]['contribution_rerank']['model_fingerprint'] == 'model'
        assert 'probability' in features[r.id]['contribution_rerank']['score_meaning']


def test_nonresearch_and_insufficient_abstract_remain_neutral():
    rows = [row(1), row(2, 'world'), row(3, abstract='Only a title.')]
    features = {r.id: {'final_score': .8} for r in rows}
    result = apply_contribution_scores([(r, .8) for r in rows], features, {1: .1})
    assert dict((r.id, score) for r, score in result) == pytest.approx({1: .31, 2: .8, 3: .8})
    assert 'contribution_rerank' not in features[2]
    assert 'contribution_rerank' not in features[3]


@pytest.mark.parametrize('values', [{1: float('nan'), 2: .5}, {1: .5}, {1: 1.1, 2: .5}])
def test_failed_batch_does_not_partially_mutate_features(values):
    rows = [row(1), row(2)]
    features = {r.id: {'final_score': .8} for r in rows}
    before = deepcopy(features)
    with pytest.raises(ValueError):
        apply_contribution_scores([(r, .8) for r in rows], features, values)
    assert features == before


def test_zero_weight_is_noop_and_invalid_weight_is_rejected():
    pairs, features = [(row(1), .8)], {1: {'final_score': .8}}
    assert apply_contribution_scores(pairs, features, {}, weight=0) == pairs
    assert features == {1: {'final_score': .8}}
    with pytest.raises(ValueError):
        apply_contribution_scores(pairs, features, {1: .5}, weight=1.1)


def test_interest_signature_binds_reader_scope_but_not_history():
    profile = {'bio': 'Design', 'keywords': ['RNA'], 'canonical_facets': {'RNA': {}},
               'negative_interests': ['diagnostics'], 'reading_history': []}
    assert interest_signature(profile) == interest_signature({**profile, 'reading_history': [1]})
    assert interest_signature(profile) != interest_signature({**profile, 'bio': 'Diagnostics'})


def test_pipeline_default_does_not_load_a_model_and_partial_configuration_fails():
    from dailydigest.config import Settings
    from dailydigest.pipeline import _apply_semantic_reranking

    pairs = [(row(1), .8)]
    assert _apply_semantic_reranking(pairs, {}, {}, Settings()) == pairs
    with pytest.raises(ValueError, match='both local paths'):
        _apply_semantic_reranking(pairs, {}, {}, Settings(contribution_reranker_path='/tmp/model.json'))


def test_bibliographic_stub_is_neutral_but_real_prose_after_it_is_scored():
    from dailydigest.rank.semantic_rerank import has_abstract

    stub = 'Science Translational Medicine, Volume 18, Issue 869 , September 2026.'
    assert not has_abstract(row(1, abstract=stub))
    assert has_abstract(row(2, abstract=stub + ' We develop a new transferable prediction method.'))
    features = {1: {'final_score': .8}}
    assert apply_contribution_scores([(row(1, abstract=stub), .8)], features, {1: .99})[0][1] == .8
    assert features == {1: {'final_score': .8}}


def test_persisted_explanation_retains_semantic_ranking_provenance():
    from dailydigest.rank.source_quality import breakdown_payload

    item = row(1)
    item.source = 'Nature'
    features = {1: {'confidence_score': .9}}
    [(item, score)] = apply_contribution_scores([(item, .9)], features, {1: .2}, model_fingerprint='model')
    payload = breakdown_payload(item, .8, final_score=score, confidence_score=.9,
                                contribution_rerank=features[1]['contribution_rerank'])
    assert payload['contribution_rerank']['rank_signal'] == .2
    assert payload['score'] == pytest.approx(score)
    assert payload['confidence_score'] == .9
    assert 'probability' in payload['contribution_rerank']['score_meaning']
