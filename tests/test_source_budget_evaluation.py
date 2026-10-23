"""The source-budget shadow trial changes one cap, never eligibility."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dailydigest.pipeline import _quality_gate
from dailydigest.rank.ranker import pick_top_per_section
from dailydigest.rank.source_budget_evaluation import evaluate_source_budget_trial


def row(index, **updates):
    return SimpleNamespace(**dict({
        'id': index, 'section': 'research', 'source': f'Unlisted Venue {index}',
        'title': f'Mechanisms of marine ecosystem diversity study {index}',
        'abstract': 'Primary experimental research with methods and quantitative results.',
        'metadata_json': '{}', 'url': f'https://example.org/{index}', 'external_id': str(index),
    }, **updates))


def settings(**updates):
    return SimpleNamespace(**dict({
        'max_preprint_research_frac': .55, 'max_low_impact_research_frac': .1,
        'low_impact_relevance_floor': .72, 'research_final_score_floor_frac': 0,
        'research_final_score_min_keep': 3, 'adaptive_relevance_floor': False,
    }, **updates))


def pick(rows, *, override=None, options=None, features=None, cap=6):
    return pick_top_per_section([(r, .90 - r.id * .001) for r in rows], {'research': cap},
                                score_features=features or {r.id: {'topic_score': .8} for r in rows},
                                settings=options or settings(),
                                experimental_unknown_journal_cap=override)


def test_explicit_default_retains_one_unknown_while_trial_recovers_same_scored_rows():
    rows = [row(i) for i in range(1, 7)]
    scored = [(r, .90 - r.id * .001) for r in rows]
    default = pick_top_per_section(scored, {'research': 6}, settings=settings())
    assert default == pick(rows) == scored[:1]
    assert pick(rows, override=6) == scored


@pytest.mark.parametrize('invalid', [-1, .5, True, '6'])
def test_invalid_experimental_cap_rejected(invalid):
    with pytest.raises(ValueError, match='nonnegative integer'):
        pick([row(1)], override=invalid)


def test_trial_preserves_measured_low_quality_budget_and_both_relevance_floors():
    rows = [row(1), row(2), row(3, venue_low_impact=True), row(4, venue_low_impact=True), row(5)]
    features = {r.id: {'topic_score': .8} for r in rows}
    features[2]['topic_score'] = .71
    features[3]['topic_score'] = .71
    selected = pick(rows, override=6, features=features)
    assert {r.id for r, _ in selected} == {1, 4, 5}
    assert {r.id for r, _ in pick(rows, override=6, features=features,
                                options=settings(max_low_impact_research_frac=0))} == {1, 5}


def test_trial_keeps_venue_diversity_and_preprint_zero():
    rows = [row(i, source='One Unlisted Venue') for i in range(1, 7)]
    # Existing last-resort diversity relaxation fills at most the hard minimum.
    assert len(pick(rows, override=6)) == 3
    rows += [row(7, source='bioRxiv')]
    assert 7 not in {r.id for r, _ in pick(rows, override=6,
                                        options=settings(max_preprint_research_frac=0))}


def test_quality_gate_retraction_and_thin_abstract_cannot_be_rescued_by_cap():
    rows = [row(1), row(2, metadata_json=json.dumps({'is_retracted': True})),
            row(3, abstract=''), row(4, title='Editorial: Ecosystem mechanisms')]
    qualified, drops = _quality_gate(rows)
    assert [r.id for r in qualified] == [1]
    assert len(drops) == 3
    assert [r.id for r, _ in pick(qualified, override=6)] == [1]


@pytest.mark.parametrize('field,value', [('eligible', False), ('features', {'topic_score': .1}),
                                       ('stages', {}), ('embedding_provenance', {'model': 'other'})])
def test_report_rejects_upstream_changes(field, value):
    original = [{'profile_id': 'x', 'url': 'a', 'eligible': True,
                 'features': {}, 'stages': {'after_quality_gate': True}, 'embedding_provenance': {}}]
    candidate = deepcopy(original)
    candidate[0][field] = value
    with pytest.raises(ValueError, match='changed upstream'):
        evaluate_source_budget_trial({}, original, candidate)


def decisions_for_trace_validation():
    corpus = {'profiles': [{'id': 'marine', 'pool': [{'url': 'a'}]}], 'cases': []}
    baseline = [{'profile_id': 'marine', 'url': 'a', 'eligible': True,
                 'selected': True, 'selected_position': 1,
                 'features': {}, 'stages': {}, 'embedding_provenance': {},
                 'selection_trace': {
                     'section_budgets': {'research': 6},
                     'effective_settings': {'low_impact_relevance_floor': .72},
                     'within_day_dropped_ids': [5],
                     'supplied_profile': {'bio': 'Marine ecosystem mechanisms'},
                     'profile_input': 'provided', 'elapsed_seconds': 1.5}}]
    candidate = deepcopy(baseline)
    candidate[0]['selection_trace']['experimental_unknown_journal_cap'] = 6
    return corpus, baseline, candidate


@pytest.mark.parametrize('field,value', [
    ('effective_settings', {'low_impact_relevance_floor': 0}),
    ('section_budgets', {'research': 12}),
    ('within_day_dropped_ids', []),
    ('supplied_profile', {'bio': 'Quantum computation'}),
    ('profile_input', 'intent'),
    ('future_policy_setting', True),
])
def test_report_rejects_fixed_selection_trace_changes(field, value):
    corpus, before, after = decisions_for_trace_validation()
    after[0]['selection_trace'][field] = value
    with pytest.raises(ValueError, match='changed fixed selection_trace'):
        evaluate_source_budget_trial(corpus, before, after)


@pytest.mark.parametrize('invalid', [None, -1, True, 6.0, '6', 5, 7])
def test_report_rejects_candidate_cap_inconsistent_with_declared_policy(invalid):
    corpus, before, after = decisions_for_trace_validation()
    after[0]['selection_trace']['experimental_unknown_journal_cap'] = invalid
    with pytest.raises(ValueError, match='must equal the research section budget'):
        evaluate_source_budget_trial(corpus, before, after)


def test_report_accepts_only_elapsed_observation_change_and_default_baseline():
    corpus, before, after = decisions_for_trace_validation()
    after[0]['selection_trace']['elapsed_seconds'] = 8.5
    assert evaluate_source_budget_trial(corpus, before, after)['release_approved'] is False
    before[0]['selection_trace']['experimental_unknown_journal_cap'] = 6
    with pytest.raises(ValueError, match='Baseline must use the default'):
        evaluate_source_budget_trial(corpus, before, after)


@pytest.mark.parametrize("reverse_priority", [False, True])
def test_paired_replay_handles_serving_picker_without_explicit_frozen_source_facts(monkeypatch, tmp_path, reverse_priority):
    from dailydigest import pipeline
    from dailydigest.rank import cross_field_evaluation, source_budget_evaluation

    rows = [row(i) for i in range(1, 7)]
    corpus = {'profiles': [{'id': 'marine', 'pool': [{'url': r.url} for r in rows]}],
              'cases': [{'profile_id': 'marine', 'url': rows[0].url, 'label': 'aligned'}]}

    def serving_selection(scored, features, **kwargs):
        picked = pipeline.pick_top_per_section(scored, {'research': 6},
                                               score_features=features, settings=settings())
        return pipeline._restore_research_order(picked, scored)

    def fake_replay(corpus, **kwargs):
        features = {r.id: {'topic_score': .8} for r in rows}
        picked = cross_field_evaluation._select_captured_research(
            [(r, .90 - r.id * .001) for r in (rows[::-1] if reverse_priority else rows)], features)
        positions = {r.url: i for i, (r, _) in enumerate(picked, 1)}
        return [{'profile_id': 'marine', 'url': r.url, 'eligible': True,
                 'selected': r.url in positions, 'selected_position': positions.get(r.url),
                 'features': features[r.id], 'stages': {}, 'embedding_provenance': {},
                 'selection_trace': {'exploration_slots': 0, 'reading_mode': 'full',
                                     'section_budgets': {'research': 6}}} for r in rows]

    monkeypatch.setattr(cross_field_evaluation, '_select_captured_research', serving_selection)
    monkeypatch.setattr(source_budget_evaluation, 'replay_offline', fake_replay)
    before, after, result = source_budget_evaluation.replay_source_budget_trial(
        corpus, state=tmp_path, model_cache=tmp_path)
    assert sum(r['selected'] for r in before) == 1
    assert sum(r['selected'] for r in after) == 6
    assert [r['url'] for r in after if r['selected_position'] == 1] == [rows[-1 if reverse_priority else 0].url]
    assert result['candidate']['marine']['unjudged'] == 5
    assert result['candidate']['marine']['off_task'] == 0
    assert result['release_approved'] is False
