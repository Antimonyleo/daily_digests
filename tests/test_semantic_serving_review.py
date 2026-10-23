"""Focused serving review: local fixtures only, no source edits or provider calls."""
from copy import deepcopy

import numpy as np
import pytest

from dailydigest.rank.semantic_rerank import (
    LocalContributionReranker, apply_contribution_scores, has_abstract,
)
from tests.test_contribution_serving import bundle as bundle, _row


STUB = 'New England Journal of Medicine, Ahead of Print.'


def test_actual_ahead_of_print_metadata_is_neutral():
    row = _row(85778, abstract=STUB)
    features = {85778: {'final_score': .13679756984114647}}
    before = deepcopy(features)
    result = apply_contribution_scores([(row, .13679756984114647)], features,
                                       {85778: .65})
    assert not has_abstract(row)
    assert result[0][1] == .13679756984114647
    assert features == before


def test_real_contribution_after_bibliographic_prefix_is_retained():
    assert has_abstract(_row(1, abstract=STUB + ' We develop a transferable protein design model.'))


@pytest.mark.parametrize('trailer', [STUB, 'Nature Methods, Volume 12, Issue 3.'])
def test_real_contribution_before_bibliographic_trailer_is_retained(trailer):
    prose = 'We develop a protein prediction method and evaluate its transferable accuracy. '
    assert has_abstract(_row(1, abstract=prose + trailer))


def test_empty_positive_scope_defaults_preserve_legacy_identity(bundle):
    from dailydigest.rank.scope import fingerprint
    from dailydigest.rank.semantic_rerank import interest_signature

    assert interest_signature({**bundle.profile, 'seed_works': [],
                               'interest_weights': {}, 'facet_weights': {}}) == fingerprint(bundle.profile)


@pytest.mark.parametrize('field,value', [
    ('seed_works', ['Designed enzyme assembly with transferable functional prediction.']),
    ('interest_weights', {'New material framework interest': 2.}),
    ('facet_weights', {'New structural nucleic acid interest': 2.}),
])
def test_positive_interest_scope_change_rejects_stale_artifact(bundle, field, value):
    profile = {**bundle.profile, field: value}
    with pytest.raises(ValueError, match='reader interest'):
        LocalContributionReranker(bundle.path, '/local', profile)
    assert not bundle.calls


def test_equal_model_evidence_keeps_existing_source_and_preference_differences():
    rows = [_row(1), _row(2)]
    features = {
        1: {'topic_score': .8, 'quality_score': .8, 'negative_interest_penalty': .2,
            'preference_weight': .5, 'confidence_score': .6},
        2: {'topic_score': .8, 'quality_score': .8, 'negative_interest_penalty': 0.,
            'preference_weight': .5, 'confidence_score': .8},
    }
    before = deepcopy(features)
    result = apply_contribution_scores([(rows[0], .4), (rows[1], .8)], features,
                                       {1: .75, 2: .75}, weight=.7)
    assert [row.id for row, _ in result] == [2, 1]
    values = {row.id: score for row, score in result}
    assert values[2] - values[1] == pytest.approx(.3 * (.8 - .4))
    for iid, feature in before.items():
        for key, value in feature.items():
            assert features[iid][key] == value


def test_semantic_signal_cannot_bypass_low_impact_raw_topic_floor():
    from dailydigest.config import Settings
    from dailydigest.rank.ranker import pick_top_per_section

    row = _row(1)
    row.source = 'Small specialty journal'
    facts = {1: {'source_bucket': 'low_impact_journal', 'is_arxiv_cs': False,
                 'is_preprint': False, 'is_published': True, 'is_high_quality': False,
                 'is_low_impact': True}}
    features = {1: {'topic_score': .70, 'final_score': .7}}
    reranked = apply_contribution_scores([(row, .7)], features, {1: 1.})
    assert reranked[0][1] == pytest.approx(.91)
    assert pick_top_per_section(reranked, {'research': 10}, score_features=features,
                                settings=Settings(_env_file=None, max_low_impact_research_frac=.2,
                                                  low_impact_relevance_floor=.8),
                                selection_facts=facts) == []


def test_semantic_signal_cannot_bypass_zero_low_impact_frequency_cap():
    from dailydigest.config import Settings
    from dailydigest.rank.ranker import pick_top_per_section

    row = _row(1)
    row.source = 'Small specialty journal'
    facts = {1: {'source_bucket': 'low_impact_journal', 'is_arxiv_cs': False,
                 'is_preprint': False, 'is_published': True, 'is_high_quality': False,
                 'is_low_impact': True}}
    features = {1: {'topic_score': .95, 'final_score': .7}}
    reranked = apply_contribution_scores([(row, .7)], features, {1: 1.})
    assert pick_top_per_section(reranked, {'research': 10}, score_features=features,
                                settings=Settings(_env_file=None, max_low_impact_research_frac=0.),
                                selection_facts=facts) == []


def test_configured_serving_failure_aborts_before_digest_or_summary_replacement(monkeypatch, tmp_path):
    from dailydigest import pipeline
    from dailydigest.config import Settings
    from dailydigest.models import Profile

    row = _row(1)
    row.source, row.title, row.authors = 'Nature', 'Designed protein method', ''
    row.url, row.external_id, row.metadata_json = 'https://example.org/1', '1', '{}'
    settings = Settings(_env_file=None, db_path=str(tmp_path / 'digest.db'),
                        contribution_reranker_path=str(tmp_path / 'missing.json'),
                        contribution_encoder_dir=str(tmp_path / 'model'),
                        within_day_dedupe=False, opportunity_enabled=False, events_enabled=False)
    monkeypatch.setattr(pipeline, 'get_settings', lambda: settings)
    monkeypatch.setattr(pipeline, 'init_db', lambda: None)
    monkeypatch.setattr(pipeline, '_digest_id', lambda: '2026-10-03')
    monkeypatch.setattr(pipeline, '_catch_up_window', lambda *args: 2)
    monkeypatch.setattr(pipeline, 'ingest_all', lambda **kwargs: 0)
    monkeypatch.setattr('dailydigest.rank.calibrate.fit_calibrator', lambda: None)
    monkeypatch.setattr('dailydigest.store.feedback_state', lambda: {})
    monkeypatch.setattr(pipeline, 'load_profile', lambda: Profile(bio='Engineering', keywords=['protein design']))
    monkeypatch.setattr('dailydigest.votes.signed_vote_count', lambda: 0)
    monkeypatch.setattr(pipeline, '_build_profile_with_rocchio', None)
    monkeypatch.setattr(pipeline, 'build_profile_matrix', lambda _: np.ones((1, 2)))
    monkeypatch.setattr(pipeline, 'recent_items', lambda **kwargs: [row])
    monkeypatch.setattr(pipeline, 'carryover_item_rows', lambda **kwargs: [])
    for name in ('exclude_reviewed_items', 'exclude_previously_shown', 'exclude_known_items', 'dedupe_ranking_candidates'):
        monkeypatch.setattr(pipeline, name, lambda rows, **kwargs: rows)
    monkeypatch.setattr('dailydigest.rank.near_dup.exclude_recent_near_duplicates', lambda rows, **kwargs: (rows, []))
    monkeypatch.setattr(pipeline, '_quality_gate', lambda rows: (rows, []))
    monkeypatch.setattr('dailydigest.rank.profile.build_attribution_context', lambda _: None)
    monkeypatch.setattr(pipeline, '_score_items_for_pipeline', lambda *args, **kwargs: (
        [(row, .8)], {1: {'topic_score': .9, 'confidence_score': .8, 'final_score': .8}}))
    monkeypatch.setattr(pipeline, '_build_neg_centroid', None)
    monkeypatch.setattr('dailydigest.rank.authors.load_watchlist', lambda _: [])
    monkeypatch.setattr('dailydigest.rank.enrich.enrich_scored', lambda rows: rows)
    writes = []

    def forbidden(*args, **kwargs):
        writes.append(True)
        pytest.fail('Configured model failure reached digest/summaries persistence or provider')

    for name in ('publish_digest', 'summarize_items', 'send_digest'):
        monkeypatch.setattr(pipeline, name, forbidden)
    monkeypatch.setattr('dailydigest.store.write_digest', forbidden)
    with pytest.raises(FileNotFoundError, match='missing.json'):
        pipeline.run_all(dry_run=True)
    assert not writes
    assert not (tmp_path / 'digest.db').exists()
