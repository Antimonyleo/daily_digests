from copy import deepcopy
from types import SimpleNamespace

from dailydigest import pipeline, summarize
from dailydigest.config import Settings
from dailydigest.models import Profile
from dailydigest.rank.contribution import rerank_snapshot_contributions


def profile():
    return Profile(bio='Biomolecular engineering', keywords=['protein engineering'], canonical_facets={
        'Protein engineering and design': {'aliases': ['protein design']},
    })


def row(iid, text):
    return SimpleNamespace(id=iid, section='research', title='Study', abstract=text, source='Nature', published_at=None)


def test_optional_pipeline_ordering_preserves_raw_scores_and_neutral_default():
    rows = [row(1, 'Here we determine the native protein structure and reveal the mechanism of cellular signaling across several complementary experimental conditions.'), row(2, 'Here we engineer proteins with enhanced activity and experimentally validate the resulting variants across several independent conditions.')]
    pairs = [(rows[0], .8), (rows[1], .79)]
    features = {1: {'topic_score': .9}, 2: {'topic_score': .9}}
    assert Settings().contribution_order_strength == 0
    assert pipeline._apply_contribution_ordering(pairs, features, profile(), Settings()) == pairs
    result = pipeline._apply_contribution_ordering(pairs, features, profile(), Settings(contribution_order_strength=.04))
    assert result == [pairs[1], pairs[0]]
    assert features[2]['contribution']['evidence'][0] in rows[1].abstract
    assert features[1]['topic_score'] == .9


def test_snapshot_ordering_preserves_original_pool_and_scores():
    rows = [row(1, 'Here we determine the native protein structure and reveal the mechanism of cellular signaling across several complementary experimental conditions.'), row(2, 'Here we engineer proteins with enhanced activity and experimentally validate the resulting variants across several independent conditions.')]
    snapshot = {'profile': profile().model_dump(), 'candidates': [
        {'item_id': r.id, 'section': r.section, 'eligible': True, 'input_position': i, 'ranking_score': .8 - i * .01,
         'metadata': {'title': r.title, 'abstract': r.abstract}, 'features': {'topic_score': .9}} for i, r in enumerate(rows)
    ]}
    before = deepcopy(snapshot)
    result = rerank_snapshot_contributions(snapshot, strength=.04)
    assert snapshot == before
    assert [c['input_position'] for c in result['candidates']] == [1, 0]
    for original, changed in zip(snapshot['candidates'], result['candidates'], strict=True):
        assert original['ranking_score'] == changed['ranking_score']
        assert original['eligible'] == changed['eligible']


def test_extractive_explanation_uses_contribution_evidence_and_marks_context():
    summarize._set_reader_context(profile())
    try:
        engineering = row(1, 'Protein engineering is promising. Here we engineer proteins with enhanced specificity and experimentally validate their performance across several conditions.')
        context = row(2, 'Protein engineering motivates this work. Here we reveal the native protein signaling mechanism and characterize its physiological cellular role.')
        assert 'engineering' in summarize._why_read(engineering)
        assert 'background context' in summarize._why_read(context)
        assert engineering.abstract.split('. ')[1] in summarize._extractive(engineering)
        prompt, payload = summarize._build_prompt([engineering])
        assert 'supporting evidence' in prompt
        assert 'contribution_evidence' in payload
        assert 'brainstorm' not in prompt
    finally:
        summarize._set_reader_context(None)


def test_background_keyword_and_prestige_do_not_create_confident_relevance_explanation():
    summarize._set_reader_context(profile())
    try:
        item = row(1, 'Protein engineering is a growing field. Here we develop a clinical model for hospital mortality and evaluate it across several patient cohorts.')
        reason = summarize._why_read(item)
        assert 'match remains unclear' in reason
        assert 'selective venue' not in reason
        assert 'it connects to your interest' not in reason
    finally:
        summarize._set_reader_context(None)


def test_summary_uses_stated_object_and_keeps_quality_claims_unverified():
    import json

    item = row(1, 'Protein design motivates many research tools. We introduce a literature assistant that retrieves existing protein papers. We tested the assistant on held-out questions.')
    summarize._set_reader_context(profile())
    try:
        assert 'literature assistant that retrieves existing protein papers' in summarize._extractive(item)
        prompt, payload = summarize._build_prompt([item])
        record = json.loads(payload.split('\n\n', 1)[1])[0]
        assert record['author_claims']['claims'][0]['object'].startswith('a literature assistant')
        assert record['article_evidence']['dimensions']['validation']['status'] == 'reported'
        assert 'not independently verified' in prompt
        assert 'unknown evidence' in prompt
    finally:
        summarize._set_reader_context(None)
