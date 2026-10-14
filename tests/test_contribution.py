from copy import deepcopy
from types import SimpleNamespace

import pytest

from dailydigest.rank.contribution import analyze_contribution, rerank_contributions


FACETS = {
    'Protein engineering and design': {'aliases': ['protein engineering', 'protein design']},
    'Protein structure and function prediction': {'aliases': ['protein structure prediction', 'AlphaFold']},
    'DNA nanotechnology': {'aliases': ['DNA origami', 'DNA nanotechnology']},
    'Self-assembled functional materials': {'aliases': ['peptide self-assembly']},
}


def analyze(text):
    return analyze_contribution('A study', text, FACETS)


def test_engineering_evidence_and_multiple_topics_are_grounded_in_abstract():
    text = 'Protein design is challenging. Here we design a DNA origami scaffold for engineered protein assembly and validate its performance across several conditions.'
    result = analyze(text)
    assert result['kind'] == 'engineering'
    assert set(result['facets']) == {'DNA nanotechnology', 'Protein engineering and design'}
    assert result['evidence'][0] in text
    assert result['evidence'][0].startswith('Here we design')


def test_natural_structure_and_signaling_are_context():
    result = analyze('Here we determine the structure of a native protein complex and reveal its role in cellular signaling using cryo-electron microscopy.')
    assert result['kind'] == 'natural_mechanism'
    assert result['priority'] == -1


def test_transferable_prediction_method_is_prioritized():
    result = analyze('Here we introduce a new model for protein structure prediction and evaluate its accuracy on an independent benchmark covering diverse folds.')
    assert result['kind'] == 'method'
    assert result['priority'] == 1


@pytest.mark.parametrize('text', [
    'Protein design has transformed biology. Here we develop a clinical risk model for hospital mortality and validate it in multiple patient cohorts.',
    'Earlier work engineered proteins for this purpose. Here we investigate the native protein signaling pathway and describe its biological mechanism.',
    'Protein engineering is promising. We did not design proteins in this study, but instead investigate cellular signaling in native protein complexes.',
    'Protein engineering offers exciting opportunities. Future work could develop a new protein design method, but here we describe an observational study.',
    'Protein engineering motivates this study. We use an existing protein design method to analyze naturally occurring sequences in a signaling pathway.',
    'Here we characterize engineered protein variants made in an earlier laboratory project and reveal the mechanism of the native signaling pathway.',
    'Here we investigate protein design constraints in native cellular complexes and describe the naturally occurring signaling mechanism.',
])
def test_background_negation_future_and_existing_method_do_not_get_engineering_boost(text):
    assert analyze(text)['priority'] <= 0


def test_existing_prediction_tool_is_application_not_new_method():
    result = analyze('We use AlphaFold to study native protein interactions and compare predicted structures with experimental observations from the cellular complex.')
    assert result['kind'] == 'application'
    assert result['priority'] == -1


def test_review_does_not_become_engineering_from_quoted_contributions():
    result = analyze('In this review, we discuss protein engineering. Recent studies developed protein design platforms and introduced powerful prediction methods.')
    assert result['kind'] == 'review'
    assert result['priority'] == -1


def test_missing_evidence_is_neutral_and_title_alone_cannot_boost():
    result = analyze_contribution('A new protein design method', '', FACETS)
    assert result['kind'] == 'unknown'
    assert result['priority'] == 0
    assert not result['evidence']


def test_off_interest_engineering_is_neutral():
    result = analyze('Here we develop a new platform for steel manufacturing and demonstrate improved efficiency using factory production data from several industrial sites.')
    assert result['priority'] == 0


def test_custom_profile_topics_work_without_biomolecular_taxonomy():
    result = analyze_contribution('Study', 'Here we develop a new method for ocean circulation prediction and validate the model using several independent satellite datasets.', {'Ocean forecasting': {'aliases': ['ocean circulation']}})
    assert result['kind'] == 'method'
    assert result['facets'] == ['Ocean forecasting']


def test_reranking_preserves_scores_eligibility_features_and_nonresearch_order():
    rows = [SimpleNamespace(id=i, section='research', title='Study', abstract=text) for i, text in enumerate([
        'Here we reveal the native protein signaling mechanism and determine the structure of the cellular complex using several complementary experiments.',
        'Here we design proteins with enhanced activity and validate the engineered protein variants across several independent experimental conditions.',
        '',
    ], 1)]
    rows.append(SimpleNamespace(id=4, section='world', title='News', abstract=''))
    scored = [(rows[0], .8), (rows[1], .79), (rows[2], .76), (rows[3], .81)]
    features = {i: {'topic_score': .9, 'final_score': s} for i, (_, s) in enumerate(scored, 1)}
    before = deepcopy(features)
    reordered = rerank_contributions(scored, features, FACETS, strength=.04)
    assert {r.id: s for r, s in reordered} == {r.id: s for r, s in scored}
    assert set(r.id for r, _ in reordered) == {1, 2, 3, 4}
    assert next(r.id for r, _ in reordered if r.section == 'research') == 2
    assert features[3]['contribution']['priority'] == 0
    assert 'contribution' not in features[4]
    for iid in features:
        assert features[iid]['topic_score'] == before[iid]['topic_score']
        assert features[iid]['final_score'] == before[iid]['final_score']


@pytest.mark.parametrize('strength', [-.1, .2, float('nan')])
def test_adjustment_is_bounded(strength):
    with pytest.raises(ValueError): rerank_contributions([], {}, FACETS, strength=strength)


@pytest.mark.parametrize('text', [
    'Here we develop a prognostic model based on serum proteins to predict hospital mortality and validate its clinical performance in several patient cohorts.',
    'Here we design experiments to determine the native protein signaling mechanism and characterize the physiological interactions in several cellular conditions.',
    'Here we present an existing protein structure prediction model and use it to study a naturally occurring cellular protein complex in several conditions.',
    'Here we develop no new protein prediction model, but analyze natural protein structures to understand their physiological signaling mechanisms.',
    'Here we synthesize peptides corresponding to the native signaling protein and determine its cellular mechanism with biochemical and structural measurements.',
    'Protein engineering is a promising area, and here we develop a model for hospital mortality using routine clinical measurements from several cohorts.',
    'Here we design a survey about protein supplement preferences among hospital patients and evaluate the questionnaire across several independent clinical cohorts.',
    'Here we introduce a coarse-grained model of native chromatin condensates and describe their naturally occurring organization and physical behavior.',
    'Here we present a tool that derives genome-level statistical evidence from identified peptides and benchmark its detection of genomes in metaproteomics.',
    'Here we design a clinical classifier based on protein abundance to predict hospital mortality and evaluate the classifier across several cohorts.',
    'Here we design a series of experiments to determine native protein signaling mechanisms across multiple complementary cellular conditions.',
    'Here we develop a new platform for mRNA quantification to measure transcript abundance and benchmark its accuracy across multiple samples.',
    'Here we develop a diagnostic platform for detecting protein variants associated with disease risk and validate it using several patient cohorts.',
    'Here we design neither proteins nor molecular devices; instead, we analyze patient outcomes across independent clinical cohorts and several hospitals.',
    'Here we engineer a portable device for measuring protein biomarkers in patient blood and assess its diagnostic accuracy across several cohorts.',
])
def test_independent_review_false_boost_regressions(text):
    assert analyze(text)['priority'] <= 0


@pytest.mark.parametrize('text', [
    'Before presenting our new protein design platform, we review available models. Here we introduce a new method for protein engineering and validate engineered variants.',
    'In this work, we used structure-guided rational design and directed evolution to produce an orthogonal protein with improved specificity for its modified substrate.',
    'To address these limitations, we developed a library of fluorescent protein binders using existing nanobodies, allowing modular and cell type-specific labeling.',
])
def test_independent_review_false_context_regressions(text):
    assert analyze(text)['priority'] == 1


def test_colloid_counting_and_rna_quantification_do_not_imply_engineering_methods():
    facets = {
        'Gene and RNA therapeutic engineering': {'aliases': ['mRNA therapeutics']},
        'Colloidal self-assembly': {'aliases': ['colloidal self-assembly']},
    }
    for text in [
        'Here we develop a new platform for mRNA quantification to measure transcript abundance and benchmark its accuracy across multiple samples.',
        'Here we develop a new method for counting colloidal particles in drinking water and validate their abundance across several natural samples.',
    ]:
        assert analyze_contribution('Study', text, facets)['priority'] <= 0
    result = analyze_contribution('Study', 'Here we engineer nanoparticles to improve mRNA delivery and validate their therapeutic performance across several independent experiments.', facets)
    assert result['priority'] == 1
