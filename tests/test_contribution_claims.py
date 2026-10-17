import pytest

from dailydigest.rank.contribution_claims import contribution_claims, contribution_text


def test_contribution_records_bind_task_and_object_to_exact_author_claim():
    abstract = ('Protein tools are widely used. Here we introduce a transferable protein design method. '
                'We evaluate the method on independent molecular targets.')
    result = contribution_claims(abstract)
    assert result['claims'][0]['role'] == 'introduced_method'
    assert result['claims'][0]['task'] == 'introduce'
    assert result['claims'][0]['object'] == 'a transferable protein design method.'
    for claim in result['claims']:
        assert abstract[claim['start']:claim['end']] == claim['quote']
        assert claim['task'] in claim['quote']
        assert claim['object'] in claim['quote']
    text = contribution_text('A study', abstract)
    assert 'widely used' not in text
    assert 'introduce a transferable' in text


@pytest.mark.parametrize('abstract', [
    'We do not introduce a new molecular predictor in this study.',
    'We may develop a molecular predictor in future work.',
    'Previous studies developed a protein predictor for these targets.',
    'Previously, we developed a molecular predictor for these targets.',
])
def test_future_negated_or_prior_claims_are_not_current_contributions(abstract):
    assert contribution_claims(abstract)['claims'] == []
    # Uncertain extraction preserves the original input, rather than a scope gate.
    assert contribution_text('Title', abstract) == 'Title. ' + abstract


def test_new_assistant_is_method_but_role_does_not_assert_core_scope():
    abstract = 'We introduce a protein literature assistant that searches existing publications.'
    result = contribution_claims(abstract)
    assert result['claims'][0]['role'] == 'introduced_method'
    assert 'literature assistant' in result['claims'][0]['object']
    assert 'scope' not in result


def test_existing_tool_can_support_a_new_engineered_system():
    abstract = ('We use an existing folding tool to evaluate RNA strands. '
                'We engineer a programmable RNA assembly with controlled geometry.')
    claims = contribution_claims(abstract)['claims']
    assert {c['role'] for c in claims} == {'application', 'engineered_system'}
    assert 'controlled geometry' in contribution_text('Title', abstract)


def test_negation_in_previous_clause_does_not_hide_positive_claim():
    abstract = 'We do not merely apply an existing tool; we introduce a new molecular predictor.'
    claims = contribution_claims(abstract)['claims']
    assert len(claims) == 1
    assert claims[0]['task'] == 'introduce'
    assert claims[0]['quote'] == 'we introduce a new molecular predictor.'


@pytest.mark.parametrize('abstract', [
    'It is not true that we developed a new protein-design model. Our study only examines records.',
    'The previous authors wrote, "We introduce a general protein-design algorithm." This review discusses it.',
    'In 2018, we developed a protein predictor. This study describes a different result.',
    'For future work, we propose a protein design algorithm.',
    'Before this study, we developed a protein predictor.',
    'The claim that we developed a protein predictor is false.',
    'The term ‘we design proteins’ appears in a cited earlier report.',
])
def test_ambiguous_attribution_or_temporal_prefix_is_neutral(abstract):
    assert contribution_claims(abstract)['claims'] == []
    assert contribution_text('Title', abstract) == 'Title. ' + abstract


def test_claim_keeps_qualifying_prefix_and_late_contribution_is_not_discarded():
    abstract = 'Although prior tools exist, we introduce a new molecular prediction method.'
    assert contribution_claims(abstract)['claims'][0]['quote'] == abstract
    many = ('We use standard microscopy. We measure cell viability. '
            'We characterize native signaling. We engineer a programmable RNA switch and validate its geometry.')
    assert 'engineer a programmable RNA switch' in contribution_text('Title', many)


def test_ambiguous_prior_claim_does_not_hide_an_unparsed_current_contribution():
    abstract = ('Before this study, we developed a protein predictor. '
                'We use a standard analysis tool. A new RNA device is constructed and validated here.')
    assert contribution_text('Title', abstract) == 'Title. ' + abstract


def test_quotation_is_not_truncated_and_sparse_input_stays_uncertain():
    abstract = 'We develop a model ' + 'with detailed evaluation ' * 90 + '.'
    claim = contribution_claims(abstract)['claims'][0]
    assert claim['quote'] == abstract
    assert contribution_claims('')['status'] == 'uncertain'
