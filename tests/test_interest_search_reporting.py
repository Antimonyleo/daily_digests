"""Search transparency must describe observations, never inferred paper supply."""
from dailydigest.pipeline import _interest_coverage_audit


def test_interest_audit_keeps_actual_query_and_empty_search_without_supply_claim():
    rows = _interest_coverage_audit({
        'profile': {'keywords': ['Induce proximity']},
        'source_health': [{'source': 'PubMed', 'diagnostics': {'queries': [
            {'interest': 'Induce proximity', 'query': '(induce[tiab] OR induced[tiab]) AND proximity[tiab]',
             'returned_count': 0, 'attempted': True},
            {'interest': 'Induce proximity', 'query': 'proximity', 'failure': 'timeout'},
        ]}}], 'candidates': [],
    })
    assert rows[0]['searches'][0] == {
        'source': 'PubMed', 'query': '(induce[tiab] OR induced[tiab]) AND proximity[tiab]',
        'status': 'completed', 'returned_count': 0,
    }
    assert rows[0]['searches'][1]['returned_count'] is None
    assert rows[0]['searches'][1]['status'] == 'failed'
    assert 'not all retrieved papers' in rows[0]['explanation']


def test_unattempted_interest_search_is_not_reported_as_completed():
    rows = _interest_coverage_audit({
        'profile': {'keywords': ['Induce proximity']},
        'source_health': [{'source': 'PubMed', 'diagnostics': {'queries': [
            {'interest': 'Induce proximity', 'query': 'proximity', 'attempted': False,
             'returned_count': 0, 'truncated': True},
        ]}}],
    })
    assert rows[0]['searches'][0]['status'] == 'not run'
    assert rows[0]['searches'][0]['returned_count'] is None
