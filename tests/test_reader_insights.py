from types import SimpleNamespace

import pytest

from dailydigest.reader_insights import match_evidence, usefulness_summary


def test_match_evidence_quotes_only_actual_whole_phrase_text():
    row = SimpleNamespace(title='A PROTAC platform for selective degradation', abstract='We tested specificity.')
    result = match_evidence(row, 'Target protein degradation', ['PROTAC'])
    assert result['term'] == 'PROTAC'
    assert result['quote'] in row.title
    assert result['topic'] == 'Target protein degradation'
    assert match_evidence(row, 'RNA', []) is None
    assert match_evidence(SimpleNamespace(title='RNAse signaling', abstract=''), 'RNA', []) is None


def test_match_evidence_is_bounded_and_does_not_invent_a_phrase_for_semantic_matches():
    row = SimpleNamespace(title='Protein design', abstract='x ' * 500 + 'asymmetric catalysis' + ' y' * 500)
    result = match_evidence(row, 'Asymmetric catalysis', [])
    assert len(result['quote']) <= 240
    assert result['quote'] in row.abstract
    assert match_evidence(row, 'Enzyme engineering', []) is None


@pytest.mark.parametrize('grades,expected', [([], (0, 0, 0)), ([100, 70, 40, 10, None], (2, 4, 1)), ([None, None], (0, 0, 2))])
def test_usefulness_counts_only_explicit_current_ratings(grades, expected):
    result = usefulness_summary([{'current_grade': grade} for grade in grades])
    assert (result['useful'], result['rated'], result['unrated']) == expected
    assert result['selected'] == len(grades)
    assert 'unknown' in result['basis'].lower()


def test_recent_study_counts_research_and_preserves_unrated_uncertainty():
    from dailydigest.reader_insights import recent_usefulness
    from dailydigest.store import DigestRow, DigestItemRow, ItemRow, VoteRow, session_scope, init_db
    init_db()
    with session_scope() as session:
        session.add(DigestRow(id='2026-10-15', item_count=3))
        items = [ItemRow(source='Fixture', section='research', title='useful', external_id='one', url='https://fixture.invalid/one'), ItemRow(source='Fixture', section='research', title='unknown', external_id='two', url='https://fixture.invalid/two'), ItemRow(source='Fixture', section='industry', title='other section', external_id='three', url='https://fixture.invalid/three')]
        session.add_all(items); session.flush()
        for index, row in enumerate(items):
            session.add(DigestItemRow(digest_id='2026-10-15', item_id=row.id, item_label=f'R{index + 1}'))
        session.add(VoteRow(item_id=items[0].id, value=1, grade=70))
    result = recent_usefulness()[0]
    assert (result['useful'], result['rated'], result['selected'], result['unrated']) == (1, 1, 2, 1)
