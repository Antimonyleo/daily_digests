"""A same-day identical title is a copy only when identifiers and authors agree."""
from datetime import datetime, timezone

import itertools

import pytest

from dailydigest.dedupe import _merge_record_copies, dedupe_ranking_candidates
from dailydigest.models import Item

TITLE = "Self-assembled DNA nanostructures for targeted intracellular delivery of mRNA"
DAY = datetime(2026, 10, 13, tzinfo=timezone.utc)


def record(source, doi="", authors="", ext=None, url=None, abstract="A" * 300):
    return Item(source=source, section="research", external_id=ext or doi or source,
                url=url or (f"https://doi.org/{doi}" if doi else f"https://example.com/{source}"),
                title=TITLE, abstract=abstract, authors=authors, published_at=DAY)


def test_different_dois_and_authors_stay_two_works():
    a = record("Nature Nanotechnology", "10.1038/s41565-026-00001-1", "Alice Smith; Bob Lee")
    b = record("ACS Nano", "10.1021/acsnano.6c09999", "Chen Wang; Dana Patel", abstract="B" * 300)
    assert len(dedupe_ranking_candidates([a, b])) == 2


def test_different_dois_without_authors_stay_two_works():
    a = record("Nature Nanotechnology", "10.1038/s41565-026-00001-1")
    b = record("ACS Nano", "10.1021/acsnano.6c09999")
    assert len(dedupe_ranking_candidates([a, b])) == 2


def test_disjoint_authors_without_dois_stay_two_works():
    a = record("Feed A", authors="Alice Smith; Bob Lee")
    b = record("Feed B", authors="Chen Wang; Dana Patel")
    assert len(dedupe_ranking_candidates([a, b])) == 2


def test_angewandte_german_and_international_twins_merge():
    english = record("Angewandte Chemie International Edition", "10.1002/anie.202412345", "Alice Smith")
    german = record("Angewandte Chemie", "10.1002/ange.202412345", "Alice Smith")
    assert len(dedupe_ranking_candidates([english, german])) == 1


def test_angewandte_twins_without_authors_merge():
    english = record("Angewandte Chemie International Edition", "10.1002/anie.202412345")
    german = record("Angewandte Chemie", "10.1002/ange.202412345")
    assert len(dedupe_ranking_candidates([english, german])) == 1


def test_unrelated_angewandte_dois_stay_two_works():
    a = record("Angewandte Chemie International Edition", "10.1002/anie.202412345")
    b = record("Angewandte Chemie", "10.1002/ange.202499999")
    assert len(dedupe_ranking_candidates([a, b])) == 2


def test_cover_copy_with_own_doi_and_same_authors_merges():
    article = record("Advanced Materials", "10.1002/adma.202601234", "Smith, A.; Lee, B.")
    cover = record("Advanced Materials", "10.1002/adma.202670012", "Alice Smith; Bob Lee", abstract="")
    assert len(dedupe_ranking_candidates([article, cover])) == 1


def test_feed_copies_without_identifiers_still_merge():
    a = record("PubMed", ext="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
               authors="Smith A, Lee B")
    b = record("Publisher RSS", ext="rss-1", url="https://example.com/article/1",
               authors="Alice Smith, Bob Lee")
    assert len(dedupe_ranking_candidates([a, b])) == 1


def test_preprint_and_journal_title_match_still_left_to_version_stage():
    preprint = record("bioRxiv", "10.1101/2026.10.01.123456", "Alice Smith; Bob Lee")
    journal = record("Nature Nanotechnology", "10.1038/s41565-026-00001-1", "Chen Wang; Dana Patel")
    assert len(dedupe_ranking_candidates([preprint, journal])) == 2


@pytest.mark.parametrize("left,right", [
    ("W. Zhang et al.", "Wei Zhang, Na Li, Bo Chen"),
    ("Wei Zhang and colleagues", "Zhang W, Li N"),
    ("Zhang Wei; Li Na", "Wei Zhang; Na Li"),
    ("TM Dao, JR Smith", "Dao, T. M.; Smith, J. R."),
])
def test_truncated_or_reordered_author_lists_still_merge_copies(left, right):
    a = record("Feed A", authors=left)
    b = record("Feed B", authors=right)
    assert len(dedupe_ranking_candidates([a, b])) == 1


def test_shared_given_names_are_not_a_shared_author():
    a = record("Nature Nanotechnology", "10.1038/s41565-026-00001-1", "Alice Smith; Bob Lee")
    b = record("ACS Nano", "10.1021/acsnano.6c09999", "Alice Jones; Bob Patel", abstract="B" * 300)
    assert len(dedupe_ranking_candidates([a, b])) == 2


@pytest.mark.parametrize("left,right", [
    ("Wei Zhang; Na Li", "Min Zhang; Jie Li"),
    ("W. Zhang; N. Li", "M. Zhang; J. Li"),
])
def test_shared_surnames_of_different_people_stay_two_works(left, right):
    a = record("Feed A", authors=left)
    b = record("Feed B", authors=right, abstract="B" * 300)
    assert len(dedupe_ranking_candidates([a, b])) == 2


def test_different_dois_need_the_same_first_author():
    a = record("Nature Nanotechnology", "10.1038/s41565-026-00001-1", "Alice Smith; Bob Lee; Carol Wu")
    b = record("ACS Nano", "10.1021/acsnano.6c09999", "Dan Ortiz; Carol Wu", abstract="B" * 300)
    assert len(dedupe_ranking_candidates([a, b])) == 2


def test_middle_names_and_initials_still_name_one_author():
    a = record("Feed A", authors="Wei Ming Zhang; Na Li")
    b = record("Feed B", authors="Zhang W, Li N")
    assert len(dedupe_ranking_candidates([a, b])) == 1


def _works(rows):
    return sorted(sorted(member.source for member in group)
                  for _item, group in _merge_record_copies(rows))


def test_copy_of_the_second_same_title_work_finds_its_own_group_in_any_order():
    other = record("Feed A", authors="Alice Smith; Bob Lee", url="https://example.com/a")
    work = record("Feed B", authors="Chen Wang; Dana Patel", url="https://example.com/b", abstract="B" * 300)
    copy = record("Feed C", authors="Chen Wang; Dana Patel", url="https://example.com/c", abstract="B" * 300)
    expected = [["Feed A"], ["Feed B", "Feed C"]]
    for order in itertools.permutations([other, work, copy]):
        assert _works(list(order)) == expected, [row.source for row in order]


def test_title_only_record_never_joins_two_distinct_works():
    a = record("Feed A", authors="Alice Smith; Bob Lee", url="https://example.com/a")
    b = record("Feed B", authors="Chen Wang; Dana Patel", url="https://example.com/b", abstract="B" * 300)
    bare = record("Feed C", url="https://example.com/c")
    for order in itertools.permutations([a, b, bare]):
        works = _works(list(order))
        assert len(works) == 2, [row.source for row in order]
        assert not any({"Feed A", "Feed B"} <= set(group) for group in works)
