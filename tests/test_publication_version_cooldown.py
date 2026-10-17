"""Publication versions share a short reading cooldown, not a permanent ban."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from dailydigest import config, store
from dailydigest.dedupe import dedupe_ranking_candidates
from dailydigest.rank import near_dup

NOW = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
TITLE = "Programmable protein assemblies enable selective delivery of therapeutic RNA"


@pytest.fixture(autouse=True)
def clock_and_vectors(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz else NOW.replace(tzinfo=None)

    monkeypatch.setattr(near_dup, "datetime", Clock)
    # Deliberately different embeddings: identity cannot depend on cosine.
    monkeypatch.setattr(near_dup, "embed_item_rows", lambda rows: np.array([
        [1., 0.] if row.source == "bioRxiv" else [0., 1.] for row in rows
    ], dtype=np.float32))


def paper(*, preprint=False, item_id=None, title=TITLE, metadata=None, authors="Alice Smith; Bob Jones"):
    doi = "10.1101/2026.01.01.123456" if preprint else "10.1038/s41586-026-12345-6"
    return store.ItemRow(id=item_id, source="bioRxiv" if preprint else "Nature",
                         section="research", external_id=doi, url=f"https://doi.org/{doi}",
                         title=title, authors=authors, abstract="We engineer programmable assemblies for RNA delivery.",
                         published_at=NOW - timedelta(days=120 if preprint else 1),
                         metadata_json=json.dumps(metadata or {}))


def shown(preprint, age, *, sent=False, viewed=False, metadata=None, title=TITLE):
    store.init_db()
    row = paper(preprint=preprint, metadata=metadata, title=title)
    day = (NOW - timedelta(days=age)).date().isoformat()
    with store.session_scope() as session:
        session.add(row); session.flush()
        iid = int(row.id)
        session.add(store.DigestRow(id=day, created_at=NOW-timedelta(days=age),
                                    sent_at=NOW-timedelta(days=age) if sent else None))
        session.add(store.DigestItemRow(digest_id=day, item_id=iid, item_label="R1", score=.8))
        if viewed:
            session.add(store.ImpressionRow(digest_id=day, item_id=iid, section="research",
                                           run_id=f"fixture-{day}", position=0, final_score=.8,
                                           selected=True, viewed=True,
                                           created_at=NOW-timedelta(days=age)))
    return iid, day


def test_same_brew_collapses_versions_with_different_dois_and_dates():
    preprint, journal = paper(preprint=True), paper()
    result = dedupe_ranking_candidates([preprint, journal])
    assert result == [journal]


@pytest.mark.parametrize("age", [1, 7, 10, 14])
@pytest.mark.parametrize("sent,viewed", [(True, False), (False, True), (False, False)])
def test_recent_versions_suppressed_without_needing_similar_embeddings(age, sent, viewed):
    iid, _ = shown(True, age, sent=sent, viewed=viewed)
    candidate = paper(item_id=101)
    kept, dropped = near_dup.exclude_recent_near_duplicates([candidate], settings=config.load_settings())
    assert kept == []
    assert dropped[0]["matched_item_id"] == iid
    assert dropped[0]["cooldown_days"] == 14


def test_explicit_link_matches_changed_title():
    shown(True, 10, sent=True, metadata={"published_doi": "10.1038/s41586-026-12345-6"})
    candidate = paper(item_id=101, title="A renamed publication with a different abstract")
    kept, dropped = near_dup.exclude_recent_near_duplicates([candidate], settings=config.load_settings())
    assert kept == [] and dropped[0]["match_type"] == "identifier"


@pytest.mark.parametrize("age", [15, 90])
def test_later_journal_publication_remains_eligible(age):
    shown(True, age, sent=True)
    candidate = paper(item_id=101)
    kept, dropped = near_dup.exclude_recent_near_duplicates([candidate], settings=config.load_settings())
    assert kept == [candidate] and dropped == []


def test_ordinary_draft_candidates_are_not_reading_history():
    store.init_db()
    with store.session_scope() as session:
        row = paper(preprint=True); session.add(row); session.flush()
        session.add(store.DigestRow(id="2026-10-12", created_at=NOW-timedelta(days=1)))
        session.flush()
        session.add(store.ImpressionRow(digest_id="2026-10-12", item_id=row.id, section="research",
                                       run_id="fixture-pool", position=0, final_score=.8,
                                       selected=False, viewed=False,
                                       created_at=NOW-timedelta(days=1)))
    candidate = paper(item_id=101)
    kept, dropped = near_dup.exclude_recent_near_duplicates([candidate], settings=config.load_settings())
    assert kept == [candidate] and dropped == []


@pytest.mark.parametrize("title,authors", [(TITLE, "Carol Lee; David Chen"),
                                          ("Protein design", "Alice Smith; Bob Jones")])
def test_generic_title_or_conflicting_authors_does_not_create_a_family(title, authors):
    preprint = paper(preprint=True, title=title)
    journal = paper(title=title, authors=authors)
    assert len(dedupe_ranking_candidates([preprint, journal])) == 2


@pytest.mark.parametrize("same_day", [False, True])
def test_distinct_greek_targets_are_not_collapsed(same_day):
    preprint = paper(preprint=True, title=TITLE+" targeting α receptors")
    journal = paper(title=TITLE+" targeting β receptors")
    if same_day:
        preprint.published_at = journal.published_at
    assert len(dedupe_ranking_candidates([preprint, journal])) == 2


def test_current_day_rebrew_can_upgrade_to_journal_version():
    _, day = shown(True, 0, sent=True)
    candidate = paper(item_id=101)
    kept, dropped = near_dup.exclude_recent_near_duplicates(
        [candidate], settings=config.load_settings(), exclude_digest_id=day,
    )
    assert kept == [candidate] and dropped == []


@pytest.mark.parametrize("metadata", [{"publication_types": ["Editorial"]}, {"is_retracted": True}])
def test_ineligible_journal_record_does_not_replace_a_valid_preprint(metadata):
    preprint = paper(preprint=True)
    journal = paper(metadata=metadata)
    assert preprint in dedupe_ranking_candidates([preprint, journal])


def test_deterministic_identity_suppression_survives_encoder_failure(monkeypatch):
    shown(True, 3, sent=True)
    candidate = paper(item_id=101)
    def failed(rows):
        raise RuntimeError("encoder unavailable")
    monkeypatch.setattr(near_dup, "embed_item_rows", failed)
    kept, dropped = near_dup.exclude_recent_near_duplicates([candidate], settings=config.load_settings())
    assert kept == [] and dropped[0]["match_type"] == "preprint_journal_title"


def test_initials_and_author_separator_formatting_do_not_split_versions():
    preprint = paper(preprint=True, authors="Dao, T. M.; Ghent, S.; Uversky, V. N.; Rahman, T.")
    journal = paper(authors="Dao TM, Ghent S, Uversky VN, Rahman T")
    assert dedupe_ranking_candidates([preprint, journal]) == [journal]


def test_full_names_and_initials_match_after_journal_provider_merge():
    preprint = paper(preprint=True, authors="Dao, T. M.; Ghent, S.; Uversky, V. N.; Rahman, T.")
    pubmed = paper(authors="Dao TM, Ghent S, Uversky VN, Rahman T")
    journal = paper(authors="Tuan Minh Dao, Sebastien Ghent, Vladimir N. Uversky, Taseef Rahman")
    journal.abstract *= 5  # richer representative absorbs the initials-only copy
    assert dedupe_ranking_candidates([preprint, pubmed, journal]) == [journal]


def test_conflicting_full_names_with_same_initials_are_not_collapsed():
    preprint = paper(preprint=True, authors="Alice Smith; Bob Jones")
    journal = paper(authors="Arthur Smith, Bill Jones")
    assert len(dedupe_ranking_candidates([preprint, journal])) == 2
