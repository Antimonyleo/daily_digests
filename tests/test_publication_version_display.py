"""U8 display: one card per unique work; every version keeps its own records."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from tests.test_web import _json_payload, _request, _text_payload

DAY = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
AF3 = ("AlphaFold3 and Intrinsically Disordered Proteins: Reliable Monomer "
       "Prediction but Unreliable Complex Prediction")
PREPRINT, JOURNAL, OTHER = 97624, 96791, 90001


def _item(store, item_id, source, doi, *, title=AF3, authors="Dao, T. M.; Ghent, S.; Uversky, V. N.; Rahman, T.",
          url=None, metadata=None, published_at=DAY):
    return store.ItemRow(
        id=item_id, source=source, section="research", external_id=doi,
        url=url or f"https://doi.org/{doi}", title=title, authors=authors,
        abstract="We benchmark AlphaFold3 on disordered proteins and complexes. " * 6,
        published_at=published_at, metadata_json=json.dumps({"doi": doi, **(metadata or {})}),
    )


def _seed(store, *, journal_first=False):
    """The real 2026-10-13 pair (R2 bioRxiv preprint, R8 PubMed/Proteins) plus one paper."""
    store.init_db()
    with store.session_scope() as s:
        s.add(_item(store, PREPRINT, "bioRxiv (recent)", "10.64898/2025.12.05.691730",
                    metadata={"preprint_type": "new results", "preprint_version": "2"}))
        s.add(_item(store, JOURNAL, "PubMed (your topics)", "10.1002/prot.70189",
                    authors="Dao TM, Ghent S, Uversky VN, Rahman T",
                    url="https://pubmed.ncbi.nlm.nih.gov/42861594/", published_at=DAY - timedelta(days=1),
                    metadata={"publication_types": ["Journal Article"], "venue": "Proteins"}))
        s.add(_item(store, OTHER, "Nature", "10.1038/s41586-026-00001",
                    title="A different paper about designed protein binders for receptor signalling"))
    first, second = (JOURNAL, PREPRINT) if journal_first else (PREPRINT, JOURNAL)
    store.write_digest("2026-10-13", [("R1", OTHER, .9), ("R2", first, .8), ("R8", second, .5)])


def _research(web, digest_id="2026-10-13"):
    sections, _ = web._load_today(digest_id, include_disabled_sections=True)
    return next(section for section in sections if section["key"] == "research")["entries"]


def _index(web, monkeypatch, digest_id="2026-10-13"):
    monkeypatch.setattr(web, "_profile_exists", lambda: True)
    monkeypatch.setattr(web, "_digest_id", lambda: digest_id)
    monkeypatch.setattr(web, "daily_tea_deck", lambda *_args: ["Tea"])
    monkeypatch.setattr(web, "mark_impressions_viewed", lambda *_args: 0)
    monkeypatch.setattr(web.votes_mod, "lr_training_status", lambda: {"vote_counts": {"good": 0, "bad": 0, "signed": 0}})
    return _text_payload(web.index(_request("GET", "/")))


def _csrf(web, method, path):
    return _request(method, path, headers={"x-csrf-token": web._CSRF_TOKEN})


def test_history_counts_unique_works_and_links_the_hidden_version(monkeypatch):
    from dailydigest import store, web

    _seed(store)
    entries = _research(web)
    assert [entry["id"] for entry in entries] == [OTHER, PREPRINT]
    (link,) = entries[1]["version_links"]
    assert link["lead"] == "Also published in" and link["venue"] == "Proteins"
    assert link["href"] == "https://doi.org/10.1002/prot.70189" and link["item_id"] == JOURNAL
    assert link["status"] == ""
    assert entries[1]["version_item_ids"] == [JOURNAL]
    # The card keeps the preprint's own source and type; nothing is copied over.
    assert entries[1]["source"] == "bioRxiv" and entries[1]["ranking"]["source_bucket"] != "published_journal"
    html = _index(web, monkeypatch)
    assert html.count('class="item"') == 2
    assert f'data-item-id="{PREPRINT}"' in html and f'data-item-id="{JOURNAL}"' not in html
    assert 'Also published in <cite class="version-venue">Proteins</cite>' in html
    assert ">DOI 10.1002/prot.70189</a>" in html
    assert ">2 shown<" in html and '<span class="section-count">2</span>' in html
    # History is grouped for display, never rewritten or refilled.
    with store.session_scope() as s:
        assert {row.item_id for row in s.query(store.DigestItemRow).filter_by(digest_id="2026-10-13")} == {
            OTHER, PREPRINT, JOURNAL}


def test_hidden_version_title_and_doi_stay_searchable(monkeypatch):
    from dailydigest import store, web

    _seed(store)
    with store.session_scope() as s:
        s.get(store.ItemRow, JOURNAL).title = "AlphaFold3 for disordered proteins: renamed journal title"
        meta = json.loads(s.get(store.ItemRow, PREPRINT).metadata_json)
        s.get(store.ItemRow, PREPRINT).metadata_json = json.dumps({**meta, "published_doi": "10.1002/prot.70189"})
    entry = _research(web)[1]
    assert entry["version_links"][0]["venue"] == "Proteins"
    assert "renamed journal title" in entry["version_search"]
    assert "10.1002/prot.70189" in entry["version_search"]
    html = _index(web, monkeypatch)
    assert 'data-version-search="AlphaFold3 for disordered proteins: renamed journal title' in html
    assert "item.dataset.versionSearch" in html and ".version-links" in html


def test_journal_primary_names_the_preprint_version(monkeypatch):
    from dailydigest import store, web

    _seed(store, journal_first=True)
    entry = _research(web)[1]
    assert entry["id"] == JOURNAL
    assert entry["version_links"][0]["lead"] == "Preprint version on"
    assert entry["version_links"][0]["venue"] == "bioRxiv"
    assert 'Preprint version on <cite class="version-venue">bioRxiv</cite>' in _index(web, monkeypatch)


def test_rating_on_other_version_is_a_note_and_controls_act_on_primary(monkeypatch):
    from dailydigest import store, votes, web

    _seed(store)
    assert votes.record_vote_by_id(JOURNAL, 1, 70)
    entry = _research(web)[1]
    assert entry["id"] == PREPRINT and entry["current_grade"] is None
    assert entry["version_rating_notes"] == [
        {"item_id": JOURNAL, "venue": "Proteins", "grade": 70, "grade_label": "Relevant"}]
    html = _index(web, monkeypatch)
    assert "You rated the <cite>Proteins</cite> version: Relevant" in html
    # Rating the card writes the primary only; the other version's vote stays.
    monkeypatch.setattr(web.votes_mod, "lr_training_status", lambda: {})
    _json_payload(web.vote(_csrf(web, "POST", f"/vote/{PREPRINT}/40"), PREPRINT, 40))
    with store.session_scope() as s:
        grades = {(row.item_id, row.grade) for row in s.query(store.VoteRow)}
    assert grades == {(JOURNAL, 70), (PREPRINT, 40)}
    entry = _research(web)[1]
    assert entry["current_grade"] == 40  # never merged or averaged with 70
    assert entry["version_rating_notes"][0]["grade_label"] == "Relevant"


def test_equal_ratings_need_no_note():
    from dailydigest import store, votes, web

    _seed(store)
    votes.record_vote_by_id(JOURNAL, 1, 70)
    votes.record_vote_by_id(PREPRINT, 1, 70)
    entry = _research(web)[1]
    assert entry["current_grade"] == 70 and entry["version_rating_notes"] == []


def test_saved_on_either_version_shows_saved_and_unsave_clears_every_version(monkeypatch):
    from dailydigest import store, web

    _seed(store)
    store.set_bookmark(JOURNAL, True)
    entry = _research(web)[1]
    assert entry["bookmarked"] is True and entry["saved_item_ids"] == [JOURNAL]
    html = _index(web, monkeypatch)
    assert f'data-saved-item-ids="{JOURNAL}"' in html and 'data-bookmarked="true"' in html
    payload = _json_payload(web.bookmark_remove(_csrf(web, "DELETE", f"/bookmark/{PREPRINT}"), PREPRINT,
                                                versions=str(JOURNAL)))
    assert payload["item_ids"] == [PREPRINT, JOURNAL]  # reported only when versions are named
    assert store.bookmarked_item_ids([PREPRINT, JOURNAL, OTHER]) == set()
    # Saving from the card writes the card's own ID only.
    web.bookmark_add(_csrf(web, "POST", f"/bookmark/{PREPRINT}"), PREPRINT)
    assert store.bookmarked_item_ids([PREPRINT, JOURNAL]) == {PREPRINT}


def test_known_behaves_like_saved_and_stays_on_original_ids():
    from dailydigest import store, web

    _seed(store)
    store.set_item_known(JOURNAL, True)
    entry = _research(web)[1]
    assert entry["known"] is True and entry["known_item_ids"] == [JOURNAL]
    web.known_remove(_csrf(web, "DELETE", f"/known/{PREPRINT}"), PREPRINT, versions=str(JOURNAL))
    assert store.known_item_ids([PREPRINT, JOURNAL]) == set()
    web.known_add(_csrf(web, "POST", f"/known/{PREPRINT}"), PREPRINT)
    assert store.known_item_ids([PREPRINT, JOURNAL]) == {PREPRINT}


@pytest.mark.parametrize("versions", ["abc", "1;2", ",".join(str(i) for i in range(1, 23))])
def test_malformed_version_list_is_rejected_before_any_change(versions):
    from dailydigest import store, web

    _seed(store)
    store.set_bookmark(PREPRINT, True)
    with pytest.raises(HTTPException) as error:
        web.bookmark_remove(_csrf(web, "DELETE", f"/bookmark/{PREPRINT}"), PREPRINT, versions=versions)
    assert error.value.status_code == 400
    assert store.bookmarked_item_ids([PREPRINT]) == {PREPRINT}


def test_recorded_link_shows_its_own_withdrawn_status_only_on_the_link(monkeypatch):
    from dailydigest import store, web

    _seed(store)
    link = {"item_id": 55555, "label": "", "source": "medRxiv", "venue": "medRxiv",
            "doi": "10.1101/2026.01.01.1", "url": "https://doi.org/10.1101/2026.01.01.1",
            "is_preprint": True, "publication_label": "Withdrawn", "retraction": "Withdrawn",
            "relation": "published_doi", "evidence": {}}
    store.write_digest_features("2026-10-13", [("R1", OTHER, .9, {"publication_versions": [link]})])
    entry = _research(web)[0]
    assert entry["id"] == OTHER and entry["publication_label"] != "Withdrawn"
    view = entry["version_links"][0]
    assert view["lead"] == "Preprint version on" and view["status"] == "Withdrawn"
    html = _index(web, monkeypatch)
    assert '<strong class="version-status">Withdrawn</strong>' in html
    assert html.count(">Withdrawn<") == 1


def test_saved_page_shows_one_card_per_work_and_removes_every_saved_version():
    from dailydigest import store, web

    _seed(store)
    store.set_bookmark(PREPRINT, True)
    store.set_bookmark(JOURNAL, True)
    store.set_bookmark(OTHER, True)
    response = web.saved_items(_request("GET", "/saved"))
    html = _text_payload(response)
    assert "2 saved items" in html and html.count('class="saved-card"') == 2
    grouped = next(item for item in response.context["items"] if len(item["saved_item_ids"]) == 2)
    assert set(grouped["saved_item_ids"]) == {PREPRINT, JOURNAL}
    assert grouped["version_links"][0]["item_id"] in {PREPRINT, JOURNAL}
    assert f'data-saved-item-ids="{",".join(str(i) for i in grouped["saved_item_ids"])}"' in html
    assert "?versions=" in html
    others = [i for i in grouped["saved_item_ids"] if i != grouped["id"]]
    web.bookmark_remove(_csrf(web, "DELETE", "/bookmark"), grouped["id"], versions=",".join(map(str, others)))
    assert store.bookmarked_item_ids([PREPRINT, JOURNAL, OTHER]) == {OTHER}
    # Undo restores exactly the versions that were saved.
    web.bookmark_add(_csrf(web, "POST", "/bookmark"), grouped["id"], versions=",".join(map(str, others)))
    assert store.bookmarked_item_ids([PREPRINT, JOURNAL, OTHER]) == {PREPRINT, JOURNAL, OTHER}


def test_saved_card_attaches_brew_recorded_link_for_an_unsaved_version():
    from dailydigest import store, web

    _seed(store)
    store.set_bookmark(PREPRINT, True)
    link = {"item_id": JOURNAL, "label": "R8", "source": "PubMed (your topics)", "venue": "Proteins",
            "doi": "10.1002/prot.70189", "url": "https://pubmed.ncbi.nlm.nih.gov/42861594/",
            "is_preprint": False, "publication_label": "", "retraction": None,
            "relation": "same_title_authors", "evidence": {}}
    store.write_digest_features("2026-10-13", [("R2", PREPRINT, .8, {"publication_versions": [link]})])
    response = web.saved_items(_request("GET", "/saved"))
    (item,) = response.context["items"]
    assert item["id"] == PREPRINT and item["saved_item_ids"] == [PREPRINT]
    assert item["version_links"][0]["venue"] == "Proteins" and not item["version_links"][0]["in_digest"]
    assert "1 saved item" in _text_payload(response)
