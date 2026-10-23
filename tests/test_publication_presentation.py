"""Publication genres and later commentary share one research-paper card."""
from __future__ import annotations

import json

from starlette.requests import Request


def _store_paper(store, *, title, doi, metadata, digest_id=None):
    with store.session_scope() as session:
        row = store.ItemRow(
            source="Nature", section="research", external_id=doi,
            url=f"https://doi.org/{doi}", title=title,
            abstract="DNA assembly experiments.",
            metadata_json=json.dumps({"doi": doi, **metadata}),
            digest_id=digest_id,
        )
        session.add(row)
        session.flush()
        return int(row.id)


def _seed(store):
    store.init_db()
    parent = _store_paper(
        store, title="Original lattice assembly", doi="10.1234/lattice",
        metadata={"publication_types": ["Research article"]},
    )
    commentary = _store_paper(
        store, title="Understanding the lattice", doi="10.1234/view",
        metadata={"publication_types": ["News and Views"],
                  "companion_of_doi": "10.1234/lattice"},
        digest_id="2026-10-22",
    )
    perspective = _store_paper(
        store, title="Future directions", doi="10.1234/perspective",
        metadata={"publication_types": ["Perspective"]},
    )
    store.write_digest("2026-10-21", [("R1", parent, .9), ("R2", perspective, .8)])
    store.write_digest("2026-10-22", [("R1", commentary, .9)])
    return parent, commentary, perspective


def test_archived_research_attaches_later_commentary_and_keeps_perspective():
    from dailydigest import store, web

    parent, commentary, perspective = _seed(store)
    sections, _ = web._load_today("2026-10-21", include_disabled_sections=True)
    entries = [entry for section in sections for entry in section["entries"]]
    assert [entry["id"] for entry in entries] == [parent, perspective]
    assert entries[0]["publication_label"] == "Research article"
    assert entries[1]["publication_label"] == "Perspective"
    child = entries[0]["related_commentaries"][0]
    assert child["title"] == "Understanding the lattice"
    assert child["publication_label"] == "News & Views"
    assert child["url"] == "https://doi.org/10.1234/view"
    tomorrow, votes = web._load_today("2026-10-22")
    assert tomorrow == []
    assert votes == {}
    # Rendering never deletes the later commentary or its historical record.
    with store.session_scope() as session:
        assert session.get(store.ItemRow, commentary) is not None


def test_publication_label_uses_row_metadata_not_old_ranking_snapshot(monkeypatch):
    from dailydigest import store, web

    _seed(store)
    original = web._breakdown_payload

    def stale_breakdown(row, snapshot):
        result = original(row, snapshot)
        result["content_type"] = "research"
        result["source_bucket"] = "published_journal"
        return result

    monkeypatch.setattr(web, "_breakdown_payload", stale_breakdown)
    sections, _ = web._load_today("2026-10-21")
    assert sections[0]["entries"][1]["publication_label"] == "Perspective"


def test_confirmed_retraction_is_visible_in_digest_and_saved_without_old_claim():
    from dailydigest import store, web

    parent, _, _ = _seed(store)
    store.set_bookmark(parent, True)
    with store.session_scope() as session:
        row = session.get(store.ItemRow, parent)
        row.summary = "This study establishes a useful experimental result."
        row.summary_signature = "old"
        row.summary_backend = "extractive"
    with store.session_scope() as session:
        row = session.get(store.ItemRow, parent)
        session.expunge(row)
    row.metadata_json = json.dumps({**store.item_metadata(row), "is_retracted": True})
    store.persist_publication_metadata([row])
    sections, _ = web._load_today("2026-10-21", include_disabled_sections=True)
    entry = next(entry for section in sections for entry in section["entries"] if entry["id"] == parent)
    assert entry["publication_label"] == "Retracted"
    assert not entry["summary"]
    request = Request({"type": "http", "method": "GET", "path": "/saved",
                       "query_string": b"", "headers": [],
                       "scheme": "http", "server": ("testserver", 80)})
    response = web.saved_items(request)
    entry = next(item for item in response.context["items"] if item["id"] == parent)
    assert entry["publication_label"] == "Retracted"
    assert "Retracted" in response.body.decode()
    assert "This study establishes a useful experimental result." not in response.body.decode()


def test_archived_generic_duplicate_uses_stored_relation_without_mutating_row():
    from dailydigest import store, web

    _seed(store)
    with store.session_scope() as session:
        duplicate = store.ItemRow(
            source="OpenAlex", section="research", external_id="W123",
            url="https://doi.org/10.1234/view", title="Understanding the lattice",
            metadata_json=json.dumps({"doi": "10.1234/view", "work_type": "article"}),
        )
        session.add(duplicate)
        session.flush()
        duplicate_id = int(duplicate.id)
        original_metadata = duplicate.metadata_json
    store.write_digest("2026-10-23", [("R1", duplicate_id, .9)])
    sections, _ = web._load_today("2026-10-23", include_disabled_sections=True)
    assert sections == []
    with store.session_scope() as session:
        assert session.get(store.ItemRow, duplicate_id).metadata_json == original_metadata


def test_companion_is_subentry_with_no_separate_rating(monkeypatch):
    from dailydigest import store, web

    parent, commentary, _ = _seed(store)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-10-21")
    monkeypatch.setattr(web, "daily_tea_deck", lambda *_args: ["Tea"])
    monkeypatch.setattr(web, "mark_impressions_viewed", lambda *_args: 0)
    monkeypatch.setattr(web.votes_mod, "lr_training_status", lambda: {
        "vote_counts": {"good": 0, "bad": 0, "signed": 0},
    })
    request = Request({"type": "http", "method": "GET", "path": "/",
                       "query_string": b"", "headers": [],
                       "scheme": "http", "server": ("testserver", 80)})
    response = web.index(request)
    html = response.body.decode()
    assert 'class="related-commentaries"' in html
    assert 'aria-label="Commentary on Original lattice assembly"' in html
    assert "News &amp; Views" in html
    assert html.count("Understanding the lattice") == 1
    assert f'data-item-id="{parent}"' in html
    assert f'data-item-id="{commentary}"' not in html


def test_saved_paper_gets_later_commentary_and_explicitly_saved_commentary_stays():
    from dailydigest import store, web

    parent, commentary, _ = _seed(store)
    store.set_bookmark(parent, True)
    store.set_bookmark(commentary, True)
    request = Request({"type": "http", "method": "GET", "path": "/saved",
                       "query_string": b"", "headers": [],
                       "scheme": "http", "server": ("testserver", 80)})
    response = web.saved_items(request)
    entries = {entry["id"]: entry for entry in response.context["items"]}
    assert set(entries) == {parent, commentary}
    assert entries[parent]["publication_label"] == "Research article"
    assert entries[parent]["related_commentaries"][0]["item_id"] == commentary
    assert entries[commentary]["publication_label"] == "News & Views"
    assert entries[commentary]["related_commentaries"] == []
    assert 'aria-label="Commentary on Original lattice assembly"' in response.body.decode()
