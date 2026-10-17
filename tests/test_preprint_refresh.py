"""Authoritative preprint revisions and withdrawals update existing reader state."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from dailydigest import store
from dailydigest.ingest import biorxiv
from dailydigest.models import Item, SourceSpec
from dailydigest.rank.source_quality import should_skip_item


def paper(version="1", **changes):
    item = Item(source="bioRxiv", section="research", external_id="10.1101/fixture",
                url="https://doi.org/10.1101/fixture", title="Programmable molecular assembly",
                abstract="We found that assembly improved activity by 30 percent under all conditions.",
                published_at=datetime.now(timezone.utc),
                metadata={"doi": "10.1101/fixture", "preprint_version": version})
    return item.model_copy(update=changes)


@pytest.mark.parametrize("abstract", [
    "We found that assembly reduced activity by 30 percent under all conditions.",
    "Assembly reduced activity.",
])
def test_newer_version_replaces_shorter_text_and_invalidates_summary_without_losing_feedback(abstract):
    store.upsert_items([paper()])
    with store.session_scope() as session:
        row = session.scalar(select(store.ItemRow))
        iid = row.id
        row.summary = "Old finding"
        row.summary_signature = "old"
        row.summary_backend = "extractive"
        session.add(store.VoteRow(item_id=iid, value=1, grade=100))
    store.set_bookmark(iid, True)
    store.upsert_items([paper("2", title="Corrected assembly", abstract=abstract)])
    with store.session_scope() as session:
        row = session.get(store.ItemRow, iid)
        assert row.title == "Corrected assembly"
        assert row.abstract == abstract
        assert row.summary == "" and row.summary_signature is None and row.summary_backend is None
        assert store.item_metadata(row)["preprint_version"] == "2"
        assert session.scalar(select(store.VoteRow)).grade == 100
    assert store.bookmarked_item_ids([iid]) == {iid}


@pytest.mark.parametrize("reverse", [False, True])
def test_one_fetch_selects_newest_version_independently_of_provider_order(reverse):
    items = [paper(), paper("2", abstract="The earlier result was not replicated.")]
    store.upsert_items(list(reversed(items)) if reverse else items)
    store.upsert_items([paper()])
    with store.session_scope() as session:
        row = session.scalar(select(store.ItemRow))
        assert row.abstract == items[1].abstract
        assert store.item_metadata(row)["preprint_version"] == "2"


def test_withdrawal_notices_reach_cached_identity_siblings_without_entering_live_supply(monkeypatch):
    original = paper()
    sibling = paper(source="OpenAlex", external_id="W1", url="https://openalex.org/W1")
    unrelated = paper(source="Other", external_id="other", url="https://example.test/other",
                      metadata={"doi": "10.1101/unrelated"})
    store.upsert_items([original, sibling, unrelated])
    with store.session_scope() as session:
        iid = session.scalar(select(store.ItemRow.id).where(store.ItemRow.source == "bioRxiv"))
        session.get(store.ItemRow, iid).summary = "A previously cached claim."
        session.add(store.VoteRow(item_id=iid, value=1, grade=100))
    store.set_bookmark(iid, True)
    payload = {"messages": [{"total": 1}], "collection": [{
        "doi": "10.1101/fixture", "title": "WITHDRAWN: Programmable molecular assembly",
        "type": "withdrawn", "version": "3", "date": "2026-10-17",
    }]}
    monkeypatch.setattr(biorxiv, "_get_json", lambda *_args: payload)
    source = biorxiv.BiorxivSource()
    spec = SourceSpec(name="bioRxiv", kind="biorxiv", section="research")
    assert source.fetch(spec) == []
    assert len(source.last_fetch_withdrawals) == 1
    assert store.mark_research_withdrawn(source.last_fetch_withdrawals) == 2
    with store.session_scope() as session:
        rows = session.scalars(select(store.ItemRow)).all()
        for row in rows:
            assert should_skip_item(row) is (row.source != "Other")
        assert session.get(store.ItemRow, iid).summary == ""
        assert session.scalar(select(store.VoteRow)).grade == 100
    assert store.bookmarked_item_ids([iid]) == {iid}
    # Ordinary stale records cannot remove an observed withdrawal.
    store.upsert_items([original])
    assert all(should_skip_item(row) for row in store.recent_items(4) if row.source != "Other")
    assert store.mark_research_withdrawn(source.last_fetch_withdrawals) == 0
    assert store.mark_research_withdrawn([paper("3", metadata={"is_retracted": False})]) == 0
    monkeypatch.setattr(biorxiv, "_get_json", lambda *_args: {"messages": [{"total": 0}], "collection": []})
    assert source.fetch(spec) == []
    assert source.last_fetch_withdrawals == []


def test_observed_withdrawals_survive_a_later_provider_failure(monkeypatch):
    payload = {"messages": [{"total": 2}], "collection": [{
        "doi": "10.1101/fixture", "title": "WITHDRAWN: Molecular assembly", "type": "withdrawn",
    }]}

    def fetch(_client, url):
        if url.endswith("/0"):
            return payload
        raise RuntimeError("Provider failed after its withdrawal notice")

    monkeypatch.setattr(biorxiv, "_get_json", fetch)
    source = biorxiv.BiorxivSource()
    with pytest.raises(RuntimeError, match="fetch failed"):
        source.fetch(SourceSpec(name="bioRxiv", kind="biorxiv", section="research"))
    assert len(source.last_fetch_withdrawals) == 1
    assert source.last_fetch_diagnostics["queries"][0]["failure"]


@pytest.mark.parametrize("reverse", [False, True])
def test_adapter_returns_latest_version_before_url_deduplication(monkeypatch, reverse):
    entries = [{"doi": "10.1101/fixture", "title": "Molecular assembly", "version": str(v),
                "abstract": f"Results from version {v}"} for v in (1, 2)]
    payload = {"messages": [{"total": 2}], "collection": entries[::-1] if reverse else entries}
    monkeypatch.setattr(biorxiv, "_get_json", lambda *_args: payload)
    source = biorxiv.BiorxivSource()
    returned = source.fetch(SourceSpec(name="bioRxiv", kind="biorxiv", section="research"))
    assert len(returned) == 1
    assert returned[0].metadata["preprint_version"] == "2"
    assert returned[0].abstract == "Results from version 2"


def test_withdrawal_cannot_retract_a_separately_published_version():
    from dailydigest.dedupe import candidate_identity_keys

    original = paper(metadata={"doi": "10.1101/fixture", "published_doi": "10.1234/published"})
    published = paper(source="Nature", external_id="10.1234/published",
                      url="https://doi.org/10.1234/published", metadata={"doi": "10.1234/published"})
    store.upsert_items([original, published])
    notice = paper(metadata={"doi": "10.1101/fixture", "published_doi": "10.1234/published",
                             "is_retracted": True})
    assert "doi:10.1234/published" in candidate_identity_keys(notice)
    assert "doi:10.1234/published" not in candidate_identity_keys(notice, include_publication_links=False)
    assert store.mark_research_withdrawn([notice]) == 1
    with store.session_scope() as session:
        rows = session.scalars(select(store.ItemRow)).all()
        assert all(should_skip_item(row) is (row.source == "bioRxiv") for row in rows)
