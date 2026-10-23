"""Metadata refresh must preserve reading until publication, except unsafe claims."""
import json

import pytest
from sqlalchemy import select

from dailydigest import pipeline, store
from dailydigest.models import Item
from dailydigest.publications import publication_label
from tests import test_daily_release_persistence

_saved_article = test_daily_release_persistence._saved_article
offline_brew = test_daily_release_persistence.offline_brew


def _incoming(**metadata):
    return Item(
        source="Nature", section="research", external_id="original",
        title="Programmable lattice assembly by experimental self-assembly",
        abstract=("A fuller measured assembly result with methods and controls. " * 6).strip(),
        url="https://example.com/original", metadata=metadata,
    )


@pytest.mark.parametrize("outcome", ["cancel", "source_failure", "publication_failure", "success"])
def test_refresh_then_interruption_keeps_published_text_until_success(monkeypatch, offline_brew, outcome):
    item_id = _saved_article(offline_brew)
    incoming = _incoming()
    stopped = [False]
    generated = []

    def ingest(**kwargs):
        store.upsert_items([incoming])
        stopped[0] = outcome == "cancel"
        if outcome == "source_failure":
            raise RuntimeError("Research source coverage is degraded")
        return 0

    def candidates(**kwargs):
        with store.session_scope() as session:
            row = session.get(store.ItemRow, item_id)
            session.expunge(row)
            return [row]

    def summarize(rows, **kwargs):
        assert rows[0].abstract == incoming.abstract
        generated.append(item_id)
        return {item_id: "New published reading text"}

    def fail_publication(*args, **kwargs):
        raise RuntimeError("Publication failed")

    monkeypatch.setattr(pipeline, "ingest_all", ingest)
    monkeypatch.setattr(pipeline, "recent_items", candidates)
    monkeypatch.setattr(pipeline, "exclude_reviewed_items", lambda rows: rows)
    monkeypatch.setattr(pipeline, "exclude_previously_shown", lambda rows, **kwargs: rows)
    monkeypatch.setattr(pipeline, "exclude_known_items", lambda rows: rows)
    monkeypatch.setattr(pipeline, "score_items", lambda rows, *args, **kwargs: [(rows[0], .9)])
    monkeypatch.setattr(pipeline, "pick_top_per_section", lambda rows, *args, **kwargs: rows)
    monkeypatch.setattr(pipeline, "summarize_items", summarize)
    if outcome == "publication_failure":
        monkeypatch.setattr(pipeline, "publish_digest", fail_publication)

    if outcome == "success":
        assert pipeline.run_all(dry_run=True) == offline_brew
        assert generated == [item_id]
    else:
        error = pipeline.BrewCancelled if outcome == "cancel" else RuntimeError
        with pytest.raises(error):
            pipeline.run_all(dry_run=True, should_cancel=lambda: stopped[0])
    with store.session_scope() as session:
        row = session.get(store.ItemRow, item_id)
        assert row.abstract == incoming.abstract
        expected = "New published reading text" if outcome == "success" else "Original saved reading text"
        assert row.summary == expected
        assert row.summary_backend == "extractive"
        if outcome != "success":
            assert row.summary_signature is None
        else:
            assert row.summary_signature and row.summary_signature != "old-profile"
        assert session.get(store.DigestRow, offline_brew).item_count == 1
        assert session.scalar(select(store.BookmarkRow.item_id)) == item_id
        assert session.scalar(select(store.VoteRow.grade)) == 70


@pytest.mark.parametrize("write_path", ["upsert", "enrichment"])
@pytest.mark.parametrize("evidence, label", [
    ({"is_retracted": True}, "Retracted"),
    ({"publication_types": ["Research Article", "Retracted Publication"]}, "Retracted"),
    ({"is_retracted": True, "publication_types": ["Withdrawn"]}, "Withdrawn"),
])
def test_explicit_retraction_clears_old_claims_but_retains_saved_history(write_path, evidence, label):
    item_id = _saved_article()
    with store.session_scope() as session:
        row = session.get(store.ItemRow, item_id)
        abstract = row.abstract
        session.expunge(row)
    if write_path == "upsert":
        # Metadata alone can establish retraction; no richer text is needed.
        store.upsert_items([_incoming(**evidence).model_copy(update={"abstract": abstract})])
    else:
        row.metadata_json = json.dumps(evidence)
        store.persist_publication_metadata([row])
    with store.session_scope() as session:
        row = session.get(store.ItemRow, item_id)
        assert row.summary == ""
        assert row.summary_signature is None and row.summary_backend is None
        assert publication_label(row) == label
        assert session.scalar(select(store.DigestItemRow.item_id)) == item_id
        assert session.scalar(select(store.BookmarkRow.item_id)) == item_id
        assert session.scalar(select(store.VoteRow.grade)) == 70


def test_retraction_notice_is_not_labelled_as_a_retracted_article():
    assert publication_label(_incoming(publication_types=["Retraction of Publication"])) == "Retraction notice"


def test_retraction_flag_requires_explicit_boolean_evidence():
    assert publication_label(_incoming(is_retracted="false", publication_types=["Research Article"])) == "Research article"
