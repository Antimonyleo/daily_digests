"""A failed or cancelled Brew must preserve the saved digest's reading text."""
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from dailydigest import pipeline, store
from tests import test_brew_runtime

offline_brew = test_brew_runtime.offline_brew


def _saved_article(digest_id="2026-10-10"):
    store.init_db()
    with store.session_scope() as session:
        row = store.ItemRow(
            source="Nature", section="research", external_id="original",
            title="Programmable lattice assembly by experimental self-assembly",
            abstract="We demonstrate a measured and validated assembly method for lattices.",
            url="https://example.com/original", published_at=datetime.now(timezone.utc),
            summary="Original saved reading text", summary_signature="old-profile",
            summary_backend="extractive", metadata_json='{"keep": true}',
        )
        session.add(row)
        session.flush()
        item_id = int(row.id)
        session.add(store.BookmarkRow(item_id=item_id))
        session.add(store.VoteRow(item_id=item_id, value=1, grade=70))
    store.write_digest(digest_id, [("R1", item_id, 0.8)])
    return item_id


def test_publish_commits_selected_summaries_with_slate_and_preserves_reading_state():
    item_id = _saved_article()
    with store.session_scope() as session:
        other = store.ItemRow(source="Test", section="research", external_id="unselected",
                              title="Unselected", url="https://example.com/unselected",
                              summary="Unselected reading text")
        session.add(other)
        session.flush()
        other_id = int(other.id)
    store.publish_digest("2026-10-10", [("R1", item_id, 0.9)], [], [],
                         summaries={item_id: "Published text", other_id: "Should not leak"},
                         summary_signatures={item_id: "new-profile"},
                         summary_backends={item_id: "api"})
    with store.session_scope() as session:
        row = session.get(store.ItemRow, item_id)
        assert (row.summary, row.summary_signature, row.summary_backend, row.score) == (
            "Published text", "new-profile", "api", 0.9)
        assert row.metadata_json == '{"keep": true}'
        assert session.get(store.ItemRow, other_id).summary == "Unselected reading text"
        assert session.scalar(select(store.BookmarkRow.item_id)) == item_id
        assert session.scalar(select(store.VoteRow.grade)) == 70
    # Existing callers without a summary payload continue to retain their cache.
    store.publish_digest("2026-10-10", [("R1", item_id, 0.7)], [], [])
    with store.session_scope() as session:
        row = session.get(store.ItemRow, item_id)
        assert (row.summary, row.summary_signature, row.summary_backend) == (
            "Published text", "new-profile", "api")


@pytest.mark.parametrize("failure", ["database", "audit_serialization"])
def test_publication_failure_rolls_back_summary_and_slate(failure):
    item_id = _saved_article()
    impressions = [("research", 999999, 0, 0.9, True)] if failure == "database" else []
    audits = {"broken": [{"value": object()}]} if failure == "audit_serialization" else {}
    expected = IntegrityError if failure == "database" else TypeError
    message = "FOREIGN KEY constraint failed" if failure == "database" else "not JSON serializable"
    with pytest.raises(expected, match=message):
        store.publish_digest("2026-10-10", [("R2", item_id, 0.9)], [], impressions,
                             summaries={item_id: "Unpublished text"},
                             summary_signatures={item_id: "new-profile"},
                             summary_backends={item_id: "api"}, audits=audits)
    with store.session_scope() as session:
        row = session.get(store.ItemRow, item_id)
        assert (row.summary, row.summary_signature, row.summary_backend, row.item_label, row.score) == (
            "Original saved reading text", "old-profile", "extractive", "R1", 0.8)
        assert session.scalar(select(store.DigestItemRow.item_label)) == "R1"
        assert session.scalar(select(store.ImpressionRow.id)) is None
        assert session.scalar(select(store.BookmarkRow.item_id)) == item_id
        assert session.scalar(select(store.VoteRow.grade)) == 70


@pytest.mark.parametrize("failure", ["cancel", "publication"])
def test_failed_or_cancelled_rebrew_preserves_existing_visible_summary(monkeypatch, offline_brew, failure):
    item_id = _saved_article(offline_brew)
    with store.session_scope() as session:
        candidate = session.get(store.ItemRow, item_id)
        session.expunge(candidate)
    monkeypatch.setattr(pipeline, "recent_items", lambda **kwargs: [candidate])
    monkeypatch.setattr(pipeline, "exclude_reviewed_items", lambda rows: rows)
    monkeypatch.setattr(pipeline, "exclude_previously_shown", lambda rows, **kwargs: rows)
    monkeypatch.setattr(pipeline, "exclude_known_items", lambda rows: rows)
    monkeypatch.setattr(pipeline, "score_items", lambda items, *args, **kwargs: [(items[0], 0.9)])
    monkeypatch.setattr(pipeline, "pick_top_per_section", lambda scored, *args, **kwargs: scored)
    monkeypatch.setattr(pipeline, "summarize_items", lambda rows, **kwargs: {item_id: "Unpublished text"})
    stopped = [False]

    def progress(stage, payload):
        if failure == "cancel" and stage == "summarize_done":
            stopped[0] = True

    def fail_publication(*args, **kwargs):
        raise RuntimeError("Publication failed")

    if failure == "publication":
        monkeypatch.setattr(pipeline, "publish_digest", fail_publication)
    expected = pipeline.BrewCancelled if failure == "cancel" else RuntimeError
    message = "cancelled before publication" if failure == "cancel" else "Publication failed"
    with pytest.raises(expected, match=message):
        pipeline.run_all(dry_run=True, progress_callback=progress,
                         should_cancel=lambda: stopped[0])
    with store.session_scope() as session:
        assert session.get(store.DigestRow, offline_brew).item_count == 1
        row = session.get(store.ItemRow, item_id)
        assert (row.summary, row.summary_signature, row.summary_backend) == (
            "Original saved reading text", "old-profile", "extractive")
