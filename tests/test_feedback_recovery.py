"""Feedback mutations preserve private state when its reason file is damaged."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import select

from dailydigest import store, votes


@pytest.fixture
def feedback(monkeypatch):
    monkeypatch.setattr(votes, "_update_rocchio", lambda *args: None)
    store.init_db()
    with store.session_scope() as session:
        item = store.ItemRow(source="test", section="research", external_id="paper",
                             title="A useful method", url="https://example.test/paper",
                             digest_id="2026-10-10", item_label="R1")
        session.add(item)
        session.flush()
        item_id = int(item.id)
    votes.record_vote_by_id(item_id, -1, 40)
    votes.record_vote_reason(item_id, "too_contextual")
    return item_id, votes._vote_reasons_path()


@pytest.mark.parametrize("content", [
    b'{"1": [', b'[]', b'null', b'{"1": "off_topic"}',
    b'{"1": [123]}', b'{"oops": []}', b'\xff',
])
@pytest.mark.parametrize("operation", ["add", "remove", "positive", "batch", "clear"])
def test_corrupt_reasons_block_mutations_without_changing_votes(feedback, content, operation):
    item_id, path = feedback
    path.write_bytes(content)
    before = store.load_feedback_events()
    actions = {
        "add": lambda: votes.record_vote_reason(item_id, "off_topic"),
        "remove": lambda: votes.remove_vote_reason(item_id, "too_contextual"),
        "positive": lambda: votes.record_vote_by_id(item_id, 1, 100),
        "batch": lambda: votes.record_votes("+R1", "2026-10-10"),
        "clear": lambda: votes.clear_rating_by_id(item_id),
    }
    with pytest.raises(votes.FeedbackStateError, match="reason"):
        actions[operation]()
    assert path.read_bytes() == content
    assert votes.get_vote_value(item_id) == -1
    assert store.load_feedback_events() == before


@pytest.mark.parametrize("operation", ["positive", "clear"])
def test_reason_write_failure_rolls_back_canonical_vote(feedback, monkeypatch, operation):
    item_id, path = feedback
    before = path.read_bytes(), store.load_feedback_events()

    def fail(_data):
        raise OSError("disk full")

    monkeypatch.setattr(votes, "_write_vote_reasons", fail)
    with pytest.raises(votes.FeedbackStateError, match="reason"):
        if operation == "clear":
            votes.clear_rating_by_id(item_id)
        else:
            votes.record_vote_by_id(item_id, 1, 100)
    assert votes.get_vote_value(item_id) == -1
    assert (path.read_bytes(), store.load_feedback_events()) == before


def test_commit_failure_restores_exact_original_reason_bytes(feedback, monkeypatch):
    item_id, path = feedback
    original = ('{ "' + str(item_id) + '" : ["too_contextual"] }\n').encode()
    path.write_bytes(original)
    before = store.load_feedback_events()
    original_scope = votes.session_scope

    @contextmanager
    def fail_commit():
        with original_scope() as session:
            yield session
            raise RuntimeError("injected commit failure")

    monkeypatch.setattr(votes, "session_scope", fail_commit)
    with pytest.raises(votes.FeedbackStateError):
        votes.record_vote_by_id(item_id, 1, 100)
    monkeypatch.setattr(votes, "session_scope", original_scope)
    assert votes.get_vote_value(item_id) == -1
    assert path.read_bytes() == original
    assert store.load_feedback_events() == before


def test_clear_removes_current_rating_reasons_but_keeps_history_and_bookmark(feedback):
    item_id, path = feedback
    assert store.set_bookmark(item_id, True)
    old_events = store.load_feedback_events()
    assert votes.clear_rating_by_id(item_id)
    assert votes.get_vote_value(item_id) is None
    assert votes.get_vote_reasons(item_id) == []
    assert json.loads(path.read_text()) == {}
    assert item_id in store.bookmarked_item_ids([item_id])
    with store.session_scope() as session:
        assert session.get(store.ItemRow, item_id) is not None
        assert session.execute(select(store.VoteRow)).scalars().all() == []
    events = store.load_feedback_events()
    assert events[:-1] == old_events
    assert events[-1]["value"] == 0
    assert events[-1]["grade"] is None
    assert votes.clear_rating_by_id(item_id)
    assert store.load_feedback_events() == events


def test_clear_missing_or_unrated_item_never_fabricates_feedback(monkeypatch):
    monkeypatch.setattr(votes, "_update_rocchio", lambda *args: None)
    store.init_db()
    with store.session_scope() as session:
        item = store.ItemRow(source="test", section="research", external_id="unrated",
                             title="Unrated", url="https://example.test/unrated")
        session.add(item)
        session.flush()
        item_id = int(item.id)
    assert votes.clear_rating_by_id(item_id)
    assert not votes.clear_rating_by_id(item_id + 1)
    assert store.load_feedback_events() == []
    assert votes.get_vote_value(item_id) is None
    assert not votes._vote_reasons_path().exists()


def test_unknown_reason_strings_are_preserved_when_updating_other_item(feedback):
    item_id, path = feedback
    path.write_text(json.dumps({str(item_id): ["too_contextual"], "987": ["future_reason"]}))
    votes.record_vote_by_id(item_id, 1, 100)
    assert json.loads(path.read_text()) == {"987": ["future_reason"]}


def test_failed_model_invalidation_does_not_clear_rating(feedback, monkeypatch):
    from dailydigest.rank.ranker import _lr_weights_path

    item_id, path = feedback
    before = path.read_bytes(), store.load_feedback_events()
    weights = _lr_weights_path()
    weights.write_bytes(b"retained model")
    unlink = Path.unlink

    def fail_weights(target, *args, **kwargs):
        if target == weights:
            raise OSError("model is read-only")
        return unlink(target, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_weights)
    with pytest.raises(votes.FeedbackStateError):
        votes.clear_rating_by_id(item_id)
    assert votes.get_vote_value(item_id) == -1
    assert (path.read_bytes(), store.load_feedback_events()) == before
    assert weights.read_bytes() == b"retained model"


@pytest.mark.parametrize("content", [
    b'{"1": ["too_contextual"], "1": ["off_topic"]}',
    b'{"01": ["too_contextual"]}',
])
def test_duplicate_or_aliased_reason_ids_block_updates_without_loss(feedback, content):
    item_id, path = feedback
    path.write_bytes(content)
    before = store.load_feedback_events()
    with pytest.raises(votes.FeedbackStateError):
        votes.record_vote_by_id(item_id, 1, 100)
    assert path.read_bytes() == content
    assert votes.get_vote_value(item_id) == -1
    assert store.load_feedback_events() == before
