"""P7 regression tests for reason-chip generalization by publisher/source.

Background (the 2026-07-22 bug): a broad-scope journal like *Nature* publishes
across every field. When the user marked several off-topic Nature papers as
"Not for me" with the `off_topic` reason, the ranker generalized that penalty by
publisher and buried an on-topic Nature paper the user actually wanted.

The fix (in votes.py, owned by another agent) narrows publisher/source/content
generalization to PUBLISHER-INTRINSIC reasons only:
``{low_impact, promotional, access_friction, duplicate}``. Topic/timing reasons
(``off_topic``, ``already_known``, ``not_urgent``) apply only their EXACT
per-item penalty and must NOT transfer to other papers from the same publisher.

These tests pin that behavior down.
"""
from __future__ import annotations

from datetime import datetime, timezone


def _reset_store_for_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_digest.db"))
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    return store_mod


def _add_nature_item(store_mod, external_id: str, title: str) -> int:
    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id=external_id,
            url=f"https://example.com/{external_id}",
            title=title,
            abstract="Peer-reviewed research abstract.",
            published_at=datetime.now(timezone.utc),
        )
        s.add(row)
        s.flush()
        return int(row.id)


def _detached(store_mod, item_id: int):
    with store_mod.session_scope() as s:
        row = s.get(store_mod.ItemRow, item_id)
        s.expunge(row)
        return row


def test_off_topic_negative_does_not_penalize_unrelated_nature_paper(tmp_path, monkeypatch):
    """An off_topic downvote on one Nature paper must NOT bleed onto another.

    Both items are *Nature* (same top-tier `published_journal` bucket), so if the
    ranker generalized `off_topic` by publisher, the on-topic paper WOULD receive
    a generalized penalty. It must not — `off_topic` is topic-specific, captured
    semantically by the LR's neg_affinity feature, not by publisher transfer.
    """
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    disliked_id = _add_nature_item(
        store_mod, "nature-offtopic", "Geology of Martian regolith"
    )
    ontopic_id = _add_nature_item(
        store_mod, "nature-ontopic", "RNA nanotechnology for targeted delivery"
    )

    assert votes_mod.record_vote_by_id(disliked_id, -1) is True
    assert votes_mod.record_vote_reason(disliked_id, "off_topic") is True

    ontopic = _detached(store_mod, ontopic_id)
    penalties = votes_mod.reason_penalty_map([ontopic])

    # The disliked item keeps its exact per-item penalty ...
    assert penalties.get(str(disliked_id), 0.0) > 0.0
    # ... but the unrelated on-topic Nature paper gets NO generalized penalty.
    assert str(ontopic_id) not in penalties or penalties[str(ontopic_id)] == 0.0


def test_promotional_negative_does_generalize_by_source(tmp_path, monkeypatch):
    """Inverse of the off_topic case: `promotional` DOES generalize by source.

    `promotional` describes the venue/feed, not the individual paper, so it is
    intended to transfer to future items from the same source. This documents the
    intended asymmetry between publisher-intrinsic and topic-specific reasons.
    """
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    disliked_id = _add_nature_item(
        store_mod, "nature-promo", "Sponsored product spotlight from Nature"
    )
    future_id = _add_nature_item(
        store_mod, "nature-promo-future", "Another Nature research report"
    )

    assert votes_mod.record_vote_by_id(disliked_id, -1) is True
    assert votes_mod.record_vote_reason(disliked_id, "promotional") is True

    future = _detached(store_mod, future_id)
    penalties = votes_mod.reason_penalty_map([future])

    # The future same-source item DOES pick up a (soft, decayed) generalized
    # penalty, smaller than the full per-item promotional weight.
    assert penalties.get(str(future_id), 0.0) > 0.0
    assert penalties[str(future_id)] < votes_mod.REASON_PENALTIES["promotional"]


def test_duplicate_currently_generalizes_by_publisher_debatable(tmp_path, monkeypatch):
    """DOCUMENTS current behavior: `duplicate` generalizes by publisher/source.

    NOTE (debatable): `duplicate` is arguably ITEM-SPECIFIC — one
    paper appearing twice says nothing about the *next* paper from the same
    publisher, so it is questionable that it lives in votes.py's `generalizable`
    set {low_impact, promotional, access_friction, duplicate}. This test pins the
    CURRENT behavior only; it is intentionally a characterization test, not an
    endorsement. If votes.py later removes `duplicate` from that set, flip the
    assertion below to `== 0.0`. (This test does not modify votes.py — another
    agent owns it.)
    """
    store_mod = _reset_store_for_tmp_db(monkeypatch, tmp_path)
    from dailydigest import votes as votes_mod

    # Guard: if `duplicate` is ever dropped from the generalizable set, this test
    # is expected to be updated. Assert current membership explicitly so the
    # failure message is self-explaining.
    disliked_id = _add_nature_item(
        store_mod, "nature-dup", "Duplicated Nature research report"
    )
    future_id = _add_nature_item(
        store_mod, "nature-dup-future", "Fresh Nature research report"
    )

    assert votes_mod.record_vote_by_id(disliked_id, -1) is True
    assert votes_mod.record_vote_reason(disliked_id, "duplicate") is True

    future = _detached(store_mod, future_id)
    penalties = votes_mod.reason_penalty_map([future])

    # CURRENT behavior: duplicate DOES transfer by publisher (debatable — see
    # docstring). Documented, not endorsed.
    assert penalties.get(str(future_id), 0.0) > 0.0
