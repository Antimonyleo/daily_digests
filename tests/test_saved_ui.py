"""Saved reading preserves scientific context and supports real bookmark undo."""

from datetime import datetime, timezone

from tests.test_web import _json_payload, _request, _text_payload


def saved_item(monkeypatch, tmp_path, summary):
    from dailydigest import store, web

    profile = tmp_path / "reader.yaml"
    profile.write_text("name: Ada\nkeywords: [assembly]\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    store.init_db()
    with store.session_scope() as session:
        item = store.ItemRow(
            source="Fixture Journal", section="research", external_id="saved-ui",
            title="A fictional assembly study", url="https://example.invalid/saved-ui",
            abstract="A supplied abstract.", summary=summary,
            published_at=datetime(2026, 10, 21, tzinfo=timezone.utc),
        )
        session.add(item)
        session.flush()
        item_id = item.id
        session.add(store.VoteRow(item_id=item_id, value=1, grade=70))
    store.set_bookmark(item_id, True)
    store.write_digest("2026-10-21", [("R1", item_id, .8)])
    return store, web, item_id


def test_saved_view_keeps_structured_finding_relevance_and_caveat_once(monkeypatch, tmp_path):
    _store, web, _item_id = saved_item(
        monkeypatch, tmp_path,
        "Key finding: Fixture finding.\nWhy read: Fixture transfer.\nCaveat: Fixture limitation.",
    )
    html = _text_payload(web.saved_items(_request("GET", "/saved")))
    for text in ("Fixture finding.", "Fixture transfer.", "Fixture limitation."):
        assert html.count(text) == 1
    assert 'class="reading-context-field"' in html
    assert 'class="copy-summary btn"' in html
    assert 'class="btn undo-saved"' in html
    assert html.index('href="/"') < html.index('href="/setup"')
    assert "Your personal library" not in html
    assert "Keep useful papers, calls, and news within reach." not in html


def test_saved_view_retains_original_unstructured_text_without_inventing_fields(monkeypatch, tmp_path):
    _store, web, _item_id = saved_item(monkeypatch, tmp_path, "Only the supplied plain summary.")
    html = _text_payload(web.saved_items(_request("GET", "/saved")))
    assert '<div class="saved-summary">Only the supplied plain summary.</div>' in html
    assert '<div class="reading-context-field">' not in html


def test_undo_bookmark_route_preserves_rating_summary_and_retained_edition(monkeypatch, tmp_path):
    store, web, item_id = saved_item(monkeypatch, tmp_path, "Retained scientific reading text.")
    headers = {"X-CSRF-Token": web._CSRF_TOKEN}
    result = web.bookmark_remove(_request("DELETE", f"/bookmark/{item_id}", headers=headers), item_id)
    assert _json_payload(result)["saved"] is False
    assert store.bookmarked_item_ids([item_id]) == set()
    result = web.bookmark_add(_request("POST", f"/bookmark/{item_id}", headers=headers), item_id)
    assert _json_payload(result)["saved"] is True
    assert store.bookmarked_item_ids([item_id]) == {item_id}
    with store.session_scope() as session:
        assert session.get(store.ItemRow, item_id).summary == "Retained scientific reading text."
        votes = session.query(store.VoteRow).filter_by(item_id=item_id).all()
        assert [(vote.value, vote.grade) for vote in votes] == [(1, 70)]
        assert session.query(store.DigestItemRow).filter_by(item_id=item_id).count() == 1
