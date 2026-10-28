"""Regressions for actions on retained and saved reading, using isolated state."""
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from tests.test_web import _json_payload, _request, _text_payload


def _overflow(web, monkeypatch):
    audits = {"2026-05-07": [{"item_id": 71, "title": "Previous pick"}],
              "2026-05-08": [{"item_id": 81, "title": "Current pick"}]}
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-08")
    monkeypatch.setattr(web, "load_digest_audit", lambda day, kind: audits.get(day, []))
    pins = []
    monkeypatch.setattr(web, "add_carryover_items",
                        lambda ids, pinned_digest_id: pins.append((ids, pinned_digest_id)) or len(ids))
    return audits, pins


def test_archive_overflow_pins_displayed_items_for_next_day(monkeypatch):
    from dailydigest import web
    audits, pins = _overflow(web, monkeypatch)
    signature = web._overflow_signature("2026-05-07", audits["2026-05-07"])
    request = _request("POST", "/overflow/save", headers={"x-csrf-token": web._CSRF_TOKEN})
    response = web.overflow_save(request, digest_id="2026-05-07", audit_signature=signature)
    assert _json_payload(response)["digest_id"] == "2026-05-07"
    assert pins == [([71], "2026-05-08")]


def test_archived_single_overflow_does_not_require_current_brew(monkeypatch):
    from dailydigest import web
    audits, pins = _overflow(web, monkeypatch)
    del audits["2026-05-08"]
    signature = web._overflow_signature("2026-05-07", audits["2026-05-07"])
    request = _request("POST", "/overflow/save/71", headers={"x-csrf-token": web._CSRF_TOKEN})
    web.overflow_save_one(request, 71, digest_id="2026-05-07", audit_signature=signature)
    assert pins == [([71], "2026-05-08")]


def test_overflow_rejects_changed_audit_and_old_client_without_identity(monkeypatch):
    from dailydigest import web
    audits, pins = _overflow(web, monkeypatch)
    signature = web._overflow_signature("2026-05-07", audits["2026-05-07"])
    audits["2026-05-07"] = [{"item_id": 72, "title": "Replacement pick"}]
    request = _request("POST", "/overflow/save", headers={"x-csrf-token": web._CSRF_TOKEN})
    with pytest.raises(HTTPException) as changed:
        web.overflow_save(request, digest_id="2026-05-07", audit_signature=signature)
    assert changed.value.status_code == 409
    with pytest.raises(HTTPException) as missing:
        web.overflow_save(request)
    assert missing.value.status_code == 409
    assert pins == []


def test_saved_library_paginates_all_items_and_exposes_feedback(monkeypatch):
    from dailydigest import store, web
    store.init_db()
    with store.session_scope() as s:
        for index in range(205):
            item = store.ItemRow(source="Nature", section="research", external_id=f"saved-{index}",
                                 url=f"https://example.com/{index}", title=f"Saved paper {index}")
            s.add(item)
            s.flush()
            s.add(store.BookmarkRow(item_id=item.id, created_at=datetime.now(timezone.utc)))
    first = _text_payload(web.saved_items(_request("GET", "/saved"), page=1))
    last = _text_payload(web.saved_items(_request("GET", "/saved"), page=5))
    assert "205 saved items" in first
    assert 'href="/saved?page=2' in first
    assert first.count('class="saved-card"') == 50
    assert last.count('class="saved-card"') == 5
    assert "Saved paper 0" in last
    assert 'data-grade="70"' in last and 'data-reason="too_contextual"' in last
    assert "Clear rating" in last
    assert store.count_bookmarks("Saved paper 204") == 1


def test_saved_empty_page_clamps_after_removal():
    from dailydigest import web
    page = _text_payload(web.saved_items(_request("GET", "/saved"), page=999))
    assert "0 saved items" in page


def test_clear_vote_route_returns_actionable_state_failure(monkeypatch):
    from dailydigest import web
    monkeypatch.setattr(web.votes_mod, "clear_rating_by_id", lambda iid: True, raising=False)
    monkeypatch.setattr(web.votes_mod, "lr_training_status", lambda: {})
    request = _request("DELETE", "/vote/42", headers={"x-csrf-token": web._CSRF_TOKEN})
    assert _json_payload(web.vote_clear(request, 42))["new_grade"] is None


def test_run_recovery_offers_reading_and_reconnect(monkeypatch):
    from dailydigest import web
    monkeypatch.setattr(web, "_profile_exists", lambda: True)
    monkeypatch.setattr(web, "_recent_digest_ids", lambda *_a, **_kw: ["2026-10-24"])
    page = _text_payload(web.run_get(_request("GET", "/run")))
    assert "Read latest saved digest" in page
    assert "Reconnect" in page
    assert "The brew may still be running" in page


def test_resuming_unknown_run_never_starts_another_brew(monkeypatch):
    from dailydigest import web
    monkeypatch.setattr(web, "_profile_exists", lambda: True)
    monkeypatch.setattr(web, "_RUN_QUEUES", {})
    started = []
    monkeypatch.setattr(web, "_kick_off_run", lambda *args: started.append(args))
    page = _text_payload(web.run_get(_request("GET", "/run"), run_id="c" * 12, resume=True))
    assert "const RESUME = true;" in page
    assert "resume=1" in page
    assert started == [] and web._RUN_QUEUES == {}


def test_old_refresh_url_is_also_resume_only(monkeypatch):
    from dailydigest import web
    monkeypatch.setattr(web, "_profile_exists", lambda: True)
    page = _text_payload(web.run_get(_request("GET", "/run"), run_id="d" * 12, autostart=True))
    assert "const RESUME = true;" in page


def test_setup_cannot_mutate_files_while_compute_is_running(tmp_path, monkeypatch):
    import asyncio
    from dailydigest import web
    path = tmp_path / "profile.yaml"
    path.write_text("bio: keep\nkeywords: [RNA]\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: path)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    web._COMPUTE_LOCK.acquire()
    try:
        request = _request("POST", "/setup", form={"_csrf_token": web._CSRF_TOKEN,
                            "topics": "DNA", "bio": "changed"})
        with pytest.raises(HTTPException) as busy:
            asyncio.run(web.setup_post(request))
        assert busy.value.status_code == 409
        assert path.read_text() == "bio: keep\nkeywords: [RNA]\n"
    finally:
        web._COMPUTE_LOCK.release()


@pytest.mark.parametrize("signature", ["☃", "z" * 64, "f" * 65])
def test_overflow_invalid_signature_is_actionable(monkeypatch, signature):
    from dailydigest import web
    _audits, pins = _overflow(web, monkeypatch)
    request = _request("POST", "/overflow/save", headers={"x-csrf-token": web._CSRF_TOKEN})
    with pytest.raises(HTTPException) as error:
        web.overflow_save(request, digest_id="2026-05-07", audit_signature=signature)
    assert error.value.status_code == 409
    assert pins == []


def test_clear_rating_cannot_race_training(monkeypatch):
    from dailydigest import web
    calls = []
    monkeypatch.setattr(web.votes_mod, "clear_rating_by_id", lambda iid: calls.append(iid))
    web._COMPUTE_LOCK.acquire()
    try:
        request = _request("DELETE", "/vote/42", headers={"x-csrf-token": web._CSRF_TOKEN})
        with pytest.raises(HTTPException) as busy:
            web.vote_clear(request, 42)
        assert busy.value.status_code == 409
        assert calls == []
    finally:
        web._COMPUTE_LOCK.release()


def test_corrupt_reasons_preserve_reading_but_block_training(monkeypatch):
    from dailydigest import web
    def corrupt():
        raise web.votes_mod.FeedbackStateError("Repair the feedback reason file before changing ratings.")
    monkeypatch.setattr(web.votes_mod, "_load_vote_reasons", corrupt)
    monkeypatch.setattr(web.votes_mod, "lr_training_status", corrupt)
    monkeypatch.setattr(web, "_profile_exists", lambda: True)
    page = _text_payload(web.saved_items(_request("GET", "/saved")))
    assert "Repair the feedback reason file" in page
    request = _request("POST", "/ranking/train", headers={"x-csrf-token": web._CSRF_TOKEN})
    with pytest.raises(HTTPException) as blocked:
        web.ranking_train(request)
    assert blocked.value.status_code == 409


def test_stop_requests_cooperative_cancellation_and_refuses_after_publish(monkeypatch):
    import threading
    from dailydigest import web
    event = threading.Event()
    run_id = "a" * 12
    monkeypatch.setattr(web, "_RUN_CANCEL", {run_id: event})
    monkeypatch.setattr(web, "_RUN_QUEUES", {run_id: []})
    monkeypatch.setattr(web, "_RUN_PUBLISHED", set())
    request = _request("POST", "/run/cancel", headers={"x-csrf-token": web._CSRF_TOKEN})
    result = _json_payload(web.run_cancel(request, run_id))
    assert result["pending"] and event.is_set()
    web._RUN_PUBLISHED.add(run_id)
    with pytest.raises(HTTPException) as finished:
        web.run_cancel(request, run_id)
    assert finished.value.status_code == 409


def test_cancelled_brew_emits_terminal_event_and_releases_locks(monkeypatch):
    import threading
    from dailydigest import web
    finished = threading.Event()
    run_id = "b" * 12
    monkeypatch.setattr(web, "_RUN_CANCEL", {})
    monkeypatch.setattr(web, "_RUN_QUEUES", {run_id: []})
    def cancelled(**kwargs):
        assert callable(kwargs["should_cancel"])
        raise web.BrewCancelled("Cancelled before publication.")
    monkeypatch.setattr(web, "run_all", cancelled)
    monkeypatch.setattr(web, "release_encoder", lambda: None)
    original_push = web._RUN_QUEUES[run_id]
    class Lock:
        def release(self):
            finished.set()
    monkeypatch.setattr(web, "acquire_compute_lock", lambda _: Lock())
    web._kick_off_run(run_id, "usual")
    assert finished.wait(2)
    # Acquiring the thread lock also waits for the worker's final bookkeeping.
    assert web._COMPUTE_LOCK.acquire(timeout=2)
    web._COMPUTE_LOCK.release()
    assert [row["stage"] for row in original_push] == ["cancelled"]
    assert not web._BREW_JOB["running"]
