"""Reader fixes preserve explicit Brew intent and all retained editions."""

import asyncio
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import HTTPException

from tests.test_web import _request, _stub_index_dependencies


@pytest.fixture
def reader(monkeypatch, tmp_path):
    from dailydigest import web

    profile = tmp_path / "profile.yaml"
    profile.write_text("name: Ada\nkeywords: [DNA origami]\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "BROWSER_MODE", False)
    monkeypatch.setattr(web, "_recent_digest_ids", lambda *_args, **_kw: [])
    monkeypatch.setattr(web, "_kick_off_run", lambda *_args: pytest.fail("Rendering must not brew"))
    return web


def prepare(web, mode="usual"):
    return asyncio.run(web.run_prepare(_request("POST", "/run", form={
        "_csrf_token": web._CSRF_TOKEN, "reading_mode": mode,
    })))


def test_a_plain_or_cross_site_autostart_link_never_starts(reader):
    response = reader.run_get(_request("GET", "/run", headers={
        "sec-fetch-site": "cross-site",
    }), autostart=True)
    assert "const AUTOSTART = false;" in response.body.decode()


@pytest.mark.parametrize("mode", ["usual", "minimal", "full"])
def test_explicit_brew_authorization_is_mode_bound_and_single_use(reader, mode):
    response = prepare(reader, mode)
    assert response.status_code == 303
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert query["reading_mode"] == [mode]
    token = query["start"][0]
    wrong_mode = "minimal" if mode != "minimal" else "full"
    wrong = reader.run_get(_request("GET", "/run"), reading_mode=wrong_mode,
                           autostart=True, start=token)
    assert "const AUTOSTART = false;" in wrong.body.decode()
    page = reader.run_get(_request("GET", "/run"), reading_mode=mode,
                          autostart=True, start=token)
    assert "const AUTOSTART = true;" in page.body.decode()
    replay = reader.run_get(_request("GET", "/run"), reading_mode=mode,
                            autostart=True, start=token)
    assert "const AUTOSTART = false;" in replay.body.decode()


def test_explicit_brew_requires_csrf_and_expiring_authorization(reader, monkeypatch):
    with pytest.raises(HTTPException) as denied:
        asyncio.run(reader.run_prepare(_request("POST", "/run", form={"reading_mode": "usual"})))
    assert denied.value.status_code == 403
    response = prepare(reader)
    token = parse_qs(urlsplit(response.headers["location"]).query)["start"][0]
    now = reader.monotonic()
    monkeypatch.setattr(reader, "monotonic", lambda: now + 301)
    expired = reader.run_get(_request("GET", "/run"), autostart=True, start=token)
    assert "const AUTOSTART = false;" in expired.body.decode()


def test_native_run_has_no_browser_notice_or_nonexistent_saved_digest(reader):
    page = reader.run_get(_request("GET", "/run")).body.decode()
    assert "20 minutes" not in page
    assert "one core" not in page
    assert "Read latest saved digest" not in page
    assert "const HAS_SAVED_DIGEST = false;" in page


def test_browser_run_retains_device_guidance_and_real_saved_link(reader, monkeypatch):
    monkeypatch.setattr(reader, "BROWSER_MODE", True)
    monkeypatch.setattr(reader, "_recent_digest_ids", lambda *_a, **_kw: ["2026-10-24"])
    page = reader.run_get(_request("GET", "/run")).body.decode()
    assert "20 minutes" in page
    assert "const HAS_SAVED_DIGEST = true;" in page
    assert 'href="/?digest_id=2026-10-24"' in page


def test_native_pages_block_foreign_framing(reader):
    from starlette.responses import Response

    async def next_response(_request):
        return Response("ok")

    response = asyncio.run(reader._loopback_host_only(_request("GET", "/healthz"), next_response))
    assert response.headers["x-frame-options"] == "SAMEORIGIN"
    assert response.headers["content-security-policy"] == "frame-ancestors 'self'"
    assert response.headers["referrer-policy"] == "same-origin"


def test_retained_edition_outside_picker_still_opens(reader, monkeypatch):
    today, archived = "2026-10-25", "2026-09-01"
    monkeypatch.setattr(reader, "_digest_id", lambda: today)
    monkeypatch.setattr(reader, "_digest_exists", lambda day: day in {today, archived})
    monkeypatch.setattr(reader, "_recent_digest_ids", lambda *_a, **_k: [today])
    _stub_index_dependencies(reader, monkeypatch)
    loaded = []
    monkeypatch.setattr(reader, "_load_today", lambda day, **_kw: (loaded.append(day) or [], {}))
    page = reader.index(_request("GET", "/"), digest_id=archived).body.decode()
    assert loaded == [archived]
    assert archived in page
