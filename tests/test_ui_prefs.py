"""Display choices are kept beside the reader's data, so backups carry them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from tests.test_web import _request, _text_payload


def _web(monkeypatch, tmp_path):
    from dailydigest import config, web

    config.reload_settings()
    monkeypatch.setattr(web, "SETTINGS", config.SETTINGS)
    profile = tmp_path / "reader.yaml"
    profile.write_text("name: Ada\nkeywords: [DNA origami]\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    return web


def test_display_choices_are_saved_to_the_data_folder_and_render_on_a_new_device(monkeypatch, tmp_path):
    web = _web(monkeypatch, tmp_path)
    headers = {"X-CSRF-Token": web._CSRF_TOKEN}
    prefs_file = Path(web.get_settings().db_path).parent / "ui_prefs.json"
    assert not prefs_file.exists()

    saved = web.ui_prefs_save(_request("POST", "/ui-prefs", headers=headers), theme="dark", display="compact")
    assert json.loads(saved.body) == {"display": "compact", "theme": "dark"}
    web.ui_prefs_save(_request("POST", "/ui-prefs", headers=headers), pip="hidden")
    assert json.loads(prefs_file.read_text()) == {"display": "compact", "pip": "hidden", "theme": "dark"}

    # A browser with nothing in localStorage starts from the saved file.
    page = _text_payload(web.saved_items(_request("GET", "/saved")))
    assert 'const filed = {"display": "compact", "pip": "hidden", "theme": "dark"};' in page
    assert "savePrefs({theme: next});" in page


def test_display_choices_need_csrf_and_known_values(monkeypatch, tmp_path):
    web = _web(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as missing:
        web.ui_prefs_save(_request("POST", "/ui-prefs"), theme="dark")
    assert missing.value.status_code == 403
    with pytest.raises(HTTPException) as unknown:
        web.ui_prefs_save(_request("POST", "/ui-prefs", headers={"X-CSRF-Token": web._CSRF_TOKEN}),
                          theme="<script>")
    assert unknown.value.status_code == 422
    # A hand-edited or damaged file never reaches the page as an unknown value.
    prefs_file = Path(web.get_settings().db_path).parent / "ui_prefs.json"
    prefs_file.write_text('{"theme": "neon", "display": "compact", "extra": 1}')
    assert web._load_ui_prefs() == {"display": "compact"}
    prefs_file.write_text('{"theme": ["dark"], "pip": {"x": 1}, "display": "compact"}')
    assert web._load_ui_prefs() == {"display": "compact"}
    prefs_file.write_text("not json")
    assert web._load_ui_prefs() == {}
