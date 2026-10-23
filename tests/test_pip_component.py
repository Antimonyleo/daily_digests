"""Shared Pip cache metadata remains bounded, inert and independent of article HTML."""

import json
import re

from tests.test_web import _request, _text_payload


def test_reading_personalization_metadata_is_bounded_and_script_safe(monkeypatch, tmp_path):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Fictional reader\nbio: Reader\nkeywords: []\ndownweight: []\n")
    hostile_note = '</script><img src=x onerror="alert(1)">'
    notes = [hostile_note, "x" * 3000] + [f"Fictional note {index}" for index in range(40)]
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2031-02-03")
    monkeypatch.setattr(web, "_load_today", lambda _digest_id: ([], {}))
    monkeypatch.setattr(web, "_digest_exists", lambda _digest_id: False)
    monkeypatch.setattr(web, "daily_tea_deck", lambda _date: notes)

    html = _text_payload(web.index(_request("GET", "/")))
    raw = re.search(
        r'<script type="application/json" id="reading-personalization">(.*?)</script>',
        html, re.S,
    ).group(1)
    metadata = json.loads(raw)
    assert metadata["name"] == "Fictional reader"
    assert len(metadata["notes"]) == 32
    assert metadata["notes"][0] == hostile_note
    assert len(metadata["notes"][1]) == 2000
    assert "<" not in raw
    assert 'id="tea-note-text" aria-live="polite">&lt;/script&gt;' in html
