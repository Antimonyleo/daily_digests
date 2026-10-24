"""The Settings form keeps interests prominent and advanced values intact."""

import asyncio
import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml

from tests.test_web import _request


class FormElements(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = {}
        self.details = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "details":
            self.details.append(attrs.get("id", ""))
        if attrs.get("id"):
            self.elements[attrs["id"]] = (attrs, tuple(self.details))

    def handle_endtag(self, tag):
        if tag == "details":
            self.details.pop()


def _render_settings(monkeypatch, tmp_path, *, errors=None, **values):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_get_opportunity_profile_path", lambda: tmp_path / "opportunities.yaml")
    form = web._load_existing_form_defaults()
    form.update(llm_backend="extractive", llm_api_key="")
    form.update(values)
    response = web.templates.TemplateResponse(
        _request("GET", "/setup"), "setup.html.j2",
        {"form": form, "errors": errors or [], "csrf_token": "test-csrf"},
    )
    return response.body.decode()


def test_settings_starts_with_interests_and_discloses_optional_fields(monkeypatch, tmp_path):
    html = _render_settings(monkeypatch, tmp_path)
    fields = FormElements(html).elements

    assert fields["topics"][1] == ()
    assert html.index('id="topics"') < html.index('id="name"')
    assert fields["bio"][1] == ()
    assert fields["dw-hidden"][1] == ("profile-options",)
    assert fields["llm_model"][1] == ("summary-options",)
    assert "open" not in fields["profile-options"][0]
    assert "open" not in fields["summary-options"][0]
    assert html.index('id="top_research"') < html.index('id="summary-options"')


def test_settings_pip_uses_the_configured_digest_day(monkeypatch, tmp_path):
    from dailydigest import web

    monkeypatch.setattr(web, "_digest_id", lambda: "2001-02-03")
    monkeypatch.setattr(web, "preview_tea_deck", lambda day: (f"A note for {day.isoformat()}",))
    html = _render_settings(monkeypatch, tmp_path)
    assert 'digestId: "2001-02-03"' in html
    assert 'notes: ["A note for 2001-02-03"]' in html


def test_setup_keeps_first_interest_ahead_of_examples_and_storage_help(monkeypatch, tmp_path):
    html = _render_settings(monkeypatch, tmp_path, topics="")

    assert html.index('id="interest-rows"') < html.index('id="priority-options"')
    assert html.index('id="topics"') < html.index('id="example-discipline"')
    assert html.index('id="topics"') < html.index('id="browser-storage-note"')
    assert '>Load example</button>' in html
    assert 'data-browser-action="open-guide"' in html
    assert 'data-browser-action="open-backup"' in html


def test_returning_settings_does_not_repeat_first_visit_instructions(monkeypatch, tmp_path):
    html = _render_settings(monkeypatch, tmp_path, topics="RNA delivery | 100")

    assert 'DailyDigest settings' in html
    assert 'Update your research interests and reading preferences.' in html
    assert 'No account is needed' not in html
    assert 'Loading an example is available only' not in html
    assert 'id="example-discipline"' not in html


@pytest.mark.parametrize("backend", ["extractive", "api", "anthropic", "claude_cli", "codex_cli"])
def test_collapsed_provider_settings_keep_configured_values(monkeypatch, tmp_path, backend):
    html = _render_settings(
        monkeypatch, tmp_path, llm_backend=backend, llm_api_key="***",
        llm_base_url="https://custom.example.test/v1", llm_model="my-custom-model",
    )
    fields = FormElements(html).elements

    assert fields["llm_base_url"][0]["value"] == "https://custom.example.test/v1"
    assert fields["llm_api_key"][0]["value"] == "***"
    assert fields["llm_model"][0]["value"] == "my-custom-model"
    assert fields["llm_model"][1] == ("summary-options",)
    assert f'value="{backend}"\n                 checked' in html


def test_settings_errors_reveal_disclosures(monkeypatch, tmp_path):
    html = _render_settings(monkeypatch, tmp_path, errors=["API model is required."])
    fields = FormElements(html).elements

    assert "API model is required." in html
    assert 'role="alert"' in html
    assert "open" in fields["profile-options"][0]
    assert "open" in fields["summary-options"][0]


def test_plain_interests_default_to_equal_weights_and_explicit_weights_survive():
    from dailydigest.web import _parse_ranked_topics

    topics, errors = _parse_ranked_topics("RNA delivery\nProtein design | 17.5\nHydrogels")

    assert errors == []
    assert topics == [("RNA delivery", 1.0), ("Protein design", 17.5), ("Hydrogels", 1.0)]
    assert _parse_ranked_topics("RNA | ")[1]
    assert _parse_ranked_topics("RNA | nope")[1]


@pytest.mark.parametrize("action,destination", [
    ("brew", "/run?reading_mode=usual&autostart=1"),
    ("save", "/"),
    ("", "/run"),
    ("https://untrusted.example/", "/run"),
])
def test_setup_action_has_one_explicit_destination_and_never_starts_during_save(
    monkeypatch, tmp_path, action, destination,
):
    from dailydigest import web

    profile = tmp_path / "profile.yaml"
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(web, "_kick_off_run", lambda *args: pytest.fail("Saving must not start a brew"))
    response = asyncio.run(web.setup_post(_request("POST", "/setup", form={
        "_csrf_token": web._CSRF_TOKEN, "topics": "DNA origami assembly",
        "llm_backend": "extractive", "after_save": action,
    })))
    assert response.status_code == 303
    assert response.headers["location"] == destination
    assert response.headers["cache-control"] == "no-store"
    assert yaml.safe_load(profile.read_text())["keywords"] == ["DNA origami assembly"]


@pytest.mark.parametrize("weights", ["RNA delivery | 40\nProtein design | 50", "RNA delivery | 60\nProtein design | 50", "RNA delivery | 49.5\nProtein design | 50.5"])
def test_percentage_editor_rejects_invalid_totals_without_writing(monkeypatch, tmp_path, weights):
    from dailydigest import web

    profile = tmp_path / "profile.yaml"
    original = "keywords: [Existing interest]\n"
    profile.write_text(original)
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    response = asyncio.run(web.setup_post(_request("POST", "/setup", form={
        "_csrf_token": web._CSRF_TOKEN, "topics": weights,
        "topic_weight_mode": "percentage", "llm_backend": "extractive",
    })))
    assert response.status_code == 400
    assert "whole-number percentages totaling 100%" in response.body.decode()
    assert profile.read_text() == original
    assert not (tmp_path / ".env").exists()


def test_percentage_editor_saves_exact_displayed_priorities(monkeypatch, tmp_path):
    from dailydigest import web

    profile = tmp_path / "profile.yaml"
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    response = asyncio.run(web.setup_post(_request("POST", "/setup", form={
        "_csrf_token": web._CSRF_TOKEN, "topics": "RNA delivery | 75\nProtein design | 25",
        "topic_weight_mode": "percentage", "llm_backend": "extractive",
    })))
    assert response.status_code == 303
    saved = yaml.safe_load(profile.read_text())["canonical_facets"]
    assert {name: spec["priority"] for name, spec in saved.items()} == {"RNA delivery": 75, "Protein design": 25}


def test_saving_plain_interests_preserves_existing_explicit_weight_and_context(monkeypatch, tmp_path):
    from dailydigest import web

    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({
        "bio": "Existing research context.", "keywords": ["RNA delivery", "Hydrogels"],
        "canonical_facets": {"RNA delivery": {"priority": 17.5, "anchors": ["RNA transport"]}},
    }))
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    response = asyncio.run(web.setup_post(_request("POST", "/setup", form={
        "_csrf_token": web._CSRF_TOKEN,
        "topics": "RNA delivery | 17.5\nHydrogels",
        "bio": "Existing research context.",
        "llm_backend": "extractive",
    })))

    assert response.status_code == 303
    saved = yaml.safe_load(profile.read_text())
    assert saved["canonical_facets"]["RNA delivery"] == {"priority": 17.5, "anchors": ["RNA transport"]}
    assert saved["canonical_facets"]["Hydrogels"]["priority"] == 1
    assert saved["bio"] == "Existing research context."


def test_compact_editor_unrelated_save_preserves_relative_precision_and_topic_metadata(monkeypatch, tmp_path):
    from dailydigest import web

    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({
        "bio": "Original goal", "keywords": ["RNA delivery", "Hydrogels"],
        "canonical_facets": {
            "RNA delivery": {"priority": 17.5, "aliases": ["RNA transport"],
                             "description": "Compare methods", "anchors": ["Original anchor"]},
            "Hydrogels": {"priority": 1},
        },
    }))
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    response = asyncio.run(web.setup_post(_request("POST", "/setup", form={
        "_csrf_token": web._CSRF_TOKEN, "topics": "RNA delivery | 17.5\nHydrogels | 1",
        "topic_weight_mode": "relative", "llm_backend": "extractive", "bio": "Updated goal",
        "topic_details": json.dumps({"RNA delivery": {"aliases": ["RNA transport"], "description": "Compare methods"}}),
    })))

    assert response.status_code == 303
    saved = yaml.safe_load(profile.read_text())
    assert saved["canonical_facets"]["RNA delivery"] == {
        "priority": 17.5, "aliases": ["RNA transport"],
        "description": "Compare methods", "anchors": ["Original anchor"],
    }
    assert saved["canonical_facets"]["Hydrogels"]["priority"] == 1
    assert saved["bio"] == "Updated goal"


def test_browser_validation_reveals_nested_closed_details():
    source = (Path(__file__).resolve().parents[1] / "templates/setup.html.j2").read_text()
    script = source.split("  // Reveal invalid fields before native validation focuses them.")[1].split("  // Backend reveal logic.")[0]
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the browser-script regression check")
    result = subprocess.run([node, "-e", '''
        const assert = require('node:assert/strict');
        let invalid;
        const form = {addEventListener: (name, fn, capture) => {
            if (name === 'invalid') { invalid = fn; assert.equal(capture, true); }
        }};
        const document = {querySelector: () => form, getElementById: () => ({})};
        const priorities = {open: false};
        let singlePriorityReveals = 0;
        const editor = {classList: {remove(name) {
            assert.equal(name, 'single-topic'); singlePriorityReveals++;
        }}};
        let priorityReveals = 0;
        const showPriorities = () => { priorityReveals++; };
    ''' + script + '''
        const outer = {tagName: 'DETAILS', open: false, parentElement: form};
        const inner = {tagName: 'DETAILS', open: false, parentElement: outer};
        const hidden = {tagName: 'DIV', hidden: true, parentElement: inner};
        invalid({target: {matches: () => false, parentElement: hidden}});
        assert.equal(inner.open, true);
        assert.equal(outer.open, true);
        assert.equal(hidden.hidden, false);
        invalid({target: {matches: selector => selector === '[data-interest-weight]', parentElement: form}});
        assert.equal(priorities.open, true);
        assert.equal(priorityReveals, 1, 'Priority CSS must update synchronously before native focus');
        assert.equal(singlePriorityReveals, 1, 'An invalid single priority must become focusable');
    '''], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, json.dumps(result.stderr)


@pytest.mark.parametrize("field", ["authors_of_interest", "seed_works", "context_keywords"])
def test_advanced_interest_lists_round_trip_and_preserve_when_omitted(monkeypatch, tmp_path, field):
    from dailydigest import web

    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({"bio": "", "keywords": ["RNA"], field: ["Existing value"]}))
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    assert web._load_existing_form_defaults()[field] == "Existing value"
    base = {"_csrf_token": web._CSRF_TOKEN, "topics": "RNA | 17.5", "llm_backend": "extractive"}
    assert asyncio.run(web.setup_post(_request("POST", "/setup", form=base))).status_code == 303
    assert yaml.safe_load(profile.read_text())[field] == ["Existing value"]
    submitted = {**base, field: "First value\n Second value \nFirst value"}
    assert asyncio.run(web.setup_post(_request("POST", "/setup", form=submitted))).status_code == 303
    saved = yaml.safe_load(profile.read_text())
    assert saved[field] == ["First value", "Second value"]
    assert saved["canonical_facets"]["RNA"]["priority"] == 17.5
    assert asyncio.run(web.setup_post(_request("POST", "/setup", form={**base, field: ""}))).status_code == 303
    assert yaml.safe_load(profile.read_text())[field] == []


@pytest.mark.parametrize("value", ["x" * 301, "\n".join(f"Value {n}" for n in range(21)), "bad\x00value"])
def test_advanced_interest_validation_preserves_form_and_saved_profile(monkeypatch, tmp_path, value):
    from dailydigest import web

    profile = tmp_path / "profile.yaml"
    original = yaml.safe_dump({"bio": "", "keywords": ["RNA"], "seed_works": ["Kept"]})
    profile.write_text(original)
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    response = asyncio.run(web.setup_post(_request("POST", "/setup", form={
        "_csrf_token": web._CSRF_TOKEN, "topics": "RNA", "llm_backend": "extractive",
        "authors_of_interest": value, "seed_works": "Unsaved title", "context_keywords": "Unsaved context",
    })))
    assert response.status_code == 400
    assert profile.read_text() == original
    assert "Unsaved title" in response.body.decode()
    assert "Unsaved context" in response.body.decode()


def test_advanced_interests_and_coverage_limits_are_discoverable(monkeypatch, tmp_path):
    html = _render_settings(monkeypatch, tmp_path)
    fields = FormElements(html).elements
    for field in ("authors_of_interest", "context_keywords"):
        assert fields[field][1] == ("advanced-interests",)
    assert fields["seed_works"][1] == ("paper-examples",)
    assert "open" not in fields["paper-examples"][0]
    assert "open" not in fields["advanced-interests"][0]
    assert "Grants.gov" in html and "EMBL" in html
    assert "not exhaustive" in html
    assert "browser companion" in html.lower()
    assert 'id="settings-save-status"' in html


def test_api_connection_check_requires_explicit_csrf_post_and_does_not_save(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from dailydigest import provider_check, web

    calls = []
    monkeypatch.setattr(provider_check, "check_api_connection", lambda **kwargs: calls.append(kwargs) or {"ok": True, "message": "Connected."})
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    values = {"llm_backend": "api", "llm_base_url": "https://example.test/v1", "llm_model": "example", "llm_api_key": "test-secret"}
    with TestClient(web.app) as client:
        assert client.request("GET", "/setup/test-connection").status_code == 405
        assert client.request("POST", "/setup/test-connection", data=values).status_code == 403
        assert calls == []
        response = client.request("POST", "/setup/test-connection", data=values, headers={"X-CSRF-Token": web._CSRF_TOKEN})
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert calls == [{"backend": "api", "base_url": "https://example.test/v1", "model": "example", "api_key": "test-secret"}]
    assert "test-secret" not in response.text
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / "profile.yaml").exists()


def test_unchanged_advanced_values_keep_legacy_entries(monkeypatch, tmp_path):
    from dailydigest import web

    profile = tmp_path / "profile.yaml"
    entries = ["Legacy abstract " * 40, "Repeated", "Repeated"]
    profile.write_text(yaml.safe_dump({"bio": "", "keywords": ["RNA"], "seed_works": entries}))
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    defaults = web._load_existing_form_defaults()
    response = asyncio.run(web.setup_post(_request("POST", "/setup", form={
        "_csrf_token": web._CSRF_TOKEN, "topics": "RNA", "llm_backend": "extractive",
        "seed_works": defaults["seed_works"],
    })))
    assert response.status_code == 303
    assert yaml.safe_load(profile.read_text())["seed_works"] == entries


@pytest.mark.parametrize("base_url,expected", [("https://saved.example/v1", "saved-secret"), ("https://other.example/v1", "")])
def test_api_connection_check_keeps_masked_saved_key_at_its_provider(monkeypatch, base_url, expected):
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from dailydigest import provider_check, web

    calls = []
    monkeypatch.setattr(web, "SETTINGS", SimpleNamespace(llm_backend="api", llm_base_url="https://saved.example/v1", llm_api_key="saved-secret"))
    monkeypatch.setattr(provider_check, "check_api_connection", lambda **kwargs: calls.append(kwargs) or {"ok": False, "message": "Test result."})
    with TestClient(web.app) as client:
        response = client.request("POST", "/setup/test-connection", data={
            "llm_backend": "api", "llm_base_url": base_url, "llm_api_key": "***", "llm_model": "example",
        }, headers={"X-CSRF-Token": web._CSRF_TOKEN})
    assert response.status_code == 200
    assert calls[0]["api_key"] == expected
    assert "saved-secret" not in response.text
