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
    assert fields["bio"][1] == ("profile-options",)
    assert fields["dw-hidden"][1] == ("profile-options",)
    assert fields["llm_model"][1] == ("summary-options",)
    assert "open" not in fields["profile-options"][0]
    assert "open" not in fields["summary-options"][0]
    assert html.index('id="top_research"') < html.index('id="summary-options"')


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
    ''' + script + '''
        const outer = {tagName: 'DETAILS', open: false, parentElement: form};
        const inner = {tagName: 'DETAILS', open: false, parentElement: outer};
        const hidden = {tagName: 'DIV', hidden: true, parentElement: inner};
        invalid({target: {parentElement: hidden}});
        assert.equal(inner.open, true);
        assert.equal(outer.open, true);
        assert.equal(hidden.hidden, false);
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
    for field in ("authors_of_interest", "seed_works", "context_keywords"):
        assert fields[field][1] == ("advanced-interests",)
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


def test_local_interest_wording_variants_are_bounded_and_never_guess_synonyms():
    source = (Path(__file__).resolve().parents[1] / "templates/setup.html.j2").read_text()
    assert "  function wordingVariants(" in source
    script = source.split("  function wordingVariants(")[1].split("  function percentages(")[0]
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for local wording variants")
    result = subprocess.run([node, "-e", "const assert = require('node:assert/strict');\n"
        + "function wordingVariants(" + script + """
        assert.deepEqual(wordingVariants('quantum error correction (QEC)'), ['quantum error correction', 'QEC']);
        assert.deepEqual(wordingVariants('QEC (quantum error correction)'), ['QEC', 'quantum error correction']);
        assert.deepEqual(wordingVariants('metal-organic frameworks'), ['metal organic frameworks']);
        assert.deepEqual(wordingVariants('single-cell analysis'), ['single cell analysis']);
        assert.deepEqual(wordingVariants('C-H activation'), []);
        assert.deepEqual(wordingVariants('KRAS-G12D inhibitor'), []);
        assert.deepEqual(wordingVariants('Drug development'), []);
        assert.deepEqual(wordingVariants('Induce proximity'), []);
        assert.deepEqual(wordingVariants('protein engineering (computational)'), []);
        assert.deepEqual(wordingVariants('word '.repeat(50)), []);
        """], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_loaded_interest_exposes_variants_without_editing_and_requires_explicit_add():
    source = (Path(__file__).resolve().parents[1] / "templates/setup.html.j2").read_text()
    helper = "function wordingVariants(" + source.split("  function wordingVariants(", 1)[1].split("  function percentages(", 1)[0]
    controls = "const aliasInput =" + source.split("    const aliasInput =", 1)[1].split("    const nameInput =", 1)[0]
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for local wording controls")
    script = """
        const assert = require('node:assert/strict');
        class Element {
            constructor() { this.children = []; this.events = {}; this.textContent = ''; }
            setAttribute() {}
            addEventListener(type, callback) { this.events[type] = callback; }
            append(...nodes) { this.children.push(...nodes); }
            replaceChildren(...nodes) { this.children = nodes; }
        }
        const aliases = {value: ''}, topic = {value: 'quantum error correction (QEC)'};
        const detailsPanel = new Element();
        const row = {querySelector(selector) {
            return selector === '[data-interest-aliases]' ? aliases
                : selector === '[data-interest-name]' ? topic : detailsPanel;
        }};
        const document = {createElement: () => new Element()};
        const fetch = () => assert.fail('Local suggestions must not contact a service');
        let saves = 0; const serialize = () => {saves++;};
    """ + helper + controls + """
        const actions = detailsPanel.children[0];
        assert.equal(actions.children.length, 1, 'Loaded interests must expose the suggestion button immediately');
        assert.equal(actions.children[0].textContent, 'Suggest wording variants');
        actions.children[0].events.click();
        assert.equal(aliases.value, ''); assert.equal(saves, 0);
        actions.children.find(button => button.textContent === 'Add “QEC”').events.click();
        assert.equal(aliases.value, 'QEC'); assert.equal(saves, 1);
        assert.equal(topic.value, 'quantum error correction (QEC)');
        aliases.value = 'one; two; three';
        actions.children[0].events.click();
        actions.children.find(button => button.textContent === 'Add “QEC”').events.click();
        assert.equal(aliases.value, 'one; two; three'); assert.equal(saves, 1);
        assert.match(detailsPanel.children[1].textContent, /up to 3/);
    """
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
