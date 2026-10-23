"""Cold-start intent is discoverable without expanding the settings form."""

import html as html_module
import re

import pytest

from tests.test_setup_interface import FormElements, _render_settings


def test_current_goal_is_visible_optional_and_compact(monkeypatch, tmp_path):
    html = _render_settings(monkeypatch, tmp_path)
    fields = FormElements(html).elements
    attrs, disclosures = fields["bio"]

    assert disclosures == ()
    assert attrs["rows"] == "2"
    assert "required" not in attrs
    assert attrs["aria-describedby"] == "bio-hint"
    assert html.index('id="topics"') < html.index('id="bio"')
    assert html.index('id="bio"') < html.index('id="search-tips"')
    assert "Research context below" not in html


def test_optional_example_papers_have_one_nearby_disclosure(monkeypatch, tmp_path):
    html = _render_settings(monkeypatch, tmp_path)
    fields = FormElements(html).elements
    attrs, disclosures = fields["seed_works"]

    assert disclosures == ("paper-examples",)
    assert "open" not in fields["paper-examples"][0]
    assert "required" not in attrs
    assert attrs["aria-describedby"] == "seed-works-hint"
    assert html.index('id="bio"') < html.index('id="paper-examples"')
    assert html.index('id="paper-examples"') < html.index('id="advanced-interests"')
    assert "identifiers and linked papers are not fetched automatically" in html
    assert "under Advanced interests below" not in html


@pytest.mark.parametrize("errors", [[], ["Correct the submitted interests."]])
def test_goal_and_example_values_render_once_and_remain_escaped(monkeypatch, tmp_path, errors):
    values = {
        "bio": 'Study <mechanisms> & compare "methods".\n研究跨领域方法。',
        "seed_works": 'A paper </textarea><script>unsafe()</script>\nAnother & related paper',
    }
    html = _render_settings(monkeypatch, tmp_path, errors=errors, **values)
    fields = FormElements(html).elements

    for name, value in values.items():
        matches = re.findall(rf'<textarea\b[^>]*\bname="{name}"[^>]*>(.*?)</textarea>', html, re.S)
        assert len(matches) == 1
        assert html_module.unescape(matches[0]) == value
    assert "<script>unsafe()</script>" not in html
    assert fields["bio"][1] == ()
    assert ("open" in fields["paper-examples"][0]) == bool(errors)
