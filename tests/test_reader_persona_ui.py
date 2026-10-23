"""Readers can refine purpose without changing existing interests or priorities."""

import json
import re
from pathlib import Path

import pytest

from tests.test_setup_interface import FormElements, _render_settings


def test_returning_reader_priorities_are_optional_and_values_survive(monkeypatch, tmp_path):
    details = {"Galaxy formation": {"aliases": ["galaxy evolution"], "description": "Reviews explaining unfamiliar methods"}}
    html = _render_settings(
        monkeypatch, tmp_path, topics="Galaxy formation | 75\nExoplanet atmospheres | 25",
        topic_details=json.dumps(details), bio="I teach physics and am new to planetary science.",
    )
    fields = FormElements(html).elements
    assert "open" not in fields["priority-options"][0]
    assert "Galaxy formation | 75\nExoplanet atmospheres | 25" in html
    assert "Reviews explaining unfamiliar methods" in html
    assert "I teach physics and am new to planetary science." in html


def test_invalid_priorities_are_revealed_without_replacing_submitted_values(monkeypatch, tmp_path):
    html = _render_settings(
        monkeypatch, tmp_path, errors=["Use whole-number percentages totaling 100%."],
        topics="Galaxy formation | 75\nExoplanet atmospheres | 50", topic_weight_mode="percentage",
    )
    assert "open" in FormElements(html).elements["priority-options"][0]
    assert "Galaxy formation | 75\nExoplanet atmospheres | 50" in html


def test_existing_context_is_discoverable_without_adding_role_or_reading_level_fields(monkeypatch, tmp_path):
    html = _render_settings(monkeypatch, tmp_path)
    profile = re.search(r'<details[^>]+id="profile-options"[^>]*>\s*<summary>(.*?)</summary>', html, re.S).group(1)
    assert "Name &amp; topics to see less often" in profile
    assert FormElements(html).elements["bio"][1] == ()
    assert "Current research or learning goal (optional)" in html
    assert html.index('for="bio"') < html.index('for="name"')
    assert "need clear reviews" in html
    assert "looking for methods I can use" in html
    assert 'name="reading_level"' not in html
    assert 'name="research_role"' not in html
    assert "new engineering methods, transferable design rules" not in html


def test_topic_extras_are_optional_and_automatic_pubmed_checks_are_disclosed(monkeypatch, tmp_path):
    html = _render_settings(monkeypatch, tmp_path)
    row_markup = html.split("row.innerHTML = ", 1)[1].split(";\n", 1)[0]
    assert "Other names (optional)" in row_markup
    assert "Matching &amp; terminology" not in row_markup
    assert "What would make a paper useful?" not in row_markup
    assert "data-interest-aliases" in row_markup
    assert "data-interest-description" in row_markup
    assert "targeted protein degradation" not in row_markup
    assert "PROTAC" not in row_markup
    assert "Suggest wording variants" not in html
    assert "After you finish typing, only the topic name is checked with PubMed" in html
    assert "your wording changes only if you accept one" in html
    assert "PubMed may not recognize specialist terms from every field" in html


def feedback_html(*reasons):
    from dailydigest import web

    template = web.templates.env.get_template("_feedback_controls.html.j2")
    return template.module.feedback_controls({
        "label": "fictional paper", "current_grade": 10, "current_reasons": list(reasons),
    })


def test_negative_feedback_starts_with_three_reasons_and_discloses_the_rest():
    html = feedback_html()
    primary, secondary = html.split('<details class="more-reasons"', 1)
    assert re.findall('data-reason="([^"]+)"', primary) == [
        "off_topic", "too_contextual", "too_technical",
    ]
    assert ">Background only</button>" in primary
    assert ">Too technical</button>" in primary
    assert '<summary class="reason-prompt">More reasons</summary>' in secondary
    assert len(re.findall('data-reason="([^"]+)"', secondary)) == 7
    assert not secondary.startswith(" open")


@pytest.mark.parametrize("reason", ["already_known", "access_friction", "low_quality"])
def test_existing_supplementary_reason_stays_visible_and_reversible(reason):
    html = feedback_html(reason)
    assert '<details class="more-reasons" open>' in html
    assert re.search(rf'data-reason="{reason}"[^>]*aria-pressed="true"', html)
    assert 'class="btn clear-rating" type="button">Clear rating' in html


def test_guide_describes_the_available_known_paper_feedback_path():
    source = (Path(__file__).resolve().parents[1] / "browser/index.html").read_text()
    guide = source.split('id="quick-guide"', 1)[1].split("</dialog>", 1)[0]
    assert "Less relevant, then More reasons → Already known" in guide
    assert "Funding and event cards offer Mark known" in guide
    assert "mark a paper as known" not in guide
    assert 'data-reason="already_known"' in feedback_html("already_known")
