"""Reader feedback: chronological editions and optional, grounded explanations."""

import json

import pytest

from dailydigest import summarize as sm
from tests.test_summarize import _row


@pytest.fixture(autouse=True)
def reader_context(monkeypatch):
    monkeypatch.setattr(sm, "_READER_FACETS", {})
    monkeypatch.setattr(sm, "_READER_KEYWORDS", [])


@pytest.mark.parametrize("section", ["research", "industry", "regulatory"])
def test_no_profile_does_not_force_a_generic_why_read(section):
    item = _row("A measured result", "Measurements compared the two sample conditions.")
    item.section = section
    summary = sm._extractive(item)
    assert "Why read:" not in summary
    # No study-specific limitation in the text, so no generic Caveat is invented.
    assert "Key finding:" in summary and "Caveat:" not in summary


def test_neutral_contribution_has_no_why_field_even_with_a_topic(monkeypatch):
    from dailydigest.rank import contribution

    monkeypatch.setattr(sm, "_READER_FACETS", {"RNA delivery": {"aliases": ["RNA delivery"]}})
    monkeypatch.setattr(contribution, "analyze_contribution", lambda *_a: {
        "priority": 0, "facets": ["RNA delivery"], "kind": "neutral", "evidence": [],
    })
    item = _row("RNA delivery survey", "RNA delivery measurements were compared.")
    assert sm._why_read(item) == ""
    assert "Why read:" not in sm._extractive(item)


def test_grounded_contribution_reason_is_preserved_and_capitalized(monkeypatch):
    from dailydigest.rank import contribution

    monkeypatch.setattr(sm, "_READER_FACETS", {"DNA origami": {"aliases": ["DNA origami"]}})
    monkeypatch.setattr(contribution, "analyze_contribution", lambda *_a: {
        "priority": 1, "facets": ["DNA origami"], "kind": "engineering",
        "evidence": ["The study introduces a programmable hinge."],
    })
    item = _row("Programmable hinge", "The study introduces a programmable hinge.")
    reason = sm._why_read(item)
    assert reason[0].isupper()
    assert "DNA origami" in reason
    assert "Why read:" not in sm._extractive(item), "Do not repeat the same evidence as the finding"


def test_why_can_add_another_supported_contribution_detail(monkeypatch):
    from dailydigest.rank import contribution

    first = "The study introduces a programmable hinge."
    second = "Measurements improved opening precision in 120 independent trials."
    additional = "A reversible latch connects the hinge to a DNA origami assembly."
    monkeypatch.setattr(sm, "_READER_FACETS", {"DNA origami": {"aliases": ["DNA origami"]}})
    monkeypatch.setattr(contribution, "analyze_contribution", lambda *_a: {
        "priority": 1, "facets": ["DNA origami"], "kind": "engineering",
        "evidence": [first, additional],
    })
    summary = sm._extractive(_row("Programmable hinges", f"{first} {second} {additional}"))
    # The measured result stands alone; Why read adds the interest-specific detail.
    assert f"Key finding: {second}" in summary
    assert f"Why read: For DNA origami: {additional}" in summary


def test_missing_abstract_uses_an_explicit_caveat_without_a_generic_why():
    summary = sm._extractive(_row("A title is not evidence", ""))
    assert "No abstract is available" in summary
    assert "Why read:" not in summary
    assert "cannot be assessed" in summary


def test_provider_can_omit_why_without_omitting_finding_or_caveat():
    summary = "Key finding: The measurements differed.\nCaveat: Only one sample was tested."
    assert sm._parse_id_summary_map(json.dumps({"1": summary})) == {1: summary}
    assert sm._parse_id_summary_map(json.dumps({"1": "Key finding: A result."})) == {}


def test_new_explanation_contract_invalidates_old_cached_summaries(monkeypatch):
    item = _row("Title", "Measured result.")
    current = sm.summary_signature(item, None)
    monkeypatch.setattr(sm, "_SUMMARY_VERSION", "summary-v8")
    assert current != sm.summary_signature(item, None)
