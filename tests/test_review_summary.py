"""Regression cases verified from the October 2026 summary review."""
import json

import pytest

from dailydigest import summarize
from dailydigest.store import ItemRow


def row(abstract):
    return ItemRow(id=1, source="Nature", section="research", title="An exciting treatment",
                   abstract=abstract, external_id="review-summary", url="https://example.test/paper")


def key_finding(summary):
    return summary.splitlines()[0].removeprefix("Key finding: ")


def test_structured_summary_strips_labels_and_prioritizes_results_and_conclusion():
    item = row("BACKGROUND: An important question remains. "
               "METHODS: The trial screened 5000 patients across 300 sites for 30 months. "
               "RESULTS: Treatment reduced symptoms. "
               "CONCLUSIONS: The effect was limited to adults.")
    finding = key_finding(summarize._extractive(item))
    assert finding == "Treatment reduced symptoms. The effect was limited to adults."


def test_figure_abbreviation_does_not_fragment_an_extracted_sentence():
    finding = key_finding(summarize._extractive(row(
        "The trial reports 300 patients across 20 sites. "
        "The screening method reports 20 samples in Fig. 2 and identifies 3 active compounds."
    )))
    assert "The screening method reports 20 samples in Fig. 2 and identifies 3 active compounds." in finding


def test_profile_contribution_excerpt_also_strips_structured_labels(monkeypatch):
    monkeypatch.setattr(summarize, "_READER_FACETS", {
        "Protein engineering": {"aliases": ["protein engineering"]},
    })
    item = row("BACKGROUND: Protein engineering has many applications. "
               "METHODS: We engineer proteins with enhanced activity and experimentally "
               "validate their performance across several conditions.")
    finding = key_finding(summarize._extractive(item))
    assert "METHODS:" not in finding and "BACKGROUND:" not in finding
    assert "We engineer proteins" in finding


def test_long_extraction_has_a_bounded_honest_excerpt():
    original = "The study reports " + "an experimentally tested formulation " * 40 + "with improved activity."
    finding = key_finding(summarize._extractive(row(original)))
    assert len(finding) <= 400
    assert finding.endswith("…")
    assert original.startswith(finding[:-1])


@pytest.mark.parametrize("abstract", ["", "   ", "METHODS: RESULTS: "])
def test_missing_abstract_has_three_explicit_fields_without_title_as_a_finding(abstract):
    summary = summarize._extractive(row(abstract))
    assert summary.startswith("Key finding: No abstract is available.")
    assert "Why read:" in summary and "Caveat:" in summary
    assert "An exciting treatment" not in summary
    assert "title alone" in summary


def test_prompt_exposes_missing_abstract_and_prohibits_invented_claims():
    prompt, payload = summarize._build_prompt([row("")])
    item = json.loads(payload.split("\n\n", 1)[1])[0]
    assert item["abstract_available"] is False
    assert "do not infer findings or numbers from the title" in prompt
    assert "No abstract is available" in prompt
    _, available = summarize._build_prompt([row("A reported result.")])
    assert json.loads(available.split("\n\n", 1)[1])[0]["abstract_available"] is True


@pytest.mark.parametrize("value", [
    {"Key finding": "a result", "Why read": "relevant", "Caveat": "limited"},
    ["a finding"], 123, True, None, "unstructured text",
    "Key finding: A result.\nWhy read: Relevant.",
    "Key finding: A result.\nWhy read: \nCaveat: Limited.",
    "Key finding: A result.\nWhy read: Relevant.\nCaveat: Limited.\nUnexpected: More.",
])
def test_parser_rejects_nonstring_or_incomplete_three_field_contract(value):
    assert summarize._parse_id_summary_map(json.dumps({"1": value})) == {}


def test_parser_accepts_valid_siblings_when_another_item_is_malformed():
    valid = "Key finding: A reported result.\nWhy read: It concerns delivery.\nCaveat: A pilot study."
    assert summarize._parse_id_summary_map(json.dumps({"1": valid, "2": {"finding": "bad"}})) == {1: valid}


def test_malformed_provider_item_uses_local_summary_and_fallback_provenance(monkeypatch):
    def provider(batch, **kwargs):
        return summarize._parse_id_summary_map('{"1": {"finding": "not a summary string"}}')

    monkeypatch.setattr(summarize, "_call_llm", provider)
    summaries, provenance = summarize._summarize_via_provider_with_provenance(
        [row("The trial reports improved outcomes in 20 adults.")], "api")
    assert provenance == {1: "extractive_fallback"}
    assert summaries[1].startswith("Key finding: The trial reports")


def test_caveat_preserves_explicit_abstract_limitation():
    limitation = "The study is limited to a small sample and lacks independent replication."
    item = row("We developed a synthesis method and evaluated 300 formulations with improved "
               "efficiency across six test conditions. " + limitation)
    caveat = summarize._extractive(item).split("Caveat: ", 1)[1]
    assert limitation in caveat
    assert "No major limitation" not in caveat


def test_preprint_caveat_keeps_warning_and_stated_limitation():
    item = row("The study is limited to a small sample of experimental designs.")
    item.source = "bioRxiv"
    caveat = summarize._extractive(item).split("Caveat: ", 1)[1]
    assert "not yet peer-reviewed" in caveat
    assert "limited to a small sample" in caveat


def test_unrecognized_limitations_are_unknown_instead_of_asserted_absent():
    item = row("The method reports improved performance across 300 formulations and six "
               "test conditions, achieving reproducible targeting of the selected material.")
    caveat = summarize._extractive(item).split("Caveat: ", 1)[1]
    assert "No major limitation" not in caveat
    assert "check" in caveat.lower()


def test_stated_limitation_excerpt_is_bounded():
    item = row("The study is limited to " + "a specific experimental condition " * 40 + ".")
    caveat = summarize._extractive(item).split("Caveat: ", 1)[1]
    assert "limited to" in caveat
    assert len(caveat) <= 300
    assert caveat.endswith("…")
