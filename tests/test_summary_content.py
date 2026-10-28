"""Summary content is fixed at the source: sourced, complete, role-appropriate fields.

Fixtures are minimal public-style feed patterns, not copies of user data.
"""
from __future__ import annotations

import json
import re

import pytest

from dailydigest import summarize as sm
from dailydigest.ingest.rss import strip_feed_metadata
from dailydigest.store import ItemRow

NO_ABSTRACT = ("Key finding: No abstract is available, so the findings cannot be assessed "
               "from the title alone.")


@pytest.fixture(autouse=True)
def no_reader_profile(monkeypatch):
    monkeypatch.setattr(sm, "_READER_FACETS", {})
    monkeypatch.setattr(sm, "_READER_KEYWORDS", [])
    monkeypatch.setattr(sm, "_READER_CONTEXT", "")


def item(abstract, *, source="Example Journal", section="research", title="A sample study",
         metadata=None, url="https://example.org/article"):
    return ItemRow(id=1, source=source, section=section, external_id="content-test", url=url,
                   title=title, abstract=abstract, metadata_json=json.dumps(metadata or {}))


def fields(summary):
    return sm._summary_parts(summary)


# --- Feed metadata -----------------------------------------------------------

@pytest.mark.parametrize("header", [
    "Example Nanotechnology, Published online: 13 October 2026; doi:10.1038/s41565-026-01234-5 ",
    # Nature feeds glue the DOI check digit to the first body word.
    "Example Methods, Published online: 13 May 2026; doi:10.1038/s41592-026-01234-8",
])
def test_metadata_header_is_removed_before_body_text_is_summarized(header):
    body = ("A nickel-modified nanopore identifies all twenty amino acids in single peptides. "
            "The device reached 98% accuracy across 1200 reads.")
    summary = sm._extractive(item(header + body))
    finding = fields(summary)["Key finding"]
    assert "Published online" not in summary and "doi:" not in summary and "10.1038" not in summary
    assert finding.startswith("A nickel-modified nanopore") or "98% accuracy" in finding
    assert all(sentence in body for sentence in sm._split_sentences(finding))


@pytest.mark.parametrize("abstract", [
    "Example Materials, Volume 38, Issue 52, 16 September 2026.",
    "Example Science, Volume 392, Issue 6804 , Page 1244-1247, June 2026.",
    "Example Journal of Medicine, Ahead of Print.",
    "Example Nanotechnology, Published online: 13 October 2026; doi:10.1038/s41565-026-01234-5",
    "Article URL: https://example.org/post Comments URL: https://news.example/item?id=1 "
    "Points: 213 # Comments: 421",
])
def test_header_only_feed_text_is_a_missing_abstract(abstract):
    assert sm._extractive(item(abstract)) == NO_ABSTRACT


def test_header_cleaning_keeps_body_sentences_that_cite_a_doi_or_numbers():
    body = ("We reanalysed the public dataset (doi:10.5555/example.123) and measured 42 binding "
            "constants in 2026 conditions.")
    assert strip_feed_metadata(body) == body
    assert "doi:10.5555/example.123" in sm._extractive(item(body))


def test_abstract_labels_and_inner_abstract_marker_do_not_fuse_sentences():
    abstract = ("A programmable triple helix forms with a preferred handedness. ABSTRACT "
                "Sequence information can direct synthetic assembly. Here, we demonstrate two "
                "hierarchical levels of sequence control within one oligomer strand.")
    cleaned = strip_feed_metadata(abstract)
    assert "ABSTRACT" not in cleaned
    assert "handedness.\nSequence information" in cleaned
    assert strip_feed_metadata("AbstractAlternative splicing shapes the proteome.") == (
        "Alternative splicing shapes the proteome.")
    assert "ABSTRACT" not in sm._extractive(item(abstract))


def test_publisher_licence_and_author_list_are_not_summary_text():
    abstract = ("Example Chem., 2026, Advance Article DOI: 10.1039/D6XX01234A, Edge Article Open "
                "Access This article is licensed under a Creative Commons Attribution 3.0 Unported "
                "Licence. Ada Example, Ben Sample, Cy Placeholder Here we develop a peptide assay "
                "that resolves 12 isobaric modifications. To cite this article before page numbers "
                "are assigned, use the DOI form of citation above. The content of this RSS Feed (c) "
                "The Example Society")
    assert strip_feed_metadata(abstract) == ("Here we develop a peptide assay that resolves 12 "
                                             "isobaric modifications.")


def test_sentence_score_gives_identifiers_and_dates_no_substance():
    header = "Journal, Published online: 13 October 2026; doi:10.1038/s41565-026-02302-1."
    result = "The assay reduced false positives by 40% across 120 samples."
    assert sm._sentence_score(header)[0] <= 0 < sm._sentence_score(result)[0]


# --- Finding selection ------------------------------------------------------

def test_structured_abstract_prefers_results_and_moves_limitations_to_caveat():
    summary = sm._extractive(item(
        "BACKGROUND: Better delivery vehicles are needed. "
        "METHODS: We screened 120 lipid formulations in 30 mice. "
        "RESULTS: The lead formulation increased spleen delivery 4-fold over the control. "
        "LIMITATIONS: Only female mice were tested."))
    parts = fields(summary)
    assert "The lead formulation increased spleen delivery 4-fold over the control." in parts["Key finding"]
    assert "Better delivery vehicles" not in parts["Key finding"]
    assert parts["Caveat"] == "Only female mice were tested."
    assert "LIMITATIONS" not in summary and "RESULTS" not in summary


def test_valid_here_we_propose_method_sentence_is_kept_when_no_result_is_stated():
    method = ("Here we propose a high-temperature ligation method that lowers misconnection "
              "rates by joining overhangs above their predicted melting temperature.")
    abstract = ("Synthetic DNA construction is essential for engineering biology. Existing "
                "assembly techniques suffer from low connection specificity. " + method)
    finding = fields(sm._extractive(item(abstract)))["Key finding"]
    assert finding == method


def test_a_measured_result_is_used_alone_instead_of_method_plus_result():
    method = ("Here we propose a high-temperature ligation method that lowers misconnection "
              "rates by joining overhangs above their predicted melting temperature.")
    result = "The method increased assembly accuracy by 40 percentage points."
    abstract = ("Existing assembly techniques suffer from low connection specificity. "
                f"{method} {result}")
    assert fields(sm._extractive(item(abstract)))["Key finding"] == result


def test_result_needing_an_acronym_defined_earlier_takes_the_shortest_defining_sentence():
    long_definition = ("We first built a genetic information compiler (GIC) that interrogates "
                       "double-stranded DNA with an exonuclease and releases activators across "
                       "many programmable logic configurations.")
    short_definition = "The compiler (GIC) converts DNA identities into RNA outputs."
    result = "GIC increased output selectivity 30-fold over the uncompiled control."
    abstract = f"{long_definition} {short_definition} {result}"
    finding = fields(sm._extractive(item(abstract, title="Rewriting DNA identity")))["Key finding"]
    assert finding == f"{short_definition} {result}"
    # The title spells out "HighT", so a result using it needs no earlier sentence.
    abstract = ("Here we propose High Temperature (HighT) assembly, which ligates overhangs above "
                "their melting temperatures. We characterize HighT and demonstrate a 40 percentage "
                "point increase in accuracy.")
    finding = fields(sm._extractive(item(abstract, title="High temperature ligation of DNA")))
    assert finding["Key finding"] == ("We characterize HighT and demonstrate a 40 percentage point "
                                      "increase in accuracy.")


def test_a_short_result_is_preferred_over_a_long_one():
    long_result = ("The platform increased throughput 12-fold across 30 enzyme pairs, " +
                   "with consistent yields under varied buffer conditions, " * 5 + "in every test.")
    short_result = "Yields improved 3-fold in 20 independent repeats."
    finding = fields(sm._extractive(item(f"{long_result} {short_result}")))["Key finding"]
    assert finding == short_result
    assert len(finding) <= 350


def test_two_long_sentences_keep_one_complete_sentence_rather_than_a_fragment():
    method = ("Here we report a modular assembly platform that " + "couples programmable "
              "scaffolds with orthogonal handles, " * 6 + "for building enzyme cascades.")
    result = ("The platform increased cascade throughput 12-fold across 30 enzyme pairs, " +
              "with consistent yields under varied buffer conditions, " * 4 + "in every test.")
    finding = fields(sm._extractive(item(f"{method} {result}")))["Key finding"]
    assert finding in {method, result}
    assert "…" not in finding


def test_background_only_research_text_is_not_reported_as_a_finding():
    abstract = ("Targeted delivery of nucleic acid therapeutics remains a major challenge for "
                "next generation vaccines. Existing methods suffer from poor tissue selectivity.")
    assert fields(sm._extractive(item(abstract)))["Key finding"].startswith(
        "The supplied abstract gives only background")


def test_truncated_feed_teaser_is_not_stored_as_a_finding():
    abstract = ("Novartis received supplemental approval for a decades-old leprosy drug, "
                "which may boost ...")
    assert sm._extractive(item(abstract, section="industry")) == NO_ABSTRACT


@pytest.mark.parametrize("metadata,abstract", [
    ({"publication_types": ["Review"]},
     "Nanopore sequencing has advanced rapidly. We summarize recent progress in protein "
     "unfolding, transport and signal detection, and outline remaining challenges. Several "
     "groups reported 95% read accuracy."),
    ({}, "Self-assembling peptides offer programmable routes to biomaterials. In this Review, we "
         "compare stimulus-responsive and dissipative peptide assemblies. Several groups reported "
         "95% read accuracy."),
])
def test_review_states_its_scope_instead_of_borrowing_a_finding(metadata, abstract):
    row = item(abstract, metadata=metadata)
    assert sm._content_role(row) == "review"
    parts = fields(sm._extractive(row))
    assert re.search(r"\bwe (?:summarize|compare)", parts["Key finding"], re.I)
    assert "95% read accuracy" not in parts["Key finding"]
    assert "Caveat" not in parts


# --- Caveat ------------------------------------------------------------------

def test_motivation_about_existing_methods_is_not_this_studys_limitation():
    abstract = ("Super-resolution imaging is rarely used in bacteriology. A likely reason is the "
                "lack of straightforward protocols for intact cells. In this work, we establish a "
                "screening platform and benchmark 12 imaging sequences inside bacterial cells.")
    summary = sm._extractive(item(abstract, source="bioRxiv"))
    assert "Caveat:" not in summary
    assert "lack of straightforward protocols" not in summary


def test_limitation_after_the_contribution_is_kept_as_a_full_sentence():
    limitation = "Our analysis is limited to a small sample of twelve sequences."
    abstract = ("Existing methods lack bacterial benchmarks. In this work, we establish a screening "
                "platform and benchmark 12 imaging sequences inside bacterial cells. " + limitation)
    assert fields(sm._extractive(item(abstract)))["Caveat"] == limitation


@pytest.mark.parametrize("section", ["opportunities", "events"])
def test_opportunity_and_event_items_never_get_a_research_caveat(section):
    abstract = ("The purpose of this funding opportunity is to solicit applications that study "
                "genomic variant interactions. Awards are limited to a small sample of centers.")
    row = item(abstract, section=section, source="Grants.gov")
    summary = sm._extractive(row)
    assert set(fields(summary)) == {"Key finding"}
    assert "purpose of this funding opportunity" in summary
    provider = ("Key finding: The call funds variant interaction centers.\n"
                "Caveat: Eligibility is limited to a small sample of institutions.")
    assert sm._finalize_provider_summary(row, provider) == (
        "Key finding: The call funds variant interaction centers.")


# --- Provider contract, fallback and cache ----------------------------------

def test_prompt_receives_cleaned_text_and_no_preprint_caveat_instruction():
    row = item("Example Nanotechnology, Published online: 13 October 2026; "
               "doi:10.1038/s41565-026-01234-5 A nanopore resolves single amino acids.")
    system, user = sm._build_prompt([row])
    payload = json.loads(user.split("\n\n", 1)[1])[0]
    assert payload["abstract"] == "A nanopore resolves single amino acids."
    assert payload["content_role"] == "research"
    assert "Caveat: None." in system
    assert "Do not mention preprint" in system
    assert "preprint not peer-reviewed" not in system


@pytest.mark.parametrize("provider,expected", [
    ("Key finding: The assay detected 500 aM RNA.\nCaveat: None.",
     "Key finding: The assay detected 500 aM RNA."),
    ("Key finding: The assay detected 500 aM RNA.\n"
     "Why read: The abstract's contribution match remains unclear; check its relevance.\n"
     "Caveat: Preprint — not yet peer-reviewed, so treat conclusions as provisional.",
     "Key finding: The assay detected 500 aM RNA."),
    ("Key finding: The assay detected 500 aM RNA.\n"
     "Why read: It gives a heat-free amplification route for point-of-care RNA sensing.\n"
     "Caveat: Only 40 saliva samples from one site were tested.",
     "Key finding: The assay detected 500 aM RNA.\n"
     "Why read: It gives a heat-free amplification route for point-of-care RNA sensing.\n"
     "Caveat: Only 40 saliva samples from one site were tested."),
])
def test_provider_fields_keep_only_specific_why_read_and_caveat(provider, expected):
    assert sm._parse_id_summary_map(json.dumps({"1": provider}))
    assert sm._finalize_provider_summary(item("The assay detected RNA in saliva samples."),
                                         provider) == expected


def test_provider_output_is_finalized_and_failure_falls_back_to_local_summary(monkeypatch):
    first = item("We developed an amplification assay and detected 500 aM viral RNA in 40 samples.")
    second = item("Example Journal, Volume 1, Issue 2, 3 May 2026.")
    second.id = 2
    calls = []

    def provider(batch, **_kwargs):
        calls.append([row.id for row in batch])
        if len(calls) == 1:
            return {1: "Key finding: The assay detected 500 aM RNA.\nCaveat: None."}
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(sm, "_BATCH_SIZE", 1)
    monkeypatch.setattr(sm, "_call_llm", provider)
    summaries, provenance = sm._summarize_via_provider_with_provenance([first, second], "api")
    assert summaries[1] == "Key finding: The assay detected 500 aM RNA."
    assert provenance == {1: "api", 2: "extractive_fallback"}
    assert summaries[2] == NO_ABSTRACT


def test_old_cached_summary_is_refreshed_by_signature_and_read_compatibly(monkeypatch):
    row = item("Example Nanotechnology, Published online: 13 October 2026; "
               "doi:10.1038/s41565-026-01234-5 A nanopore resolves single amino acids in 30 "
               "peptides.")
    legacy = ("Key finding: Example Nanotechnology, Published online: 13 October 2026; "
              "doi:10.1038/s41565-026-01234-5 A nanopore resolves single amino acids in 30 peptides.\n"
              "Why read: it may contain details worth checking against your profile.\n"
              "Caveat: Limitations were not assessed; check the linked article.")
    current = sm.summary_signature(row, None)
    monkeypatch.setattr(sm, "_SUMMARY_VERSION", "summary-v9")
    assert sm.summary_signature(row, None) != current  # the pipeline cache misses and resummarizes
    monkeypatch.undo()
    assert sm._extractive(row) == ("Key finding: A nanopore resolves single amino acids in 30 "
                                   "peptides.")
    stored = legacy
    assert sm.normalize_cached_summary(stored, "research") == (
        "Key finding: A nanopore resolves single amino acids in 30 peptides.")
    assert stored == legacy  # display normalization never rewrites the stored history
    single_line = "Key finding: A result was measured. Why read: x. Caveat: A pilot study of 8 mice."
    assert sm.normalize_cached_summary(single_line) == (
        "Key finding: A result was measured.\nWhy read: x.\nCaveat: A pilot study of 8 mice.")
    assert sm.normalize_cached_summary("Key point: unlabeled legacy text") == (
        "Key point: unlabeled legacy text")

