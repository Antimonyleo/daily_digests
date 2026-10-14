"""Article identity and completed contributions survive incidental wording."""
from types import SimpleNamespace

import pytest

from dailydigest.ingest.rss import RSSSource, _entry_metadata
from dailydigest.models import SourceSpec
from dailydigest.rank import source_quality as quality
from dailydigest.rank.contribution import analyze_contribution


def row(title="Programmable DNA assemblies", abstract=None, **metadata):
    return SimpleNamespace(
        title=title, abstract=abstract or "We develop a programmable DNA assembly method and evaluate its experimental performance.",
        source="Nature", section="research", metadata=metadata,
    )


@pytest.mark.parametrize("title", [
    "Correction of sequence errors in programmable DNA assemblies",
    "Correction of protein design biases using experimental measurements",
])
def test_error_correction_research_remains_eligible(title):
    assert not quality.should_skip_item(row(title))


@pytest.mark.parametrize("title", [
    "Correction", "Correction: Programmable DNA assemblies", "Correction to programmable DNA assemblies",
    "Author Correction: Programmable DNA assemblies", "Publisher Correction: Programmable DNA assemblies",
    "Erratum: Programmable DNA assemblies", "Retraction Note: Programmable DNA assemblies",
])
def test_notice_titles_remain_excluded(title):
    assert quality.should_skip_item(row(title))


@pytest.mark.parametrize("phrase", ["from a new perspective", "using oligonucleotide primers"])
def test_incidental_genre_words_do_not_change_method_type_or_score(phrase):
    original = row()
    incidental = row(abstract=original.abstract + " We analyze the results " + phrase + ".")
    assert quality.content_type(incidental) == "method"
    assert quality.quality_adjusted_score(incidental, .75) == quality.quality_adjusted_score(original, .75)


@pytest.mark.parametrize("declared,expected", [("Editorial", "editorial"), ("Review", "review"), ("Perspective", "editorial")])
def test_explicit_genre_outranks_method_words(declared, expected):
    assert quality.content_type(row(publication_types=["Journal Article", declared])) == expected


@pytest.mark.parametrize("title,expected", [("Perspective: DNA assembly", "editorial"), ("Primer: DNA assembly", "review"), ("Review of DNA assembly", "review")])
def test_structural_genre_titles_remain_recognized(title, expected):
    assert quality.content_type(row(title)) == expected


@pytest.mark.parametrize("field,value", [("dc_type", "Editorial"), ("prism_section", "News and Views"), ("prism_genre", "Correction")])
def test_rss_record_type_is_preserved_with_provenance_and_enforced(field, value):
    metadata = _entry_metadata({field: value, "prism_doi": "10.9999/test"})
    assert metadata["doi"] == "10.9999/test"
    assert metadata["publication_types"]
    assert metadata["publication_type_evidence"] == [{"field": field, "value": value}]
    assert quality.should_skip_item(row(**metadata))


def test_rss_topical_categories_and_correction_links_do_not_exclude_originals():
    metadata = _entry_metadata({"tags": [{"term": "Editorial"}], "dc_subject": "Correction",
                                "prism_section": "Protein design", "dc_relation": "ErratumIn"})
    assert "publication_types" not in metadata
    assert not quality.should_skip_item(row(**metadata))


def test_namespaced_rss_notice_retains_record_evidence(monkeypatch):
    xml = b'''<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel><title>Journal</title><item><title>Programmable DNA assembly methods</title>
      <link>https://example.org/paper</link><description>We develop a method for DNA assembly.</description>
      <dc:type>Editorial</dc:type></item></channel></rss>'''
    monkeypatch.setattr("dailydigest.ingest.rss._http_get_bytes", lambda url: xml)
    item = RSSSource().fetch(SourceSpec(name="Nature", kind="rss", section="research", url="https://example.org/feed"))[0]
    assert item.metadata["publication_type_evidence"] == [{"field": "dc_type", "value": "Editorial"}]
    assert quality.should_skip_item(item)


@pytest.mark.parametrize("value", ["Letter", "Journal Article", "Research Article", "Protocol"])
def test_rss_primary_research_types_remain_eligible(value):
    assert not quality.should_skip_item(row(**_entry_metadata({"dc_type": value})))


FACETS = {"Protein engineering and design": {"aliases": ["protein design"]}}


@pytest.mark.parametrize("abstract", [
    "We used an existing protein language model to engineer proteins with improved catalytic performance, and experimentally validated the designed enzymes.",
    "We engineered proteins with improved catalytic performance using an existing protein language model, and experimentally validated the designed enzymes.",
])
def test_completed_engineering_output_survives_tool_use_paraphrase(abstract):
    result = analyze_contribution("Engineering proteins", abstract, FACETS)
    assert result["kind"] == "engineering"
    assert result["priority"] == 1
    assert result["evidence"] == [abstract]


@pytest.mark.parametrize("abstract", [
    "We used an existing protein model to study native protein interactions and characterize cellular signaling with experimental measurements.",
    "We used an existing protein model to design experiments for determining native protein signaling mechanisms in multiple cellular conditions.",
    "We used an existing protein model that could engineer proteins in future work, but here describe naturally occurring sequences.",
    "We used an existing protein model to engineer proteins, but did not produce or validate any designed molecules in this study.",
])
def test_tool_use_without_completed_engineered_output_is_not_promoted(abstract):
    assert analyze_contribution("A study", abstract, FACETS)["priority"] <= 0


def test_unknown_venue_evidence_does_not_change_conservative_bucket():
    article = row()
    article.source = "Unlisted Specialist Journal"
    payload = quality.breakdown_payload(article, .8)
    assert payload["source_bucket"] == "low_impact_journal"
    assert payload["venue_evidence"]["status"] == "unknown"
    assert "unknown_venue" in payload["quality_tags"]
    assert "Venue quality unverified" in payload["tags"]


@pytest.mark.parametrize("flag,status", [("venue_low_impact", "observed_low_impact"), ("venue_high_impact", "observed_high_impact")])
def test_observed_venue_evidence_is_separate_from_article_quality(flag, status):
    article = row()
    setattr(article, flag, True)
    assert quality.venue_evidence(article)["status"] == status
    assert "not verified article quality" in quality.venue_evidence(article)["meaning"]


def test_unmatched_notice_provenance_cannot_follow_another_declared_type():
    article = row(publication_types=["Editorial"], publication_type_evidence=[
        {"field": "dc_type", "value": "Correction"}, {"field": "dc_relation", "value": "Editorial"},
        {"field": ["dc_type"], "value": "Editorial"}, None,
    ])
    assert quality.research_exclusion_metadata(article) == {"publication_types": ["editorial"]}
def test_venue_evidence_preserves_stale_observation_provenance():
    from types import SimpleNamespace

    from dailydigest.rank.source_quality import venue_evidence

    row = SimpleNamespace(section="research", source="Nature", venue_low_impact=True,
                          enrichment_evidence={"status": "stale",
                                               "observed_at": "2026-09-20T12:00:00+00:00",
                                               "refresh_status": "sources_failed"})
    evidence = venue_evidence(row)
    assert evidence["status"] == "observed_low_impact"
    assert evidence["observation_status"] == "stale"
    assert evidence["observed_at"] == row.enrichment_evidence["observed_at"]
    assert evidence["refresh_status"] == "sources_failed"

