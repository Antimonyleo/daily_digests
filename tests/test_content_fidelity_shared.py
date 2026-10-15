"""Article identity and completed contributions survive incidental wording."""
from types import SimpleNamespace

import pytest

from dailydigest.ingest.rss import RSSSource
from dailydigest.ingest import rss
from dailydigest.models import SourceSpec
from dailydigest.rank import source_quality as quality


def _entry_metadata(entry):
    return rss._entry_metadata(entry)


def row(title="Programmable DNA assemblies", abstract=None, **metadata):
    return SimpleNamespace(
        title=title, abstract=abstract or "We develop a programmable DNA assembly method and evaluate its experimental performance.",
        source="Nature", section="research", metadata=metadata, id=1, url="https://example.test/paper",
    )


@pytest.mark.parametrize("title", [
    "Correction of sequence errors in programmable DNA assemblies",
    "Correction of protein design biases using experimental measurements",
])
def test_error_correction_research_remains_eligible(title):
    from dailydigest.pipeline import _quality_gate

    article = row(title)
    assert not quality.should_skip_item(article)
    assert _quality_gate([article])[0] == [article]


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



@pytest.mark.parametrize("metadata", [
    {"is_retracted": True}, {"publication_types": ["Retracted Publication"]},
    {"publication_types": ["Editorial"]}, {"publication_types": ["News"]},
    {"publication_types": ["Published Erratum"]}, {"work_type": "correction"},
])
def test_explicit_publication_exclusions_reach_serving_quality_gate(metadata):
    from dailydigest.pipeline import _quality_gate

    article = row(**metadata)
    assert quality.is_non_research_content(article)
    assert quality.should_skip_item(article)
    assert _quality_gate([article])[0] == []


def test_malformed_or_topical_metadata_does_not_exclude_primary_research():
    for metadata in [{"is_retracted": "false"}, {"publication_types": "Editorial"},
                     {"publication_types": [None, {"genre": "Editorial"}]},
                     {"work_type": "journal-article"}]:
        assert not quality.should_skip_item(row(**metadata))
