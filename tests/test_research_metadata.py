"""Authoritative publication metadata overrides topic-like editorial titles."""

import json
from types import SimpleNamespace

from defusedxml.ElementTree import fromstring
import pytest

from dailydigest.ingest.openalex import OpenAlexSource
from dailydigest.ingest.pubmed import PubMedSource
from dailydigest.ingest.biorxiv import BiorxivSource
from dailydigest.models import SourceSpec
from dailydigest.rank.source_quality import should_skip_item


@pytest.mark.parametrize("published,expected", [
    ("https://doi.org/10.9999/JOURNAL.123", "10.9999/journal.123"),
    ("10.9999/journal.123", "10.9999/journal.123"),
    ("NA", None), (True, None), (False, None), (None, None),
    ("published", None), ("https://example.org/10.9999/paper", None),
])
def test_biorxiv_preserves_only_explicit_published_doi(monkeypatch, published, expected):
    monkeypatch.setattr("dailydigest.ingest.biorxiv._get_json", lambda *args: {
        "collection": [{"doi": "10.1101/123456", "title": "Molecular design",
                        "published": published}], "messages": [{"total": 1}],
    })
    item = BiorxivSource().fetch(SourceSpec(
        name="bioRxiv", kind="biorxiv", section="research",
    ))[0]
    assert item.metadata.get("published_doi") == expected


def _pubmed(publication_types, correction=""):
    types = "".join(f"<PublicationType>{t}</PublicationType>" for t in publication_types)
    xml = f"""<PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
      <ArticleTitle>DNA nanotechnology reshapes molecular design</ArticleTitle>
      <Abstract><AbstractText>We discuss a new method, dataset and mechanism for DNA self assembly.</AbstractText></Abstract>
      <Journal><Title>Nature</Title></Journal><PublicationTypeList>{types}</PublicationTypeList>
    </Article>{correction}</MedlineCitation></PubmedArticle>"""
    return PubMedSource()._parse_article(
        fromstring(xml), SourceSpec(name="PubMed", kind="pubmed", section="research"),
    )


@pytest.mark.parametrize("publication_type", [
    "Editorial", "News", "Comment", "Published Erratum", "Retracted Publication",
    "Retraction of Publication", "Retraction Notice", "Expression of Concern",
])
def test_pubmed_non_research_types_survive_ingest_and_cannot_ride_venue_prestige(publication_type):
    item = _pubmed(["Journal Article", publication_type])
    assert publication_type in item.metadata["publication_types"]
    assert should_skip_item(item)
    stored = SimpleNamespace(section=item.section, title=item.title, abstract=item.abstract,
                             source=item.source, metadata_json=json.dumps(item.metadata))
    assert should_skip_item(stored)


@pytest.mark.parametrize("publication_type", ["Journal Article", "Review", "Systematic Review", "Letter"])
def test_pubmed_research_reviews_and_letters_are_not_blanket_banned(publication_type):
    item = _pubmed([publication_type])
    assert item.metadata["publication_types"] == [publication_type]
    assert not should_skip_item(item)


def test_pubmed_retraction_relation_rejects_original_paper_without_retraction_in_title():
    item = _pubmed(["Journal Article"], '<CommentsCorrectionsList><CommentsCorrections RefType="RetractionIn"><PMID>456</PMID></CommentsCorrections></CommentsCorrectionsList>')
    assert item.metadata["is_retracted"] is True
    assert should_skip_item(item)


def test_retracted_and_republished_original_is_excluded():
    item = _pubmed(["Journal Article"], '<CommentsCorrectionsList><CommentsCorrections RefType="RetractedandRepublishedIn"><PMID>456</PMID></CommentsCorrections></CommentsCorrectionsList>')
    assert item.metadata["is_retracted"] is True
    assert should_skip_item(item)


@pytest.mark.parametrize("relation", ["ErratumIn", "RetractedandRepublishedFrom"])
def test_erratum_original_and_corrected_replacement_remain_eligible(relation):
    item = _pubmed(["Journal Article"], f'<CommentsCorrectionsList><CommentsCorrections RefType="{relation}"><PMID>456</PMID></CommentsCorrections></CommentsCorrectionsList>')
    assert "is_retracted" not in item.metadata
    assert not should_skip_item(item)


def test_reference_retraction_metadata_does_not_retract_the_citing_article():
    item = _pubmed(["Journal Article"], '<CommentsCorrectionsList><CommentsCorrections RefType="Cites"><PMID>456</PMID></CommentsCorrections></CommentsCorrectionsList>')
    assert "is_retracted" not in item.metadata
    assert not should_skip_item(item)


@pytest.mark.parametrize("work_type,retracted,skip", [
    ("article", True, True), ("editorial", False, True), ("news", False, True),
    ("comment", False, True), ("erratum", False, True), ("retraction", False, True),
    ("article", False, False), ("review", False, False),
])
def test_openalex_preserves_explicit_type_and_retraction(monkeypatch, work_type, retracted, skip):
    work = {"id": "https://openalex.org/W1", "title": "DNA nanostructure mechanism and method",
            "doi": "https://doi.org/10.9999/metadata", "type": work_type,
            "is_retracted": retracted, "primary_location": {"source": {"display_name": "Nature"}}}
    monkeypatch.setattr("dailydigest.ingest.openalex._get_json", lambda *args: {
        "results": [work], "meta": {"count": 1, "next_cursor": None},
    })
    item = OpenAlexSource()._fetch_works(
        SourceSpec(name="OpenAlex", kind="openalex", section="research"),
        2, {}, cap=10, upgrade_venue=True,
    )[0]
    assert item.metadata["is_retracted"] is retracted
    assert item.metadata["work_type"] == work_type
    assert should_skip_item(item) is skip


def test_absent_retraction_metadata_remains_unknown(monkeypatch):
    work = {"id": "https://openalex.org/W1", "title": "DNA nanostructure mechanism and method"}
    monkeypatch.setattr("dailydigest.ingest.openalex._get_json", lambda *args: {
        "results": [work], "meta": {"count": 1, "next_cursor": None},
    })
    item = OpenAlexSource()._fetch_works(
        SourceSpec(name="OpenAlex", kind="openalex", section="research"), 2, {}, cap=10,
    )[0]
    assert "is_retracted" not in item.metadata
    assert "work_type" not in item.metadata
