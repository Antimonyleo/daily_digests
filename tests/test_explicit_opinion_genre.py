"""Explicit opinion evidence cannot masquerade as primary research."""
from types import SimpleNamespace

import pytest

from dailydigest.dedupe import dedupe_ranking_candidates
from dailydigest.models import Item
from dailydigest.pipeline import _quality_gate
from dailydigest.rank import source_quality as quality


# Exact self-description excerpt from the captured PubMed 41791978 abstract.
SELF_DESCRIPTION = "In this opinion article, we critically assess where quantum computing may deliver realistic near-term value for biomedicine"


def row(abstract=SELF_DESCRIPTION, metadata=None, section="research"):
    return SimpleNamespace(
        id=1, title="Quantum computing for biotechnological innovation",
        abstract=abstract, metadata=metadata or {}, section=section,
        source="Nature", url="https://example.test/opinion", external_id="opinion",
    )


def test_captured_self_description_overrides_generic_journal_review_metadata():
    article = row(
        "Quantum computing supports structural biology. " + SELF_DESCRIPTION + ".",
        {"publication_types": ["Journal Article", "Review"]},
    )
    assert quality.content_type(article) == "editorial"
    assert _quality_gate([article])[0] == []
    evidence = quality.research_exclusion_metadata(article)
    assert evidence["publication_types"] == ["opinion"]
    assert evidence["publication_type_evidence"][0]["field"] == "abstract_self_description"


@pytest.mark.parametrize("metadata", [
    {"work_type": "opinion"}, {"publication_types": ["Journal Article", "Opinion"]},
])
def test_declared_opinion_is_not_rescued_by_method_words(metadata):
    article = row("We discuss methods and structural biology in this essay.", metadata)
    assert quality.should_skip_item(article)
    assert quality.content_type(article) == "editorial"


@pytest.mark.parametrize("abstract", [
    "This is not an opinion article. We report experimental measurements.",
    "In this research article, we measure opinion formation using controlled experiments.",
    "We tested a claim from an opinion article using quantitative methods.",
    'The source states: "' + SELF_DESCRIPTION + '." We test this claim experimentally.',
    '"Earlier work was limited. ' + SELF_DESCRIPTION + '." We test this claim experimentally.',
    "“Earlier work was limited. " + SELF_DESCRIPTION + ".” We test this claim experimentally.",
    "‘Earlier work was limited. " + SELF_DESCRIPTION + ".’ We test this claim experimentally.",
    "'Earlier work was limited. " + SELF_DESCRIPTION + ".' We test this claim experimentally.",
])
def test_negated_quoted_or_studied_opinions_do_not_exclude_research(abstract):
    article = row(abstract, {"publication_types": ["Journal Article"]})
    assert not quality.should_skip_item(article)
    assert quality.content_type(article) != "editorial"
    assert not quality.research_exclusion_metadata(article)


@pytest.mark.parametrize("section", ["industry", "world", "ai"])
def test_opinion_exclusion_is_research_only(section):
    article = row(section=section)
    assert not quality.should_skip_item(article)
    assert not quality.research_exclusion_metadata(article)


@pytest.mark.parametrize("declared", [False, True])
def test_explicit_opinion_follows_strong_identity_with_honest_evidence(declared):
    metadata = {"doi": "10.9999/opinion"}
    if declared:
        metadata["publication_types"] = ["Opinion"]
    else:
        metadata["publication_types"] = ["Review"]
        metadata["publication_type_evidence"] = [{"field": "dc_type", "value": "Review"}]
    evidence = Item(source="PubMed", section="research", external_id="old",
                    url="https://pubmed.ncbi.nlm.nih.gov/123/",
                    title="Quantum computing for biotechnological innovation",
                    abstract="Method discussion." if declared else SELF_DESCRIPTION,
                    metadata=metadata)
    preferred = Item(source="Nature", section="research", external_id="rich",
                     url="https://doi.org/10.9999/opinion", title=evidence.title,
                     abstract="A detailed discussion of computational methods. " * 40,
                     metadata={"publication_types": ["Journal Article"]})
    unrelated = preferred.model_copy(update={
        "url": "https://doi.org/10.9999/other", "external_id": "other", "metadata": {},
    })
    dedupe_ranking_candidates([preferred, evidence, unrelated])
    assert quality.should_skip_item(preferred)
    assert not quality.should_skip_item(unrelated)
    if not declared:
        assert preferred.metadata["publication_type_evidence"][0]["field"] == "abstract_self_description"
        assert {record["field"] for record in preferred.metadata["publication_type_evidence"]} == {
            "abstract_self_description", "dc_type",
        }
        assert quality.content_type(preferred) == "editorial"
