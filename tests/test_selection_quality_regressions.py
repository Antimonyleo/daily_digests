"""Selection evidence must agree with scoring and durable work identities."""
from types import SimpleNamespace

import pytest

from dailydigest.dedupe import candidate_identity_keys, dedupe_ranking_candidates
from dailydigest.models import Item, SourceSpec
from dailydigest.rank import source_quality as quality


def research(source, **attributes):
    return SimpleNamespace(source=source, section="research",
                           title="Programmable DNA assembly method",
                           abstract="An experimentally evaluated method.", **attributes)


def test_verified_low_impact_venue_cannot_receive_flagship_score_bonus():
    flagged = research("Nature", venue_low_impact=True)
    minor = research("Unlisted Journal")
    assert quality.source_bucket(flagged) == "low_impact_journal"
    assert quality.quality_adjusted_score(flagged, .76) == pytest.approx(
        quality.quality_adjusted_score(minor, .76))


def test_verified_high_impact_unknown_venue_does_not_keep_unknown_venue_penalty():
    verified = research("Unlisted Journal", venue_high_impact=True)
    assert quality.source_bucket(verified) == "published_journal"
    # Evidence removes the unknown-venue penalty; it does not invent a tier bonus.
    assert quality.quality_adjusted_score(verified, .76) == pytest.approx(.76)


def test_preprint_tier_misconfiguration_cannot_earn_published_venue_bonus(monkeypatch):
    monkeypatch.setattr(quality, "_configured_sources_cached", lambda: {
        "biorxiv": SourceSpec(name="bioRxiv", kind="biorxiv", section="research",
                             quality_tier="high", prestige_score=.95),
    })
    preprint = research("bioRxiv")
    assert quality.source_bucket(preprint) == "bio_med_preprint"
    # At .76 the existing preprint penalty fades to zero; no novelty language.
    assert quality.quality_adjusted_score(preprint, .76) == pytest.approx(.76)


def item(source, url, external_id="copy", metadata=None):
    return Item(source=source, section="research", external_id=external_id,
                url=url, title=f"Research copy {source}", metadata=metadata or {})


def test_publisher_doi_digits_cannot_create_arxiv_identity_or_transfer_retraction():
    original = item("Nature", "https://doi.org/10.9999/2026.12345")
    unrelated = item("arXiv", "https://arxiv.org/abs/2026.12345", "2026.12345",
                     {"is_retracted": True})
    assert not any(key.startswith("arxiv:") for key in candidate_identity_keys(original))
    assert len(dedupe_ranking_candidates([original, unrelated])) == 2
    assert not quality.should_skip_item(original)


@pytest.mark.parametrize("url", [
    "https://notarxiv.org/abs/2401.12345",
    "https://publisher.example/articles/2401.12345",
    "https://publisher.example/search?q=2401.12345",
])
def test_unrelated_urls_cannot_claim_arxiv_identity(url):
    assert not any(key.startswith("arxiv:") for key in candidate_identity_keys(item("Publisher", url)))


def test_neighboring_pubmed_ids_do_not_collide_via_truncated_legacy_arxiv_match():
    # Previously the unrestricted legacy regex parsed 'gov/4281704' from both
    # eight-digit PMIDs, silently deleting unrelated retrieved papers.
    first = item("PubMed", "https://pubmed.ncbi.nlm.nih.gov/42817048/", "42817048")
    second = item("PubMed", "https://pubmed.ncbi.nlm.nih.gov/42817041/", "42817041")
    assert len(dedupe_ranking_candidates([first, second])) == 2
    assert not any(key.startswith("arxiv:") for key in candidate_identity_keys(first))


@pytest.mark.parametrize("url,external", [
    ("https://arxiv.org/abs/2401.12345v2", ""),
    ("https://arxiv.org/pdf/2401.12345v1.pdf", ""),
    ("https://export.arxiv.org/abs/cond-mat/9901234", ""),
    ("", "2401.12345v3"),
])
def test_actual_arxiv_records_keep_version_independent_identity(url, external):
    expected = "cond-mat/9901234" if "cond-mat" in url else "2401.12345"
    assert f"arxiv:{expected}" in candidate_identity_keys(item("arXiv cond-mat", url, external))


def test_published_doi_relationship_collapses_preprint_with_journal_without_title_match():
    preprint = item("bioRxiv", "https://doi.org/10.1101/2026.01.01.123456",
                    metadata={"published_doi": "https://doi.org/10.1038/published-paper"})
    journal = item("Nature", "https://doi.org/10.1038/published-paper")
    assert dedupe_ranking_candidates([preprint, journal]) == [journal]


@pytest.mark.parametrize("published_doi", ["", "NA", "not yet published", None])
def test_missing_published_doi_relationship_does_not_merge_unrelated_preprints(published_doi):
    first = item("bioRxiv", "https://doi.org/10.1101/one", metadata={"published_doi": published_doi})
    second = item("bioRxiv", "https://doi.org/10.1101/two", metadata={"published_doi": published_doi})
    assert len(dedupe_ranking_candidates([first, second])) == 2
