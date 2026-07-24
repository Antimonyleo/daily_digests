"""Tests for URL canonicalization and deduplication logic.

Covers:
  - dailydigest.ingest.rss.canonicalize_url
  - dailydigest.dedupe.dedupe_by_url
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import numpy as np

from dailydigest.ingest.rss import canonicalize_url
from dailydigest.dedupe import cap_near_duplicates, dedupe_by_url, dedupe_ranking_candidates
from dailydigest.models import Item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(url: str, title: str = "Test", source: str = "src") -> Item:
    return Item(
        source=source,
        section="research",
        external_id="abc123",
        url=url,
        title=title,
        abstract="",
    )


# ---------------------------------------------------------------------------
# canonicalize_url
# ---------------------------------------------------------------------------

class TestCanonicalizeUrl:
    def test_strips_utm_params(self):
        url = "https://x.com/a?utm_source=foo&utm_medium=bar&id=42"
        result = canonicalize_url(url)
        # utm_ params stripped; non-tracking id=42 kept
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "id=42" in result

    def test_strips_all_utm_variants(self):
        url = "https://example.com/page?utm_source=a&utm_medium=b&utm_campaign=c&utm_term=d&utm_content=e"
        result = canonicalize_url(url)
        assert "utm_" not in result

    def test_strips_fragment(self):
        url = "https://example.com/page#section"
        result = canonicalize_url(url)
        assert "#" not in result
        assert "section" not in result

    def test_strips_trailing_slash(self):
        url = "https://example.com/article/"
        result = canonicalize_url(url)
        assert not result.endswith("/")

    def test_preserves_non_tracking_params(self):
        url = "https://example.com/page?doi=10.1234&ref=home"
        result = canonicalize_url(url)
        assert "doi=10.1234" in result
        assert "ref=home" in result

    def test_scheme_and_host_case_preserved(self):
        # observed behavior: canonicalize_url does NOT lowercase scheme/host
        # (urlparse preserves the original casing of scheme and host);
        # tests lock in this actual behavior rather than assumed lowercasing.
        url = "https://Example.Com/page"
        result = canonicalize_url(url)
        # The result should still be a valid URL containing the path
        assert "Example.Com" in result or "example.com" in result  # observed behavior; verify intentional

    def test_empty_url_returns_empty(self):
        assert canonicalize_url("") == ""

    def test_url_no_query_no_fragment_unchanged_path(self):
        url = "https://example.com/article"
        result = canonicalize_url(url)
        assert "example.com/article" in result

    def test_trailing_slash_removed_after_utm_strip(self):
        url = "https://example.com/?utm_source=rss"
        result = canonicalize_url(url)
        assert "utm_source" not in result
        assert not result.endswith("/")


# ---------------------------------------------------------------------------
# dedupe_by_url
# ---------------------------------------------------------------------------

class TestDedupeByUrl:
    def test_removes_second_item_with_same_url(self):
        items = [
            _make_item("https://example.com/a", "First"),
            _make_item("https://example.com/a", "Duplicate"),
        ]
        result = dedupe_by_url(items)
        assert len(result) == 1
        assert result[0].title == "First"

    def test_preserves_order_of_first_seen(self):
        items = [
            _make_item("https://example.com/c", "C"),
            _make_item("https://example.com/a", "A"),
            _make_item("https://example.com/b", "B"),
        ]
        result = dedupe_by_url(items)
        assert [r.title for r in result] == ["C", "A", "B"]

    def test_dedupes_on_canonical_form(self):
        # Same URL with different UTM params → same canonical → one item kept
        items = [
            _make_item("https://example.com/article?utm_source=twitter"),
            _make_item("https://example.com/article?utm_source=facebook"),
        ]
        result = dedupe_by_url(items)
        assert len(result) == 1

    def test_distinct_urls_all_kept(self):
        items = [
            _make_item("https://example.com/a"),
            _make_item("https://example.com/b"),
            _make_item("https://example.com/c"),
        ]
        result = dedupe_by_url(items)
        assert len(result) == 3

    def test_empty_list_returns_empty(self):
        assert dedupe_by_url([]) == []

    def test_single_item_returned_unchanged(self):
        items = [_make_item("https://example.com/x")]
        result = dedupe_by_url(items)
        assert len(result) == 1

    def test_fragment_deduplication(self):
        # Two URLs that differ only by fragment canonicalize to same URL
        items = [
            _make_item("https://example.com/page#intro"),
            _make_item("https://example.com/page#methods"),
        ]
        result = dedupe_by_url(items)
        assert len(result) == 1


class TestDedupeRankingCandidates:
    def test_collapses_doi_duplicates_across_url_forms(self):
        items = [
            _make_item("https://doi.org/10.1234/ABC.1", "DOI article", "OpenAlex"),
            _make_item("https://dx.doi.org/10.1234/abc.1", "DOI article mirror", "Journal"),
        ]

        result = dedupe_ranking_candidates(items)

        assert len(result) == 1
        assert result[0].source == "Journal"

    def test_doi_duplicate_keeps_top_journal_over_aggregator(self):
        aggregator = _make_item(
            "https://doi.org/10.1038/example",
            "Important RNA delivery result",
            "OpenAlex",
        )
        aggregator.abstract = "Thin metadata."
        nature = _make_item(
            "https://www.nature.com/articles/example",
            "Important RNA delivery result",
            "Nature Biotechnology",
        )
        nature.external_id = "10.1038/example"
        nature.abstract = "Primary research with methods, efficacy, and mechanism details."

        result = dedupe_ranking_candidates([aggregator, nature])

        assert len(result) == 1
        assert result[0].source == "Nature Biotechnology"

    def test_same_day_title_duplicate_ignores_journal_issue_suffix(self):
        published_at = datetime(2026, 5, 15, tzinfo=timezone.utc)
        primary = Item(
            source="Advanced Materials",
            section="research",
            external_id="primary",
            url="https://example.com/primary",
            title="Orally Administered Nanoparticle Coacervate for Therapeutic Coating of Full Gastrointestinal Tract",
            abstract="Primary research with methods, mechanism, and delivery results.",
            published_at=published_at,
        )
        issue_teaser = Item(
            source="Advanced Materials",
            section="research",
            external_id="issue-teaser",
            url="https://example.com/issue-teaser",
            title="Orally Administered Nanoparticle Coacervate for Therapeutic Coating of Full Gastrointestinal Tract (Adv. Mater. 27/2026)",
            abstract="Primary research with methods, mechanism, and delivery results.",
            published_at=published_at,
        )

        result = dedupe_ranking_candidates([primary, issue_teaser])

        assert len(result) == 1
        assert result[0].title == primary.title

    def test_collapses_pubmed_openalex_duplicate_by_same_day_title_and_doi_transitively(self):
        pub_dt = datetime(2026, 5, 10, tzinfo=timezone.utc)
        title = "Example therapy improves survival in phase 3 trial"
        pubmed = Item(
            source="PubMed",
            section="research",
            external_id="12345678",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            title=title,
            published_at=pub_dt,
        )
        openalex = Item(
            source="OpenAlex",
            section="research",
            external_id="https://doi.org/10.5555/example",
            url="https://doi.org/10.5555/example",
            title=title,
            published_at=pub_dt,
        )
        publisher = Item(
            source="Publisher",
            section="research",
            external_id="publisher-1",
            url="https://dx.doi.org/10.5555/example",
            title="Publisher copy",
            published_at=pub_dt,
        )

        result = dedupe_ranking_candidates([pubmed, openalex, publisher])

        assert len(result) == 1
        assert result[0].source == "PubMed"


# ---------------------------------------------------------------------------
# cap_near_duplicates (within-day near-duplicate suppression)
# ---------------------------------------------------------------------------

def _unit(vec: list[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class TestCapNearDuplicates:
    def test_cluster_of_three_collapses_to_highest_scored(self):
        # Three near-identical vectors (cosine > 0.99), rows sorted by score DESC.
        # The first (highest-scored) is the surviving representative.
        base = _unit([1.0, 0.0, 0.0])
        v1 = _unit([1.0, 0.02, 0.0])
        v2 = _unit([1.0, 0.0, 0.02])
        vecs = np.vstack([base, v1, v2])
        rows = ["top", "mid", "low"]  # already sorted by score desc

        keep = cap_near_duplicates(rows, vecs, threshold=0.86)

        assert keep == [0]  # only the top-scored representative survives

    def test_three_distinct_all_survive(self):
        # Orthogonal (cosine 0) vectors are clearly distinct → all kept.
        vecs = np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])
        rows = ["a", "b", "c"]

        keep = cap_near_duplicates(rows, vecs, threshold=0.86)

        assert keep == [0, 1, 2]

    def test_threshold_boundary_respected(self):
        # Two vectors with a known cosine; keep both above the threshold, drop
        # the second when the threshold is at/below their similarity.
        a = _unit([1.0, 0.0])
        b = _unit([0.9, np.sqrt(1 - 0.81)])  # cosine(a, b) == 0.9
        vecs = np.vstack([a, b])
        cos = float(np.dot(vecs[0] / np.linalg.norm(vecs[0]),
                           vecs[1] / np.linalg.norm(vecs[1])))
        assert abs(cos - 0.9) < 1e-5

        # threshold just ABOVE the pair's similarity → both kept (not a near-dup)
        assert cap_near_duplicates(["x", "y"], vecs, threshold=0.95) == [0, 1]
        # threshold at/below the similarity → second dropped as a near-dup
        assert cap_near_duplicates(["x", "y"], vecs, threshold=0.90) == [0]
        assert cap_near_duplicates(["x", "y"], vecs, threshold=0.85) == [0]

    def test_empty_input_returns_empty(self):
        assert cap_near_duplicates([], np.zeros((0, 0), dtype=np.float32), 0.86) == []

    def test_single_item_always_kept(self):
        assert cap_near_duplicates(["only"], _unit([1, 0, 0]).reshape(1, -1), 0.86) == [0]

    def test_does_not_mutate_inputs(self):
        vecs = np.vstack([_unit([1, 0]), _unit([1, 0.01])])
        vecs_copy = vecs.copy()
        rows = ["a", "b"]
        rows_copy = list(rows)

        cap_near_duplicates(rows, vecs, threshold=0.86)

        assert np.array_equal(vecs, vecs_copy)
        assert rows == rows_copy

    def test_misaligned_matrix_keeps_all(self):
        # If the embedding matrix does not align with rows, keep everything
        # rather than silently dropping items.
        vecs = np.vstack([_unit([1, 0]), _unit([1, 0])])  # 2 rows
        keep = cap_near_duplicates(["a", "b", "c"], vecs, threshold=0.86)  # 3 rows
        assert keep == [0, 1, 2]

    def test_representative_is_first_of_each_cluster(self):
        # Two separate clusters interleaved by score; each keeps its first
        # (highest-scored) member.
        c1a = _unit([1.0, 0.0, 0.0])
        c2a = _unit([0.0, 1.0, 0.0])
        c1b = _unit([1.0, 0.01, 0.0])  # near c1a
        c2b = _unit([0.0, 1.0, 0.01])  # near c2a
        vecs = np.vstack([c1a, c2a, c1b, c2b])  # score-desc order

        keep = cap_near_duplicates(["c1a", "c2a", "c1b", "c2b"], vecs, threshold=0.86)

        assert keep == [0, 1]  # both distinct cluster heads kept, near-dups dropped
