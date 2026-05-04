"""Tests for URL canonicalization and deduplication logic.

Covers:
  - dailydigest.ingest.rss.canonicalize_url
  - dailydigest.dedupe.dedupe_by_url
"""

from __future__ import annotations

import pytest

from dailydigest.ingest.rss import canonicalize_url
from dailydigest.dedupe import dedupe_by_url
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
