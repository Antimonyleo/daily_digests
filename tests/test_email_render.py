from __future__ import annotations

import pytest

from dailydigest.email_render import (
    content_type_label,
    reason_line,
    render_digest,
    safe_url,
)
from dailydigest.store import ItemRow


@pytest.fixture(autouse=True)
def _stub_features(monkeypatch):
    """Keep render_digest hermetic: no DB read for persisted features by default.

    Individual tests override the return value to inject primary_facet etc.
    """
    import dailydigest.email_render as er

    monkeypatch.setattr(er, "load_digest_features", lambda digest_id: {})
    return monkeypatch


def _research_row(**overrides) -> ItemRow:
    kwargs = {
        "source": "Nature",
        "section": "research",
        "external_id": "facet-1",
        "url": "https://example.com/facet-1",
        "title": "A primary RNA delivery study",
        "abstract": "Primary result.",
    }
    kwargs.update(overrides)
    row = ItemRow(**kwargs)
    row.id = overrides.get("id", 1)
    row.item_label = overrides.get("item_label", "R1")
    return row


def test_render_digest_escapes_feed_controlled_html():
    row = ItemRow(
        source='Source <img src=x onerror="alert(1)">',
        section="research",
        external_id="xss",
        url="javascript:alert(1)",
        title='<script>alert("x")</script>',
        abstract="Abstract.",
    )
    row.id = 1
    row.item_label = "R1"

    html = render_digest(
        "2026-05-05",
        {"research": [(row, 0.9, '<img src=x onerror="alert(2)">')]},
    )

    assert '<script>alert("x")</script>' not in html
    assert '<img src=x onerror="alert(2)">' not in html
    assert 'Source <img src=x onerror="alert(1)">' not in html
    assert "&lt;script&gt;" in html
    assert 'href="javascript:alert(1)"' not in html
    assert 'href="#"' in html


def test_safe_url_allows_only_http_and_https_links():
    assert safe_url("https://example.com/a?b=1") == "https://example.com/a?b=1"
    assert safe_url("http://example.com") == "http://example.com"
    assert safe_url("javascript:alert(1)") == "#"
    assert safe_url("data:text/html,hi") == "#"
    assert safe_url("/relative") == "#"


def test_content_type_label_only_tags_non_research_types():
    assert content_type_label("review") == "Review"
    assert content_type_label("method") == "Method"
    assert content_type_label("dataset") == "Dataset"
    assert content_type_label("clinical") == "Clinical"
    # Primary research / plain article and anything else get no badge.
    assert content_type_label("research") == ""
    assert content_type_label("article") == ""
    assert content_type_label("editorial") == ""
    assert content_type_label("") == ""
    assert content_type_label(None) == ""


def test_reason_line_uses_primary_facet_and_high_profile_suffix():
    assert reason_line("RNA nanotechnology") == "Shown for RNA nanotechnology"
    assert (
        reason_line("RNA nanotechnology", high_profile=True)
        == "Shown for RNA nanotechnology · high-profile journal"
    )
    assert (
        reason_line("RNA nanotechnology", high_profile=True, journal="Nature")
        == "Shown for RNA nanotechnology · Nature"
    )


def test_reason_line_falls_back_and_never_renders_broken_line():
    # Empty facet with no fallback signals -> omit entirely.
    assert reason_line("") == ""
    assert reason_line(None, tags=[], why_shown=[]) == ""
    # Empty facet but a why_shown signal -> fall back to it.
    assert reason_line("", why_shown=["Reliable source"]) == "Shown for Reliable source"
    # Underscored tags are humanized.
    assert reason_line("", tags=["high_quality_source"]) == "Shown for high quality source"
    # Never produce "Shown for ." from an empty facet.
    assert "Shown for ." not in reason_line("")


def test_render_digest_renders_reason_for_primary_facet(_stub_features):
    row = _research_row()
    _stub_features.setattr(
        "dailydigest.email_render.load_digest_features",
        lambda digest_id: {
            1: {
                "primary_facet": "RNA nanotechnology",
                "content_type": "research",
                "source_bucket": "published_journal",
                "tags": ["high_quality_source"],
            }
        },
    )

    html = render_digest("2026-06-15", {"research": [(row, 0.9, "A summary.")]})

    assert "Shown for RNA nanotechnology" in html
    # Published journal -> high-profile suffix (uses the journal name).
    assert "Nature" in html
    # research content_type must NOT produce a type badge.
    assert ">Review<" not in html
    assert ">Method<" not in html


def test_render_digest_renders_review_label(_stub_features):
    row = _research_row(title="A comprehensive review of delivery")
    _stub_features.setattr(
        "dailydigest.email_render.load_digest_features",
        lambda digest_id: {
            1: {
                "primary_facet": "RNA nanotechnology",
                "content_type": "review",
                "source_bucket": "published_journal",
            }
        },
    )

    html = render_digest("2026-06-15", {"research": [(row, 0.9, "A summary.")]})

    assert ">Review<" in html


def test_render_digest_plain_research_has_no_type_badge_and_no_broken_reason(_stub_features):
    row = _research_row(source="OpenAlex")
    _stub_features.setattr(
        "dailydigest.email_render.load_digest_features",
        lambda digest_id: {
            1: {
                "primary_facet": "",
                "content_type": "research",
                "source_bucket": "aggregator",
                "tags": [],
                "why_shown": [],
            }
        },
    )

    html = render_digest("2026-06-15", {"research": [(row, 0.9, "A summary.")]})

    assert ">Review<" not in html
    assert ">Method<" not in html
    assert ">Dataset<" not in html
    assert ">Clinical<" not in html
    # Empty facet + no fallback signals must never render a broken reason line.
    assert "Shown for ." not in html


def test_render_digest_shows_structured_opportunity_facts(_stub_features):
    row = _research_row(
        source="Grants.gov",
        section="opportunities",
        item_label="F1",
        metadata_json=(
            '{"status":"open","deadline":"2026-10-01",'
            '"amount_min":100000,"amount_max":500000,"currency":"USD",'
            '"official":true,"eligibility_tags":[]}'
        ),
    )

    html = render_digest(
        "2026-06-15", {"opportunities": [(row, 0.9, "Official funding call.")]}
    )

    assert "Funding &amp; Opportunities" in html
    assert "$100,000–$500,000" in html
    assert "2026-10-01" in html
    assert "Verified official source" in html
