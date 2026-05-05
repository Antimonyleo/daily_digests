from __future__ import annotations

from dailydigest.email_render import render_digest, safe_url
from dailydigest.store import ItemRow


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
