"""Reader polish: clear venue names, a working compact view, an explained
digest size, the source notice at the end and an even About panel."""
import re

import pytest

from tests.test_top_picks import _digest
from tests.test_web import _request, _text_payload


@pytest.mark.parametrize("feed,shown", [
    ("Angew. Chem. Int. Ed.", "Angewandte Chemie International Edition"),
    ("Chem. Soc. Rev.", "Chemical Society Reviews"),
    ("Small (Wiley)", "Small (Wiley)"),
    ("Genomics, proteomics & bioinformatics", "Genomics, Proteomics & Bioinformatics"),
    ("Journal of the american chemical society", "Journal of the American Chemical Society"),
    ("bioRxiv (recent)", "bioRxiv"),
    ("eLife", "eLife"),
    ("arXiv q-bio.QM", "arXiv q-bio.QM"),
    ("Nature Nanotechnology", "Nature Nanotechnology"),
    ("Grants.gov (fellowships and career awards)", "Grants.gov (fellowships and career awards)"),
])
def test_venue_names_show_the_journal_title(feed, shown):
    from dailydigest.web import _display_source_name
    assert _display_source_name(feed) == shown


def _page(tmp_path, monkeypatch, coverage=None):
    from dailydigest import web
    if coverage is not None:
        monkeypatch.setattr(web.health, "coverage_summary", lambda: coverage)
    _digest(tmp_path, monkeypatch, {"research": [0.9, 0.8], "world": [0.3]})
    return _text_payload(web.index(_request("GET", "/")))


def test_venue_leads_the_meta_line(tmp_path, monkeypatch):
    text = _page(tmp_path, monkeypatch)
    meta = re.search(r'<div class="meta">(.*?)</div>', text, re.S).group(1)
    assert meta.lstrip().startswith('<span class="venue">Example</span>')
    assert ".item .meta .venue { color: var(--text); font-weight: 600; }" in text


def test_compact_finding_uses_the_full_card_width(tmp_path, monkeypatch):
    text = _page(tmp_path, monkeypatch)
    assert "grid-template-columns: 72px 1fr" not in text
    assert 'html[data-display="compact"] .summary-primary { grid-template-columns: minmax(0, 1fr); gap: 0; }' in text
    assert ':not(:has(.item-details[open])) .summary { display: -webkit-box;' in text


def test_digest_size_choice_explains_each_size(tmp_path, monkeypatch):
    text = _page(tmp_path, monkeypatch)
    assert 'aria-describedby="reading-mode-hint"' in text
    assert 'data-hint="Every qualified pick">Full<' in text
    assert 'data-hint="Up to 15 best picks" selected>Standard<' in text
    assert 'data-hint="5 research picks + 1 from each other section">Short<' in text
    assert '<span class="brew-size-hint" id="reading-mode-hint">Up to 15 best picks</span>' in text
    assert 'sizeSelect?.addEventListener("change", syncSizeHint);' in text


def test_unavailable_source_note_sits_at_the_end_of_the_digest(tmp_path, monkeypatch):
    coverage = {"total": 2, "responded": 1, "unavailable": 1, "limited": 0, "unknown": 0,
                "checked_at": "", "checked_label": "Time not recorded", "checked_stale": False,
                "public_oldest_label": "", "public_stale": False, "rows": [], "authors": []}
    text = _page(tmp_path, monkeypatch, coverage)
    note = '1 source was unavailable in the latest check. <a href="#source-details" data-open-source-details>See source coverage</a>'
    assert text.count(note) == 1
    assert text.index('id="digest-content"') < text.index(note) < text.index('<details class="about-brew">')
    assert text.index(note) < text.index("Your reading history and ratings are saved locally.")


def test_about_panel_sections_share_one_heading_style(tmp_path, monkeypatch):
    text = _page(tmp_path, monkeypatch)
    for heading in ('<h3 class="about-heading">Your feedback</h3>',
                    '<h3 class="about-heading">Latest source check</h3>',
                    '<h3 class="ranking-kicker about-heading">Personalization</h3>'):
        assert heading in text
    assert "<strong>Latest source check</strong>" not in text
    assert 'id="vote-threshold-part"' in text and '<span id="vote-threshold">' in text
