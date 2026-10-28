"""Long pages offer a way back to the top that Pip never covers."""
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def test_back_to_top_starts_hidden_and_returns_focus_to_the_page():
    base = (TEMPLATES / "_base.html.j2").read_text()
    assert '<button class="app-utility back-to-top" id="back-to-top" type="button" aria-label="Back to top" hidden>' in base
    assert "backToTop.hidden = window.scrollY < window.innerHeight;" in base
    assert 'document.getElementById("page-content").focus({ preventScroll: true });' in base
    assert "prefers-reduced-motion: reduce" in base
    assert ".settings-actions, .back-to-top { display: none !important; }" in base


def test_a_dropped_pip_clears_both_fixed_controls():
    pip = (TEMPLATES / "_pip.js.j2").read_text()
    assert 'for (const id of ["display-menu-toggle", "back-to-top"])' in pip
