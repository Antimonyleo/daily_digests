"""Tests for profile-driven retrieval terms and personalized summaries."""

from __future__ import annotations

from types import SimpleNamespace

import dailydigest.summarize as sm


def _profile():
    return SimpleNamespace(
        bio="Researcher in nucleic acid nanotechnology and protein design.",
        keywords=["DNA nanotechnology", "protein design", "mRNA therapeutics"],
        authors_of_interest=["Jane Roe"],
    )


def _row(title, abstract="", source="bioRxiv", section="research"):
    return SimpleNamespace(id=1, title=title, abstract=abstract, source=source,
                           section=section, published_at=None)


def test_matched_interests_personalizes_why_read():
    sm._set_reader_context(_profile())
    row = _row("A new DNA nanotechnology scaffold", "self-assembly of nucleic acids")
    why = sm._why_read(row)
    assert "DNA nanotechnology" in why
    sm._set_reader_context(None)


def test_no_profile_falls_back_to_generic_why_read():
    sm._set_reader_context(None)
    row = _row("Some unrelated finance paper", "markets and trading")
    why = sm._why_read(row)
    assert "your interest" not in why


def test_reader_context_included_in_llm_prompt():
    sm._set_reader_context(_profile())
    sys_prompt, _ = sm._build_prompt([_row("X", "y")])
    assert "protein design" in sys_prompt and "Reader profile" in sys_prompt
    sm._set_reader_context(None)


def test_synthesis_empty_on_normal_run():
    # days <= 2 -> no synthesis regardless of backend
    assert sm.synthesize_catch_up([_row("X")], days=2) == ""
    assert sm.synthesize_catch_up([], days=10) == ""


def test_synthesis_empty_without_llm(monkeypatch):
    from dailydigest.config import get_settings
    monkeypatch.setattr(get_settings(), "llm_backend", "extractive")
    monkeypatch.setattr(get_settings(), "llm_api_key", "")
    assert sm.synthesize_catch_up([_row("X"), _row("Y")], days=8) == ""


def test_profile_search_terms_from_profile(monkeypatch):
    from dailydigest.ingest import _terms
    monkeypatch.setattr(_terms, "load_profile", lambda: _profile(), raising=False)
    # load_profile is imported inside the function; patch config instead
    import dailydigest.config as cfg
    monkeypatch.setattr(cfg, "load_profile", lambda *a, **k: _profile())
    # All core keywords are always returned even when max_terms is smaller — the
    # cap only bounds the context tail, so no core interest is silently dropped.
    terms = _terms.profile_search_terms(2)
    assert terms == ["DNA nanotechnology", "protein design", "mRNA therapeutics"]
    assert _terms.watched_author_names(5) == ["Jane Roe"]


def test_profile_search_terms_include_context_keywords(monkeypatch):
    """Context keywords must still drive retrieval even though they're down-weighted
    for research relevance."""
    from dailydigest.ingest import _terms
    from dailydigest.models import Profile

    prof = Profile(
        bio="x",
        keywords=["DNA nanotechnology"],
        context_keywords=["FDA approval", "gene therapies"],
    )
    monkeypatch.setattr(_terms, "load_profile", lambda: prof, raising=False)
    import dailydigest.config as cfg
    monkeypatch.setattr(cfg, "load_profile", lambda *a, **k: prof)
    terms = _terms.profile_search_terms(12)
    assert "DNA nanotechnology" in terms
    assert "FDA approval" in terms
    assert "gene therapies" in terms
