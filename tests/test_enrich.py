"""Tests for OpenAlex citation enrichment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dailydigest import config as config_mod
from dailydigest.rank import enrich as enrich_mod
from dailydigest.rank.enrich import (
    citation_score,
    derive_doi,
    enrich_scored,
    venue_quality_score,
)
from dailydigest.store import ItemRow


def _item(title: str, url: str, published_at=None) -> ItemRow:
    return ItemRow(
        id=abs(hash(title)) % 100000,
        source="Test",
        section="research",
        external_id=title,
        url=url,
        title=title,
        abstract="abstract",
        published_at=published_at or datetime.now(timezone.utc),
    )


def _settings(**overrides):
    return config_mod.load_settings().model_copy(update=overrides)


def test_derive_doi_from_url():
    row = _item("A", "https://doi.org/10.1038/s41586-024-12345-6")
    assert derive_doi(row) == "10.1038/s41586-024-12345-6"


def test_derive_doi_none_when_absent():
    assert derive_doi(_item("A", "https://www.nature.com/articles/xyz")) is None


def test_citation_score_monotonic_and_bounded():
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pub = now - timedelta(days=30)
    low = citation_score(1, pub, now=now)
    high = citation_score(30, pub, now=now)
    assert 0.0 < low < high <= 1.0
    assert citation_score(0, pub, now=now) == 0.0
    assert citation_score(None, pub, now=now) == 0.0


def test_citation_velocity_rewards_young_papers():
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    young = citation_score(10, now - timedelta(days=30), now=now)  # 10/month
    old = citation_score(10, now - timedelta(days=365), now=now)   # ~0.8/month
    assert young > old


def test_enrich_disabled_is_noop():
    scored = [(_item("A", "https://doi.org/10.1000/x"), 0.5)]
    out = enrich_scored(scored, settings=_settings(citation_enrichment=False))
    assert out == scored


def test_venue_quality_score_bounded_and_monotonic():
    assert venue_quality_score(None) is None
    assert venue_quality_score(-1) is None
    low = venue_quality_score(0.5)
    high = venue_quality_score(20)
    assert 0.0 < low < high <= 1.0


def test_enrich_boosts_and_reorders():
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pub = now - timedelta(days=30)
    a = _item("A", "https://doi.org/10.1000/aaa", published_at=pub)
    b = _item("B", "https://doi.org/10.1000/bbb", published_at=pub)
    # B starts slightly behind A but is far more cited → should overtake.
    scored = [(a, 0.60), (b, 0.55)]

    def fake_fetch(dois, email):
        # bare-int form is still accepted for backward compatibility
        return {"10.1000/aaa": 0, "10.1000/bbb": 60}

    out = enrich_scored(
        scored,
        settings=_settings(citation_enrichment=True),
        fetcher=fake_fetch,
        now=now,
    )
    assert [row.title for row, _ in out][0] == "B"


def test_venue_impact_penalizes_low_impact_journal():
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pub = now - timedelta(days=30)
    hi = _item("HI", "https://doi.org/10.1000/hi", published_at=pub)
    lo = _item("LO", "https://doi.org/10.1000/lo", published_at=pub)
    # Equal start, equal (zero) citations, but very different venue impact.
    scored = [(hi, 0.60), (lo, 0.60)]

    def fake_fetch(dois, email):
        return {
            "10.1000/hi": {"cited_by_count": 0, "venue_impact": 25.0},
            "10.1000/lo": {"cited_by_count": 0, "venue_impact": 0.4},
        }

    out = enrich_scored(
        scored,
        settings=_settings(citation_enrichment=True, venue_quality_weight=0.18),
        fetcher=fake_fetch,
        now=now,
    )
    titles = [row.title for row, _ in out]
    scores = {row.title: s for row, s in out}
    assert titles[0] == "HI"
    assert scores["HI"] > 0.60 > scores["LO"]  # high venue boosted, low penalized


def test_enrich_no_dois_is_noop():
    scored = [(_item("A", "https://example.com/no-doi"), 0.5)]
    called = {"n": 0}

    def fake_fetch(dois, email):
        called["n"] += 1
        return {}

    out = enrich_scored(
        scored, settings=_settings(citation_enrichment=True), fetcher=fake_fetch
    )
    assert out == scored
    assert called["n"] == 0  # never hit the network when no DOIs present
