from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from dailydigest.rank import enrich


def article():
    return SimpleNamespace(id=1, url="https://doi.org/10.1234/example", external_id="",
                           section="research", source="OpenAlex", title="Research method",
                           metadata_json="{}", published_at=datetime.now(timezone.utc) - timedelta(days=30))


SETTINGS = SimpleNamespace(citation_enrichment=True, citation_polite_email="", venue_quality_weight=.18)


@pytest.mark.parametrize("failure, expected", [(False, "missing"), (True, "works_failed")])
def test_fetch_distinguishes_healthy_missing_from_failed_batch(monkeypatch, failure, expected):
    def get(*args):
        if failure:
            raise OSError("offline")
        return {"results": []}
    monkeypatch.setattr(enrich, "_get_json_bounded", get)
    result = enrich.fetch_openalex(["10.1234/example"])
    assert result["10.1234/example"]["status"] == expected


def test_sources_failure_is_partial_not_fresh(monkeypatch):
    def get(client, url, params):
        if url == enrich._OPENALEX_SOURCES_URL:
            raise OSError("offline sources")
        return {"results": [{"doi": "https://doi.org/10.1234/example", "cited_by_count": 999,
                              "primary_location": {"source": {"id": "https://openalex.org/S1",
                                                                 "display_name": "Nature"}}}]}
    monkeypatch.setattr(enrich, "_get_json_bounded", get)
    assert enrich.fetch_openalex(["10.1234/example"])["10.1234/example"]["status"] == "sources_failed"


@pytest.mark.parametrize("refresh", ["works_failed", "sources_failed", "missing", "sources_missing", "venue_missing"])
def test_stale_observation_survives_without_boost_or_refresh_laundering(monkeypatch, refresh):
    row = article()
    observed = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    old = {"cited_by_count": 999, "venue_impact": 99, "venue": "Nature",
           "observed_at": observed, "status": "stale"}
    monkeypatch.setattr(enrich, "load_cached_enrichment", lambda ids: {1: old})
    monkeypatch.setattr(enrich, "save_enrichment", lambda entries: pytest.fail("Failed refresh overwrote cache"))
    def fetch(*args):
        return {"10.1234/example": {"status": refresh, "cited_by_count": 1000, "venue": "Nature"}}
    enrich.prepare_enrichment([row], settings=SETTINGS, fetcher=fetch)
    result = enrich.enrich_scored([(row, .6)], settings=SETTINGS, prepared_only=True)
    assert result[0][1] == .6
    assert not getattr(row, "venue_name", None)
    assert not getattr(row, "venue_high_impact", False)
    assert row.enrichment_evidence["observed_at"] == observed
    assert row.enrichment_evidence["status"] == "stale"
    assert row.enrichment_evidence["refresh_status"] == refresh


def test_successful_refresh_replaces_stale_observation(monkeypatch):
    row = article()
    monkeypatch.setattr(enrich, "load_cached_enrichment", lambda ids: {1: {"status": "stale", "venue": "old"}})
    saved = []
    monkeypatch.setattr(enrich, "save_enrichment", saved.append)
    enrich.prepare_enrichment([row], settings=SETTINGS, fetcher=lambda *args: {
        "10.1234/example": {"status": "fresh", "venue": "new", "cited_by_count": 1}})
    assert row.enrichment_evidence["status"] == "fresh"
    assert row.enrichment_evidence["observed_at"]
    assert row.venue_name == "new"
    assert saved[0][1]["venue"] == "new"


def test_cache_bounds_keep_original_timestamp_and_reject_unknown_age():
    from dailydigest.store import ItemEnrichmentRow, ItemRow, init_db, session_scope
    init_db()
    now = datetime.now(timezone.utc)
    with session_scope() as session:
        for iid, age in [(1, 2), (2, 20), (3, 31)]:
            session.add(ItemRow(id=iid, source="fixture", section="research", external_id=str(iid),
                                title="fixture", url=f"https://example.org/{iid}"))
            session.flush()
            session.add(ItemEnrichmentRow(item_id=iid, cited_by_count=7, fetched_at=now-timedelta(days=age)))
    cached = enrich.load_cached_enrichment([1, 2, 3])
    assert cached[1]["status"] == "fresh"
    assert cached[2]["status"] == "stale"
    assert datetime.fromisoformat(cached[2]["observed_at"]) == now-timedelta(days=20)
    assert cached[3]["status"] == "expired"


@pytest.mark.parametrize("status, expected_low", [("stale", True), ("expired", False)])
def test_only_bounded_low_venue_veto_survives_healthy_miss(monkeypatch, status, expected_low):
    row = article()
    monkeypatch.setattr(enrich, "load_cached_enrichment", lambda ids: {1: {
        "status": status, "venue_impact": .1, "venue": "Nature", "cited_by_count": 900,
        "observed_at": "2026-09-01T00:00:00+00:00"}})
    monkeypatch.setattr(enrich, "save_enrichment", lambda entries: pytest.fail("Healthy miss overwrote known venue"))
    enrich.prepare_enrichment([row], settings=SETTINGS, fetcher=lambda *args: {})
    assert bool(getattr(row, "venue_low_impact", False)) is expected_low
    assert not getattr(row, "venue_name", None)
    assert enrich.enrich_scored([(row, .6)], settings=SETTINGS, prepared_only=True)[0][1] == .6


def test_healthy_sources_without_impact_are_not_complete(monkeypatch):
    monkeypatch.setattr(enrich, "_get_json_bounded", lambda client, url, params:
        {"results": []} if url == enrich._OPENALEX_SOURCES_URL else {"results": [{
            "doi": "https://doi.org/10.1234/example", "cited_by_count": 3,
            "primary_location": {"source": {"id": "https://openalex.org/S1"}}}]})
    assert enrich.fetch_openalex(["10.1234/example"])["10.1234/example"]["status"] == "sources_missing"


def test_malformed_works_response_is_failure_not_healthy_absence(monkeypatch):
    monkeypatch.setattr(enrich, "_get_json_bounded", lambda *args: {})
    assert enrich.fetch_openalex(["10.1234/example"])["10.1234/example"]["status"] == "works_failed"


def test_partial_work_cannot_overwrite_a_known_citation_observation(monkeypatch):
    monkeypatch.setattr(enrich, "_get_json_bounded", lambda *args: {"results": [{
        "doi": "https://doi.org/10.1234/example",
        "primary_location": {"source": {"id": "https://openalex.org/S1"}}}]})
    assert enrich.fetch_openalex(["10.1234/example"])["10.1234/example"]["status"] == "works_failed"


def test_malformed_source_statistics_preserve_source_failure_identity(monkeypatch):
    monkeypatch.setattr(enrich, "_get_json_bounded", lambda client, url, params: {"results": [{
        "id": "https://openalex.org/S1", "summary_stats": ["malformed"]}]} if url == enrich._OPENALEX_SOURCES_URL
        else {"results": [{"doi": "https://doi.org/10.1234/example", "cited_by_count": 3,
                           "primary_location": {"source": {"id": "https://openalex.org/S1"}}}]})
    assert enrich.fetch_openalex(["10.1234/example"])["10.1234/example"]["status"] == "sources_failed"


@pytest.mark.parametrize("status", ["stale", "expired"])
def test_reused_row_cannot_keep_old_positive_venue_admission(status):
    row = article()
    enrich._apply_venue_evidence(row, {"status": "fresh", "venue": "Nature", "venue_impact": 99})
    assert row.venue_name == "Nature" and row.venue_high_impact
    enrich._apply_venue_evidence(row, {"status": status, "venue": "Nature", "venue_impact": 99})
    assert not row.venue_name and not row.venue_high_impact


def test_partial_legacy_refresh_does_not_erase_known_venue(monkeypatch):
    row = article()
    monkeypatch.setattr(enrich, "load_cached_enrichment", lambda ids: {1: {
        "status": "stale", "venue_impact": .1, "venue": "Known venue", "cited_by_count": 9}})
    monkeypatch.setattr(enrich, "save_enrichment", lambda entries: pytest.fail("Partial refresh overwrote known venue"))
    enrich.prepare_enrichment([row], settings=SETTINGS, fetcher=lambda *args: {"10.1234/example": 10})
    assert row.venue_low_impact
    assert row.enrichment_evidence["refresh_status"] == "sources_missing"
