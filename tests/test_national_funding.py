"""UKRI and SNSF funding calls read from official feeds and call pages (fixtures only)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from dailydigest.models import SourceSpec

TODAY = date(2026, 10, 27)
FIXTURES = Path(__file__).parent / "fixtures" / "sources"
FUNDING_KEYS = {
    "opportunity_type", "status", "official", "verified_at", "funder", "open_date", "deadline",
    "deadline_timezone", "deadlines", "currency", "eligibility", "eligibility_tags",
    "eligibility_description", "official_id",
}


def _spec(**kwargs) -> SourceSpec:
    return SourceSpec(**{"name": "Test", "kind": "test", "section": "opportunities", "lookahead_days": 365, **kwargs})


def _serve(monkeypatch, module, pages: dict[str, str]) -> list[str]:
    calls: list[str] = []

    def fake_get_text(url: str) -> str:
        calls.append(url)
        if url not in pages:
            request = httpx.Request("GET", url)
            raise httpx.HTTPStatusError("404", request=request, response=httpx.Response(404, request=request))
        return (FIXTURES / pages[url]).read_text(encoding="utf-8")

    monkeypatch.setattr(module, "_get_text", fake_get_text)
    monkeypatch.setattr(module, "_today", lambda: TODAY)
    return calls


# -- UKRI ---------------------------------------------------------------------

UKRI_PAGES = {
    "https://www.ukri.org/opportunity/feed/": "ukri-feed.xml",
    "https://www.ukri.org/opportunity/early-independence-clinician-scientist-fellowship":
        "ukri-opportunity-fellowship.html",
    "https://www.ukri.org/opportunity/knowledge-transfer-partnership-2026-to-2027-round-4":
        "ukri-opportunity-ktp.html",
    "https://www.ukri.org/opportunity/engineering-biology-programme-mission-hubs": "ukri-opportunity-upcoming.html",
    "https://www.ukri.org/opportunity/data-for-metascience-sandpit-2027": "ukri-opportunity-closed.html",
    "https://www.ukri.org/opportunity/advancing-solutions-for-critical-mineral-production-and-recycling":
        "ukri-opportunity-stale.html",
}


def test_ukri_keeps_open_and_forthcoming_calls_verified_on_their_pages(monkeypatch):
    from dailydigest.ingest import national_funding

    calls = _serve(monkeypatch, national_funding, UKRI_PAGES)
    items = national_funding.UKRIFundingSource().fetch(_spec(name="UKRI", kind="ukri_funding"))

    by_id = {item.external_id: item for item in items}
    assert set(by_id) == {"ukri-181592", "ukri-206800", "ukri-206500"}
    # Business-only Innovate UK and invite-only calls are never fetched.
    assert not any("healthy-soils" in url or "invite-only" in url for url in calls)

    fellowship = by_id["ukri-181592"]
    meta = fellowship.metadata
    assert FUNDING_KEYS <= set(meta)
    assert fellowship.url == "https://www.ukri.org/opportunity/early-independence-clinician-scientist-fellowship"
    assert meta["opportunity_type"] == "fellowship" and meta["status"] == "open"
    assert meta["funder"] == "Medical Research Council (MRC)" and fellowship.authors == meta["funder"]
    assert meta["open_date"] == "2026-10-21" and meta["deadline"] == "2027-01-13"
    assert meta["deadlines"] == [{"type": "application", "date": "2027-01-13", "timezone": "Europe/London"}]
    assert meta["currency"] == "GBP" and meta["amount_max"] == 1_500_000 and "amount_min" not in meta
    assert meta["eligibility"] == "" and meta["eligibility_tags"] == []
    assert meta["eligibility_description"].startswith(
        "This funding opportunity is open to organisations with standard eligibility. Check if your organisation is eligible.")
    assert "registered healthcare professional" not in meta["eligibility_description"]
    assert "Councils" not in fellowship.abstract and "MRC" in fellowship.abstract

    ktp = by_id["ukri-206800"].metadata
    assert ktp["deadline"] == "2026-12-02" and "amount_max" not in ktp, "A fund total is not an award size"
    assert ktp["eligibility_description"].startswith("To lead a project your organisation must be")
    assert "Not part of the eligibility summary" not in ktp["eligibility_description"]

    upcoming = by_id["ukri-206500"].metadata
    assert upcoming["status"] == "forthcoming" and upcoming["deadline"] is None and upcoming["deadlines"] == []
    assert upcoming["open_date"] == "2027-02-01" and upcoming["amount_max"] == 15_000_000


def test_ukri_drops_calls_closing_beyond_the_lookahead(monkeypatch):
    from dailydigest.ingest import national_funding

    _serve(monkeypatch, national_funding, UKRI_PAGES)
    items = national_funding.UKRIFundingSource().fetch(_spec(name="UKRI", kind="ukri_funding", lookahead_days=60))

    assert {item.external_id for item in items} == {"ukri-206800"}


def test_ukri_caps_detail_page_fetches(monkeypatch):
    from dailydigest.ingest import national_funding

    calls = _serve(monkeypatch, national_funding, UKRI_PAGES)
    monkeypatch.setattr(national_funding.UKRIFundingSource, "MAX_DETAIL_PAGES", 1)
    items = national_funding.UKRIFundingSource().fetch(_spec(name="UKRI", kind="ukri_funding"))

    assert len(calls) == 2 and [item.external_id for item in items] == ["ukri-181592"]


def test_ukri_broken_feed_raises(monkeypatch):
    from dailydigest.ingest import national_funding

    monkeypatch.setattr(national_funding, "_get_text", lambda url: "<html><body>Maintenance</body></html>")
    with pytest.raises(RuntimeError):
        national_funding.UKRIFundingSource().fetch(_spec(name="UKRI", kind="ukri_funding"))


# -- SNSF ---------------------------------------------------------------------

SNSF_PAGES = {
    "https://www.snf.ch/api/rss/en/call/feed.xml": "snsf-feed.xml",
    "https://www.snf.ch/en/B9wb55x8qyNWmy1n/news/snsf-starting-grants-2027": "snsf-call-starting-grants.html",
    "https://www.snf.ch/en/3P7pzbPX0iyOtYDo/news/call-for-projects-ambizione": "snsf-call-ambizione.html",
    "https://www.snf.ch/en/aK0GyCwNg3knzGat/news/call-for-postdocmobility": "snsf-call-postdoc-mobility.html",
    "https://www.snf.ch/en/gaGYS8iom20zvCCM/news/biodiversa-new-call-opening": "snsf-call-no-deadline.html",
}


def test_snsf_reads_recent_calls_and_their_official_deadlines(monkeypatch):
    from dailydigest.ingest import national_funding

    calls = _serve(monkeypatch, national_funding, SNSF_PAGES)
    items = national_funding.SNSFFundingSource().fetch(_spec(name="SNSF", kind="snsf_funding"))

    assert not any("OldCall" in url for url in calls), "Calls older than a year are not fetched"
    by_id = {item.external_id: item for item in items}
    assert set(by_id) == {"snsf-B9wb55x8qyNWmy1n", "snsf-3P7pzbPX0iyOtYDo"}

    starting = by_id["snsf-B9wb55x8qyNWmy1n"]
    meta = starting.metadata
    assert FUNDING_KEYS <= set(meta)
    assert meta["deadline"] == "2027-05-04", "JSON-LD Application Deadline"
    assert meta["deadlines"] == [{"type": "application", "date": "2027-05-04", "timezone": "Europe/Zurich"}]
    assert meta["currency"] == "CHF" and meta["status"] == "open" and meta["opportunity_type"] == "grant"
    assert meta["funder"] == "Swiss National Science Foundation (SNSF)" and meta["official_id"] == "B9wb55x8qyNWmy1n"
    assert meta["eligibility_description"] == "" and "amount_max" not in meta
    assert starting.abstract.startswith("Grants for researchers wanting to lead")
    assert "com@snf.ch" not in starting.model_dump_json()

    ambizione = by_id["snsf-3P7pzbPX0iyOtYDo"].metadata
    assert ambizione["deadline"] == "2026-11-03", "Labelled HTML deadline when JSON-LD has none"


def test_snsf_caps_detail_page_fetches(monkeypatch):
    from dailydigest.ingest import national_funding

    calls = _serve(monkeypatch, national_funding, SNSF_PAGES)
    monkeypatch.setattr(national_funding.SNSFFundingSource, "MAX_DETAIL_PAGES", 1)
    items = national_funding.SNSFFundingSource().fetch(_spec(name="SNSF", kind="snsf_funding"))

    # The newest call is read first.
    assert calls[1:] == ["https://www.snf.ch/en/B9wb55x8qyNWmy1n/news/snsf-starting-grants-2027"]
    assert [item.external_id for item in items] == ["snsf-B9wb55x8qyNWmy1n"]


def test_snsf_empty_feed_raises(monkeypatch):
    from dailydigest.ingest import national_funding

    monkeypatch.setattr(national_funding, "_get_text",
                        lambda url: '<?xml version="1.0"?><rss version="2.0"><channel><title>x</title></channel></rss>')
    with pytest.raises(RuntimeError):
        national_funding.SNSFFundingSource().fetch(_spec(name="SNSF", kind="snsf_funding"))


@pytest.mark.parametrize("source,pages,kind", [
    ("UKRIFundingSource", UKRI_PAGES, "ukri_funding"), ("SNSFFundingSource", SNSF_PAGES, "snsf_funding")])
def test_every_detail_page_failing_is_an_outage_not_an_empty_result(monkeypatch, source, pages, kind):
    from dailydigest.ingest import national_funding

    feed = {url: name for url, name in pages.items() if name.endswith(".xml")}
    _serve(monkeypatch, national_funding, feed)  # every opportunity page returns 404
    adapter = getattr(national_funding, source)()
    with pytest.raises(RuntimeError, match="opportunity pages could be read"):
        adapter.fetch(_spec(name="Funder", kind=kind))
    assert adapter.last_fetch_diagnostics["detail_failures"] == adapter.last_fetch_diagnostics["detail_pages"] > 0

    # One readable page is a partial result: kept, with the failures recorded.
    _serve(monkeypatch, national_funding, pages)
    adapter.fetch(_spec(name="Funder", kind=kind))
    assert adapter.last_fetch_diagnostics["detail_failures"] < adapter.last_fetch_diagnostics["detail_pages"]


def test_funding_requests_send_the_descriptive_user_agent(monkeypatch):
    from dailydigest.ingest import national_funding

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url):
            request = httpx.Request("GET", url)
            return httpx.Response(200, text="ok", request=request)

    monkeypatch.setattr(national_funding.httpx, "Client", FakeClient)
    assert national_funding._get_text("https://www.ukri.org/opportunity/feed/") == "ok"
    agent = seen["headers"]["User-Agent"]
    assert agent.startswith("dailydigest/") and "bot" not in agent.casefold()
