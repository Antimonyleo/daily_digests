"""Event adapters for organisers that publish listings as web pages (fixtures only)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from dailydigest.models import SourceSpec

TODAY = date(2026, 10, 27)
FIXTURES = Path(__file__).parent / "fixtures" / "sources"
EVENT_KEYS = {
    "record_type", "event_type", "status", "official", "verified_at", "event_start", "event_end",
    "deadline", "deadline_timezone", "deadlines", "location", "format", "organizer", "official_id",
}


def _spec(**kwargs) -> SourceSpec:
    return SourceSpec(**{"name": "Test", "kind": "test", "section": "events", "lookahead_days": 365, **kwargs})


def _not_found(url: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", url)
    return httpx.HTTPStatusError("404", request=request, response=httpx.Response(404, request=request))


@pytest.fixture
def serve(monkeypatch):
    """Serve fixture files for URLs and record every request."""
    from dailydigest.ingest import event_pages, events_api

    calls: list[str] = []
    monkeypatch.setattr(event_pages, "_today", lambda: TODAY)
    monkeypatch.setattr(events_api, "_today", lambda: TODAY)

    def install(pages: dict[str, str]) -> list[str]:
        def fake_get_text(url, params=None):
            calls.append(url)
            if url not in pages:
                raise _not_found(url)
            return (FIXTURES / pages[url]).read_text(encoding="utf-8")

        def fake_get_json(url, params=None):
            calls.append(url)
            return json.loads((FIXTURES / pages[url]).read_text(encoding="utf-8"))

        monkeypatch.setattr(event_pages, "_get_text", fake_get_text)
        monkeypatch.setattr(event_pages, "_get_json", fake_get_json)
        return calls

    return install


def _by_id(items):
    return {item.external_id: item for item in items}


def _assert_event_contract(item):
    meta = item.metadata
    assert EVENT_KEYS <= set(meta)
    assert meta["record_type"] == "event" and meta["status"] == "open" and meta["official"] is True
    assert meta["format"] in {"in_person", "online", "hybrid"}


# -- shared parsing -------------------------------------------------------------

@pytest.mark.parametrize(("text", "start", "end"), [
    ("January 11–14, 2027", date(2027, 1, 11), date(2027, 1, 14)),
    ("March 29–April 01, 2027", date(2027, 3, 29), date(2027, 4, 1)),
    ("October 14 – 16, 2026", date(2026, 10, 14), date(2026, 10, 16)),
    ("January 18 -20, 2027", date(2027, 1, 18), date(2027, 1, 20)),
    ("Oct 27 - Oct 31, 2026", date(2026, 10, 27), date(2026, 10, 31)),
    ("4–6 November 2026", date(2026, 11, 4), date(2026, 11, 6)),
    ("29 November – 3 December 2026", date(2026, 11, 29), date(2026, 12, 3)),
    ("28 December – 2 January 2027", date(2026, 12, 28), date(2027, 1, 2)),
    ("November 16, 2026", date(2026, 11, 16), date(2026, 11, 16)),
    ("31–4 May 2027", None, None),
    ("October, 2028", None, None),
    ("To be announced", None, None),
])
def test_date_ranges_printed_by_organisers(text, start, end):
    from dailydigest.ingest.event_pages import _date_range

    assert _date_range(text) == (start, end)


def test_locations_read_as_city_and_country():
    from dailydigest.ingest.event_pages import _city_country

    assert _city_country("Fairmont Banff Springs, Banff, AB, Canada") == "Banff, Canada"
    assert _city_country("La Jolla, California, USA") == "La Jolla, USA"
    assert _city_country("NCBS, Bengaluru, India") == "Bengaluru, India"
    assert _city_country("Sorbonne Université, 4 Pl. Jussieu, 75005 Paris") == "Sorbonne Université, Paris"
    assert _city_country("Fairmont Singapore, Singapore, Singapore") == "Fairmont Singapore, Singapore"
    assert _city_country("London, UK") == "London, UK"


def test_event_requests_send_a_plain_descriptive_user_agent(monkeypatch):
    """ISSCR's robots.txt refuses AI-crawler agents; the adapter never sends one."""
    from dailydigest.ingest import event_pages

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, params=None):
            return httpx.Response(200, text="{}", request=httpx.Request("GET", url))

    monkeypatch.setattr(event_pages.httpx, "Client", FakeClient)
    event_pages._get_json("https://www.isscr.org/upcoming-events", {"format": "json"})
    agent = seen["headers"]["User-Agent"]
    assert agent == "dailydigest/0.1 (+https://github.com/Antimonyleo/daily_digests)"
    assert not any(name in agent.casefold() for name in ("claude", "gpt", "bot", "anthropic"))
    assert seen["follow_redirects"] is True and seen["timeout"] == 30.0


# -- ISSCR -------------------------------------------------------------------------

def test_isscr_reads_upcoming_events_in_their_local_time_zone(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.ISSCR_URL: "isscr-upcoming.json"})
    items = event_pages.ISSCREventsSource().fetch(_spec(name="ISSCR", kind="isscr_events", lookahead_days=730))
    by_id = _by_id(items)

    assert set(by_id) == {
        "isscr-6a87490f4bd3db1534a3c9fc", "isscr-67872a938d3cfc6f1e58df6d", "isscr-687e8f2718cc6d6904085994",
        "isscr-69d7d52d3d882552c33f167f", "isscr-687e901f9415d133be50a5bb", "isscr-test-kyoto-2027",
    }, "Past and cancelled rows are dropped"
    for item in items:
        _assert_event_contract(item)

    webinar = by_id["isscr-6a87490f4bd3db1534a3c9fc"]
    assert webinar.metadata["format"] == "online" and webinar.metadata["location"] == "Online"
    assert webinar.metadata["event_type"] == "webinar" and webinar.metadata["event_start"] == "2026-11-10"
    assert webinar.url == "https://www.isscr.org/upcoming-events/organoid-models-webinar"

    san_diego = by_id["isscr-67872a938d3cfc6f1e58df6d"].metadata
    assert (san_diego["event_start"], san_diego["event_end"]) == ("2026-12-16", "2026-12-18")
    assert san_diego["location"] == "San Diego, United States" and san_diego["format"] == "in_person"
    assert san_diego["event_type"] == "symposium" and san_diego["deadlines"] == []

    annual = by_id["isscr-687e8f2718cc6d6904085994"].metadata
    assert annual["event_type"] == "", "An unlabelled kind is left for review, not filtered"
    assert by_id["isscr-687e901f9415d133be50a5bb"].metadata["location"] == "København, Denmark"
    # "tz-Asia/Kyoto" is not a zone name; the venue longitude still gives Japan's calendar day.
    assert by_id["isscr-test-kyoto-2027"].metadata["event_start"] == "2027-03-01"


def test_isscr_lookahead_and_broken_listing(serve, monkeypatch):
    from dailydigest.ingest import event_pages

    serve({event_pages.ISSCR_URL: "isscr-upcoming.json"})
    near = event_pages.ISSCREventsSource().fetch(_spec(name="ISSCR", kind="isscr_events", lookahead_days=60))
    assert [item.external_id for item in near] == ["isscr-6a87490f4bd3db1534a3c9fc", "isscr-67872a938d3cfc6f1e58df6d"]

    # A page model without the events collection is a broken listing, not "no events".
    monkeypatch.setattr(event_pages, "_get_json", lambda url, params=None: {"collection": {}})
    with pytest.raises(RuntimeError):
        event_pages.ISSCREventsSource().fetch(_spec(name="ISSCR", kind="isscr_events"))


# -- Keystone -------------------------------------------------------------------------

def test_keystone_reads_listing_and_deadlines_from_meeting_pages(serve):
    from dailydigest.ingest import event_pages

    a1 = "https://www.keystonesymposia.org/conferences/conference-listing/meeting/A12027"
    calls = serve({event_pages.KEYSTONE_URL: "keystone-listing.html", a1: "keystone-meeting-A12027.html"})
    items = event_pages.KeystoneEventsSource().fetch(_spec(name="Keystone Symposia", kind="keystone_events"))
    by_id = _by_id(items)

    assert list(by_id) == ["keystone-J22027", "keystone-A12027", "keystone-C52027"], \
        "Ordered by start; sold-out and out-of-window meetings are dropped"
    for item in items:
        _assert_event_contract(item)

    a1_item = by_id["keystone-A12027"]
    meta = a1_item.metadata
    assert a1_item.url == a1 and a1_item.title == "Cancer Immunotherapy: Bridging Basic Research and Clinical Translation"
    # Struck-through (passed) and hidden deadlines are ignored; the latest open route decides.
    assert meta["deadlines"] == [
        {"type": "early registration", "date": "2026-11-23"},
        {"type": "poster abstract", "date": "2027-01-04"},
    ]
    assert meta["deadline"] == "2027-01-04"
    assert meta["location"] == "Breckenridge, United States" and meta["format"] == "in_person"
    assert meta["event_type"] == "conference" and meta["official_id"] == "A12027"
    assert "Cancer immunotherapy has advanced dramatically" in a1_item.abstract
    assert "Jane Example" not in a1_item.abstract and "@" not in a1_item.abstract

    # Without a meeting page, the listing's calendar link still gives the early-registration date.
    j2 = by_id["keystone-J22027"]
    assert j2.metadata["deadlines"] == [{"type": "early registration", "date": "2026-11-12"}]
    assert j2.metadata["location"] == "Banff, Canada" and j2.metadata["event_type"] == ""
    assert "Joint with: Innate Immunity" in j2.abstract
    assert by_id["keystone-C52027"].metadata["location"] == "Taipei, Taiwan"
    assert by_id["keystone-C52027"].metadata["event_end"] == "2027-04-01"
    assert calls[0] == event_pages.KEYSTONE_URL and len(calls) == 4


def test_keystone_detail_fetches_are_capped(serve, monkeypatch):
    from dailydigest.ingest import event_pages

    calls = serve({event_pages.KEYSTONE_URL: "keystone-listing.html"})
    monkeypatch.setattr(event_pages.KeystoneEventsSource, "MAX_DETAIL_PAGES", 1)
    items = event_pages.KeystoneEventsSource().fetch(_spec(name="Keystone Symposia", kind="keystone_events"))

    assert len(items) == 3 and len(calls) == 2


def test_keystone_without_meeting_rows_raises(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.KEYSTONE_URL: "nature-conferences.html"})
    with pytest.raises(RuntimeError):
        event_pages.KeystoneEventsSource().fetch(_spec(name="Keystone Symposia", kind="keystone_events"))


# -- CSHL -------------------------------------------------------------------------------

def test_cshl_meetings_with_abstract_deadlines(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.CSHL_URL: "cshl-meetings.html"})
    items = event_pages.CSHLEventsSource().fetch(_spec(name="CSHL Meetings", kind="cshl_events"))
    by_id = _by_id(items)

    assert list(by_id) == ["cshl-biome-2026", "cshl-data-2026", "cshl-fusion-2026", "cshl-nat-2027"]
    for item in items:
        _assert_event_contract(item)
    microbiome = by_id["cshl-biome-2026"]
    assert microbiome.metadata["event_start"] == "2026-10-27", "A meeting starting today is kept"
    # A passed abstract deadline does not close the meeting, but it is not offered as a deadline.
    assert microbiome.metadata["deadline"] is None and microbiome.metadata["deadlines"] == []
    assert microbiome.url == "https://meetings.cshl.edu/meetings.aspx?meet=biome"
    nat = by_id["cshl-nat-2027"].metadata
    assert nat["deadline"] == "2027-01-15" and nat["deadlines"] == [{"type": "abstract", "date": "2027-01-15"}]
    assert nat["location"] == "Cold Spring Harbor, United States" and nat["official_id"] == "nat"
    assert by_id["cshl-fusion-2026"].title == "Cell & Membrane Fusion"


def test_cshl_without_meeting_rows_raises(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.CSHL_URL: "keystone-listing.html"})
    with pytest.raises(RuntimeError):
        event_pages.CSHLEventsSource().fetch(_spec(name="CSHL Meetings", kind="cshl_events"))


# -- Nature Conferences -------------------------------------------------------------------

def test_nature_conferences_filters_past_entries_and_reads_labels(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.NATURE_URL: "nature-conferences.html"})
    items = event_pages.NatureConferencesSource().fetch(_spec(name="Nature Conferences", kind="nature_conferences"))
    by_title = {item.title: item for item in items}

    assert set(by_title) == {
        "RNA at the Bench and Bedside V", "Digital Biomarkers",
        "Aging Across Scales: From Biology to Population Health",
    }, "Past and undated (month-only) entries are dropped"
    for item in items:
        _assert_event_contract(item)
    rna = by_title["RNA at the Bench and Bedside V"]
    assert rna.url == "https://natureconferences.streamgo.live/rna-at-the-bench-and-bedside-v/lobby"
    assert rna.metadata["event_start"] == "2026-12-08" and rna.metadata["event_end"] == "2026-12-10"
    assert rna.metadata["location"] == "La Jolla, USA" and rna.metadata["format"] == "in_person"
    assert rna.abstract.startswith("Partner: Ionis Pharmaceuticals")
    online = by_title["Digital Biomarkers"].metadata
    assert online["format"] == "online" and online["event_start"] == "2027-02-03"
    unlinked = by_title["Aging Across Scales: From Biology to Population Health"]
    assert unlinked.url == "https://conferences.nature.com"

    again = event_pages.NatureConferencesSource().fetch(_spec(name="Nature Conferences", kind="nature_conferences"))
    assert [item.external_id for item in again] == [item.external_id for item in items], "Stable ids"
    assert len({item.external_id for item in items}) == 3


def test_nature_conferences_without_entries_raises(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.NATURE_URL: "cshl-meetings.html"})
    with pytest.raises(RuntimeError):
        event_pages.NatureConferencesSource().fetch(_spec(name="Nature Conferences", kind="nature_conferences"))


# -- Wellcome Connecting Science ---------------------------------------------------------------

def test_wellcome_connecting_science_keeps_events_with_an_open_route(serve):
    from dailydigest.ingest import event_pages

    base = "https://coursesandconferences.wellcomeconnectingscience.org/event/"
    serve({
        event_pages.WELLCOME_CS_URL: "wellcome-conferences.html",
        f"{base}genome-informatics-20261202": "wellcome-event-genome-informatics.html",
    })
    items = event_pages.WellcomeConnectingScienceSource().fetch(
        _spec(name="Wellcome Connecting Science", kind="wellcome_cs_events"))
    by_url = {item.url: item for item in items}

    # "Deadlines: all closed" is dropped; Organoids (Sep 2027) is within 365 days.
    assert set(by_url) == {
        f"{base}virus-genomics-evolution-and-bioinformatics-20261111",
        f"{base}genome-informatics-20261202",
        f"{base}organoids-and-beyond-from-development-to-disease-20270920",
    }
    for item in items:
        _assert_event_contract(item)
    genome = by_url[f"{base}genome-informatics-20261202"]
    meta = genome.metadata
    assert meta["deadlines"] == [
        {"type": "in-person registration", "date": "2026-11-02"},
        {"type": "virtual registration", "date": "2026-11-23"},
    ]
    assert meta["deadline"] == "2026-11-23", "The last open route is the last chance to act"
    assert meta["location"] == "Hinxton, United Kingdom" and meta["format"] == "hybrid"
    assert meta["event_type"] == "conference"
    assert genome.abstract.startswith("Methods for analysis of large data sets")
    assert "Jane Example" not in genome.abstract and "@" not in genome.abstract
    assert "Not part of the summary" not in genome.abstract
    assert genome.external_id.startswith("wcs-")

    organoids = by_url[f"{base}organoids-and-beyond-from-development-to-disease-20270920"].metadata
    assert organoids["deadline"] == "2027-09-13" and len(organoids["deadlines"]) == 5
    # No event page was available: format still reflects the virtual route.
    assert organoids["format"] == "hybrid" and organoids["location"] == ""


def test_wellcome_drops_events_whose_routes_have_all_passed(serve, monkeypatch):
    from dailydigest.ingest import event_pages, events_api

    serve({event_pages.WELLCOME_CS_URL: "wellcome-conferences.html"})
    later = date(2026, 11, 3)  # Virus Genomics' only listed route closed on 2 November.
    monkeypatch.setattr(event_pages, "_today", lambda: later)
    monkeypatch.setattr(events_api, "_today", lambda: later)
    items = event_pages.WellcomeConnectingScienceSource().fetch(
        _spec(name="Wellcome Connecting Science", kind="wellcome_cs_events"))

    assert [item.title for item in items] == [
        "Genome Informatics", "Organoids and Beyond: From Development to Disease"]


# -- Controlled Release Society ------------------------------------------------------------------

def test_crs_reads_the_upcoming_events_view(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.CRS_URL: "crs-events.html"})
    items = event_pages.CRSEventsSource().fetch(_spec(name="Controlled Release Society", kind="crs_events"))
    by_title = {item.title: item for item in items}

    assert list(by_title) == [
        "Ocular Delivery Webinar: Scavenging Toxic Aldehydes in Diabetic Retinal Disease",
        "CRS Italy Workshop 2026 Bridging Nanomedicines, Biomaterials and the Immune System",
        "CRS 2027 Annual Meeting & Exposition",
    ], "Cancelled entries are dropped"
    for item in items:
        _assert_event_contract(item)
    webinar = by_title["Ocular Delivery Webinar: Scavenging Toxic Aldehydes in Diabetic Retinal Disease"].metadata
    assert webinar["format"] == "online" and webinar["location"] == "Online" and webinar["event_type"] == "webinar"
    workshop = by_title["CRS Italy Workshop 2026 Bridging Nanomedicines, Biomaterials and the Immune System"].metadata
    assert (workshop["event_start"], workshop["event_end"]) == ("2026-10-29", "2026-10-31")
    assert workshop["location"] == "Aurum - Largo Gardone Riviera, Pescara" and workshop["event_type"] == "workshop"
    annual = by_title["CRS 2027 Annual Meeting & Exposition"]
    # The page shows the society's calendar day, not the UTC timestamp.
    assert annual.metadata["event_start"] == "2027-07-12" and annual.metadata["event_end"] == "2027-07-15"
    assert annual.metadata["location"] == "Toronto, Canada" and annual.metadata["format"] == "in_person"
    assert annual.url == "https://www.controlledreleasesociety.org/events/crs-2027-annual-meeting-exposition"
    assert annual.external_id.startswith("crs-")


def test_crs_without_events_view_raises(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.CRS_URL: "nature-conferences.html"})
    with pytest.raises(RuntimeError):
        event_pages.CRSEventsSource().fetch(_spec(name="Controlled Release Society", kind="crs_events"))


# -- FEBS -----------------------------------------------------------------------------------------

def test_febs_calendar_pages_and_privacy(serve):
    from dailydigest.ingest import event_pages

    calls = serve({
        event_pages.FEBS_URL: "febs-calendar.html",
        f"{event_pages.FEBS_URL}page/2/": "febs-calendar-page2.html",
    })
    items = event_pages.FEBSEventsSource().fetch(_spec(name="FEBS Events", kind="febs_events"))
    by_id = _by_id(items)

    assert calls == [event_pages.FEBS_URL, f"{event_pages.FEBS_URL}page/2/"]
    # "31–4 May 2027" is a typo on the page and is skipped; page 2 is beyond the lookahead.
    assert list(by_id) == ["febs-15173", "febs-15944", "febs-15951"]
    for item in items:
        _assert_event_contract(item)

    barcelona = by_id["febs-15173"]
    assert barcelona.url == "https://premc.org/blood-brain-barrier-conference"
    assert barcelona.metadata["event_start"] == "2026-11-03" and barcelona.metadata["location"] == "Barcelona"
    assert barcelona.metadata["organizer"] == "External organiser (listed by FEBS)"
    assert barcelona.metadata["event_type"] == "conference" and barcelona.metadata["official_id"] == "15173"
    assert "brain barrier research" not in barcelona.abstract, "The sentence naming the chairs is dropped"
    assert "Ada Example" not in barcelona.abstract and "@" not in barcelona.abstract
    assert "global collaboration" in barcelona.abstract

    bengaluru = by_id["febs-15944"]
    assert bengaluru.metadata["location"] == "Bengaluru, India"
    assert "frontiers of synthetic biology" in bengaluru.abstract

    course = by_id["febs-15951"].metadata
    assert course["organizer"] == "FEBS" and course["event_type"] == "course"
    assert course["location"] == "Dresden, Germany" and course["event_end"] == "2027-05-28"


def test_febs_stops_at_a_missing_page_and_raises_on_an_empty_first_page(serve):
    from dailydigest.ingest import event_pages

    calls = serve({event_pages.FEBS_URL: "febs-calendar.html"})
    items = event_pages.FEBSEventsSource().fetch(_spec(name="FEBS Events", kind="febs_events"))
    assert len(items) == 3 and calls[-1].endswith("page/2/")

    serve({event_pages.FEBS_URL: "cshl-meetings.html"})
    with pytest.raises(RuntimeError):
        event_pages.FEBSEventsSource().fetch(_spec(name="FEBS Events", kind="febs_events"))


def test_listing_http_failure_propagates(serve):
    from dailydigest.ingest import event_pages

    serve({})
    with pytest.raises(httpx.HTTPStatusError):
        event_pages.CSHLEventsSource().fetch(_spec(name="CSHL Meetings", kind="cshl_events"))


# -- Engineering Conferences International -------------------------------------------------

def test_place_with_country_names_north_american_venues():
    from dailydigest.ingest.event_pages import _place_with_country

    assert _place_with_country("Estes Park, Colorado") == "Estes Park, United States"
    assert _place_with_country("Princeton Marriott at Forrestal, Princeton, NJ") == "Princeton, United States"
    assert _place_with_country("Montreal, P.Q., Canada") == "Montreal, Canada"
    assert _place_with_country("Yours Truly DC, Washington, DC") == "Washington DC, United States"
    assert _place_with_country("Sheraton Porto Hotel & Spa") == "Sheraton Porto Hotel & Spa"
    assert _place_with_country("British Columbia") == "Canada"
    assert _place_with_country("Cetraro (Calabria), Italy") == "Cetraro (Calabria), Italy"
    assert _place_with_country("Malta") == "Malta" and _place_with_country("") == ""


def test_eci_reads_every_fact_from_the_single_conference_list(serve):
    from dailydigest.ingest import event_pages

    calls = serve({event_pages.ECI_URL: "eci-conferences.html"})
    items = event_pages.ECIEventsSource().fetch(_spec(name="Engineering Conferences International", kind="eci_events"))
    by_id = _by_id(items)

    assert calls == [event_pages.ECI_URL], "One request: no conference page is fetched"
    # Past 2026 meetings, "TBA, 2027", "February 2028", "Date TBA" and 2028 meetings beyond
    # the lookahead are dropped; the navigation menu's links are not entries.
    assert list(by_id) == ["eci-27am", "eci-27ac", "eci-27ak", "eci-27aw", "eci-27ah", "eci-27ae"]
    for item in items:
        _assert_event_contract(item)
        assert item.metadata["organizer"] == "Engineering Conferences International"
        assert item.metadata["event_type"] == "conference" and item.metadata["format"] == "in_person"

    enzyme = by_id["eci-27ae"]
    assert enzyme.title == "Enzyme Engineering XXIX"
    assert enzyme.url == ("https://engconf.org/conferences/biotechnology/"
                          "enzyme-engineering-xxix-bridging-fundamental-discovery-and-industrial-deployment")
    meta = enzyme.metadata
    assert (meta["event_start"], meta["event_end"]) == ("2027-10-17", "2027-10-22")
    assert meta["location"] == "Montreal, Canada" and meta["official_id"] == "27AE"
    assert enzyme.abstract == "Engineering Conferences International · Biotechnology"
    # Chairs are listed under each title; no names reach the item.
    assert all("University" not in item.abstract and "Lewis" not in item.abstract for item in items)

    ceramics = by_id["eci-27ak"]
    assert ceramics.title == "Ultra-High Temperature Ceramics: Materials for Extreme Environment Applications VII"
    assert ceramics.metadata["location"] == "Estes Park, United States"
    assert by_id["eci-27aw"].title == "Micro-Nano Plastics (MNP) in Water: Engineered-Solutions"
    assert by_id["eci-27am"].title.startswith("Microbial Engineering IV – Natural to Synthetic")
    assert by_id["eci-27ah"].metadata["location"] == "Whitefish, USA"


def test_eci_lookahead_and_unrecognisable_page(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.ECI_URL: "eci-conferences.html"})
    items = event_pages.ECIEventsSource().fetch(
        _spec(name="Engineering Conferences International", kind="eci_events", lookahead_days=120))
    assert [item.external_id for item in items] == ["eci-27am"]

    serve({event_pages.ECI_URL: "cshl-meetings.html"})
    with pytest.raises(RuntimeError):
        event_pages.ECIEventsSource().fetch(_spec(name="Engineering Conferences International", kind="eci_events"))


# -- Biophysical Society ------------------------------------------------------------------------

def test_biophysical_society_reads_the_upcoming_events_list(serve):
    from dailydigest.ingest import event_pages

    calls = serve({event_pages.BPS_URL: "bps-events-calendar.html"})
    items = event_pages.BiophysicalSocietyEventsSource().fetch(
        _spec(name="Biophysical Society Meetings", kind="biophysics_events"))

    assert calls == [event_pages.BPS_URL]
    # The 2028 Annual Meeting (no page yet) starts beyond the 365-day lookahead.
    assert [item.title for item in items] == [
        "Molecular Biophysics in East Africa",
        "Oakdale Quantitative Biology Day 2026",
        "Biophysical Society 41st Annual Meeting",
        "Ion Channels: Structure to Medicine",
        "Protein Motion: Experiments Meet Simulation",
    ]
    for item in items:
        _assert_event_contract(item)
        assert item.metadata["organizer"] == "Biophysical Society"
    by_title = {item.title: item for item in items}

    annual = by_title["Biophysical Society 41st Annual Meeting"]
    assert annual.url == "https://bps.example.org/meetings-events/annual-meeting/2027-annual-meeting"
    assert (annual.metadata["event_start"], annual.metadata["event_end"]) == ("2027-02-20", "2027-02-24")
    assert annual.metadata["location"] == "Lakeside, United States"
    assert annual.metadata["event_type"] == "conference"
    assert annual.abstract == "Biophysical Society · Annual Meeting"

    thematic = by_title["Protein Motion: Experiments Meet Simulation"]
    assert thematic.metadata["location"] == "Eastfield, Germany" and thematic.metadata["event_type"] == "conference"
    assert by_title["Ion Channels: Structure to Medicine"].metadata["location"] == "Northbridge, United Kingdom"
    networking = by_title["Oakdale Quantitative Biology Day 2026"].metadata
    assert networking["location"] == "Oakdale, United States"
    assert networking["event_type"] == "", "A networking event's kind is left for review"
    assert by_title["Molecular Biophysics in East Africa"].metadata["location"] == "Riverton, Kenya"


def test_biophysical_society_lookahead_and_unrecognisable_page(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.BPS_URL: "bps-events-calendar.html"})
    items = event_pages.BiophysicalSocietyEventsSource().fetch(
        _spec(name="Biophysical Society Meetings", kind="biophysics_events", lookahead_days=60))
    assert len(items) == 2

    serve({event_pages.BPS_URL: "nature-conferences.html"})
    with pytest.raises(RuntimeError):
        event_pages.BiophysicalSocietyEventsSource().fetch(
            _spec(name="Biophysical Society Meetings", kind="biophysics_events"))


# -- FASEB Science Research Conferences -----------------------------------------------------------

FASEB_AGGREGATION = ("https://www.faseb.org/meetings-and-events/science-research-conferences/"
                     "protein-aggregation-conformer-heterogeneity-2027")


def test_faseb_reads_venue_and_deadlines_from_conference_pages(serve):
    from dailydigest.ingest import event_pages

    calls = serve({event_pages.FASEB_SRC_URL: "faseb-src-listing.html",
                   FASEB_AGGREGATION: "faseb-src-protein-aggregation.html"})
    items = event_pages.FASEBConferencesSource().fetch(
        _spec(name="FASEB Science Research Conferences", kind="faseb_src_events"))
    by_url = {item.url: item for item in items}

    # The registration-platform page is not read; a missing page leaves the listing facts.
    assert calls[0] == event_pages.FASEB_SRC_URL and len(calls) == 4
    assert not any("events.faseb.org" in url for url in calls)
    assert len(items) == 4
    for item in items:
        _assert_event_contract(item)
        assert item.metadata["organizer"] == "FASEB" and item.metadata["event_type"] == "conference"

    aggregation = by_url[FASEB_AGGREGATION]
    meta = aggregation.metadata
    assert aggregation.title == "Protein Aggregation: Conformer Heterogeneity"
    assert (meta["event_start"], meta["event_end"]) == ("2027-07-19", "2027-07-22")
    assert meta["location"] == "Princeton, United States" and meta["format"] == "in_person"
    assert meta["deadlines"] == [
        {"type": "early registration and early abstract", "date": "2027-05-24"},
        {"type": "abstract edit & submission/registration cancellation", "date": "2027-06-29"},
        {"type": "regular registration", "date": "2027-07-12"},
    ]
    assert meta["deadline"] == "2027-07-12"
    assert "protein aggregation and protein misfolding" in aggregation.abstract

    phosphatases = by_url["https://events.faseb.org/d/zxq1ym"].metadata
    assert phosphatases["event_start"] == "2026-12-14" and phosphatases["location"] == ""
    assert phosphatases["deadlines"] == []


def test_faseb_detail_fetches_are_capped_and_lookahead_applies(serve, monkeypatch):
    from dailydigest.ingest import event_pages

    calls = serve({event_pages.FASEB_SRC_URL: "faseb-src-listing.html"})
    monkeypatch.setattr(event_pages.FASEBConferencesSource, "MAX_DETAIL_PAGES", 1)
    items = event_pages.FASEBConferencesSource().fetch(
        _spec(name="FASEB Science Research Conferences", kind="faseb_src_events"))
    assert len(items) == 4 and len(calls) == 2

    calls.clear()
    items = event_pages.FASEBConferencesSource().fetch(
        _spec(name="FASEB Science Research Conferences", kind="faseb_src_events", lookahead_days=90))
    assert [item.url for item in items] == ["https://events.faseb.org/d/zxq1ym"] and len(calls) == 1


def test_faseb_without_conference_cards_raises(serve):
    from dailydigest.ingest import event_pages

    serve({event_pages.FASEB_SRC_URL: "cshl-meetings.html"})
    with pytest.raises(RuntimeError):
        event_pages.FASEBConferencesSource().fetch(
            _spec(name="FASEB Science Research Conferences", kind="faseb_src_events"))
