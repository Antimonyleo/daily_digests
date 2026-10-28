"""EU and NIH funding, EMBL and EMBO events, and matching funders to a reader's country."""
from __future__ import annotations

import hashlib
from datetime import date
from types import SimpleNamespace

import pytest

from dailydigest.models import SourceSpec

TODAY = date(2026, 10, 27)


def _spec(**kwargs) -> SourceSpec:
    return SourceSpec(**{"name": "Test", "kind": "test", "section": "opportunities", **kwargs})


# -- EU Funding & Tenders -----------------------------------------------------

def _eu_topic(identifier, title, deadlines, start="2026-09-01", **extra):
    return {"metadata": {
        "identifier": [identifier], "title": [title], "startDate": [f"{start}T00:00:00.000+0000"],
        "deadlineDate": [f"{d}T17:00:00.000+0000" for d in deadlines],
        "url": [f"https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/{identifier}"],
        "status": ["31094502"], **{key: [value] for key, value in extra.items()}}}


def test_eu_topics_keep_only_future_deadlines_and_official_pages(monkeypatch):
    from dailydigest.ingest import eu_funding

    pages = {
        1: {"totalResults": 150, "results": [
            # The portal still marks this 2023 topic "open"; its deadline decides.
            _eu_topic("ERC-2023-POC", "ERC Proof of Concept", ["2023-03-16"]),
            _eu_topic("HORIZON-MSCA-2027-PF-01-01", "MSCA Postdoctoral Fellowships",
                      ["2027-09-09"], start="2027-04-08", typesOfAction="HORIZON TMA MSCA Postdoctoral Fellowships"),
            _eu_topic("HORIZON-CL4-2027-02-MAT-PROD-32", "Advanced materials for process industries",
                      ["2026-10-01", "2027-02-02", "2027-09-02"], deadlineModel="two-stage", callTitle="INDUSTRY"),
        ]},
        2: {"totalResults": 150, "results": [
            _eu_topic("HORIZON-CL4-2027-02-MAT-PROD-32", "Duplicate copy", ["2027-02-02"]),
        ]},
    }
    calls = []
    monkeypatch.setattr(eu_funding, "_search", lambda page, programme: calls.append((page, programme)) or pages[page])
    monkeypatch.setattr(eu_funding, "_today", lambda: TODAY)

    items = eu_funding.EUFundingSource().fetch(_spec(name="EU Funding & Tenders", kind="eu_funding", lookahead_days=400))

    assert calls == [(1, eu_funding.HORIZON_EUROPE), (2, eu_funding.HORIZON_EUROPE)]
    by_id = {item.external_id: item for item in items}
    assert set(by_id) == {"HORIZON-MSCA-2027-PF-01-01", "HORIZON-CL4-2027-02-MAT-PROD-32"}
    fellowship = by_id["HORIZON-MSCA-2027-PF-01-01"].metadata
    assert fellowship["opportunity_type"] == "fellowship" and fellowship["status"] == "forthcoming"
    two_stage = by_id["HORIZON-CL4-2027-02-MAT-PROD-32"]
    assert two_stage.title == "Advanced materials for process industries"
    assert two_stage.metadata["deadline"] == "2027-02-02", "The next open stage is the actionable deadline"
    assert [row["date"] for row in two_stage.metadata["deadlines"]] == ["2027-02-02", "2027-09-02"]
    assert two_stage.url.endswith("/topic-details/HORIZON-CL4-2027-02-MAT-PROD-32")
    assert two_stage.metadata["official"] is True and two_stage.metadata["eligibility_tags"] == []


def test_eu_search_is_a_cors_simple_multipart_request(monkeypatch):
    import httpx

    from dailydigest.ingest import eu_funding

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, **kwargs):
            seen.update(kwargs, url=url)
            return httpx.Response(200, json={"totalResults": 0, "results": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(eu_funding.httpx, "Client", FakeClient)
    eu_funding._search(1, eu_funding.HORIZON_EUROPE)
    assert set(seen["files"]) == {"query", "languages", "displayFields", "sort"}
    assert "Content-Type" not in seen["headers"], "httpx sets multipart/form-data, which needs no preflight"
    assert seen["params"]["apiKey"] == "SEDIA"


# -- NIH Guide ----------------------------------------------------------------

def _nih(docnum, doctype="PAR", **source):
    return {"_source": {"docnum": docnum, "doctype": doctype, "title": f"Notice {docnum}",
                        "expdate": "2027-11-16T00:00:00.000Z", "opendate": "2025-04-19T00:00:00.000Z",
                        "reldate": "2024-12-16T00:00:00.000Z", "purpose": "<p>Nanotechnology cancer interventions</p>",
                        "primaryIC": "NCI", "ac": ["R01"], **source}}


def test_nih_announcements_link_officially_and_keep_due_dates_honest(monkeypatch):
    from dailydigest.ingest import nih_guide
    from dailydigest.opportunities import opportunity_display

    hits = [
        _nih("PAR-25-336", filename="PAR-25-336.html", appreceiptdate="Multiple dates, see announcement."),
        _nih("RFA-AI-28-031", doctype="RFA", filename=None, ggid="361050", appreceiptdate="2027-05-29T00:00:00.000Z",
             opendate="2027-04-28T00:00:00.000Z"),
        _nih("PA-25-100", doctype="PA", filename="PA-25-100.html", ac=["F32"]),
        _nih("NOT-OD-26-001", doctype="NOT", filename="NOT-OD-26-001.html"),
        _nih("PAR-24-001", filename="PAR-24-001.html", expdate="2026-01-07T00:00:00.000Z"),
        _nih("PAR-24-002"),  # No official page at all.
    ]
    monkeypatch.setattr(nih_guide, "_search", lambda term, rows, today: {"data": {"hits": {"hits": hits}}})
    monkeypatch.setattr(nih_guide, "_today", lambda: TODAY)

    items = nih_guide.NIHGuideSource().fetch(_spec(name="NIH Guide", kind="nih_guide", query="nanoparticle", lookahead_days=400))

    by_id = {item.external_id: item for item in items}
    assert set(by_id) == {"PAR-25-336", "RFA-AI-28-031", "PA-25-100"}
    par = by_id["PAR-25-336"]
    assert par.url == "https://grants.nih.gov/grants/guide/pa-files/PAR-25-336.html"
    assert par.metadata["deadline"] is None and par.metadata["expires"] == "2027-11-16"
    assert opportunity_display(par.metadata)["deadline"] == "See the announcement for due dates (expires 2027-11-16)"
    assert par.abstract == "Nanotechnology cancer interventions"
    rfa = by_id["RFA-AI-28-031"]
    assert rfa.url == "https://www.grants.gov/search-results-detail/361050"
    assert rfa.metadata["deadline"] == "2027-05-29" and rfa.metadata["status"] == "forthcoming"
    assert by_id["PA-25-100"].metadata["opportunity_type"] == "fellowship"


# -- EMBL and EMBO events -----------------------------------------------------

def _embl(post_id, start, end=None, **fields):
    row = {"id": post_id,
           "guid": {"rendered": f"https://www.embl.org/about/info/course-and-conference-office/?post_type=vf_event&#038;p={post_id}"},
           "link": f"https://www.embl.org/about/info/course-and-conference-office/events/ees{post_id}/",
           "title": {"rendered": f"Chemical biology &amp; {post_id}"},
           "vf_event_start_date": {"value": start}, "vf_event_end_date": {"value": end or start},
           "vf_event_summary": {"value": "<p>Chemical tools for biology.</p>"},
           "vf_event_canceled": {"value": "no"}, "vf_event_event_type": {"value": "conference"},
           "vf_event_location": {"value": ["67", "364"], "formatted": ["EMBL Heidelberg", "Virtual"]},
           "vf_event_venue": {"value": "EMBL Advanced Training Centre"}}
    row.update({key: {"value": value} for key, value in fields.items()})
    return row


def test_embl_listing_keeps_saved_event_identity_and_reads_every_route(monkeypatch):
    from dailydigest.ingest import events_api

    page = [
        _embl(1, "20270215", "20270218", vf_event_submission_closing="20261101",
              vf_event_registration_closing="20270110"),
        _embl(2, "20270301", vf_event_canceled="yes"),
        _embl(3, "20270401", vf_event_registration_closing="20261001"),
        _embl(4, "20251001"),
    ]
    requests = []
    monkeypatch.setattr(events_api, "_get_json", lambda url, params=None: requests.append(params) or page)
    monkeypatch.setattr(events_api, "_today", lambda: TODAY)

    items = events_api.EMBLEventsSource().fetch(_spec(name="EMBL Events", kind="embl_events", section="events", lookahead_days=365))

    assert [item.title for item in items] == ["Chemical biology & 1"]
    event = items[0]
    # Same hash of the same guid the RSS adapter stored, so ratings and saves carry over.
    guid = "https://www.embl.org/about/info/course-and-conference-office/?post_type=vf_event&p=1"
    assert event.external_id == hashlib.sha1(guid.encode()).hexdigest()[:16]
    assert event.url == "https://www.embl.org/about/info/course-and-conference-office/events/ees1"
    meta = event.metadata
    assert (meta["event_start"], meta["event_end"], meta["deadline"]) == ("2027-02-15", "2027-02-18", "2027-01-10")
    assert meta["format"] == "hybrid" and meta["event_type"] == "conference"
    assert "Germany" in meta["location"]
    assert len(requests) == 1, "A short page is the last page"


def test_embo_events_drop_contacts_and_closed_registration(monkeypatch):
    from dailydigest.ingest import events_api

    rows = [
        {"event_id": "ws27-01", "title": "RNA nanotechnology workshop", "start_date": "2027-03-02",
         "end_date": "2027-03-05", "event_type": "EMBO Workshop", "city": "Bangalore", "country": "India",
         "website": "https://meetings.example.org/rna/", "event_medium": "in-person", "deadline_date": "2027-01-15",
         "registration_status": "open", "topic_areas": "Chemical Biology | RNA", "event_keywords": "RNA origami",
         "contact_name": "Private Person", "contact_email": "person@example.org"},
        {"event_id": "pc27-02", "title": "Virtual course", "start_date": "2027-02-01", "event_type": "EMBO Training",
         "website": "https://embo.example.org/course", "event_medium": "virtual", "registration_status": None},
        {"event_id": "ws27-03", "title": "Closed workshop", "start_date": "2027-04-01",
         "website": "https://x.example.org", "registration_status": "closed"},
    ]
    monkeypatch.setattr(events_api, "_get_json", lambda url, params=None: {"data": rows, "page": 1})
    monkeypatch.setattr(events_api, "_today", lambda: TODAY)

    items = events_api.EMBOEventsSource().fetch(_spec(name="EMBO Events", kind="embo_events", section="events", lookahead_days=365))

    assert [item.external_id for item in items] == ["embo-ws27-01", "embo-pc27-02"]
    workshop, course = items
    assert workshop.metadata["location"] == "Bangalore, India" and workshop.metadata["event_type"] == "workshop"
    assert workshop.metadata["deadline"] == "2027-01-15"
    flattened = repr(workshop.model_dump())
    assert "person@example.org" not in flattened and "Private Person" not in flattened
    assert course.metadata["format"] == "online" and course.metadata["event_type"] == "course"


# -- Funders matched to the reader's country ------------------------------------

@pytest.mark.parametrize("country,regions", [
    ("United States", {"us"}), ("USA", {"us"}), ("Germany", {"eu"}), ("United Kingdom", {"eu"}),
    ("Switzerland", {"eu"}), ("Japan", None), ("", None),
])
def test_country_maps_to_funding_regions(country, regions):
    from dailydigest.opportunities import funding_regions
    assert funding_regions(country) == regions


@pytest.mark.parametrize("country,expected", [
    ("Germany", {"EU Funding & Tenders", "PubMed", "EMBO Events"}),
    ("United States", {"Grants.gov", "NIH Guide", "PubMed", "EMBO Events"}),
    ("Japan", {"Grants.gov", "NIH Guide", "EU Funding & Tenders", "PubMed", "EMBO Events"}),
])
def test_reader_fetches_the_funders_for_their_country(monkeypatch, country, expected):
    from dailydigest import opportunities, pipeline

    specs = [
        _spec(name="Grants.gov", kind="grants_gov", applicant_regions=["us"]),
        _spec(name="NIH Guide", kind="nih_guide", applicant_regions=["us"]),
        _spec(name="EU Funding & Tenders", kind="eu_funding", applicant_regions=["eu"]),
        _spec(name="PubMed", kind="pubmed", section="research"),
        _spec(name="EMBO Events", kind="embo_events", section="events"),
    ]
    monkeypatch.setattr(opportunities, "load_opportunity_profile", lambda path: SimpleNamespace(country=country))
    chosen = pipeline._reader_funding_sources(specs, SimpleNamespace(opportunity_profile_path="unused"))
    assert {spec.name for spec in chosen} == expected


def test_missing_opportunity_profile_keeps_every_funder(monkeypatch):
    from dailydigest import opportunities, pipeline

    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(opportunities, "load_opportunity_profile", missing)
    specs = [_spec(name="Grants.gov", kind="grants_gov", applicant_regions=["us"])]
    assert pipeline._reader_funding_sources(specs, SimpleNamespace(opportunity_profile_path="x")) == specs


def test_event_regions_match_whole_words():
    from dailydigest.opportunities import _region_matches
    assert _region_matches(["Europe"], "Lisbon, Portugal")
    assert _region_matches(["Asia"], "Bangalore, India")
    assert not _region_matches(["Europe"], "Fukuoka, Japan")
    assert not _region_matches(["Asia"], "Indianapolis, Indiana")


def test_configured_sources_load_with_their_adapters():
    from dailydigest.config import load_sources
    from dailydigest.ingest.base import dispatch_source

    specs = {spec.name: spec for spec in load_sources("config/sources.yaml")}
    for name, kind in {"NIH Guide": "nih_guide", "EU Funding & Tenders": "eu_funding",
                       "EMBL Events": "embl_events", "EMBO Events": "embo_events"}.items():
        assert specs[name].kind == kind
        assert dispatch_source(specs[name]) is not None
    assert specs["Grants.gov"].applicant_regions == ["us"]
    assert specs["EU Funding & Tenders"].applicant_regions == ["eu"]


def test_same_notice_or_event_from_two_providers_is_one_pick():
    from dailydigest.dedupe import candidate_identity_keys
    from dailydigest.models import Item

    nih = Item(source="NIH Guide", section="opportunities", external_id="PAR-25-336", title="TTNCI (R01)",
               url="https://grants.nih.gov/grants/guide/pa-files/PAR-25-336.html", metadata={"official_id": "PAR-25-336"})
    grants = Item(source="Grants.gov", section="opportunities", external_id="PAR-25-336", title="TTNCI R01",
                  url="https://www.grants.gov/search-results-detail/357748", metadata={"official_id": "PAR-25-336"})
    assert set(candidate_identity_keys(nih)) & set(candidate_identity_keys(grants))
    embo = Item(source="EMBO Events", section="events", external_id="embo-ees27-01", title="Chemical Biology 2027",
                url="https://www.embl.org/events/ees27-01/", metadata={"record_type": "event", "event_start": "2027-09-01"})
    embl = Item(source="EMBL Events", section="events", external_id="abc", title="Chemical biology 2027",
                url="https://www.embl.org/about/info/course-and-conference-office/events/ees27-01",
                metadata={"record_type": "event", "event_start": "2027-09-01"})
    assert set(candidate_identity_keys(embo)) & set(candidate_identity_keys(embl))
    other_year = embl.model_copy(update={"metadata": {"record_type": "event", "event_start": "2028-09-01"}})
    assert not set(candidate_identity_keys(embo)) & set(candidate_identity_keys(other_year))
