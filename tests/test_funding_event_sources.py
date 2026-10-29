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
    monkeypatch.setattr(eu_funding, "_search", lambda page, programme, division="": calls.append((page, programme)) or pages[page])
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
    ("United States", {"us"}), ("USA", {"us"}), ("Germany", {"de", "eu"}), ("United Kingdom", {"gb", "eu"}),
    ("Switzerland", {"ch", "eu"}), ("Japan", {"jp"}), ("DE", {"de", "eu"}), ("Narnia", None), ("", None),
    ("U.S.", {"us"}), ("U.S.A.", {"us"}), ("U.K.", {"gb", "eu"}), (" u.s ", {"us"}),
])
def test_country_maps_to_funding_regions(country, regions):
    from dailydigest.opportunities import funding_regions
    assert funding_regions(country) == regions


@pytest.mark.parametrize("country,expected", [
    ("Germany", {"EU Funding & Tenders", "PubMed", "EMBO Events", "MSCA Postdoctoral Fellowships"}),
    ("United States", {"Grants.gov", "NIH Guide", "PubMed", "EMBO Events", "MSCA Postdoctoral Fellowships"}),
    ("Japan", {"PubMed", "EMBO Events", "MSCA Postdoctoral Fellowships"}),
    ("Narnia", {"Grants.gov", "NIH Guide", "EU Funding & Tenders", "PubMed", "EMBO Events",
                "MSCA Postdoctoral Fellowships"}),
])
def test_reader_fetches_the_funders_for_their_country(monkeypatch, country, expected):
    from dailydigest import opportunities, pipeline

    specs = [
        _spec(name="Grants.gov", kind="grants_gov", applicant_regions=["us"]),
        _spec(name="NIH Guide", kind="nih_guide", applicant_regions=["us"]),
        _spec(name="EU Funding & Tenders", kind="eu_funding", applicant_regions=["eu"]),
        _spec(name="PubMed", kind="pubmed", section="research"),
        _spec(name="EMBO Events", kind="embo_events", section="events"),
        _spec(name="MSCA Postdoctoral Fellowships", kind="eu_funding", section="events", calls=True,
              applicant_regions=["international"]),
    ]
    monkeypatch.setattr(opportunities, "load_opportunity_profile", lambda path: SimpleNamespace(country=country))
    chosen = pipeline._reader_sources(specs, SimpleNamespace(opportunity_profile_path="unused"))
    assert {spec.name for spec in chosen} == expected


def test_website_leaves_out_local_only_sources(monkeypatch):
    import sys

    from dailydigest import pipeline

    specs = [_spec(name="CRS", kind="crs_events", section="events"),
             _spec(name="Biophysical Society Meetings", kind="biophysics_events", section="events",
                   local_only=True)]
    assert pipeline._reader_sources(specs, SimpleNamespace(opportunity_profile_path="x")) == specs
    monkeypatch.setattr(sys, "platform", "emscripten")
    chosen = pipeline._reader_sources(specs, SimpleNamespace(opportunity_profile_path="x"))
    assert [spec.name for spec in chosen] == ["CRS"]


def test_missing_opportunity_profile_keeps_every_funder(monkeypatch):
    from dailydigest import opportunities, pipeline

    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(opportunities, "load_opportunity_profile", missing)
    specs = [_spec(name="Grants.gov", kind="grants_gov", applicant_regions=["us"])]
    assert pipeline._reader_sources(specs, SimpleNamespace(opportunity_profile_path="x")) == specs


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


# -- Fellowships and awards are calls in Events & Calls --------------------------

def _settings(**sections):
    return SimpleNamespace(include_opportunities=sections.get("opportunities", True),
                           include_events=sections.get("events", True))


@pytest.mark.parametrize("opportunities_on,events_on,section", [
    (True, True, "events"), (True, False, "opportunities"), (False, True, "events"), (False, False, None),
])
def test_call_sources_follow_the_sections_the_reader_turned_on(monkeypatch, opportunities_on, events_on, section):
    from dailydigest import pipeline

    monkeypatch.setattr(pipeline, "_section_enabled",
                        lambda name, settings=None: {"opportunities": opportunities_on, "events": events_on}.get(name, True))
    spec = _spec(name="Fellowships", kind="grants_gov", section="events", calls=True)
    placed = pipeline._call_section(spec)
    assert (placed.section if placed else None) == section
    assert pipeline._call_section(_spec(name="Grants", kind="grants_gov")) .section == "opportunities"


@pytest.mark.parametrize("events_on,section", [(True, "events"), (False, "opportunities")])
def test_fellowships_and_awards_from_funding_sources_become_calls(monkeypatch, events_on, section):
    from dailydigest import pipeline
    from dailydigest.models import Item

    monkeypatch.setattr(pipeline, "_section_enabled", lambda name, settings=None: events_on if name == "events" else True)
    spec = _spec(name="NIH Guide", kind="nih_guide")
    items = [Item(source="NIH Guide", section="opportunities", external_id=number, title=number, url=f"https://x/{number}",
                  metadata={"opportunity_type": kind}) for number, kind in (("F32", "fellowship"), ("R01", "grant"), ("PRIZE", "award"))]
    routed = {item.external_id: item for item in pipeline._route_calls(spec, items)}
    assert routed["R01"].section == "opportunities" and "record_type" not in routed["R01"].metadata
    for number in ("F32", "PRIZE"):
        assert routed[number].section == section and routed[number].metadata["record_type"] == "call"
    event = Item(source="EMBO", section="events", external_id="e", title="e", url="https://x/e",
                 metadata={"record_type": "event", "opportunity_type": "fellowship"})
    assert pipeline._route_calls(spec, [event])[0].metadata["record_type"] == "event"


def test_calls_for_other_countries_are_not_shown_but_international_ones_are():
    from dailydigest.opportunities import OpportunityProfile, assess_opportunity

    reader = OpportunityProfile(career_stage="postdoc", institution_type="university", country="Japan", applicant_role="fellow")
    uk_only = {"record_type": "call", "opportunity_type": "fellowship", "applicant_regions": ["gb"]}
    anywhere = {**uk_only, "applicant_regions": ["international"]}
    assert assess_opportunity(uk_only, reader, today=TODAY).actionable is False
    assert assess_opportunity(anywhere, reader, today=TODAY).actionable is True
    assert assess_opportunity({**uk_only, "applicant_regions": ["jp"]}, reader, today=TODAY).actionable is True


def test_msca_fellowship_source_keeps_individual_fellowships_only(monkeypatch):
    from dailydigest.ingest import eu_funding

    page = {"totalResults": 3, "results": [
        _eu_topic("HORIZON-MSCA-2027-PF-01-01", "MSCA Postdoctoral Fellowships 2027", ["2027-09-08"], start="2027-04-07"),
        _eu_topic("HORIZON-MSCA-2027-DN-01-01", "MSCA Doctoral Networks 2027", ["2027-09-23"]),
        _eu_topic("HORIZON-MSCA-2027-SE-01-01", "MSCA Staff Exchanges 2027", ["2027-04-15"]),
    ]}
    seen = []
    monkeypatch.setattr(eu_funding, "_search", lambda page_no, programme, division="": seen.append(division) or page)
    monkeypatch.setattr(eu_funding, "_today", lambda: TODAY)
    spec = _spec(name="MSCA Postdoctoral Fellowships", kind="eu_funding", section="events", calls=True,
                 condition="msca_fellowships", lookahead_days=365)
    items = eu_funding.EUFundingSource().fetch(spec)
    assert [item.external_id for item in items] == ["HORIZON-MSCA-2027-PF-01-01"]
    assert items[0].metadata["opportunity_type"] == "fellowship" and seen == [eu_funding.MSCA]


# -- Hand-kept fellowships, awards and meetings --------------------------------

def test_curated_calls_state_only_official_deadlines_and_their_countries(monkeypatch):
    from dailydigest.ingest import curated
    from dailydigest.opportunities import opportunity_display

    rows = [
        {"id": "hfsp-postdoc", "title": "HFSP Postdoctoral Fellowships", "funder": "HFSP", "type": "fellowship",
         "url": "https://www.hfsp.org/funding/postdoctoral-fellowships", "applicant_regions": ["international"],
         "career_stage": ["postdoc"], "summary": "Frontier life-science research abroad.", "cycle": "annual",
         "verified_on": "2026-10-28"},
        {"id": "ukri-flf", "title": "UKRI Future Leaders Fellowships", "funder": "UKRI", "type": "fellowship",
         "url": "https://www.ukri.org/opportunity/future-leaders-fellowships/", "applicant_regions": ["gb"],
         "next_deadline": "2027-02-01", "eligibility_note": "Hosted at a UK research organisation."},
        {"id": "leverhulme-ecf", "title": "Leverhulme Early Career Fellowships", "url": "https://www.leverhulme.ac.uk/x",
         "applicant_regions": ["gb"], "next_deadline": "2027-02-18", "deadline_evidence": "search_snippet: closing-dates table"},
        {"id": "old-round", "title": "Past round", "url": "https://x.example.org", "next_deadline": "2026-01-01"},
        {"id": "", "title": "No id", "url": "https://x.example.org"},
    ]
    monkeypatch.setattr(curated, "_load", lambda name: rows)
    monkeypatch.setattr(curated, "_today", lambda: TODAY)
    items = curated.CuratedCallsSource().fetch(_spec(name="Fellowships and awards", kind="curated_calls",
                                                     section="events", calls=True, lookahead_days=365))
    by_id = {item.external_id: item for item in items}
    assert set(by_id) == {"curated-hfsp-postdoc", "curated-ukri-flf", "curated-old-round", "curated-leverhulme-ecf"}
    assert by_id["curated-leverhulme-ecf"].metadata["deadline"] is None, "Only deadlines read on the official page"
    hfsp = by_id["curated-hfsp-postdoc"].metadata
    assert hfsp["status"] == "recurring" and hfsp["deadline"] is None and hfsp["applicant_regions"] == ["international"]
    assert opportunity_display(hfsp)["deadline"] == "Recurring call; see the official page for this round"
    assert opportunity_display(hfsp)["kind"] == "call"
    ukri = by_id["curated-ukri-flf"].metadata
    assert ukri["deadline"] == "2027-02-01" and ukri["status"] == "open" and ukri["applicant_regions"] == ["gb"]
    assert by_id["curated-old-round"].metadata["deadline"] is None, "A past round leaves the recurring scheme"
    assert hfsp["official"] is True and hfsp["verified_at"].startswith("2026-10-28")
    assert ukri["official"] is False and ukri["verified_at"] is None, "No check date: not shown as verified"


def test_curated_meetings_keep_upcoming_dates_only(monkeypatch):
    from dailydigest.ingest import curated

    rows = [
        {"id": "grc-drug-carriers-2027", "title": "Drug Carriers in Medicine and Biology", "organizer": "Gordon Research Conferences",
         "url": "https://www.grc.org/drug-carriers-in-medicine-and-biology-conference/2027/", "start": "2027-08-08",
         "end": "2027-08-13", "location": "Waterville Valley, NH, United States", "format": "in_person",
         "event_type": "conference", "deadline": "2027-07-11", "deadline_type": "application", "topics": "Drug delivery.",
         "evidence_kind": "page", "verified_on": "2026-10-28"},
        {"id": "mrs-spring-2027", "title": "MRS Spring Meeting", "organizer": "Materials Research Society",
         "url": "https://www.mrs.org/meetings-events/spring-meetings-exhibits/2027-mrs-spring-meeting", "start": "2027-04-05",
         "end": "2027-04-09", "location": "Seattle, United States", "deadline": "2026-11-01", "evidence_kind": "search_snippet"},
        {"id": "past", "title": "Past meeting", "url": "https://x.example.org", "start": "2026-06-01"},
    ]
    monkeypatch.setattr(curated, "_load", lambda name: rows)
    monkeypatch.setattr(curated, "_today", lambda: TODAY)
    items = curated.CuratedEventsSource().fetch(_spec(name="Flagship meetings", kind="curated_events", section="events",
                                                      lookahead_days=548))
    assert [item.external_id for item in items] == ["curated-grc-drug-carriers-2027", "curated-mrs-spring-2027"]
    snippet = items[1].metadata
    assert snippet["official"] is False and snippet["deadline"] is None, "Search-result dates say Verify source"
    meta = items[0].metadata
    assert (meta["event_start"], meta["deadline"], meta["event_type"]) == ("2027-08-08", "2027-07-11", "conference")
    assert meta["deadlines"] == [{"type": "application", "date": "2027-07-11"}]
    # Hand-kept facts carry the date they were checked, never the brew time.
    assert meta["official"] is True and meta["verified_at"] == "2026-10-28T00:00:00+00:00"
    assert snippet["verified_at"] is None


def test_source_access_issue_lists_only_public_source_lines(monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("report_failure", Path(__file__).resolve().parents[1] / "browser/report_failure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("DAILYDIGEST_ACCESS_SUMMARY", "Browser source access from x\n- Grants.gov: cors_blocked\nstray log line")
    body = module._access_body("https://github.com/o/r/actions/runs/1")
    assert "- Grants.gov: cors_blocked" in body and "stray log line" not in body


def test_calls_render_funding_facts_inside_events_and_calls(tmp_path, monkeypatch):
    import json
    import re

    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web
    from tests.test_web import _request, _text_payload

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    monkeypatch.setenv("INCLUDE_EVENTS", "true")
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    monkeypatch.setattr(web, "SETTINGS", config_mod.SETTINGS)
    store_mod._ENGINE = store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2027-01-05")
    store_mod.init_db()
    records = {
        "conference": {"record_type": "event", "event_start": "2027-03-02", "event_end": "2027-03-05",
                       "location": "Bangalore, India", "format": "in_person", "status": "open", "official": True},
        "fellowship": {"record_type": "call", "opportunity_type": "fellowship", "status": "recurring",
                       "official": True, "applicant_regions": ["international"]},
    }
    features = []
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2027-01-05", item_count=2))
        for rank, (name, metadata) in enumerate(records.items(), start=1):
            row = store_mod.ItemRow(source="Example", section="events", external_id=name, url=f"https://example.com/{name}",
                                    title=f"{name} pick", abstract="A fictional abstract.", score=0.5, digest_id="2027-01-05",
                                    item_label=f"E{rank}", metadata_json=json.dumps(metadata))
            s.add(row)
            s.flush()
            features.append((f"E{rank}", int(row.id), 0.5, {"score": 0.5, "rank_score": 0.5, "confidence_score": 0.5,
                                                            "why_shown": [], "tags": []}))
    store_mod.write_digest_features("2027-01-05", features)
    text = _text_payload(web.index(_request("GET", "/")))
    kinds = re.findall(r'data-opportunity-kind="(\w+)"', text)
    assert sorted(kinds) == ["call", "event"]
    call = text[text.index('data-opportunity-kind="call"'):]
    call = call[:call.index("</div>")]
    assert "Amount:" in call and "Recurring call; see the official page for this round" in call and "Official call" in call
    event = text[text.index('data-opportunity-kind="event"'):]
    assert "Where:" in event[:event.index("</div>")]


@pytest.mark.parametrize("cap,calls_shown", [(1, 0), (2, 1), (5, 2), (10, 4)])
def test_calls_get_about_forty_percent_of_events_and_calls(cap, calls_shown):
    import json
    from types import SimpleNamespace

    from dailydigest.rank.ranker import _pick_events_and_calls

    def row(n, kind):
        return SimpleNamespace(id=n, source=f"S{n}", metadata_json=json.dumps({"record_type": kind}))

    meetings = [(row(n, "event"), 0.9 - n / 100) for n in range(10)]
    calls = [(row(100 + n, "call"), 0.3 - n / 100) for n in range(10)]
    picked = _pick_events_and_calls(meetings + calls, cap)
    assert len(picked) == cap
    assert sum(1 for item, _score in picked if item.id >= 100) == calls_shown


def test_health_and_ingest_judge_the_same_sources_in_their_filled_sections(monkeypatch):
    from types import SimpleNamespace

    from dailydigest import pipeline

    specs = [_spec(name="Grants.gov", kind="grants_gov"),
             _spec(name="MSCA Postdoctoral Fellowships", kind="eu_funding", section="events", calls=True),
             _spec(name="EMBO Events", kind="embo_events", section="events")]
    monkeypatch.setattr(pipeline, "load_sources", lambda: specs)
    funding_only = SimpleNamespace(include_opportunities=True, include_events=False,
                                   opportunity_profile_path="unused")
    chosen = {spec.name: spec.section for spec in pipeline._fetched_sources(funding_only)}
    # With Events off, fellowships are fetched into Funding, so their health counts.
    assert chosen == {"Grants.gov": "opportunities", "MSCA Postdoctoral Fellowships": "opportunities"}
