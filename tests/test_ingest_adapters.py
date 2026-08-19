from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from dailydigest.models import SourceSpec


def _spec(**kwargs) -> SourceSpec:
    defaults = {"name": "Test", "kind": "test", "section": "research"}
    return SourceSpec(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# ClinicalTrials — pagination
# ---------------------------------------------------------------------------


def test_clinicaltrials_follows_next_page_token(monkeypatch):
    """Pagination must follow nextPageToken until exhausted."""
    from dailydigest.ingest.clinicaltrials import ClinicalTrialsSource

    page1 = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT0001", "briefTitle": "Trial one"},
                    "descriptionModule": {"briefSummary": "Summary one"},
                    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
                    "statusModule": {
                        "lastUpdatePostDateStruct": {"date": "2026-05-01"}
                    },
                }
            }
        ],
        "nextPageToken": "tok2",
    }
    page2 = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT0002", "briefTitle": "Trial two"},
                    "descriptionModule": {"briefSummary": "Summary two"},
                    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
                    "statusModule": {
                        "lastUpdatePostDateStruct": {"date": "2026-05-02"}
                    },
                }
            }
        ],
        # No nextPageToken — last page
    }
    calls: list[dict] = []

    def fake_get_json(url, params):
        calls.append(dict(params))
        return page1 if "pageToken" not in params else page2

    monkeypatch.setattr("dailydigest.ingest.clinicaltrials._get_json", fake_get_json)

    items = ClinicalTrialsSource().fetch(_spec(name="ClinicalTrials", kind="clinicaltrials"))

    assert len(items) == 2
    assert items[0].external_id == "NCT0001"
    assert items[1].external_id == "NCT0002"
    assert any("pageToken" in c for c in calls), "second page was never fetched"


def test_clinicaltrials_stops_at_max_items(monkeypatch):
    from dailydigest.ingest.clinicaltrials import ClinicalTrialsSource

    def fake_get_json(url, params):
        return {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": f"NCT{i:04d}",
                            "briefTitle": f"Trial {i}",
                        },
                        "descriptionModule": {},
                        "sponsorCollaboratorsModule": {},
                        "statusModule": {},
                    }
                }
                for i in range(50)
            ],
            "nextPageToken": "keep_going",
        }

    monkeypatch.setattr("dailydigest.ingest.clinicaltrials._get_json", fake_get_json)

    ct = ClinicalTrialsSource()
    items = ct.fetch(_spec(name="CT", kind="clinicaltrials"))
    assert len(items) <= ct.MAX_ITEMS


# ---------------------------------------------------------------------------
# BioRxiv — DOI normalization for dedup
# ---------------------------------------------------------------------------


def test_biorxiv_external_id_is_canonical_doi(monkeypatch):
    """external_id must be a bare canonical DOI so it matches OpenAlex dedup keys."""
    from dailydigest.ingest.biorxiv import BiorxivSource

    payload = {
        "messages": [{"total": 1, "status": "ok"}],
        "collection": [
            {
                "doi": "10.1101/2026.01.01.123456",
                "title": "A novel preprint",
                "abstract": "We describe a method.",
                "authors": "Smith J",
                "date": "2026-01-01",
            }
        ],
    }

    def fake_get_json(client, url):
        # biorxiv URL ends with /{cursor}; first call is cursor=0
        return payload if url.endswith("/0") else {"collection": []}

    monkeypatch.setattr("dailydigest.ingest.biorxiv._get_json", fake_get_json)

    items = BiorxivSource().fetch(_spec(name="bioRxiv", kind="biorxiv", server="biorxiv"))

    assert len(items) == 1
    assert items[0].external_id == "10.1101/2026.01.01.123456"
    assert not items[0].external_id.startswith("http")


# ---------------------------------------------------------------------------
# OpenAlex — URL fallback
# ---------------------------------------------------------------------------


def test_openalex_url_fallback_uses_https_openalex_url(monkeypatch):
    """When DOI and landing_page_url are absent, item.url must be a proper HTTPS URL."""
    from dailydigest.ingest.openalex import OpenAlexSource

    work_without_doi = {
        "id": "https://openalex.org/W9999",
        "title": "A mysterious work",
        "doi": None,
        "primary_location": None,
        "abstract_inverted_index": None,
        "authorships": [],
        "publication_date": "2026-05-01",
    }
    payload = {
        "results": [work_without_doi],
        "meta": {"next_cursor": None},
    }

    monkeypatch.setattr(
        "dailydigest.ingest.openalex._get_json",
        lambda url, params, headers: payload,
    )

    items = OpenAlexSource().fetch(_spec(name="OpenAlex", kind="openalex", query="RNA"))

    assert len(items) == 1
    assert items[0].url.startswith("https://")
    assert "openalex.org" in items[0].url


def test_openalex_bounds_publication_window_at_today(monkeypatch):
    """OpenAlex must not admit records with erroneous future publication dates."""
    from dailydigest.ingest.openalex import OpenAlexSource

    captured: dict[str, str] = {}

    def fake_get_json(url, params, headers):
        captured["filter"] = params["filter"]
        return {"results": [], "meta": {"next_cursor": None}}

    monkeypatch.setattr("dailydigest.ingest.openalex._get_json", fake_get_json)

    OpenAlexSource().fetch(_spec(name="OpenAlex", kind="openalex", query="RNA"), days=2)

    today = datetime.now(timezone.utc).date().isoformat()
    assert f"to_publication_date:{today}" in captured["filter"]


def test_openalex_skips_work_with_no_url_at_all(monkeypatch):
    from dailydigest.ingest.openalex import OpenAlexSource

    work_no_url = {
        "id": None,
        "title": "Ghost work",
        "doi": None,
        "primary_location": None,
        "abstract_inverted_index": None,
        "authorships": [],
        "publication_date": "2026-05-01",
    }
    payload = {"results": [work_no_url], "meta": {"next_cursor": None}}

    monkeypatch.setattr(
        "dailydigest.ingest.openalex._get_json",
        lambda url, params, headers: payload,
    )

    items = OpenAlexSource().fetch(_spec(name="OpenAlex", kind="openalex"))
    assert len(items) == 0


def test_rss_transport_failure_is_reported_to_pipeline(monkeypatch):
    """A failed feed is not a healthy zero-result fetch."""
    from dailydigest.ingest.rss import RSSSource

    def fail(_url):
        raise OSError("broken runtime")

    monkeypatch.setattr("dailydigest.ingest.rss._http_get_bytes", fail)

    with pytest.raises(RuntimeError, match="RSS fetch failed"):
        RSSSource().fetch(
            _spec(name="Broken feed", kind="rss", url="https://example.com/feed")
        )


def test_rss_download_rejects_an_oversized_stream(monkeypatch):
    from dailydigest.ingest import rss as rss_mod

    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"123"
            yield b"456"

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(rss_mod, "_MAX_FEED_BYTES", 5)
    monkeypatch.setattr(rss_mod.httpx, "Client", Client)

    with pytest.raises(RuntimeError, match="safety limit"):
        rss_mod._http_get_bytes("https://example.com/feed")


def test_openalex_venues_tags_items_with_real_journal(monkeypatch):
    """openalex_venues attributes each item to its real journal (venue) name so it
    earns venue prestige, and filters by the configured venue ids."""
    from dailydigest.ingest.openalex import OpenAlexVenuesSource

    work = {
        "id": "https://openalex.org/W1",
        "title": "A DNA origami nanodevice",
        "doi": "https://doi.org/10.1021/acsnano.6b00001",
        "primary_location": {"source": {"display_name": "ACS Nano"}},
        "abstract_inverted_index": None,
        "authorships": [],
        "publication_date": "2026-07-06",
    }
    captured = {}

    def fake_get_json(url, params, headers):
        captured["filter"] = params.get("filter", "")
        return {"results": [work], "meta": {"next_cursor": None}}

    monkeypatch.setattr("dailydigest.ingest.openalex._get_json", fake_get_json)
    monkeypatch.setattr(
        "dailydigest.ingest.openalex._get_crossref_json",
        lambda url, params, headers: {"message": {"items": []}},
    )

    spec = _spec(
        name="ACS journals (via OpenAlex)",
        kind="openalex_venues",
        venue_ids=["S145476921", "S143846845"],
    )
    items = OpenAlexVenuesSource().fetch(spec, days=2)

    assert len(items) == 1
    # Item is attributed to its real journal, NOT the aggregator feed name.
    assert items[0].source == "ACS Nano"
    # The venue filter was applied.
    assert "primary_location.source.id:S145476921|S143846845" in captured["filter"]


def test_openalex_repairs_acs_title_spacing_from_crossref(monkeypatch):
    """OpenAlex collapses Crossref title line breaks for some ACS papers."""
    from dailydigest.ingest import openalex as openalex_mod

    openalex_mod._recent_acs_titles.cache_clear()

    work = {
        "id": "https://openalex.org/W-ACS",
        "title": (
            "Brightly FluorescentSelf-Assembled Supra-J-AggregateNanoparticles "
            "for Bioanalysis"
        ),
        "doi": "https://doi.org/10.1021/acsnano.6c04900",
        "primary_location": {"source": {"display_name": "ACS Nano"}},
        "abstract_inverted_index": None,
        "authorships": [],
        "publication_date": "2026-08-12",
    }
    monkeypatch.setattr(
        openalex_mod,
        "_get_json",
        lambda url, params, headers: {
            "results": [work],
            "meta": {"next_cursor": None},
        },
    )
    monkeypatch.setattr(
        openalex_mod,
        "_get_crossref_json",
        lambda url, params, headers: {
            "message": {
                "items": [
                    {
                        "DOI": "10.1021/acsnano.6c04900",
                        "title": [
                            "Brightly Fluorescent\nSelf-Assembled "
                            "Supra-J-Aggregate\nNanoparticles for Bioanalysis"
                        ],
                    }
                ]
            }
        },
        raising=False,
    )

    items = openalex_mod.OpenAlexSource()._fetch_works(
        _spec(name="ACS flagships (via OpenAlex)", kind="openalex_venues"),
        2,
        {"User-Agent": "test"},
        cap=10,
        use_venue_source=True,
    )

    assert items[0].title == (
        "Brightly Fluorescent Self-Assembled Supra-J-Aggregate "
        "Nanoparticles for Bioanalysis"
    )


def test_openalex_title_cleanup_failure_keeps_the_paper(monkeypatch):
    from dailydigest.ingest import openalex as openalex_mod

    openalex_mod._recent_acs_titles.cache_clear()
    work = {
        "id": "https://openalex.org/W-ACS-FALLBACK",
        "title": "FluorescentSelf-Assembled Nanoparticles",
        "doi": "https://doi.org/10.1021/acsnano.6c99999",
        "primary_location": {"source": {"display_name": "ACS Nano"}},
        "abstract_inverted_index": None,
        "authorships": [],
        "publication_date": "2026-08-12",
    }
    monkeypatch.setattr(
        openalex_mod,
        "_get_json",
        lambda url, params, headers: {
            "results": [work],
            "meta": {"next_cursor": None},
        },
    )

    def crossref_offline(url, params, headers):
        raise RuntimeError("offline")

    monkeypatch.setattr(
        openalex_mod,
        "_get_crossref_json",
        crossref_offline,
    )

    items = openalex_mod.OpenAlexSource()._fetch_works(
        _spec(name="ACS flagships (via OpenAlex)", kind="openalex_venues"),
        2,
        {"User-Agent": "test"},
        cap=10,
        use_venue_source=True,
    )

    assert len(items) == 1
    assert items[0].title == "FluorescentSelf-Assembled Nanoparticles"


def test_openalex_profile_driven_upgrades_recognized_venue(monkeypatch):
    """A profile-driven OpenAlex hit from a flagship venue is re-attributed to that
    venue; an unknown venue keeps the aggregator source name."""
    from dailydigest.ingest.openalex import OpenAlexSource

    flagship = {
        "id": "https://openalex.org/W2",
        "title": "Self-assembled photonic crystal",
        "doi": "https://doi.org/10.1021/jacs.6b00002",
        "primary_location": {"source": {"display_name": "Journal of the American Chemical Society"}},
        "abstract_inverted_index": None,
        "authorships": [],
        "publication_date": "2026-07-06",
    }
    obscure = {
        "id": "https://openalex.org/W3",
        "title": "Something niche",
        "doi": "https://doi.org/10.9999/obscure.1",
        "primary_location": {"source": {"display_name": "Journal of Obscure Results"}},
        "abstract_inverted_index": None,
        "authorships": [],
        "publication_date": "2026-07-06",
    }
    monkeypatch.setattr(
        "dailydigest.ingest.openalex._get_json",
        lambda url, params, headers: {"results": [flagship, obscure], "meta": {"next_cursor": None}},
    )
    items = OpenAlexSource()._fetch_works(
        _spec(name="OpenAlex (your topics)", kind="openalex"),
        2,
        {"User-Agent": "t"},
        cap=10,
        query="photonic crystals",
        upgrade_venue=True,
    )
    by_title = {i.title: i.source for i in items}
    assert by_title["Self-assembled photonic crystal"] == "Journal of the American Chemical Society"
    assert by_title["Something niche"] == "OpenAlex (your topics)"


# ---------------------------------------------------------------------------
# Grants.gov — official structured opportunities
# ---------------------------------------------------------------------------


def test_grants_gov_fetches_official_details_and_amounts(monkeypatch):
    from dailydigest.ingest.grants_gov import GrantsGovSource

    search_payload = {
        "data": {
            "oppHits": [
                {
                    "id": 123,
                    "number": "RFA-RNA-26-001",
                    "title": "RNA delivery research programme",
                    "agencyName": "National Institutes of Health",
                    "openDate": "08/01/2026",
                    "closeDate": "10/01/2026",
                    "oppStatus": "posted",
                }
            ]
        }
    }
    detail_payload = {
        "data": {
            "id": 123,
            "opportunityNumber": "RFA-RNA-26-001",
            "opportunityTitle": "RNA delivery research programme",
            "synopsis": {
                "agencyName": "National Institutes of Health",
                "synopsisDesc": "<p>Supports new RNA delivery systems.</p>",
                "postingDate": "Aug 1, 2026 12:00:00 AM EDT",
                "responseDateDesc": "October 1, 2026 at 11:59 PM ET",
                "awardFloor": "100000",
                "awardCeiling": "500000",
                "costSharing": False,
                "applicantTypes": [
                    {"description": "Private institutions of higher education"}
                ],
                "fundingInstruments": [{"description": "Grant"}],
            },
        }
    }
    calls = []

    def fake_post(url, payload):
        calls.append((url, payload))
        return search_payload if url.endswith("/search2") else detail_payload

    monkeypatch.setattr("dailydigest.ingest.grants_gov._post_json", fake_post)
    monkeypatch.setattr(
        "dailydigest.ingest.grants_gov._today", lambda: date(2026, 8, 10)
    )
    monkeypatch.setattr(
        "dailydigest.ingest.grants_gov.profile_search_terms",
        lambda limit: ["RNA nanotechnology"],
    )

    items = GrantsGovSource().fetch(
        _spec(
            name="Grants.gov",
            kind="grants_gov",
            section="opportunities",
            profile_driven=True,
            lookahead_days=180,
        )
    )

    assert len(items) == 1
    item = items[0]
    assert item.external_id == "RFA-RNA-26-001"
    assert item.metadata["official"] is True
    assert item.metadata["status"] == "open"
    assert item.metadata["deadline"] == "2026-10-01"
    assert item.metadata["amount_min"] == 100000
    assert item.metadata["amount_max"] == 500000
    assert item.metadata["currency"] == "USD"
    assert "Private institutions" in item.metadata["eligibility"]
    assert len(calls) == 2


def test_openalex_venue_source_honours_configured_work_types(monkeypatch):
    """A preprint repository needs type:preprint, not the default type:article.

    OpenAlex records ChemRxiv output as ``type:preprint``, so the hardcoded
    ``type:article`` filter returned zero items for that whole channel.
    """
    from dailydigest.ingest.openalex import OpenAlexVenuesSource

    seen = {}

    def fake_get(url, params, headers):
        seen["filter"] = params["filter"]
        return {"results": [], "meta": {"next_cursor": None}}

    monkeypatch.setattr("dailydigest.ingest.openalex._get_json", fake_get)

    OpenAlexVenuesSource().fetch(
        _spec(
            name="ChemRxiv (via OpenAlex)",
            kind="openalex_venues",
            section="research",
            venue_ids=["S4393918830"],
            openalex_types="preprint",
        )
    )
    assert "type:preprint" in seen["filter"]
    assert "type:article" not in seen["filter"]
    assert "primary_location.source.id:S4393918830" in seen["filter"]

    OpenAlexVenuesSource().fetch(
        _spec(
            name="ACS flagships (via OpenAlex)",
            kind="openalex_venues",
            section="research",
            venue_ids=["S145476921"],
        )
    )
    assert "type:article" in seen["filter"]


def test_grants_gov_eligibility_order_is_stable_across_fetches(monkeypatch):
    """Grants.gov shuffles applicantTypes per call.

    The order leaked into the opportunity snapshot hash, so every grant looked
    materially changed on every brew and was re-surfaced in every digest.
    """
    from dailydigest.ingest.grants_gov import GrantsGovSource

    shuffled = [
        ["Small businesses", "County governments", "State governments"],
        ["State governments", "Small businesses", "County governments"],
    ]

    def _payloads(applicant_types):
        return (
            {
                "data": {
                    "oppHits": [
                        {
                            "id": 77,
                            "number": "RFA-ORDER-26-001",
                            "title": "Stable ordering programme",
                            "closeDate": "10/01/2026",
                            "oppStatus": "posted",
                        }
                    ]
                }
            },
            {
                "data": {
                    "opportunityNumber": "RFA-ORDER-26-001",
                    "opportunityTitle": "Stable ordering programme",
                    "synopsis": {
                        "agencyName": "NIH",
                        "applicantTypes": [
                            {"description": name} for name in applicant_types
                        ],
                        "fundingInstruments": [{"description": "Grant"}],
                    },
                }
            },
        )

    monkeypatch.setattr(
        "dailydigest.ingest.grants_gov._today", lambda: date(2026, 8, 10)
    )
    monkeypatch.setattr(
        "dailydigest.ingest.grants_gov.profile_search_terms", lambda limit: ["RNA"]
    )

    seen = []
    for applicant_types in shuffled:
        search_payload, detail_payload = _payloads(applicant_types)
        monkeypatch.setattr(
            "dailydigest.ingest.grants_gov._post_json",
            lambda url, payload, s=search_payload, d=detail_payload: (
                s if url.endswith("/search2") else d
            ),
        )
        item = GrantsGovSource().fetch(
            _spec(
                name="Grants.gov",
                kind="grants_gov",
                section="opportunities",
                profile_driven=True,
                lookahead_days=180,
            )
        )[0]
        seen.append((item.metadata["eligibility"], item.metadata["eligibility_tags"]))

    assert seen[0] == seen[1]
    assert seen[0][1] == ["County governments", "Small businesses", "State governments"]


def test_event_rss_keeps_only_upcoming_open_events(monkeypatch):
    from dailydigest.ingest.events_rss import EventsRSSSource

    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Official events</title>
      <item>
        <guid>event-1</guid><title>RNA nanotechnology workshop</title>
        <link>https://example.org/events/rna</link>
        <description><![CDATA[
          Date: 17 - 25 Aug 2026<br/>
          Location: EMBL Heidelberg<br/>
          Deadline(s): Application: 12 Aug 2026<br/>
          A practical workshop for RNA designers.
        ]]></description>
        <pubDate>Mon, 10 Aug 2026 08:00:00 GMT</pubDate>
      </item>
      <item>
        <guid>event-2</guid><title>Closed course</title>
        <link>https://example.org/events/closed</link>
        <description><![CDATA[
          Date: 20 - 22 Aug 2026<br/>
          Location: Online<br/>
          Deadline(s): Application: Closed
        ]]></description>
      </item>
    </channel></rss>"""
    detail_pages = {
        "https://example.org/events/rna": b"""
          <main><p>Date: <span>17 - 25 Aug 2026</span></p>
          <p>Location: <span>EMBL Heidelberg</span></p>
          <p>Deadline(s):</p><p>Application: <span>12 Aug 2026</span></p></main>
        """,
        "https://example.org/events/closed": b"""
          <main><p>Date: <span>20 - 22 Aug 2026</span></p>
          <p>Location: <span>Online</span></p>
          <p>Deadline(s):</p><p>Application: <span>Closed</span></p></main>
        """,
    }

    def fake_get(url):
        return feed if url == "https://example.org/feed" else detail_pages[url]

    monkeypatch.setattr("dailydigest.ingest.events_rss._http_get_bytes", fake_get)
    monkeypatch.setattr(
        "dailydigest.ingest.events_rss._today", lambda: date(2026, 8, 10)
    )

    items = EventsRSSSource().fetch(
        SourceSpec(
            name="EMBL Events",
            kind="events_rss",
            section="events",
            url="https://example.org/feed",
            lookahead_days=365,
        )
    )

    assert len(items) == 1
    assert items[0].title == "RNA nanotechnology workshop"
    assert items[0].metadata["official"] is True
    assert items[0].metadata["status"] == "open"
    assert items[0].metadata["event_start"] == "2026-08-17"
    assert items[0].metadata["event_end"] == "2026-08-25"
    assert items[0].metadata["deadline"] == "2026-08-12"
    assert items[0].metadata["location"] == "EMBL Heidelberg"
    assert items[0].metadata["format"] == "in_person"


def test_grants_gov_supports_forecast_schema(monkeypatch):
    from dailydigest.ingest.grants_gov import GrantsGovSource

    def fake_post(url, payload):
        if url.endswith("/search2"):
            return {
                "data": {
                    "oppHits": [
                        {
                            "id": "363474",
                            "number": "RFA-RM-28-006",
                            "title": "Transformative RNA technologies",
                            "agency": "National Institutes of Health",
                            "openDate": "08/05/2026",
                            "closeDate": "",
                            "oppStatus": "forecasted",
                        }
                    ]
                }
            }
        return {
            "data": {
                "opportunityNumber": "RFA-RM-28-006",
                "opportunityTitle": "Transformative RNA technologies",
                "agencyDetails": {"agencyName": "National Institutes of Health"},
                "forecast": {
                    "forecastDesc": "Forthcoming support for new RNA technologies.",
                    "postingDate": "Aug 05, 2026 12:00:00 AM EDT",
                    "estApplicationResponseDate": "Feb 15, 2027 12:00:00 AM EST",
                    "awardCeiling": "500000",
                    "costSharing": False,
                    "applicantTypes": [
                        {"description": "Private institutions of higher education"}
                    ],
                    "fundingInstruments": [{"description": "Cooperative Agreement"}],
                },
            }
        }

    monkeypatch.setattr("dailydigest.ingest.grants_gov._post_json", fake_post)
    monkeypatch.setattr(
        "dailydigest.ingest.grants_gov._today", lambda: date(2026, 8, 10)
    )
    monkeypatch.setattr(
        "dailydigest.ingest.grants_gov.profile_search_terms", lambda limit: ["RNA"]
    )

    items = GrantsGovSource().fetch(
        SourceSpec(
            name="Grants.gov",
            kind="grants_gov",
            section="opportunities",
            profile_driven=True,
            lookahead_days=365,
        )
    )

    assert len(items) == 1
    assert items[0].metadata["status"] == "forthcoming"
    assert items[0].metadata["deadline"] == "2027-02-15"
    assert items[0].metadata["amount_max"] == 500000
    assert items[0].authors == "National Institutes of Health"
    assert "forthcoming support" in items[0].abstract.casefold()


# ---------------------------------------------------------------------------
# FDA — most-recent submission by date
# ---------------------------------------------------------------------------


def test_fda_picks_most_recent_submission(monkeypatch):
    """Abstract and date should come from the submission with the highest status date."""
    from dailydigest.ingest.fda import FDASource

    entry = {
        "application_number": "NDA123456",
        "sponsor_name": "BioPharm Inc",
        "products": [{"brand_name": "Drugex", "active_ingredients": [{"name": "compound-A"}]}],
        "submissions": [
            {
                "submission_type": "ORIG",
                "submission_status": "AP",
                "submission_status_date": "20260101",
                "submission_class_code_description": "Original approval",
            },
            {
                "submission_type": "SUPPL",
                "submission_status": "AP",
                "submission_status_date": "20260501",  # more recent
                "submission_class_code_description": "Label update",
            },
        ],
    }

    def fake_get_json(url, params):
        return {"results": [entry]}

    monkeypatch.setattr("dailydigest.ingest.fda._get_json", fake_get_json)

    items = FDASource().fetch(_spec(name="openFDA", kind="fda_api", section="regulatory"))

    assert len(items) == 1
    assert "20260501" in items[0].abstract
    assert "Label update" in items[0].abstract


# ---------------------------------------------------------------------------
# Config — load_profile error handling
# ---------------------------------------------------------------------------


def test_load_profile_raises_readable_error_on_bad_yaml(tmp_path, monkeypatch):
    import pytest

    bad_profile = tmp_path / "bad.yaml"
    bad_profile.write_text("- this is a list not a mapping\n")

    monkeypatch.setenv("PROFILE_PATH", str(bad_profile))

    from dailydigest import config as config_mod
    config_mod.reload_settings()

    with pytest.raises(ValueError, match="must be a YAML mapping"):
        config_mod.load_profile(str(bad_profile))


def test_load_profile_raises_readable_error_on_missing_bio(tmp_path):
    import pytest

    bad_profile = tmp_path / "nobio.yaml"
    bad_profile.write_text("keywords: [RNA]\n")

    from dailydigest import config as config_mod

    with pytest.raises(ValueError, match="invalid"):
        config_mod.load_profile(str(bad_profile))
