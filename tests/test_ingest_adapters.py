from __future__ import annotations

from datetime import datetime, timezone

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
