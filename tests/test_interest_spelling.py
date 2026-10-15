"""Explicit interest spelling suggestions; no live services or private state."""
from urllib.parse import urlencode

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("topic,correction", [
    ("induced provimity", "induced proximity"),
    ("assymetric catalysis", "asymmetric catalysis"),
    ("perovksite solar cells", "perovskite solar cells"),
])
def test_spelling_service_uses_only_the_explicit_topic(monkeypatch, topic, correction):
    from dailydigest.interest_spelling import suggest_topic_spelling

    calls = []

    def get(client, url, *, params):
        calls.append((url, params, client.timeout.read))
        return httpx.Response(200, request=httpx.Request("GET", url), text=(
            f"<eSpellResult><CorrectedQuery>{correction}</CorrectedQuery><ERROR/></eSpellResult>"
        ))

    monkeypatch.setattr(httpx.Client, "get", get)
    assert suggest_topic_spelling(topic) == correction
    assert calls == [("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi",
                      {"db": "pubmed", "term": topic}, 8.0)]


@pytest.mark.parametrize("topic,correction", [
    ("induced proximity", ""),
    ("RNA delivery", "rna delivery"),
    ("KRAS G12D inhibtor", "KRAS G12C inhibitor"),
    ("PROTAC desgin", "protein design"),
    ("CRISPR Cas9 desgin", "CRISPR Cas8 design"),
    ("C-H activaton", "C-N activation"),
    ("RNA desgin", "RNA design"),
])
def test_spelling_preserves_identifiers_and_does_not_claim_unchanged_text_is_correct(
    monkeypatch, topic, correction,
):
    from dailydigest.interest_spelling import suggest_topic_spelling

    monkeypatch.setattr(httpx.Client, "get", lambda client, url, **kw: httpx.Response(
        200, request=httpx.Request("GET", url),
        text=f"<eSpellResult><CorrectedQuery>{correction}</CorrectedQuery></eSpellResult>",
    ))
    expected = correction if topic == "RNA desgin" else None
    assert suggest_topic_spelling(topic) == expected


@pytest.mark.parametrize("xml", [
    "invalid XML", "<html/>", "<eSpellResult/>",
    "<eSpellResult><ERROR>Unavailable</ERROR></eSpellResult>",
])
def test_bad_spelling_response_is_unavailable(monkeypatch, xml):
    from dailydigest.interest_spelling import SpellingUnavailable, suggest_topic_spelling

    monkeypatch.setattr(httpx.Client, "get", lambda client, url, **kw: httpx.Response(
        200, request=httpx.Request("GET", url), text=xml,
    ))
    with pytest.raises(SpellingUnavailable):
        suggest_topic_spelling("induced provimity")


def test_spelling_timeout_does_not_retry(monkeypatch):
    from dailydigest.interest_spelling import SpellingUnavailable, suggest_topic_spelling

    calls = []

    def timeout(*args, **kwargs):
        calls.append(True)
        raise httpx.ReadTimeout("fixture timeout")

    monkeypatch.setattr(httpx.Client, "get", timeout)
    with pytest.raises(SpellingUnavailable):
        suggest_topic_spelling("induced provimity")
    assert len(calls) == 1


def test_spelling_route_is_explicit_csrf_protected_and_does_not_save(monkeypatch, tmp_path):
    from dailydigest import web
    import dailydigest.interest_spelling as spelling

    calls = []
    monkeypatch.setattr(spelling, "suggest_topic_spelling", lambda topic: calls.append(topic) or "induced proximity")
    profile = tmp_path / "profile.yaml"
    profile.write_text("bio: Original reader\nkeywords: [induced provimity]\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    before = profile.read_bytes()
    with TestClient(web.app) as client:
        path = "/setup/check-wording?" + urlencode({"topic": "induced provimity"})
        assert client.request("GET", path).status_code == 405
        assert client.request("POST", path).status_code == 403
        assert calls == []
        response = client.request("POST", path, headers={"x-csrf-token": web._CSRF_TOKEN})
        assert response.status_code == 200
        assert response.json() == {"topic": "induced provimity", "suggestion": "induced proximity"}
        assert response.headers["cache-control"] == "no-store"
        assert calls == ["induced provimity"]
    assert profile.read_bytes() == before


@pytest.mark.parametrize("topic", ["", "   ", "x" * 161])
def test_spelling_route_rejects_empty_or_long_topics_before_network(monkeypatch, topic):
    from dailydigest import web
    import dailydigest.interest_spelling as spelling

    def forbidden(*args):
        pytest.fail("Invalid topic must not reach the provider")

    monkeypatch.setattr(spelling, "suggest_topic_spelling", forbidden)
    with TestClient(web.app) as client:
        response = client.request("POST", "/setup/check-wording", params={"topic": topic},
                               headers={"x-csrf-token": web._CSRF_TOKEN})
    assert response.status_code in {400, 422}


def test_spelling_route_reports_unavailability_without_provider_details(monkeypatch):
    from dailydigest import web
    import dailydigest.interest_spelling as spelling

    def unavailable(topic):
        raise spelling.SpellingUnavailable("internal provider details")

    monkeypatch.setattr(spelling, "suggest_topic_spelling", unavailable)
    with TestClient(web.app) as client:
        response = client.request("POST", "/setup/check-wording", params={"topic": "assymetric catalysis"},
                               headers={"x-csrf-token": web._CSRF_TOKEN})
    assert response.status_code == 503
    assert "internal provider" not in response.text


def test_accepted_correction_saves_the_same_topic_for_search_and_ranking(monkeypatch, tmp_path):
    import json
    import yaml
    from dailydigest import config, web
    from dailydigest.ingest._terms import profile_search_term_groups

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump({"bio": "Reader", "keywords": ["induced provimity"],
        "authors_of_interest": ["Ada"], "seed_works": ["Representative title"],
        "context_keywords": ["Background topic"],
        "canonical_facets": {"induced provimity": {"priority": 17,
            "anchors": ["induced provimity", "Validated protein interaction design"],
            "reader_note": "Retain existing metadata"}}}))
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    form = {"_csrf_token": web._CSRF_TOKEN, "topics": "induced proximity | 17",
            "topic_details": json.dumps({"induced proximity": {
                "previous_name": "induced provimity",
                "aliases": ["molecular glue"], "description": "Validated design methods."}}),
            "llm_backend": "extractive", "user_tz": "UTC"}
    with TestClient(web.app) as client:
        response = client.request("POST", "/setup", data=form, follow_redirects=False)
    assert response.status_code == 303
    profile = config.load_profile(str(profile_path))
    assert profile.keywords == ["induced proximity"]
    assert profile.authors_of_interest == ["Ada"]
    assert profile.seed_works == ["Representative title"]
    assert profile.context_keywords == ["Background topic"]
    assert list(profile.canonical_facets) == ["induced proximity"]
    facet = profile.canonical_facets['induced proximity']
    assert facet.priority == 17
    assert facet.anchors == ["induced proximity", "Validated protein interaction design"]
    assert yaml.safe_load(profile_path.read_text())["canonical_facets"]["induced proximity"]["reader_note"] == "Retain existing metadata"
    assert facet.aliases == ["molecular glue"]
    assert facet.description == "Validated design methods."
    monkeypatch.setattr(config, "load_profile", lambda: profile)
    assert profile_search_term_groups(include_aliases=False) == (["induced proximity"], ["Background topic"])


@pytest.mark.parametrize("previous", [123, "", "x" * 161, "bad|name", "bad\nname", "Other interest"])
def test_correction_origin_is_bounded_and_cannot_duplicate_a_retained_interest(previous):
    import json
    from dailydigest.web import _parse_topic_details

    details, errors = _parse_topic_details(json.dumps({
        "Corrected interest": {"previous_name": previous},
    }), [("Corrected interest", 75), ("Other interest", 25)])
    assert errors and details == {}
