from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import stat
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pytest
import yaml
from fastapi import HTTPException
from starlette.requests import Request


def _request(
    method: str = "GET",
    path: str = "/",
    *,
    headers: dict[str, str] | None = None,
    form: dict[str, str] | None = None,
    json_body: dict | None = None,
) -> Request:
    body = b""
    request_headers = {"host": "testserver"}
    if form is not None:
        body = urlencode(form).encode("utf-8")
        request_headers["content-type"] = "application/x-www-form-urlencoded"
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        request_headers["content-type"] = "application/json"
    request_headers.update(headers or {})
    header_items = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in request_headers.items()
    ]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": header_items,
        "client": ("testclient", 123),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _json_payload(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


def _text_payload(response) -> str:
    return response.body.decode("utf-8")


def test_setup_post_accepts_urlencoded_form_without_multipart(tmp_path, monkeypatch):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    env_path = tmp_path / ".env"
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_ENV_PATH", env_path)

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "name": "Hao",
                    "bio": "Researcher in RNA nanotechnology.",
                    "topics": "RNA nanotechnology | 17\nprotein design | 15",
                    "downweight": "celebrity news",
                    "llm_backend": "extractive",
                    "llm_base_url": "https://api.openai.com/v1",
                    "llm_api_key": "",
                    "llm_model": "gpt-4o-mini",
                    "user_tz": "America/Phoenix",
                },
            )
        )
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/run"
    saved_profile = profile_path.read_text()
    assert "name: Hao" in saved_profile
    assert "RNA nanotechnology" in saved_profile
    assert "priority: 17.0" in saved_profile
    assert "LLM_BACKEND=extractive" in env_path.read_text()
    assert "USER_TZ=America/Phoenix" in env_path.read_text()


def test_env_writer_restricts_existing_file_permissions_on_posix(tmp_path):
    from dailydigest import web

    env_path = tmp_path / ".env"
    env_path.write_text("LLM_API_KEY=old-secret\n")
    if os.name == "posix":
        env_path.chmod(0o644)

    web._write_env_file(env_path, {"LLM_API_KEY": "new-secret"})

    assert "LLM_API_KEY=new-secret" in env_path.read_text()
    if os.name == "posix":
        assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_setup_can_explicitly_remove_a_saved_api_key(tmp_path, monkeypatch):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_API_KEY=old-secret\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_ENV_PATH", env_path)
    monkeypatch.setattr(
        web,
        "SETTINGS",
        web.SETTINGS.model_copy(update={"llm_api_key": "old-secret"}),
    )

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA nanotechnology | 10",
                    "llm_backend": "extractive",
                    "llm_api_key": "***",
                    "remove_llm_api_key": "true",
                },
            )
        )
    )

    assert response.status_code == 303
    assert web._read_env_file(env_path)["LLM_API_KEY"] == ""


def test_setup_cannot_remove_key_while_selecting_api_backend(monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(
        web,
        "SETTINGS",
        web.SETTINGS.model_copy(update={"llm_api_key": "old-secret"}),
    )
    form = web._load_existing_form_defaults()
    form.update(
        {
            "topics": "RNA nanotechnology | 10",
            "llm_backend": "api",
            "llm_base_url": "https://api.example.com/v1",
            "llm_model": "example-model",
            "llm_api_key": "***",
            "remove_llm_api_key": "true",
        }
    )

    assert "API backend requires an API key." in web._validate_setup(form)


def test_global_host_guard_rejects_non_loopback_requests():
    from dailydigest import web

    async def downstream(_request):
        raise AssertionError("foreign-host request reached a route")

    response = asyncio.run(
        web._loopback_host_only(
            _request("GET", "/healthz", headers={"host": "evil.example"}),
            downstream,
        )
    )

    assert response.status_code == 403


def test_run_stream_rejects_unknown_run_instead_of_heartbeating_forever():
    from dailydigest import web

    web._RUN_QUEUES.clear()
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            web.run_stream(
                _request("GET", "/run/stream"), "abcdef123456"
            )
        )

    assert excinfo.value.status_code == 404


def test_setup_rejects_invalid_browser_timezone_before_writing(tmp_path, monkeypatch):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA nanotechnology | 10",
                    "llm_backend": "extractive",
                    "user_tz": "not/a-timezone",
                },
            )
        )
    )

    assert response.status_code == 400
    assert "Browser timezone" in _text_payload(response)
    assert not profile_path.exists()


def test_setup_defaults_load_canonical_weights_and_handle_missing_priority(
    tmp_path, monkeypatch
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "bio: Researcher.\n"
        "canonical_facets:\n"
        "  protein design: {anchors: [protein design], priority: 18}\n"
        "  RNA delivery: {anchors: [RNA delivery]}\n"
    )
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    form = web._load_existing_form_defaults()

    assert form["topics"] == "protein design | 18\nRNA delivery | 1"


def test_setup_defaults_replace_removed_backend_with_extractive(monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(
        web,
        "SETTINGS",
        web.SETTINGS.model_copy(update={"llm_backend": "legacy_cli"}),
    )

    assert web._load_existing_form_defaults()["llm_backend"] == "extractive"


def test_setup_defaults_prepopulate_optional_sections_and_ai_count(monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(
        web,
        "SETTINGS",
        web.SETTINGS.model_copy(
            update={
                "include_industry": True,
                "include_ai": False,
                "include_regulatory": True,
                "include_world": False,
                "top_ai": 7,
            }
        ),
    )

    form = web._load_existing_form_defaults()

    assert form["include_industry"] == "true"
    assert form["include_ai"] == "false"
    assert form["include_regulatory"] == "true"
    assert form["include_world"] == "false"
    assert form["top_ai"] == "7"


def test_setup_page_renders_accessible_optional_section_switches(monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(
        web,
        "SETTINGS",
        web.SETTINGS.model_copy(
            update={
                "include_industry": True,
                "include_ai": False,
                "include_regulatory": True,
                "include_world": False,
            }
        ),
    )

    response = web.setup_get(_request("GET", "/setup"))
    html = _text_payload(response)

    assert len(re.findall(r"<input[^>]+role=\"switch\"", html)) == 6
    assert 'id="include_industry"' in html
    assert 'id="include_ai"' in html
    assert 'id="include_regulatory"' in html
    assert 'id="include_world"' in html
    assert 'id="include_opportunities"' in html
    assert 'id="include_events"' in html
    assert 'id="opportunity-description"' not in html
    assert 'id="opportunity-citizenship"' not in html
    assert 'id="requires-travel-support"' not in html
    assert 'id="user-timezone"' in html
    assert "Intl.DateTimeFormat().resolvedOptions().timeZone" in html
    assert "Optional matching preferences" in html
    assert "weighted research interests above are reused" in html
    assert "About you for opportunity matching" in html
    assert 'id="top_ai"' in html
    assert "AI tools &amp; methods" in html
    assert "Clinical &amp; Regulatory" in html
    assert 'id="top_research"' in html
    assert 'name="include_research"' not in html


def test_setup_rejects_enabled_opportunity_section_without_profile(
    tmp_path, monkeypatch
):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(
        web, "_get_opportunity_profile_path", lambda: tmp_path / "opportunities.yaml"
    )
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA nanotechnology | 10",
                    "llm_backend": "extractive",
                    "include_events": "true",
                    "top_events": "4",
                },
            )
        )
    )

    assert response.status_code == 400
    assert "Career stage is required" in _text_payload(response)
    assert not (tmp_path / "opportunities.yaml").exists()


def test_setup_persists_private_opportunity_profile_when_enabled(
    tmp_path, monkeypatch
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    opportunity_path = tmp_path / "opportunities.yaml"
    env_path = tmp_path / ".env"
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_get_opportunity_profile_path", lambda: opportunity_path)
    monkeypatch.setattr(web, "_ENV_PATH", env_path)

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA nanotechnology | 10",
                    "llm_backend": "extractive",
                    "include_opportunities": "true",
                    "top_opportunities": "5",
                    "include_events": "true",
                    "top_events": "4",
                    "opportunity_career_stage": "postdoctoral researcher",
                    "opportunity_institution_type": "nonprofit university",
                    "opportunity_country": "United States",
                    "opportunity_applicant_role": "fellow or co-investigator",
                    "opportunity_types": "fellowship,travel_support",
                    "event_types": "conference,workshop",
                    "event_regions": "North America,online",
                    "event_formats": "in_person,online",
                    "minimum_lead_days": "14",
                },
            )
        )
    )

    assert response.status_code == 303
    saved = yaml.safe_load(opportunity_path.read_text())
    assert saved["career_stage"] == "postdoctoral researcher"
    assert saved["opportunity_types"] == ["fellowship", "travel_support"]
    assert "description" not in saved
    assert "requires_travel_support" not in saved
    assert "citizenship_or_residency" not in saved
    assert saved["minimum_lead_days"] == 14
    if os.name == "posix":
        assert stat.S_IMODE(profile_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(opportunity_path.stat().st_mode) == 0o600
    env = web._read_env_file(env_path)
    assert env["INCLUDE_OPPORTUNITIES"] == "true"
    assert env["INCLUDE_EVENTS"] == "true"


def test_setup_preserves_legacy_opportunity_values_not_shown_in_simplified_form(
    tmp_path, monkeypatch
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    opportunity_path = tmp_path / "opportunities.yaml"
    opportunity_path.write_text(
        yaml.safe_dump(
            {
                "description": "A detailed legacy eligibility description that remains private.",
                "career_stage": "postdoctoral researcher",
                "institution_type": "nonprofit university",
                "country": "United States",
                "applicant_role": "fellow",
                "citizenship_or_residency": "US permanent resident",
                "requires_travel_support": True,
            }
        )
    )
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(
        web, "_get_opportunity_profile_path", lambda: opportunity_path
    )
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA nanotechnology | 10",
                    "llm_backend": "extractive",
                    "include_opportunities": "true",
                    "opportunity_career_stage": "assistant professor",
                    "opportunity_institution_type": "nonprofit university",
                    "opportunity_country": "United States",
                    "opportunity_applicant_role": "principal investigator",
                },
            )
        )
    )

    assert response.status_code == 303
    saved = yaml.safe_load(opportunity_path.read_text())
    assert saved["career_stage"] == "assistant professor"
    assert saved["description"].startswith("A detailed legacy")
    assert saved["citizenship_or_residency"] == "US permanent resident"
    assert saved["requires_travel_support"] is True


def test_setup_rejects_invalid_structured_opportunity_fields_before_writing(
    tmp_path, monkeypatch
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    opportunity_path = tmp_path / "opportunities.yaml"
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(
        web, "_get_opportunity_profile_path", lambda: opportunity_path
    )
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA nanotechnology | 10",
                    "llm_backend": "extractive",
                    "include_opportunities": "true",
                    "top_opportunities": "5",
                    "opportunity_career_stage": "x",
                    "opportunity_institution_type": "university",
                    "opportunity_country": "United States",
                    "opportunity_applicant_role": "principal investigator",
                },
            )
        )
    )

    assert response.status_code == 400
    assert "Career stage" in _text_payload(response)
    assert not profile_path.exists()
    assert not opportunity_path.exists()


def test_setup_post_requires_one_to_ten_weighted_topics(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA nanotechnology | 17\nRNA nanotechnology | 10",
                    "llm_backend": "extractive",
                },
            )
        )
    )
    assert response.status_code == 400
    assert "duplicates" in _text_payload(response)
    assert not (tmp_path / "profile.yaml").exists()


def test_setup_post_persists_checked_and_unchecked_optional_sections(
    tmp_path, monkeypatch
):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    env_path = tmp_path / ".env"
    monkeypatch.setattr(web, "_ENV_PATH", env_path)

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA nanotechnology | 10",
                    "llm_backend": "extractive",
                    "include_industry": "true",
                    "top_industry": "5",
                    "include_ai": "false",
                    "top_ai": "7",
                    "include_regulatory": "true",
                    "top_regulatory": "4",
                    "include_world": "false",
                    "top_world": "3",
                },
            )
        )
    )

    assert response.status_code == 303
    env = web._read_env_file(env_path)
    assert env["INCLUDE_INDUSTRY"] == "true"
    assert env["INCLUDE_AI"] == "false"
    assert env["INCLUDE_REGULATORY"] == "true"
    assert env["INCLUDE_WORLD"] == "false"
    assert env["TOP_AI"] == "7"
    assert web.SETTINGS.include_industry is True
    assert web.SETTINGS.include_ai is False
    assert web.SETTINGS.include_regulatory is True
    assert web.SETTINGS.include_world is False
    assert web.SETTINGS.top_ai == 7


def test_legacy_setup_post_preserves_optional_section_settings(tmp_path, monkeypatch):
    from dailydigest import web

    current = web.SETTINGS.model_copy(
        update={
            "include_industry": True,
            "include_ai": False,
            "include_regulatory": True,
            "include_world": False,
        }
    )
    monkeypatch.setattr(web, "SETTINGS", current)
    for name, value in (
        ("INCLUDE_INDUSTRY", "true"),
        ("INCLUDE_AI", "false"),
        ("INCLUDE_REGULATORY", "true"),
        ("INCLUDE_WORLD", "false"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    env_path = tmp_path / ".env"
    monkeypatch.setattr(web, "_ENV_PATH", env_path)

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA nanotechnology | 10",
                    "llm_backend": "extractive",
                },
            )
        )
    )

    assert response.status_code == 303
    env = web._read_env_file(env_path)
    assert env["INCLUDE_INDUSTRY"] == "true"
    assert env["INCLUDE_AI"] == "false"
    assert env["INCLUDE_REGULATORY"] == "true"
    assert env["INCLUDE_WORLD"] == "false"


def test_setup_rejects_enabled_section_with_zero_items(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA nanotechnology | 10",
                    "llm_backend": "extractive",
                    "include_ai": "true",
                    "top_ai": "0",
                },
            )
        )
    )

    assert response.status_code == 400
    assert "AI tools and methods items must be between 1 and 30" in _text_payload(
        response
    )


def test_setup_validation_error_preserves_optional_section_switches(
    tmp_path, monkeypatch
):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "missing a weight",
                    "llm_backend": "extractive",
                    "include_industry": "true",
                    "include_ai": "false",
                    "include_regulatory": "true",
                    "include_world": "false",
                },
            )
        )
    )
    html = _text_payload(response)

    assert response.status_code == 400
    assert re.search(r'id="include_industry"[^>]*checked', html)
    assert not re.search(r'id="include_ai"[^>]*checked', html)
    assert re.search(r'id="include_regulatory"[^>]*checked', html)
    assert not re.search(r'id="include_world"[^>]*checked', html)


def test_setup_validation_error_does_not_echo_submitted_api_key(
    tmp_path, monkeypatch
):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(
        web,
        "SETTINGS",
        web.SETTINGS.model_copy(update={"llm_api_key": ""}),
    )

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "missing a weight",
                    "llm_backend": "api",
                    "llm_base_url": "https://api.example.test/v1",
                    "llm_api_key": "super-secret-test-key",
                    "llm_model": "example-model",
                },
            )
        )
    )
    html = _text_payload(response)

    assert response.status_code == 400
    assert "super-secret-test-key" not in html
    assert 'id="llm_api_key" name="llm_api_key"' in html
    assert 'placeholder="Provider API key" value=""' in html
    assert "Re-enter the API key after correcting the form" in html


def test_setup_post_requires_at_least_one_research_item(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA | 10",
                    "llm_backend": "extractive",
                    "top_research": "0",
                },
            )
        )
    )

    assert response.status_code == 400
    assert "Research items must be between 1 and 30" in _text_payload(response)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "at least one"),
        ("topic without weight", "must use"),
        ("topic | 0", "greater than 0"),
        ("\n".join(f"topic {i} | 1" for i in range(11)), "at most 10"),
    ],
)
def test_ranked_topic_parser_rejects_invalid_input(raw, message):
    from dailydigest import web

    _topics, errors = web._parse_ranked_topics(raw)
    assert any(message in error for error in errors)


def test_setup_post_preserves_profile_fields_it_does_not_edit(tmp_path, monkeypatch):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "bio: Existing bio.\n"
        "keywords: [old topic]\n"
        "canonical_facets:\n"
        "  protein design:\n"
        "    anchors: [de novo protein design]\n"
        "    aliases: [protein engineering]\n"
        "    priority: 10\n"
        "context_keywords: [biotech news]\n"
        "interest_weights: {old topic: 2.0}\n"
        "negative_interests: {clinical pathology: 1.0}\n"
        "authors_of_interest: [Ada Lovelace]\n"
    )
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Updated bio.",
                    "topics": "protein design | 18",
                    "llm_backend": "extractive",
                },
            )
        )
    )

    assert response.status_code == 303
    saved = web.yaml.safe_load(profile_path.read_text())
    # Saving a weight-only facet edit must not replace a more specific active
    # retrieval phrase that the simplified form does not display.
    assert saved["keywords"] == ["old topic"]
    assert saved["canonical_facets"]["protein design"]["anchors"] == [
        "de novo protein design"
    ]
    assert saved["canonical_facets"]["protein design"]["aliases"] == [
        "protein engineering"
    ]
    assert saved["canonical_facets"]["protein design"]["priority"] == 18.0
    assert "interest_weights" not in saved
    assert saved["context_keywords"] == ["biotech news"]
    assert saved["negative_interests"] == {"clinical pathology": 1.0}
    assert saved["authors_of_interest"] == ["Ada Lovelace"]


def test_reordering_same_facets_preserves_specific_retrieval_keywords():
    from dailydigest import web

    profile = {
        "keywords": ["specific alpha", "specific beta"],
        "canonical_facets": {"Facet A": {}, "Facet B": {}},
    }

    keywords = web._keywords_after_topic_edit(
        profile, [("Facet B", 8.0), ("Facet A", 10.0)]
    )

    assert keywords == ["specific alpha", "specific beta"]


def test_profile_name_post_updates_existing_profile(tmp_path, monkeypatch):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "bio: Existing bio.\nkeywords:\n- RNA\ndownweight: []\n"
    )
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)

    response = asyncio.run(
        web.profile_name_post(
            _request(
                "POST",
                "/profile/name",
                form={"_csrf_token": web._CSRF_TOKEN, "name": "Hao"},
            )
        )
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "name: Hao" in Path(profile_path).read_text()


def test_index_redirects_to_setup_when_local_profile_missing(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")

    response = web.index(_request("GET", "/"))

    assert response.status_code == 302
    assert response.headers["location"] == "/setup"


def test_index_escapes_feed_html_and_rejects_unsafe_links(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-05", item_count=1))
        s.add(
            store_mod.ItemRow(
                source='Source <img src=x onerror="alert(1)">',
                section="research",
                external_id="xss-web",
                url="javascript:alert(1)",
                title="<script>alert(1)</script>",
                summary="<b>summary</b>",
                digest_id="2026-05-05",
                item_label="R1",
            )
        )

    response = web.index(_request("GET", "/"))
    text = _text_payload(response)

    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in text
    assert 'Source <img src=x onerror="alert(1)">' not in text
    assert "<b>summary</b>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert 'href="javascript:alert(1)"' not in text
    assert 'href="#"' in text


def test_index_renders_reader_card_hierarchy_and_feedback_controls(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-05", item_count=1))
        item = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="ranking-web",
            url="https://example.com/ranking-web",
            title="First-in-class RNA delivery study",
            abstract="A breakthrough primary result.",
            published_at=datetime.now(timezone.utc),
            summary=(
                "Key finding: A delivery method improved tissue targeting.\n"
                "Why read: It may matter for RNA therapeutic design.\n"
                "Caveat: None obvious from the feed text."
            ),
            score=0.91,
            digest_id="2026-05-05",
            item_label="R1",
        )
        s.add(item)
        s.flush()
        item_id = int(item.id)
    store_mod.write_digest_features(
        "2026-05-05",
        [
            (
                "R1",
                item_id,
                0.91,
                {
                    "score": 0.91,
                    "tags": ["High-quality source", "Fresh signal"],
                    "why_shown": ["Reliable source"],
                    "content_type": "article",
                    "source_bucket": "published_journal",
                    "selection_reason": "protected published-journal slot",
                    "primary_facet": "RNA nanotechnology",
                    "topic": 0.82,
                    "source": 0.95,
                    "novelty": 0.55,
                    "penalty": 0.0,
                },
            )
        ],
    )
    store_mod.write_digest_audit(
        "2026-05-05",
        "missed_top_journals",
        [
            {
                "title": "Missed journal article",
                "source": "Science",
                "url": "https://example.com/missed",
                "score": 0.88,
            }
        ],
    )
    store_mod.write_digest_audit(
        "2026-05-05",
        "candidate_funnel",
        [
            {
                "recent_items": 20,
                "after_reviewed_filter": 18,
                "after_previously_shown_filter": 17,
                "after_quality_gate": 15,
                "after_cross_source_dedupe": 12,
                "quality_gate_drops": [
                    {"reason": "thin abstract from non-protected source", "title": "Thin"}
                ],
            }
        ],
    )

    response = web.index(_request("GET", "/"))
    text = _text_payload(response)

    assert response.status_code == 200
    assert "Today’s spotlight" not in text
    assert text.count("First-in-class RNA delivery study") == 1
    assert "Today’s cup" in text
    assert "minute digest" in text
    assert "RNA nanotechnology" in text
    assert "Brew again" in text
    assert 'id="reading-mode-select"' in text
    assert '<details class="section-block"' in text
    assert 'class="section-toggle"' in text
    assert '<details class="about-brew"' in text
    assert "Today’s source mix" in text
    assert "Not shown today" in text
    assert "Missed journal article" in text
    assert "Brew diagnostics" in text
    assert "thin abstract from non-protected source" in text
    assert "after dedupe" in text
    assert "Top-journal audit" not in text
    assert "Lead story" not in text
    assert "Ranked for High-quality source, Fresh signal; selected via protected published-journal slot." not in text
    assert "Why this recommendation?" in text
    assert "editorial-signals" not in text
    assert "score-bars" not in text
    assert "High-quality source" in text
    assert 'data-filter="priority"' in text
    assert 'data-filter="unreviewed"' in text
    assert 'data-filter="published"' in text
    assert 'data-filter="preprints"' in text
    assert 'data-filter="ai-cs"' in text
    assert 'data-filter-group="status"' in text
    assert 'data-filter-group="source"' in text
    assert 'data-filter-group="section"' not in text
    assert 'bucket === "preprint_other"' in text
    assert 'class="summary-primary"' in text
    assert '<details class="item-details">' in text
    assert ">Details</summary>" in text
    assert "Why it matters, caveat and recommendation details" not in text
    assert "Key finding" in text
    # 4-level graded feedback
    assert "Must read" in text
    assert "Relevant" in text
    assert "Hmmm" in text
    assert "Not for me" in text
    assert 'data-grade="100"' in text and 'data-grade="10"' in text
    assert "No response saved yet." not in text
    assert 'class="vote-pct"' not in text
    assert "Too promotional" in text
    assert "Choose Seen or Not for me" not in text
    assert "Choose Hmmm or Not for me" in text
    assert "Update my ranking" in text
    assert "Learned" not in text


def test_reader_can_save_and_remove_an_item_from_the_digest(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-05", item_count=1))
        item = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="bookmark-web",
            url="https://example.com/bookmark-web",
            title="Save this RNA paper",
            digest_id="2026-05-05",
            item_label="R1",
        )
        s.add(item)
        s.flush()
        item_id = int(item.id)

    before = _text_payload(web.index(_request("GET", "/")))
    assert f'data-item-id="{item_id}"' in before
    assert 'data-bookmarked="false"' in before
    assert re.search(
        r'<div class="item-head">\s*<div class="item-heading">.*?'
        r'<a class="title-link".*?>Save this RNA paper</a>\s*</div>\s*'
        r'<button class="bookmark-btn"[^>]+aria-label="Save for later"',
        before,
        re.DOTALL,
    )
    assert '<span class="bookmark-label">Save</span>' in before

    headers = {"X-CSRF-Token": web._CSRF_TOKEN}
    added = web.bookmark_add(
        _request("POST", f"/bookmark/{item_id}", headers=headers), item_id
    )
    assert _json_payload(added) == {"ok": True, "item_id": item_id, "saved": True}
    after_add = _text_payload(web.index(_request("GET", "/")))
    assert 'data-bookmarked="true"' in after_add
    assert 'aria-label="Remove from saved items"' in after_add
    assert '<span class="bookmark-label">Saved</span>' in after_add

    removed = web.bookmark_remove(
        _request("DELETE", f"/bookmark/{item_id}", headers=headers), item_id
    )
    assert _json_payload(removed) == {"ok": True, "item_id": item_id, "saved": False}
    assert store_mod.bookmarked_item_ids([item_id]) == set()


def test_saved_archive_is_searchable_and_escapes_item_content(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)

    store_mod.init_db()
    with store_mod.session_scope() as s:
        rna = store_mod.ItemRow(
            source="Nature Nanotechnology",
            section="research",
            external_id="archive-rna",
            url="https://example.com/archive-rna",
            title="RNA origami archive paper",
            summary="A programmable nanostructure.",
        )
        unsafe = store_mod.ItemRow(
            source="Unsafe",
            section="industry",
            external_id="archive-unsafe",
            url="javascript:alert(1)",
            title="<script>alert(1)</script>",
        )
        s.add_all([rna, unsafe])
        s.flush()
        rna_id, unsafe_id = int(rna.id), int(unsafe.id)
    store_mod.set_bookmark(rna_id, True)
    store_mod.set_bookmark(unsafe_id, True)

    response = web.saved_items(_request("GET", "/saved"), q="RNA")
    text = _text_payload(response)

    assert response.status_code == 200
    assert "Saved reading" in text
    assert "RNA origami archive paper" in text
    assert "&lt;script&gt;" not in text
    assert 'name="q"' in text
    assert "1 saved item" in text

    all_items = _text_payload(web.saved_items(_request("GET", "/saved"), q=""))
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in all_items
    assert 'href="javascript:alert(1)"' not in all_items
    assert 'href="#"' in all_items


def test_index_renders_reason_line_and_content_type_label(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-05", item_count=3))
        review = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="facet-review",
            url="https://example.com/facet-review",
            title="A comprehensive review of RNA delivery",
            abstract="Review.",
            digest_id="2026-05-05",
            item_label="R1",
        )
        primary = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="facet-primary",
            url="https://example.com/facet-primary",
            title="Primary RNA delivery study",
            abstract="Primary.",
            digest_id="2026-05-05",
            item_label="R2",
        )
        plain = store_mod.ItemRow(
            source="OpenAlex",
            section="research",
            external_id="facet-plain",
            url="https://example.com/facet-plain",
            title="Plain aggregator item",
            abstract="Thin.",
            digest_id="2026-05-05",
            item_label="R3",
        )
        s.add_all([review, primary, plain])
        s.flush()
        review_id, primary_id, plain_id = int(review.id), int(primary.id), int(plain.id)

    store_mod.write_digest_features(
        "2026-05-05",
        [
            (
                "R1",
                review_id,
                0.9,
                {
                    "primary_facet": "RNA nanotechnology",
                    "content_type": "review",
                    "source_bucket": "published_journal",
                    "tags": ["high_quality_source"],
                },
            ),
            (
                "R2",
                primary_id,
                0.9,
                {
                    "primary_facet": "RNA nanotechnology",
                    "content_type": "research",
                    "source_bucket": "published_journal",
                    "tags": ["high_quality_source"],
                },
            ),
            (
                "R3",
                plain_id,
                0.4,
                {
                    "primary_facet": "",
                    "content_type": "research",
                    "source_bucket": "aggregator",
                    "tags": [],
                    "why_shown": [],
                },
            ),
        ],
    )

    response = web.index(_request("GET", "/"))
    text = _text_payload(response)

    assert response.status_code == 200
    # Reason line from primary_facet, with high-profile journal suffix.
    assert "Shown for RNA nanotechnology" in text
    # Review content type gets a visible badge; plain research does not.
    assert 'class="type-label">Review<' in text
    # Exactly one type badge overall (only the review item), so primary research
    # is not silently tagged.
    assert text.count('class="type-label"') == 1
    # Empty-facet plain item never renders a broken reason line.
    assert "Shown for ." not in text


def test_index_marks_latest_run_impressions_viewed(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-05", item_count=1))
        item = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="viewed-web",
            url="https://example.com/viewed-web",
            title="Viewed flag item",
            abstract="A primary result.",
            digest_id="2026-05-05",
            item_label="R1",
        )
        s.add(item)
        s.flush()
        item_id = int(item.id)

    run_id = store_mod.write_impressions(
        "2026-05-05",
        [("research", item_id, 0, 0.9)],
        model_version="v-test",
    )

    response = web.index(_request("GET", "/"))
    assert response.status_code == 200

    with store_mod.session_scope() as s:
        rows = (
            s.query(store_mod.ImpressionRow)
            .filter_by(digest_id="2026-05-05", run_id=run_id)
            .all()
        )
        assert rows
        assert all(r.viewed is True for r in rows)


def test_index_uses_confidence_not_relative_rank_for_priority_filtering(
    tmp_path, monkeypatch
):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-05", item_count=1))
        item = store_mod.ItemRow(
            source="OpenAlex",
            section="research",
            external_id="weak-relative",
            url="https://example.com/weak-relative",
            title="Weak relative winner",
            abstract="Thin metadata.",
            score=0.99,
            digest_id="2026-05-05",
            item_label="R1",
        )
        s.add(item)
        s.flush()
        item_id = int(item.id)

    store_mod.write_digest_features(
        "2026-05-05",
        [
            (
                "R1",
                item_id,
                0.99,
                {
                    "score": 0.99,
                    "rank_score": 0.99,
                    "confidence_score": 0.41,
                    "source_bucket": "aggregator",
                    "tags": ["Aggregator source"],
                    "why_shown": [],
                },
            )
        ],
    )

    response = web.index(_request("GET", "/"))
    text = _text_payload(response)

    assert response.status_code == 200
    assert 'data-priority="skim"' in text
    assert "Lead story" not in text
    assert "Must read first" not in text


def test_load_today_uses_latest_vote_when_legacy_duplicates_exist(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy_web.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            source VARCHAR NOT NULL,
            section VARCHAR NOT NULL,
            external_id VARCHAR NOT NULL,
            url VARCHAR NOT NULL,
            title TEXT NOT NULL,
            abstract TEXT DEFAULT '',
            authors TEXT DEFAULT '',
            published_at DATETIME,
            fetched_at DATETIME,
            summary TEXT DEFAULT '',
            score FLOAT,
            digest_id VARCHAR,
            item_label VARCHAR
        );
        CREATE TABLE votes (
            id INTEGER PRIMARY KEY,
            item_id INTEGER NOT NULL,
            value INTEGER NOT NULL,
            created_at DATETIME
        );
        INSERT INTO items (
            id, source, section, external_id, url, title, abstract, authors,
            published_at, fetched_at, summary, score, digest_id, item_label
        ) VALUES (
            1, 'Legacy', 'research', 'legacy-web', 'https://example.com/legacy-web',
            'Legacy item', 'Abstract', '', '2026-05-12 00:00:00',
            '2026-05-12 00:00:00', '', 0.5, '2026-05-12', 'R1'
        );
        INSERT INTO votes (id, item_id, value, created_at)
        VALUES
            (1, 1, 1, '2026-05-12 10:00:00'),
            (2, 1, -1, '2026-05-12 11:00:00');
        """
    )
    conn.commit()
    conn.close()

    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(db_path))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

    sections, current_vote = web._load_today("2026-05-12")

    assert current_vote == {1: -1}
    assert sections[0]["entries"][0]["current_vote"] == -1


def test_load_today_hides_disabled_stored_sections(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    monkeypatch.setattr(
        web,
        "SETTINGS",
        web.SETTINGS.model_copy(
            update={"include_industry": False, "top_industry": 6}
        ),
    )

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-13", item_count=2))
        s.add_all(
            [
                store_mod.ItemRow(
                    source="Nature",
                    section="research",
                    external_id="visible-research",
                    url="https://example.com/research",
                    title="Visible research",
                    digest_id="2026-05-13",
                    item_label="R1",
                ),
                store_mod.ItemRow(
                    source="Industry",
                    section="industry",
                    external_id="hidden-industry",
                    url="https://example.com/industry",
                    title="Hidden industry",
                    digest_id="2026-05-13",
                    item_label="I1",
                ),
            ]
        )

    sections, _current_vote = web._load_today("2026-05-13")

    assert [section["key"] for section in sections] == ["research"]
    assert sections[0]["entries"][0]["title"] == "Visible research"


def test_setup_post_rejects_missing_csrf_token(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            web.setup_post(
                _request(
                    "POST",
                    "/setup",
                    form={
                        "bio": "Researcher.",
                        "topics": "RNA | 10",
                        "llm_backend": "extractive",
                    },
                )
            )
        )

    assert excinfo.value.status_code == 403
    assert not (tmp_path / "profile.yaml").exists()
    assert not (tmp_path / ".env").exists()


def test_setup_post_rejects_env_newline_injection(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "topics": "RNA | 10",
                    "llm_backend": "api",
                    "llm_base_url": "https://api.openai.com/v1",
                    "llm_model": "gpt-4o-mini\nTOP_RESEARCH=30",
                },
            )
        )
    )

    assert response.status_code == 400
    assert "cannot contain line breaks" in _text_payload(response)
    assert not (tmp_path / "profile.yaml").exists()
    assert not (tmp_path / ".env").exists()


def test_vote_api_requires_csrf_and_latest_vote_wins(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

    store_mod.init_db()
    with store_mod.session_scope() as s:
        item = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="vote-api",
            url="https://example.com/vote-api",
            title="Vote API",
            abstract="Abstract.",
        )
        s.add(item)
        s.flush()
        item_id = int(item.id)

    with pytest.raises(HTTPException) as excinfo:
        web.vote(_request("POST", f"/vote/{item_id}/1"), item_id, 1)
    assert excinfo.value.status_code == 403

    headers = {"X-CSRF-Token": web._CSRF_TOKEN}
    r1 = web.vote(_request("POST", f"/vote/{item_id}/1", headers=headers), item_id, 1)
    r2 = web.vote(_request("POST", f"/vote/{item_id}/-1", headers=headers), item_id, -1)
    r3 = web.vote(_request("POST", f"/vote/{item_id}/0", headers=headers), item_id, 0)
    p1 = _json_payload(r1)
    p2 = _json_payload(r2)
    p3 = _json_payload(r3)

    assert p1["ok"] is True
    assert p1["item_id"] == item_id
    assert p1["new_value"] == 1
    assert p1["ranking_status"]["vote_counts"]["good"] == 1
    assert p2["ok"] is True
    assert p2["item_id"] == item_id
    assert p2["new_value"] == -1
    assert p2["ranking_status"]["vote_counts"]["bad"] == 1
    assert p3["ok"] is True
    assert p3["item_id"] == item_id
    assert p3["new_value"] == 0
    assert p3["ranking_status"]["vote_counts"]["neutral"] == 1

    with store_mod.session_scope() as s:
        rows = s.execute(store_mod.select(store_mod.VoteRow)).scalars().all()
        assert len(rows) == 1
        assert rows[0].value == 0

    with pytest.raises(HTTPException) as bad_value:
        web.vote(_request("POST", f"/vote/{item_id}/7", headers=headers), item_id, 7)
    assert bad_value.value.status_code == 400
    with pytest.raises(HTTPException) as missing_item:
        web.vote(_request("POST", "/vote/999999/1", headers=headers), 999999, 1)
    assert missing_item.value.status_code == 404


def test_vote_reason_api_requires_csrf_and_persists_reason(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

    store_mod.init_db()
    with store_mod.session_scope() as s:
        item = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="vote-reason-api",
            url="https://example.com/vote-reason-api",
            title="Vote Reason API",
            abstract="Abstract.",
        )
        s.add(item)
        s.flush()
        item_id = int(item.id)

    with pytest.raises(HTTPException) as excinfo:
        web.vote_reason(
            _request("POST", f"/vote/{item_id}/reason/low_impact"),
            item_id,
            "low_impact",
        )
    assert excinfo.value.status_code == 403

    headers = {"X-CSRF-Token": web._CSRF_TOKEN}
    web.vote(_request("POST", f"/vote/{item_id}/-1", headers=headers), item_id, -1)
    response = web.vote_reason(
        _request("POST", f"/vote/{item_id}/reason/low_impact", headers=headers),
        item_id,
        "low_impact",
    )

    assert response.status_code == 200
    assert _json_payload(response)["reasons"] == ["low_impact"]
    deleted = web.vote_reason_delete(
        _request("DELETE", f"/vote/{item_id}/reason/low_impact", headers=headers),
        item_id,
        "low_impact",
    )
    assert deleted.status_code == 200
    assert _json_payload(deleted)["reasons"] == []
    with pytest.raises(HTTPException) as bad_reason:
        web.vote_reason(
            _request("POST", f"/vote/{item_id}/reason/not_a_reason", headers=headers),
            item_id,
            "not_a_reason",
        )
    assert bad_reason.value.status_code == 400


def test_ranking_status_api_reports_vote_counts_without_training(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import votes as votes_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

    class DummyRanker:
        def load(self):
            return True

    monkeypatch.setattr(votes_mod, "LRRanker", DummyRanker)

    store_mod.init_db()
    with store_mod.session_scope() as s:
        items = [
            store_mod.ItemRow(
                source="Test",
                section="research",
                external_id=f"status-{idx}",
                url=f"https://example.com/status-{idx}",
                title=f"Status {idx}",
            )
            for idx in range(3)
        ]
        s.add_all(items)
        s.flush()
        s.add(store_mod.VoteRow(item_id=items[0].id, value=1))
        s.add(store_mod.VoteRow(item_id=items[1].id, value=-1))
        s.add(store_mod.VoteRow(item_id=items[2].id, value=0))

    response = web.ranking_status(_request("GET", "/ranking/status"))

    assert response.status_code == 200
    status = _json_payload(response)["status"]
    assert status["vote_counts"] == {
        "good": 1,
        "bad": 1,
        "neutral": 1,
        "signed": 2,
        "total": 3,
    }
    assert status["model_trained"] is True
    assert status["training_status"] == "needs_votes"
    assert status["ranking_status"] == "cosine_baseline"


def test_ranking_train_api_uses_monkeypatched_dataset_and_ranker(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import votes as votes_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    monkeypatch.setattr(votes_mod, "MIN_VOTES_FOR_LR", 2)

    class DummyRanker:
        trained = False

        def load(self):
            return self.trained

    monkeypatch.setattr(votes_mod, "LRRanker", DummyRanker)
    release_train = threading.Event()

    def _slow_train():
        release_train.wait(5)
        return {"ok": True, "trained": True, "status": votes_mod.lr_training_status()}

    monkeypatch.setattr(votes_mod, "train_lr_ranker", _slow_train)

    store_mod.init_db()
    with store_mod.session_scope() as s:
        items = [
            store_mod.ItemRow(
                source="Test",
                section="research",
                external_id=f"train-{idx}",
                url=f"https://example.com/train-{idx}",
                title=f"Train {idx}",
            )
            for idx in range(2)
        ]
        s.add_all(items)
        s.flush()
        s.add(store_mod.VoteRow(item_id=items[0].id, value=1))
        s.add(store_mod.VoteRow(item_id=items[1].id, value=-1))

    with pytest.raises(HTTPException) as excinfo:
        web.ranking_train(_request("POST", "/ranking/train"))
    assert excinfo.value.status_code == 403
    web._TRAIN_JOB["running"] = False
    web._TRAIN_JOB["last_result"] = None

    response = web.ranking_train(
        _request("POST", "/ranking/train", headers={"X-CSRF-Token": web._CSRF_TOKEN})
    )

    assert response.status_code == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert payload["started"] is True
    assert payload["running"] is True
    assert payload["message"] == "Ranking training started."

    # A second click while training is in progress should return immediately
    # instead of starting another fit or hanging the UI.
    response2 = web.ranking_train(
        _request("POST", "/ranking/train", headers={"X-CSRF-Token": web._CSRF_TOKEN})
    )
    assert response2.status_code == 200
    payload2 = _json_payload(response2)
    assert payload2["ok"] is True
    assert payload2["started"] is False
    assert payload2["running"] is True
    release_train.set()
    for _ in range(100):
        if not web._TRAIN_JOB["running"]:
            break
        time.sleep(0.01)
    assert web._TRAIN_JOB["running"] is False


def test_ranking_train_thread_clears_running_after_exception(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import votes as votes_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    monkeypatch.setattr(votes_mod, "MIN_VOTES_FOR_LR", 1)
    monkeypatch.setattr(
        votes_mod,
        "lr_training_status",
        lambda: {
            "can_train": True,
            "remaining_votes_for_lr": 0,
            "vote_counts": {"good": 1, "bad": 0, "neutral": 0, "signed": 1, "total": 1},
            "min_votes_for_lr": 1,
            "model_trained": False,
            "training_status": "ready",
            "ranking_status": "cosine_baseline",
        },
    )

    def _raise_train():
        raise RuntimeError("boom")

    monkeypatch.setattr(votes_mod, "train_lr_ranker", _raise_train)
    released = threading.Event()
    monkeypatch.setattr(web, "release_encoder", released.set)
    web._TRAIN_JOB["running"] = False
    web._TRAIN_JOB["last_result"] = None

    response = web.ranking_train(
        _request("POST", "/ranking/train", headers={"X-CSRF-Token": web._CSRF_TOKEN})
    )

    assert response.status_code == 200
    for _ in range(100):
        if not web._TRAIN_JOB["running"]:
            break
        time.sleep(0.01)
    assert web._TRAIN_JOB["running"] is False
    assert web._TRAIN_JOB["last_result"]["ok"] is False
    assert web._TRAIN_JOB["last_result"]["reason"] == "training_error"
    assert released.is_set()


def test_ranking_train_does_not_overlap_another_embedding_job(monkeypatch):
    from dailydigest import votes as votes_mod
    from dailydigest import web

    monkeypatch.setattr(
        votes_mod,
        "lr_training_status",
        lambda: {
            "can_train": True,
            "remaining_votes_for_lr": 0,
            "vote_counts": {"good": 3, "bad": 3, "neutral": 0, "signed": 6, "total": 6},
            "min_votes_for_lr": 6,
            "model_trained": False,
            "training_status": "ready",
            "ranking_status": "cosine_baseline",
        },
    )
    web._TRAIN_JOB["running"] = False
    web._TRAIN_JOB["last_result"] = None
    assert web._COMPUTE_LOCK.acquire(blocking=False)
    try:
        response = web.ranking_train(
            _request("POST", "/ranking/train", headers={"X-CSRF-Token": web._CSRF_TOKEN})
        )
    finally:
        web._COMPUTE_LOCK.release()

    payload = _json_payload(response)
    assert payload["ok"] is False
    assert payload["started"] is False
    assert payload["running"] is True
    assert "brew or ranking job" in payload["message"].lower()


def test_brew_releases_embedding_runtime_after_failure(monkeypatch):
    from dailydigest import web

    released = threading.Event()

    def fail_run_all(**_kwargs):
        raise RuntimeError("expected failure")

    monkeypatch.setattr(web, "run_all", fail_run_all)
    monkeypatch.setattr(web, "release_encoder", released.set)
    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()
    web._ensure_run("cleanup-test")

    web._kick_off_run("cleanup-test", "usual")

    assert released.wait(2)
    for _ in range(100):
        if not web._BREW_JOB["running"]:
            break
        threading.Event().wait(0.01)
    assert web._BREW_JOB["running"] is False
    assert web._COMPUTE_LOCK.acquire(blocking=False)
    web._COMPUTE_LOCK.release()


def test_background_refresh_releases_embedding_runtime(monkeypatch):
    from dailydigest import web

    released = []
    monkeypatch.setattr(web, "run_all", lambda **_kwargs: None)
    monkeypatch.setattr(web, "release_encoder", lambda: released.append(True))

    web._run_pipeline_dry_run()

    assert released == [True]
    assert web._BREW_JOB["running"] is False
    assert web._COMPUTE_LOCK.acquire(blocking=False)
    web._COMPUTE_LOCK.release()


def test_run_start_rejects_foreign_origin_and_detects_duplicate(monkeypatch):
    from dailydigest import web

    runs = []
    monkeypatch.setattr(
        web,
        "_kick_off_run",
        lambda run_id, reading_mode: runs.append((run_id, reading_mode)),
    )
    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()

    headers = {"X-CSRF-Token": web._CSRF_TOKEN}

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            web.run_start(
                _request(
                    "POST",
                    "/run/start",
                    json_body={"run_id": "abcdef123456"},
                    headers={**headers, "Origin": "https://evil.example"},
                )
            )
        )
    assert excinfo.value.status_code == 403

    first = asyncio.run(
        web.run_start(
            _request(
                "POST",
                "/run/start",
                json_body={"run_id": "abcdef123456", "reading_mode": "minimal"},
                headers=headers,
            )
        )
    )
    second = asyncio.run(
        web.run_start(
            _request(
                "POST",
                "/run/start",
                json_body={"run_id": "abcdef123456", "reading_mode": "minimal"},
                headers=headers,
            )
        )
    )

    assert _json_payload(first) == {"ok": True, "run_id": "abcdef123456"}
    assert _json_payload(second) == {
        "ok": True,
        "run_id": "abcdef123456",
        "already_started": True,
    }
    assert runs == [("abcdef123456", "minimal")]

    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()


def test_run_start_defaults_to_usual_and_rejects_unknown_reading_mode(monkeypatch):
    from dailydigest import web

    runs = []
    monkeypatch.setattr(
        web,
        "_kick_off_run",
        lambda run_id, reading_mode: runs.append((run_id, reading_mode)),
    )
    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()
    headers = {"X-CSRF-Token": web._CSRF_TOKEN}

    response = asyncio.run(
        web.run_start(
            _request(
                "POST",
                "/run/start",
                json_body={"run_id": "012345abcdef"},
                headers=headers,
            )
        )
    )
    assert _json_payload(response) == {"ok": True, "run_id": "012345abcdef"}
    assert runs == [("012345abcdef", "usual")]

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            web.run_start(
                _request(
                    "POST",
                    "/run/start",
                    json_body={"run_id": "badcafe12345", "reading_mode": "bottomless"},
                    headers=headers,
                )
            )
        )
    assert excinfo.value.status_code == 400
    assert runs == [("012345abcdef", "usual")]

    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()


def test_abandoned_run_queues_are_bounded():
    from dailydigest import web

    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()
    for index in range(web._MAX_RETAINED_RUNS + 5):
        web._ensure_run(f"abandoned-{index}")

    assert len(web._RUN_QUEUES) == web._MAX_RETAINED_RUNS
    assert "abandoned-0" not in web._RUN_QUEUES

    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()


def test_run_page_asks_for_reading_depth_before_brewing(monkeypatch, tmp_path):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("bio: Reader\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)

    response = web.run_get(_request("GET", "/run"))
    body = _text_payload(response)

    assert "How are we feeling today?" in body
    assert 'name="reading_mode"' in body
    assert 'value="full"' in body
    assert 'value="usual"' in body
    assert 'value="minimal"' in body
    assert "5 research picks + 1 per other section" in body
    assert body.index("How are we feeling today?") < body.index("Brew today’s digest")
    assert "morning tea" not in body.lower()


def test_setup_and_completion_pages_use_time_neutral_copy(monkeypatch, tmp_path):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("bio: Reader\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)

    setup = _text_payload(web.setup_get(_request("GET", "/setup")))
    complete = _text_payload(
        web.done(_request("GET", "/done"), digest_id="2026-08-10", n=3)
    )

    assert "Save settings and choose today’s digest" in setup
    assert "Your digest is ready" in complete
    assert "morning cup" not in (setup + complete).lower()


def test_main_page_exposes_one_click_reading_modes_and_settings_can_return(
    monkeypatch, tmp_path
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Reader\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2031-02-03")
    monkeypatch.setattr(web, "_load_today", lambda _digest_id: ([], {}))
    monkeypatch.setattr(web, "_digest_exists", lambda _digest_id: False)
    expected_tea_notes = web.daily_tea_deck(date(2031, 2, 3))
    main = _text_payload(web.index(_request("GET", "/")))
    settings = _text_payload(web.setup_get(_request("GET", "/setup")))

    assert "Welcome back, Ada" in main
    assert "new Date().getHours()" in main
    assert "How are we feeling today?" in main
    assert 'name="reading_mode"' in main
    assert 'value="full"' in main
    assert 'value="usual"' in main
    assert 'value="minimal"' in main
    assert 'action="/run"' in main
    assert 'name="autostart" value="1"' in main
    assert "window.location.href = `/run?reading_mode=" not in main
    assert "Pip’s tea break" in main
    assert 'id="tea-note-text"' in main
    assert 'id="tea-another"' in main
    assert 'class="tea-leaf-drop"' in main
    assert 'class="tea-leaf-shape" aria-hidden="true"' in main
    assert "Drop a tea leaf into Pip’s cup for another science fact or joke" in main
    assert "Add tea leaf" not in main
    assert 'id="tea-leaf-count">0</span>' in main
    assert 'const teaLeafStorageKey = "dailydigest-tea-leaves:"' in main
    assert "Math.min(10, teaLeaves + 1)" in main
    assert "Tea break over—time to read?" in main
    assert 'teaButton.setAttribute("title", "Tea break complete")' in main
    assert "teaButton.disabled = true" in main
    assert "localStorage.setItem(teaLeafStorageKey, String(teaLeaves))" in main
    assert "teaIndex = (teaIndex + 1) % TEA_NOTES.length" in main
    assert 'class="tea-pet" id="tea-pet" role="button" tabindex="0"' in main
    assert 'data-note-kind=' in main
    assert ".tea-break-copy { display: block; min-width: 0; }" in main
    assert ".tea-break-copy { max-width: 80ch; }" not in main
    assert "teaBreak.dataset.noteKind" in main
    assert "pet-steam-rise 1.9s ease-out 2 both" in main
    assert ".tea-break.is-changing .tea-leaf-shape { animation: leaf-drop 0.62s ease-in 1; }" in main
    assert '@media (hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference)' in main
    assert 'teaPet.addEventListener("pointermove"' in main
    assert 'teaPet.addEventListener("pointerleave"' in main
    assert 'teaPet.addEventListener("pointerdown"' in main
    assert "teaPet.setPointerCapture(event.pointerId)" in main
    assert 'teaPet.addEventListener("keydown"' in main
    assert 'id="pip-move"' in main
    assert 'id="pip-move-help"' in main
    assert 'class="pip-drag-copy"' in main
    assert main.index('class="pip-stage"') < main.index('class="tea-dialog"')
    assert '<div class="tea-dialog">\n      <span class="tea-break-copy"' in main
    assert 'window.matchMedia("(prefers-reduced-motion: reduce)")' in main
    assert 'role="progressbar"' in main
    assert 'aria-label="Digest reading progress"' in main
    assert 'id="reading-progress-fill"' in main
    assert 'window.addEventListener("scroll", scheduleReadingProgress, { passive: true })' in main
    assert 'document.querySelector(".digest-card")' in main
    tea_notes_match = re.search(r"const TEA_NOTES = (\[.*?\]);", main)
    assert tea_notes_match is not None
    tea_notes = json.loads(tea_notes_match.group(1))
    assert len(tea_notes) == 15
    assert len(set(tea_notes)) == 15
    assert tea_notes == list(expected_tea_notes)
    assert (
        '<span class="tea-break-copy" id="tea-note-text" aria-live="polite">'
        f"{expected_tea_notes[0]}</span>"
    ) in main
    assert "Summaries: Extractive (local, no AI)" in main
    assert 'href="/"' in settings
    assert "Back to today’s digest" in settings


def test_main_page_choice_arrives_at_run_page_and_starts_once(monkeypatch, tmp_path):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("bio: Reader\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)

    response = web.run_get(
        _request("GET", "/run"), reading_mode="minimal", autostart=True
    )
    body = _text_payload(response)

    assert 'value="minimal" checked' in body
    assert "const AUTOSTART = true;" in body
    assert "if (AUTOSTART) startBrew();" in body


def test_settings_offer_native_api_and_signed_in_cli_summarizers(monkeypatch, tmp_path):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("bio: Reader\nkeywords: [RNA]\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)

    body = _text_payload(web.setup_get(_request("GET", "/setup")))

    assert 'value="api"' in body
    assert "OpenAI-compatible API" in body
    assert 'value="anthropic"' in body
    assert "Anthropic Claude API" in body
    assert 'value="claude_cli"' in body
    assert "Signed-in Claude Code" in body
    assert 'value="codex_cli"' in body
    assert "Signed-in Codex" in body
    assert "API billing" in body
    assert "chat subscription" in body.lower()
    assert "short reader profile" in body


def test_setup_saves_signed_in_cli_backend_without_api_credentials(tmp_path, monkeypatch):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    env_path = tmp_path / ".env"
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_ENV_PATH", env_path)

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "RNA nanotechnology researcher.",
                    "topics": "RNA nanotechnology | 10",
                    "llm_backend": "claude_cli",
                    "llm_base_url": "",
                    "llm_api_key": "",
                    "llm_model": "haiku",
                },
            )
        )
    )

    assert response.status_code == 303
    saved = env_path.read_text()
    assert "LLM_BACKEND=claude_cli" in saved
    assert "LLM_MODEL=haiku" in saved
    assert "LLM_API_KEY" not in saved


def test_theme_controls_and_safe_pwa_assets_are_exposed(monkeypatch, tmp_path):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("bio: Reader\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2031-02-03")
    monkeypatch.setattr(web, "_load_today", lambda _digest_id: ([], {}))
    monkeypatch.setattr(web, "_digest_exists", lambda _digest_id: False)

    page = _text_payload(web.index(_request("GET", "/")))
    assert 'rel="manifest" href="/manifest.webmanifest"' in page
    assert 'id="theme-toggle"' in page
    assert '>Night theme</button>' in page
    assert 'id="display-toggle"' in page
    assert '>Compact view</button>' in page
    assert 'id="install-app"' in page
    assert '>Add to desktop</button>' in page
    assert "It does not install or start the DailyDigest server" in page
    assert 'aria-describedby="install-app-help"' in page
    assert 'compact ? "Comfortable view" : "Compact view"' in page
    assert "localStorage.getItem(\"dailydigest-theme\")" in page
    assert "localStorage.getItem(\"dailydigest-display\")" in page
    assert page.index("dailydigest-theme") < page.index("<style>")
    assert "@media (max-width: 680px)" in page
    assert ".app-utilities {" in page
    assert "display: flex; flex-direction: column" in page
    assert "position: static; flex-direction: row; justify-content: center" in page
    assert "navigator.serviceWorker.register" in page
    assert 'data-theme="light"' in page
    assert "@media (prefers-reduced-motion: no-preference)" in page
    assert "animation-iteration-count: 1 !important" in page

    manifest_response = web.web_manifest(_request("GET", "/manifest.webmanifest"))
    assert manifest_response.media_type == "application/manifest+json"
    manifest = _json_payload(manifest_response)
    assert manifest["name"] == "DailyDigest"
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "/"
    assert manifest["icons"][0]["src"] == "/app-icon.svg"
    assert {icon["sizes"] for icon in manifest["icons"]} >= {"192x192", "512x512"}

    icon = web.app_icon(_request("GET", "/app-icon.svg"))
    assert icon.media_type == "image/svg+xml"
    assert "<svg" in _text_payload(icon)

    worker = web.service_worker(_request("GET", "/sw.js"))
    worker_text = _text_payload(worker)
    assert worker.media_type == "application/javascript"
    assert "/manifest.webmanifest" in worker_text
    assert "/app-icon.svg" in worker_text
    assert "SAFE_ASSETS.includes(url.pathname)" in worker_text
    assert 'key.startsWith(CACHE_PREFIX)' in worker_text
    assert 'caches.match(event.request)' not in worker_text


def test_compact_view_changes_information_hierarchy(monkeypatch, tmp_path):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("bio: Reader\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2031-02-03")
    monkeypatch.setattr(web, "_load_today", lambda _digest_id: ([], {}))
    monkeypatch.setattr(web, "_digest_exists", lambda _digest_id: False)

    page = _text_payload(web.index(_request("GET", "/")))

    assert 'html[data-display="compact"] .subhead { display: none; }' in page
    assert 'html[data-display="compact"] .feedback-note { display: none; }' in page
    assert 'html[data-display="compact"] .reason-line { display: none; }' in page
    assert 'html[data-display="compact"] .item-heading { display: flex;' in page
    assert 'html[data-display="compact"] .vote-title { display: none; }' in page
    assert 'html[data-display="compact"] .item a.title-link { flex: 1 1 420px; font-size: 16px;' in page
    assert 'html[data-display="compact"] .today-cup { grid-template-columns: 1fr; }' in page
    assert ".wrap { max-width: 1040px; padding: 22px 14px 78px; }" in page
    assert "@media (min-width: 1440px)" in page
    assert "position: fixed; left: max(12px, calc(50% - 714px)); bottom: 18px" in page


def test_opportunity_calendar_export_uses_structured_dates(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "calendar.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="EMBL Events",
            section="events",
            external_id="calendar-event",
            url="https://example.org/event",
            title="RNA design workshop, practical session",
            abstract="A hands-on workshop.",
            metadata_json=(
                '{"event_start":"2026-09-10","event_end":"2026-09-12",'
                '"deadline":"2026-08-20","official":true}'
            ),
        )
        s.add(row)
        s.flush()
        item_id = int(row.id)

    response = web.calendar_item(_request("GET", f"/calendar/{item_id}.ics"), item_id)
    text = _text_payload(response)

    assert response.status_code == 200
    assert response.media_type == "text/calendar"
    assert "DTSTART;VALUE=DATE:20260910" in text
    assert "DTEND;VALUE=DATE:20260913" in text
    assert "RNA design workshop\\, practical session" in text
