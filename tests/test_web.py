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
from urllib.parse import parse_qs, urlencode, urlsplit

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


def _stub_index_dependencies(web, monkeypatch) -> None:
    """Keep route-only tests independent of persisted user state."""
    monkeypatch.setattr(web, "daily_tea_deck", lambda *_args, **_kwargs: ["Tea"])
    monkeypatch.setattr(web, "load_digest_audit", lambda *_args: [])
    monkeypatch.setattr(web, "carryover_item_ids", lambda: set())
    monkeypatch.setattr(web, "mark_impressions_viewed", lambda *_args: 0)
    monkeypatch.setattr(web, "_summarizer_label", lambda *_args: "Extractive")
    monkeypatch.setattr(web.votes_mod, "lr_training_status", lambda: {})


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

    assert form["topics"] == "protein design | 18\nRNA delivery | 9.0"


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
    assert "research interests above are reused" in html
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
                # Written by a newer release before the reader went back.
                "preferred_funders": ["NIH", "HFSP"],
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
    assert saved["preferred_funders"] == ["NIH", "HFSP"]


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
                    "topics": "RNA | invalid",
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
                    "topics": "RNA | invalid",
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
        ("topic |", "must use"),
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


def test_index_can_render_a_retained_digest(tmp_path, monkeypatch):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")
    monkeypatch.setattr(web, "_digest_exists", lambda value: value == "2026-05-04")
    monkeypatch.setattr(
        web,
        "_recent_digest_ids",
        lambda _today_id, limit=3: ["2026-05-04"],
    )
    _stub_index_dependencies(web, monkeypatch)
    loaded: list[tuple[str, bool]] = []

    def _load_retained(
        digest_id: str, *, include_disabled_sections: bool = False
    ):
        loaded.append((digest_id, include_disabled_sections))
        return [], {}

    monkeypatch.setattr(web, "_load_today", _load_retained)

    response = web.index(_request("GET", "/"), digest_id="2026-05-04")

    assert response.status_code == 200
    assert loaded == [("2026-05-04", True)]
    page = _text_payload(response)
    assert "2026-05-04" in page
    assert "A digest was brewed for 2026-05-04" in page
    assert "A digest was brewed for today" not in page


def test_index_renders_time_machine_for_available_recent_brews(
    tmp_path, monkeypatch
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    available = {"2026-05-05", "2026-05-04"}
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")
    monkeypatch.setattr(web, "_digest_exists", lambda value: value in available)
    monkeypatch.setattr(
        web,
        "_recent_digest_ids",
        lambda _today_id, limit=3: ["2026-05-05", "2026-05-04"],
    )
    _stub_index_dependencies(web, monkeypatch)
    monkeypatch.setattr(web, "_load_today", lambda _value, **_kwargs: ([], {}))

    response = web.index(_request("GET", "/"), digest_id="2026-05-04")

    assert response.status_code == 200
    page = _text_payload(response)
    assert 'id="digest-date" aria-label="Choose digest date"' in page
    assert '<option value="/"' in page
    assert '<option value="/?digest_id=2026-05-04" selected>' in page
    assert 'May 04' in page or 'May 4' in page
    assert "2026-05-03" not in page


def test_index_time_machine_uses_latest_actual_brews_across_gaps(
    tmp_path, monkeypatch
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")
    monkeypatch.setattr(
        web,
        "_recent_digest_ids",
        lambda _today_id, limit=3: ["2026-05-05", "2026-05-03", "2026-05-01"],
    )
    monkeypatch.setattr(web, "_digest_exists", lambda _value: True)
    _stub_index_dependencies(web, monkeypatch)
    monkeypatch.setattr(web, "_load_today", lambda _value, **_kwargs: ([], {}))

    page = _text_payload(web.index(_request("GET", "/")))

    assert '<option value="/?digest_id=2026-05-03"' in page
    assert '<option value="/?digest_id=2026-05-01"' in page
    assert 'May 3' in page and 'May 1' in page


def test_recent_digest_ids_are_ordered_by_edition_date_not_rebrew_time():
    from dailydigest import web
    from dailydigest.store import DigestRow, init_db, session_scope

    init_db()
    with session_scope() as session:
        session.add_all(
            [
                DigestRow(
                    id="2026-05-05",
                    created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
                ),
                DigestRow(
                    id="2026-05-03",
                    created_at=datetime(2026, 5, 6, tzinfo=timezone.utc),
                ),
                DigestRow(
                    id="2026-05-01",
                    created_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
                ),
            ]
        )

    assert web._recent_digest_ids("2026-05-06") == [
        "2026-05-05",
        "2026-05-03",
        "2026-05-01",
    ]


@pytest.mark.parametrize('viewed,available,previous', [
    ('2026-05-05', ['2026-05-05', '2026-05-03', '2026-05-01'], '2026-05-03'),
    ('2026-05-03', ['2026-05-05', '2026-05-03', '2026-05-01'], '2026-05-01'),
    ('2026-05-03', ['2026-05-03'], None),
    ('2026-05-05', ['2026-05-03'], '2026-05-03'),
])
def test_explicit_current_and_previous_brew_navigation(tmp_path, monkeypatch, viewed, available, previous):
    from dailydigest import web
    profile_path = tmp_path / 'profile.yaml'
    profile_path.write_text('name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n')
    monkeypatch.setattr(web, '_get_profile_path', lambda: profile_path)
    monkeypatch.setattr(web, '_digest_id', lambda: '2026-05-05')
    monkeypatch.setattr(web, '_recent_digest_ids', lambda _today, limit=3: available)
    monkeypatch.setattr(web, '_digest_exists', lambda value: value in available)
    _stub_index_dependencies(web, monkeypatch)
    monkeypatch.setattr(web, '_load_today', lambda *_args, **_kwargs: ([], {}))
    response = web.index(_request('GET', '/'), digest_id=viewed)
    page = _text_payload(response)
    assert 'id="digest-date"' in page
    assert 'value="/"' in page
    if previous is None:
        assert 'aria-label="Previous digest"' not in page
    else:
        assert f'href="/?digest_id={previous}" rel="prev"' in page
        assert 'aria-label="Previous digest"' in page


def test_index_opens_a_retained_digest_outside_the_recent_picker(
    tmp_path, monkeypatch
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    available = {"2026-05-05", "2026-05-01"}
    loaded: list[str] = []
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")
    monkeypatch.setattr(web, "_digest_exists", lambda value: value in available)
    monkeypatch.setattr(
        web,
        "_recent_digest_ids",
        lambda _today_id, limit=3: ["2026-05-05"],
    )
    _stub_index_dependencies(web, monkeypatch)

    def _load_digest(digest_id: str, **_kwargs):
        loaded.append(digest_id)
        return [], {}

    monkeypatch.setattr(web, "_load_today", _load_digest)

    response = web.index(_request("GET", "/"), digest_id="2026-05-01")

    assert response.status_code == 200
    assert loaded == ["2026-05-01"]
    assert "2026-05-01" in _text_payload(response)


@pytest.mark.parametrize("today_brewed", [False, True])
def test_empty_today_prominently_links_latest_saved_digest_without_switching_dates(
    tmp_path, monkeypatch, today_brewed
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nkeywords: [RNA]\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")
    monkeypatch.setattr(web, "_recent_digest_ids", lambda *_args, **_kwargs: ["2026-05-03"])
    monkeypatch.setattr(web, "_digest_exists", lambda value: today_brewed if value == "2026-05-05" else True)
    _stub_index_dependencies(web, monkeypatch)
    loaded = []

    def load(day, **_kwargs):
        loaded.append(day)
        return [], {}

    monkeypatch.setattr(web, "_load_today", load)
    response = web.index(_request("GET", "/"))
    html = _text_payload(response)

    assert loaded == ["2026-05-05"]
    assert 'action="/run"' in html
    if today_brewed:
        assert 'Open latest saved digest' not in html
    else:
        assert html.count('No digest has been brewed today.') == 1
        assert re.search(r'href="/\?digest_id=2026-05-03"[^>]*>Open latest saved digest', html)


@pytest.mark.parametrize("rated", [0, 1])
def test_digest_feedback_shows_statistics_only_after_ratings(tmp_path, monkeypatch, rated):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nkeywords: [RNA]\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_recent_digest_ids", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(web, "_digest_exists", lambda _day: False)
    monkeypatch.setattr(web, "_load_today", lambda *_args, **_kwargs: ([], {}))
    _stub_index_dependencies(web, monkeypatch)
    response = web.index(_request("GET", "/"))
    context = dict(response.context)
    context["feedback_summary"] = {"rated": rated, "useful": rated, "unrated": 3 - rated}
    context["ranking_status"] = {
        **context["ranking_status"],
        "vote_counts": {"good": rated, "bad": 0, "signed": rated},
        "learning_message": f"{rated} helpful ratings",
    }
    html = _text_payload(web.templates.TemplateResponse(
        _request("GET", "/"), "digest_web.html.j2", context,
    ))

    if rated:
        assert '1 useful of 1 rated picks in this digest' in html
        assert '<p id="learning-summary-message">1 helpful ratings</p>' in html
    else:
        assert '0 useful of 0 rated picks' not in html
        assert '<p id="learning-summary-message">Rate a paper to help shape future digests.</p>' in html


@pytest.mark.parametrize("digest_exists", [True, False])
def test_index_reports_enabled_empty_funding_only_after_a_brew(
    tmp_path, monkeypatch, digest_exists
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")
    monkeypatch.setattr(web, "_digest_exists", lambda _value: digest_exists)
    _stub_index_dependencies(web, monkeypatch)
    monkeypatch.setattr(
        web,
        "_load_today",
        lambda _value, **_kwargs: ([], {}),
    )
    monkeypatch.setattr(
        web,
        "section_enabled",
        lambda _settings, section: section == "opportunities",
    )

    response = web.index(_request("GET", "/"))

    assert response.status_code == 200
    page = _text_payload(response)
    funding_status = "No qualified new funding calls were selected in this brew."
    if digest_exists:
        assert "Funding &amp; Opportunities" in page
        assert funding_status in page
    else:
        assert funding_status not in page


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
    assert "Today’s cup" not in text
    assert "minute digest" not in text
    assert "RNA nanotechnology" in text
    assert ">Brew</button>" in text
    assert 'id="digest-search"' in text
    assert 'id="topic-filter"' in text
    assert 'class="contents-nav"' in text
    assert 'class="copy-summary btn"' in text
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
    assert "Why this pick?" in text
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
    assert "Very relevant" in text
    assert "Relevant" in text
    assert "Less relevant" in text
    assert "Not relevant" in text
    assert 'data-grade="100"' in text and 'data-grade="10"' in text
    assert "No response saved yet." not in text
    assert 'class="vote-pct"' not in text
    assert "Too promotional" in text
    assert "Choose Seen or Not for me" not in text
    assert "Choose Less relevant or Not relevant" in text
    assert "Refresh calibration" in text
    assert "Learned" not in text

    # Older summaries remain readable in full, without repeating their parsed
    # context underneath when they lack the current Key finding field.
    with store_mod.session_scope() as session:
        session.get(store_mod.ItemRow, item_id).summary = (
            "Key point: Legacy finding.\nWhy read: Legacy relevance.\nCaveat: Legacy limitation."
        )
    legacy = _text_payload(web.index(_request("GET", "/")))
    assert legacy.count("Legacy finding.") == 1
    assert legacy.count("Legacy relevance.") == 1
    assert legacy.count("Legacy limitation.") == 1


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
    # The title owns the reading width; Save sits in the card's action row.
    assert re.search(
        r'<div class="item-head">\s*<div class="item-heading">.*?'
        r'<a class="title-link".*?>Save this RNA paper</a>\s*</div>\s*</div>',
        before,
        re.DOTALL,
    )
    assert re.search(
        r'<div class="vote-panel">.*?<div class="item-actions">\s*'
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


def test_known_button_is_opportunity_only_and_toggles(tmp_path, monkeypatch):
    """Manual "mark known" is offered where items legitimately recur: funding."""
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    monkeypatch.setenv("INCLUDE_OPPORTUNITIES", "true")
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    web.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-06")

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-06", item_count=2))
        grant = store_mod.ItemRow(
            source="Grants.gov",
            section="opportunities",
            external_id="known-web",
            url="https://example.com/known-web",
            title="RNA delivery funding call",
            digest_id="2026-05-06",
            item_label="F1",
        )
        paper = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="known-web-paper",
            url="https://example.com/known-web-paper",
            title="A research paper",
            digest_id="2026-05-06",
            item_label="R1",
        )
        s.add_all([grant, paper])
        s.flush()
        grant_id, paper_id = int(grant.id), int(paper.id)

    before = _text_payload(web.index(_request("GET", "/")))
    assert before.count('class="bookmark-btn known-btn"') == 1
    assert '<span class="known-label">Mark known</span>' in before
    # Research items never recur, so they get no extra button.
    blocks = {
        chunk.split('"', 1)[0]: chunk
        for chunk in before.split('class="item" data-item-id="')[1:]
    }
    assert "known-btn" in blocks[str(grant_id)]
    assert "known-btn" not in blocks[str(paper_id)]

    headers = {"X-CSRF-Token": web._CSRF_TOKEN}
    marked = web.known_add(
        _request("POST", f"/known/{grant_id}", headers=headers), grant_id
    )
    assert _json_payload(marked) == {"ok": True, "item_id": grant_id, "known": True}
    assert store_mod.known_item_ids([grant_id]) == {grant_id}

    after = _text_payload(web.index(_request("GET", "/")))
    assert 'data-known="true"' in after
    assert '<span class="known-label">Known</span>' in after

    cleared = web.known_remove(
        _request("DELETE", f"/known/{grant_id}", headers=headers), grant_id
    )
    assert _json_payload(cleared) == {"ok": True, "item_id": grant_id, "known": False}
    assert store_mod.known_item_ids([grant_id]) == set()

    with pytest.raises(HTTPException):
        web.known_add(_request("POST", "/known/999999", headers=headers), 999999)


def test_overflow_shelf_renders_and_saves_for_tomorrow(tmp_path, monkeypatch):
    """Trimmed-but-qualified picks are shown and can be pinned for the next brew."""
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    web.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-07")

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-07", item_count=1))
        shown = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="overflow-shown",
            url="https://example.com/shown",
            title="The one that fit",
            digest_id="2026-05-07",
            item_label="R1",
        )
        trimmed = store_mod.ItemRow(
            source="Science",
            section="research",
            external_id="overflow-trimmed",
            url="https://example.com/trimmed",
            title="The one that did not fit",
        )
        s.add_all([shown, trimmed])
        s.flush()
        trimmed_id = int(trimmed.id)
        s.add(
            store_mod.DigestAuditRow(
                digest_id="2026-05-07",
                audit_type="reading_mode_overflow",
                payload_json=json.dumps(
                    {
                        "item_id": trimmed_id,
                        "section": "research",
                        "source": "Science",
                        "title": "The one that did not fit",
                        "url": "https://example.com/trimmed",
                        "score": 0.91,
                        "topic_score": 0.74,
                    }
                ),
            )
        )

    before = _text_payload(web.index(_request("GET", "/")))
    assert "1 more qualified pick didn’t fit today’s reading mode" in before
    assert "The one that did not fit" in before
    assert "Save all for tomorrow" in before
    # Near-miss rows are gradable: they sit exactly on the cutoff the ranker
    # keeps getting wrong, so they are the scarcest training signal available.
    assert f'class="overflow-row" data-item-id="{trimmed_id}"' in before
    assert before.count('class="overflow-vote"') == 4

    headers = {"X-CSRF-Token": web._CSRF_TOKEN}
    signature = web._overflow_signature("2026-05-07", web.load_digest_audit("2026-05-07", "reading_mode_overflow"))
    response = web.overflow_save(_request("POST", "/overflow/save", headers=headers), digest_id="2026-05-07", audit_signature=signature)
    payload = _json_payload(response)
    assert payload == {
        "ok": True,
        "digest_id": "2026-05-07",
        "saved": 1,
        "added": 1,
    }
    assert store_mod.carryover_item_ids() == {trimmed_id}

    # Reload keeps the pinned state so the button cannot double-arm.
    after = _text_payload(web.index(_request("GET", "/")))
    assert "Saved for tomorrow" in after

    # Rating a near-miss positively must also PIN it: eligibility alone cannot
    # bring it back once its publication date leaves the candidate window.
    pinned = web.overflow_save_one(
        _request("POST", f"/overflow/save/{trimmed_id}", headers=headers), trimmed_id,
        digest_id="2026-05-07", audit_signature=signature,
    )
    assert _json_payload(pinned)["ok"] is True
    assert trimmed_id in store_mod.carryover_item_ids()
    with pytest.raises(HTTPException) as not_overflow:
        web.overflow_save_one(
            _request("POST", "/overflow/save/424242", headers=headers), 424242,
            digest_id="2026-05-07", audit_signature=signature,
        )
    assert not_overflow.value.status_code == 404

    # A positive grade on a near-miss keeps it eligible for tomorrow; a negative
    # one retires it (store.exclude_reviewed_items).
    votes_mod = __import__("dailydigest.votes", fromlist=["votes"])
    web.vote(_request("POST", f"/vote/{trimmed_id}/70", headers=headers), trimmed_id, 70)
    with store_mod.session_scope() as s:
        rows = [r for r in s.query(store_mod.ItemRow).all() if int(r.id) == trimmed_id]
        assert store_mod.exclude_reviewed_items(rows) == rows
        for r in rows:
            s.expunge(r)
    web.vote(_request("POST", f"/vote/{trimmed_id}/10", headers=headers), trimmed_id, 10)
    with store_mod.session_scope() as s:
        rows = [r for r in s.query(store_mod.ItemRow).all() if int(r.id) == trimmed_id]
        assert store_mod.exclude_reviewed_items(rows) == []
        for r in rows:
            s.expunge(r)
    assert votes_mod.vote_counts()["total"] >= 1

    # A day with no overflow audit -> 404 rather than a silent no-op.
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-08")
    with pytest.raises(HTTPException) as excinfo:
        web.overflow_save(_request("POST", "/overflow/save", headers=headers),
                          digest_id="2026-05-08", audit_signature=signature)
    assert excinfo.value.status_code == 404


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
    assert len(re.findall(r"<main(?:\s|>)", text)) == 1

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
    # Match attribution is shown once; the source remains in metadata.
    assert text.count("<strong>RNA nanotechnology</strong>") == 2
    assert text.count("Related by meaning; check the connection") == 2
    assert "RNA nanotechnology</strong> · Nature" not in text
    # Review content type gets a visible badge; plain research does not.
    assert 'class="type-label">Review<' in text
    # Exactly one type badge overall (only the review item), so primary research
    # is not silently tagged.
    assert text.count('class="type-label"') == 1
    # Empty-facet plain item never renders a broken reason line.
    assert text.count('class="match-topic') == 2


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


def test_load_today_hides_disabled_sections_unless_rendering_archive(
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

    archived_sections, _current_vote = web._load_today(
        "2026-05-13", include_disabled_sections=True
    )

    assert [section["key"] for section in archived_sections] == [
        "research",
        "industry",
    ]


def test_load_today_keeps_an_archived_slate_when_a_later_brew_reuses_an_item(
    tmp_path, monkeypatch
):
    """History comes from digest_items, not the mutable ItemRow back-pointer."""
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
        row = store_mod.ItemRow(
            source="Nature",
            section="research",
            external_id="reused-entry",
            url="https://example.com/reused",
            title="Reused entry",
        )
        s.add(row)
        s.flush()
        item_id = int(row.id)

    store_mod.write_digest("2026-05-13", [("R1", item_id, 0.71)])
    store_mod.write_digest("2026-05-14", [("R4", item_id, 0.42)])

    archived, _current_vote = web._load_today("2026-05-13")

    entries = [entry for section in archived for entry in section["entries"]]
    assert [entry["title"] for entry in entries] == ["Reused entry"]
    assert entries[0]["label"] == "R1"
    assert entries[0]["score_raw"] == pytest.approx(0.71)


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
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False

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
    # Preference memory needs no persisted model: below the vote threshold the
    # ranking is the cosine baseline and nothing is "trained".
    assert status["model_trained"] is False
    assert status["training_status"] == "needs_votes"
    assert status["ranking_status"] == "cosine_baseline"


def test_ranking_status_does_not_claim_personalization_in_cosine_mode(
    tmp_path, monkeypatch
):
    from dailydigest import config as config_mod
    from dailydigest import votes as votes_mod

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    monkeypatch.setenv("SCORING_MODE", "cosine")
    config_mod.reload_settings()
    monkeypatch.setattr(
        votes_mod,
        "vote_counts",
        lambda: {
            "good": 30,
            "bad": 0,
            "neutral": 0,
            "signed": 30,
            "total": 30,
        },
    )

    status = votes_mod.lr_training_status()

    assert status["scoring_mode"] == "cosine"
    assert status["model_trained"] is False
    assert status["ranking_status"] == "cosine_baseline"
    assert status["training_status"] == "disabled"


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


def test_run_stream_replays_progress_to_multiple_refreshes():
    from dailydigest import web

    run_id = "feedface1234"
    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()
    events = web._ensure_run(run_id)
    events.extend(
        [
            {"stage": "rank_start", "payload": {"candidates": 42}},
            {"stage": "done", "payload": {"digest_id": "2026-09-02"}},
        ]
    )

    async def collect() -> str:
        response = await web.run_stream(
            _request("GET", f"/run/stream?run_id={run_id}"), run_id
        )
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    async def collect_both() -> tuple[str, str]:
        return await asyncio.gather(collect(), collect())

    first, second = asyncio.run(collect_both())

    assert '"stage": "rank_start"' in first
    assert '"stage": "done"' in first
    assert first == second
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

    assert "Choose your digest size" in body
    assert 'name="reading_mode"' in body
    assert 'value="full"' in body
    assert 'value="usual"' in body
    assert 'value="minimal"' in body
    assert "5 research picks + 1 per other section" in body
    assert body.index("Choose your digest size") < body.index("Brew today’s digest")
    assert "morning tea" not in body.lower()


def test_run_page_reuses_valid_run_id_so_refresh_reconnects(monkeypatch, tmp_path):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("bio: Reader\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)

    response = web.run_get(
        _request("GET", "/run"),
        reading_mode="usual",
        autostart=True,
        run_id="abcdef123456",
    )
    body = _text_payload(response)

    assert 'const RUN_ID = "abcdef123456";' in body
    assert "resume=1\\u0026run_id=abcdef123456" in body


def test_setup_and_completion_pages_use_time_neutral_copy(monkeypatch, tmp_path):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("bio: Reader\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)

    setup = _text_payload(web.setup_get(_request("GET", "/setup")))
    complete = _text_payload(
        web.done(_request("GET", "/done"), digest_id="2026-08-10", n=3)
    )

    assert "Save changes" in setup
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
    assert 'aria-label="Choose digest size"' in main
    for label in ("Short", "Standard", "Full"):
        assert f'>{label}</option>' in main
    assert 'name="reading_mode"' in main
    assert 'value="full"' in main
    assert 'value="usual"' in main
    assert 'value="minimal"' in main
    assert 'action="/run"' in main
    assert 'name="autostart" value="1"' not in main
    assert "window.location.href = `/run?reading_mode=" not in main
    assert 'aria-label="Pip"' in main
    assert 'id="tea-note-text"' in main
    assert 'id="tea-another"' in main
    assert 'class="tea-leaf-drop"' in main
    assert 'class="tea-leaf-shape" aria-hidden="true"' in main
    assert "Drop a tea leaf into Pip’s cup for another science fact or joke" in main
    assert "Add tea leaf" not in main
    assert 'id="tea-leaf-count">0</span>' in main
    assert 'const teaLeafStorageKey = "dailydigest-tea-leaves:"' in main
    assert "Math.min(10, teaLeaves + 1)" in main
    assert "Your cup is full. I’ll keep it warm while you read." in main
    assert 'teaButton.setAttribute("title", "Your cup is full")' in main
    assert "teaButton.disabled = true" in main
    assert "localStorage.setItem(teaLeafStorageKey, String(teaLeaves))" in main
    assert "TEA_NOTES[teaLeaves % TEA_NOTES.length]" in main
    assert 'class="tea-pet" id="tea-pet" role="button" tabindex="0"' in main
    assert 'data-note-kind=' in main
    assert ".tea-break-copy { display: block; min-width: 0; }" in main
    assert ".tea-break-copy { max-width: 80ch; }" not in main
    assert "teaBreak.dataset.noteKind" in main
    assert 'class="pet-rim"' in main
    assert "pet-steam-rise 1.5s ease-out 1 both" in main
    assert "thankReader" not in main
    assert "pet-hop" not in main
    assert "note-pop" not in main
    # Hold the leaf gone until the contented reaction ends, then regrow.
    assert (
        ".tea-break.is-feeding .tea-leaf-shape { animation: leaf-drop 0.48s ease-in 1 forwards; }"
        in main
    )
    assert ".tea-break.is-regrowing .tea-leaf-shape { animation: leaf-regrow" in main
    assert "const LEAF_REST_MS = 420;" in main
    assert '@media (hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference)' in main
    assert 'teaPet.addEventListener("pointermove"' in main
    assert 'stage.addEventListener("pointerleave"' in main
    assert 'teaPet.addEventListener("pointerdown"' in main
    assert "teaPet.setPointerCapture(event.pointerId)" in main
    assert 'teaPet.addEventListener("keydown"' in main
    assert 'id="pip-move"' not in main
    assert 'id="pip-move-help"' in main
    assert 'class="pip-drag-copy"' not in main
    assert main.index('class="pip-stage"') < main.index('class="tea-dialog"')
    assert '<div class="tea-dialog" id="pip-dialog">' in main
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
    assert "Extractive (local, no AI)" in main
    assert 'href="/"' in settings
    assert "Back to today’s digest" in settings


def test_main_page_choice_arrives_at_run_page_and_starts_once(monkeypatch, tmp_path):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("bio: Reader\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)

    prepared = asyncio.run(web.run_prepare(_request("POST", "/run", form={
        "_csrf_token": web._CSRF_TOKEN, "reading_mode": "minimal",
    })))
    start = parse_qs(urlsplit(prepared.headers["location"]).query)["start"][0]
    response = web.run_get(_request("GET", "/run"), reading_mode="minimal", autostart=True, start=start)
    body = _text_payload(response)

    assert 'value="minimal" checked' in body
    assert "const AUTOSTART = true;" in body
    assert "if (AUTOSTART) startBrew();" in body
    assert 'Choose your digest size' in body
    for label in ("Short", "Standard", "Full"):
        assert f'<span class="mode-title">{label}</span>' in body


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
    assert "first 400 characters of your current research or learning goal" in body
    assert "up to 20 core topics" in body


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
    assert '<details class="app-utilities" id="display-menu">' in page
    assert 'id="display-menu-toggle" aria-controls="display-options">Display</summary>' in page
    assert 'id="pip-toggle"' in page
    assert 'role="group" aria-label="Display preferences"' in page
    assert 'displayButton.textContent = "Compact view"' in page
    # Saved choices apply before any style loads (this browser first, then
    # the data/ui_prefs.json copy), so the page never flashes the other theme.
    assert "localStorage.getItem(`dailydigest-${key}`)" in page
    assert 'const filed = {};' in page
    assert page.index('pick("theme")') < page.index("<style>")
    assert page.index('pick("display")') < page.index("<style>")
    assert ".app-utilities {" in page
    assert "display: flex; flex-direction: column" in page
    utilities = re.search(r"\.app-utilities \{([^}]+)\}", page).group(1)
    # Display stays at the viewport's bottom-right instead of scrolling away.
    assert "position: fixed" in utilities
    assert "env(safe-area-inset-bottom" in utilities and "right:" in utilities
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


def test_setup_research_examples_are_placeholders_and_keep_saved_values(monkeypatch, tmp_path):
    from dailydigest import web

    monkeypatch.setattr(web, '_get_profile_path', lambda: tmp_path/'missing-profile.yaml')
    rendered = web.setup_get(_request('GET', '/setup')).body.decode()
    assert 'placeholder="DNA nanotechnology' in rendered
    assert 'id="use-example-profile"' in rendered
    assert 'superlattice materials' in rendered
    assert 'id="topics"' in rendered and 'superlattice materials</textarea>' not in rendered
    path = tmp_path/'profile.yaml'
    path.write_text('bio: My own interests\nkeywords: [Custom topic]\n')
    monkeypatch.setattr(web, '_get_profile_path', lambda: path)
    rendered = web.setup_get(_request('GET', '/setup')).body.decode()
    assert 'Custom topic | 1</textarea>' in rendered
    assert 'My own interests</textarea>' in rendered
    assert 'id="use-example-profile"' not in rendered


def test_first_visit_puts_blank_interests_before_optional_examples(monkeypatch, tmp_path):
    from dailydigest import web

    monkeypatch.setattr(web, '_get_profile_path', lambda: tmp_path / 'missing-profile.yaml')
    rendered = web.setup_get(_request('GET', '/setup')).body.decode()
    assert 'Welcome to DailyDigest' in rendered
    assert rendered.index('id="interest-rows"') < rendered.index('id="example-discipline"')
    assert 'value="chemistry"' in rendered and 'value="astronomy"' in rendered
    assert '<details class="settings-disclosure" id="search-tips">' in rendered
    assert 'Nothing is saved until' in rendered


def test_shared_feedback_labels_describe_relevance_without_changing_scores():
    from jinja2 import Environment, FileSystemLoader
    environment = Environment(loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'), autoescape=True)
    controls = environment.get_template('_feedback_controls.html.j2').module.feedback_controls
    html = controls({'label': 'R1', 'current_grade': 40, 'current_reasons': []})
    labels = re.findall(r'data-grade="(\d+)"[^>]*>\s*<span class="vote-label">([^<]+)</span>', html)
    assert labels == [('100', 'Very relevant'), ('70', 'Relevant'), ('40', 'Less relevant'), ('10', 'Not relevant')]
    assert 'data-grade="40" type="button"' in html
    assert 'aria-pressed="true"' in html
    assert 'Less relevant and Not relevant lower future recommendations' in html


@pytest.mark.parametrize("first_run", [True, False])
def test_setup_timezone_detection_is_only_enabled_on_first_run(
    tmp_path, monkeypatch, first_run
):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(web, "_profile_exists", lambda: not first_run)
    response = web.setup_get(_request(path="/setup"))
    assert response.status_code == 200
    expected = "true" if first_run else "false"
    assert f'data-detect-timezone="{expected}"' in _text_payload(response)


def test_future_dated_digest_after_timezone_change_stays_in_date_order(tmp_path, monkeypatch):
    from dailydigest import web
    profile_path = tmp_path / 'profile.yaml'
    profile_path.write_text('name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n')
    monkeypatch.setattr(web, '_get_profile_path', lambda: profile_path)
    monkeypatch.setattr(web, '_digest_id', lambda: '2026-05-05')
    available = ['2026-05-05', '2026-05-03']
    monkeypatch.setattr(web, '_recent_digest_ids', lambda _today, limit=3: list(available))
    monkeypatch.setattr(web, '_digest_exists', lambda value: value in {*available, '2026-05-06'})
    _stub_index_dependencies(web, monkeypatch)
    monkeypatch.setattr(web, '_load_today', lambda *_args, **_kwargs: ([], {}))
    page = _text_payload(web.index(_request('GET', '/'), digest_id='2026-05-06'))
    assert '-1 days' not in page and 'from -1' not in page
    picker = page[page.index('<option value="/?digest_id=2026-05-06" selected>'):]
    assert picker.index('2026-05-06') < picker.index('value="/"') < picker.index('digest_id=2026-05-03')
