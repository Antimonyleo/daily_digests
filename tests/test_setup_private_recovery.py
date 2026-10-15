"""A Settings save must never replace unreadable private profile files."""

import asyncio

import pytest
from fastapi import HTTPException

from tests.test_web import _request


@pytest.mark.parametrize("filename", ["profile.yaml", "opportunities.yaml"])
@pytest.mark.parametrize("damaged", ["description: [unfinished\n", "- not a YAML mapping\n"])
def test_setup_rejects_unreadable_profiles_before_any_private_write(monkeypatch, tmp_path, filename, damaged):
    from dailydigest import web

    profile = tmp_path / "profile.yaml"
    opportunity = tmp_path / "opportunities.yaml"
    env = tmp_path / ".env"
    profile.write_text("bio: Existing reader\nkeywords: [RNA]\nprivate_metadata: Retain original\n")
    opportunity.write_text("description: Existing private context\ncitizenship_or_residency: Retain original\n")
    env.write_text("# Existing settings\nCUSTOM_KEEP=retained\nLLM_BACKEND=extractive\n")
    (tmp_path / filename).write_text(damaged)
    originals = {path: path.read_bytes() for path in (profile, opportunity, env)}
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_get_opportunity_profile_path", lambda: opportunity)
    monkeypatch.setattr(web, "_ENV_PATH", env)
    writes = []
    original_yaml_writer = web._write_private_yaml
    original_env_writer = web._write_env_file

    def yaml_write(*args, **kwargs):
        writes.append("yaml")
        return original_yaml_writer(*args, **kwargs)

    def env_write(*args, **kwargs):
        writes.append("env")
        return original_env_writer(*args, **kwargs)

    monkeypatch.setattr(web, "_write_private_yaml", yaml_write)
    monkeypatch.setattr(web, "_write_env_file", env_write)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(web.setup_post(_request("POST", "/setup", form={
            "_csrf_token": web._CSRF_TOKEN,
            "name": "Changed reader", "topics": "RNA | 17.5", "llm_backend": "extractive",
            "include_opportunities": "true", "top_opportunities": "5", "include_events": "false",
            "opportunity_career_stage": "postdoctoral researcher",
            "opportunity_institution_type": "nonprofit university",
            "opportunity_country": "United States",
            "opportunity_applicant_role": "fellow or co-investigator",
        })))
    assert exc.value.status_code == 409
    assert filename in exc.value.detail
    assert writes == [], "Both profiles must be read before either profile or .env is written"
    assert {path: path.read_bytes() for path in originals} == originals
