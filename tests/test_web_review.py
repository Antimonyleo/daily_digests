from __future__ import annotations

import asyncio
import errno

import pytest
from fastapi import HTTPException

from tests.test_web import _request


@pytest.mark.parametrize("host", ["127.attacker.example", "127.0.0.1.attacker.example", "127.999.1.2"])
def test_loopback_guard_rejects_hostnames_with_loopback_prefix(host):
    from dailydigest import web

    async def downstream(_request):
        pytest.fail("untrusted host reached a private route")

    response = asyncio.run(
        web._loopback_host_only(_request(headers={"host": host}), downstream)
    )
    assert response.status_code == 403


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.0.0.2", "::1", "testserver"])
def test_loopback_guard_accepts_local_hosts(host):
    from dailydigest import web

    assert web._is_local_host(host)


@pytest.mark.parametrize("origin", ["http://localhost:9000", "https://localhost:8000", "http://["])
def test_origin_check_rejects_other_ports_schemes_and_malformed_origins(origin):
    from dailydigest import web

    with pytest.raises(HTTPException) as error:
        web._require_local_origin(
            _request(headers={"host": "localhost:8000", "origin": origin})
        )
    assert error.value.status_code == 403


@pytest.mark.parametrize("host, origin", [("localhost", "http://localhost:80"), ("[::1]:8000", "http://[::1]:8000")])
def test_origin_check_accepts_same_origin_with_equivalent_default_port(host, origin):
    from dailydigest import web

    web._require_local_origin(_request(headers={"host": host, "origin": origin}))


def test_non_ascii_csrf_form_token_is_rejected_cleanly():
    from dailydigest import web

    with pytest.raises(HTTPException) as error:
        web._require_csrf(_request(), {"_csrf_token": "☕"})
    assert error.value.status_code == 403


@pytest.mark.parametrize("body", [b"[]", b"null", b'"brew"', b"{", b"\xff"])
def test_run_start_rejects_invalid_json_without_starting_work(monkeypatch, body):
    from dailydigest import web

    runs = []
    monkeypatch.setattr(web, "_kick_off_run", lambda *args: runs.append(args))
    request = _request("POST", "/run/start", headers={"x-csrf-token": web._CSRF_TOKEN})
    request._body = body
    with pytest.raises(HTTPException) as error:
        asyncio.run(web.run_start(request))
    assert error.value.status_code == 400
    assert runs == []


def test_form_post_rejects_invalid_utf8():
    from dailydigest import web

    request = _request("POST", "/setup")
    request._body = b"name=\xff"
    with pytest.raises(HTTPException) as error:
        asyncio.run(web._read_urlencoded_form(request))
    assert error.value.status_code == 400


def test_private_write_preserves_original_when_atomic_replace_fails(tmp_path, monkeypatch):
    from dailydigest import web

    path = tmp_path / "profile.yaml"
    path.write_text("bio: keep this profile\n")

    def fail_replace(*args):
        raise OSError(errno.EIO, "simulated filesystem failure")

    monkeypatch.setattr(web.os, "replace", fail_replace)
    with pytest.raises(OSError):
        web._atomic_private_write(path, "bio: replacement\n")
    assert path.read_text() == "bio: keep this profile\n"
    assert list(tmp_path.iterdir()) == [path]


def test_private_write_cleans_temporary_file_after_flush_failure(tmp_path, monkeypatch):
    from dailydigest import web

    def fail_fsync(*args):
        raise OSError(errno.EIO, "simulated flush failure")

    monkeypatch.setattr(web.os, "fsync", fail_fsync)
    with pytest.raises(OSError):
        web._atomic_private_write(tmp_path / "profile.yaml", "bio: private\n")
    assert list(tmp_path.iterdir()) == []


def test_private_write_supports_bind_mounted_files(tmp_path, monkeypatch):
    from dailydigest import web

    path = tmp_path / ".env"
    path.write_text("LLM_BACKEND=extractive\n")

    def busy_mount(*args):
        raise OSError(errno.EBUSY, "bind-mounted file")

    monkeypatch.setattr(web.os, "replace", busy_mount)
    web._atomic_private_write(path, "LLM_BACKEND=api\n")
    assert path.read_text() == "LLM_BACKEND=api\n"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("content", ["bio: [invalid yaml", "- not a profile mapping\n", "[]"])
def test_rename_preserves_unreadable_existing_profile(tmp_path, monkeypatch, content):
    from dailydigest import web

    path = tmp_path / "profile.yaml"
    path.write_text(content)
    monkeypatch.setattr(web, "_get_profile_path", lambda: path)
    request = _request(
        "POST", "/profile/name", form={"name": "Reader", "_csrf_token": web._CSRF_TOKEN}
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(web.profile_name_post(request))
    assert error.value.status_code == 409
    assert path.read_text() == content


@pytest.mark.parametrize("enabled_section", ["include_opportunities", "include_events"])
@pytest.mark.parametrize("content", ["citizenship_or_residency: [truncated", "[]"])
def test_setup_preserves_all_files_when_opportunity_profile_is_unreadable(
    tmp_path, monkeypatch, enabled_section, content
):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    opportunity_path = tmp_path / "opportunities.yaml"
    env_path = tmp_path / ".env"
    original_profile = "bio: Original profile\nkeywords: [RNA]\n"
    original_env = "LLM_BACKEND=extractive\n"
    profile_path.write_text(original_profile)
    opportunity_path.write_text(content)
    env_path.write_text(original_env)
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_get_opportunity_profile_path", lambda: opportunity_path)
    monkeypatch.setattr(web, "_ENV_PATH", env_path)
    request = _request(
        "POST", "/setup", form={
            "_csrf_token": web._CSRF_TOKEN,
            "bio": "Updated profile",
            "topics": "RNA nanotechnology | 10",
            "llm_backend": "extractive",
            enabled_section: "true",
            "opportunity_career_stage": "assistant professor",
            "opportunity_institution_type": "nonprofit university",
            "opportunity_country": "United States",
            "opportunity_applicant_role": "principal investigator",
        },
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(web.setup_post(request))

    assert error.value.status_code == 409
    assert profile_path.read_text() == original_profile
    assert opportunity_path.read_text() == content
    assert env_path.read_text() == original_env


def _setup_save_case(tmp_path, monkeypatch, *, existing=True):
    from dailydigest import web

    paths = [tmp_path / 'profile.yaml', tmp_path / 'opportunities.yaml', tmp_path / '.env']
    contents = [b'bio: Original profile\r\nkeywords: [RNA]\r\n',
                b'description: Original private details\r\n',
                b'# Keep original formatting\r\nLLM_BACKEND=extractive\r\nLLM_API_KEY=old-secret\r\n']
    for path, content in zip(paths, contents, strict=True):
        if existing:
            path.write_bytes(content)
    monkeypatch.setattr(web, '_get_profile_path', lambda: paths[0])
    monkeypatch.setattr(web, '_get_opportunity_profile_path', lambda: paths[1])
    monkeypatch.setattr(web, '_ENV_PATH', paths[2])
    request = _request('POST', '/setup', form={
        '_csrf_token': web._CSRF_TOKEN, 'bio': 'Updated profile',
        'topics': 'RNA delivery | 10', 'llm_backend': 'extractive',
        'llm_api_key': 'submitted-secret', 'include_opportunities': 'true',
        'opportunity_career_stage': 'assistant professor',
        'opportunity_institution_type': 'nonprofit university',
        'opportunity_country': 'United States', 'opportunity_applicant_role': 'principal investigator',
    })
    return request, paths, contents


@pytest.mark.parametrize('stage', ['opportunity', 'env'])
@pytest.mark.parametrize('existing', [True, False])
def test_setup_rolls_back_partial_config_writes_without_reloading(tmp_path, monkeypatch, stage, existing):
    from dailydigest import web

    request, paths, originals = _setup_save_case(tmp_path, monkeypatch, existing=existing)
    reloaded = []
    monkeypatch.setattr(web, 'reload_settings', lambda: reloaded.append(True))
    before_key = web.os.environ.get('LLM_API_KEY')
    if stage == 'opportunity':
        write = web._write_private_yaml

        def fail_opportunity(path, payload):
            write(path, payload)
            if path == paths[1]:
                raise OSError('write failed; submitted-secret')

        monkeypatch.setattr(web, '_write_private_yaml', fail_opportunity)
    else:
        write = web._write_env_file

        def fail_env(path, updates):
            write(path, updates)
            raise OSError('write failed; submitted-secret')

        monkeypatch.setattr(web, '_write_env_file', fail_env)

    with pytest.raises((OSError, HTTPException)) as error:
        asyncio.run(web.setup_post(request))
    for path, original in zip(paths, originals, strict=True):
        if existing:
            assert path.read_bytes() == original
        else:
            assert not path.exists()
    assert reloaded == []
    assert web.os.environ.get('LLM_API_KEY') == before_key
    assert isinstance(error.value, HTTPException)
    assert error.value.status_code == 503
    assert 'submitted-secret' not in error.value.detail


def test_setup_reports_partial_state_if_exception_rollback_also_fails(tmp_path, monkeypatch):
    from dailydigest import web

    request, paths, originals = _setup_save_case(tmp_path, monkeypatch)
    write_env, write_private = web._write_env_file, web._atomic_private_write
    reloaded = []
    monkeypatch.setattr(web, 'reload_settings', lambda: reloaded.append(True))

    def fail_env(path, updates):
        write_env(path, updates)
        raise OSError('write failed; submitted-secret')

    def fail_rollback(path, content):
        if path == paths[0] and content == originals[0]:
            raise OSError('restore failed; submitted-secret')
        write_private(path, content)

    monkeypatch.setattr(web, '_write_env_file', fail_env)
    monkeypatch.setattr(web, '_atomic_private_write', fail_rollback)
    with pytest.raises((OSError, HTTPException)) as error:
        asyncio.run(web.setup_post(request))
    assert paths[1].read_bytes() == originals[1]
    assert paths[2].read_bytes() == originals[2]
    assert reloaded == []
    assert isinstance(error.value, HTTPException)
    assert error.value.status_code == 500
    assert 'could not be restored' in error.value.detail
    assert 'submitted-secret' not in error.value.detail


def test_private_write_restores_exact_bytes_on_bind_mounted_files(tmp_path, monkeypatch):
    from dailydigest import web

    path = tmp_path / '.env'
    path.write_bytes(b'changed\n')

    def busy_mount(*args):
        raise OSError(errno.EBUSY, 'bind-mounted file')

    monkeypatch.setattr(web.os, 'replace', busy_mount)
    original = b'# original\r\nLLM_BACKEND=extractive\r\n'
    web._atomic_private_write(path, original)
    assert path.read_bytes() == original
