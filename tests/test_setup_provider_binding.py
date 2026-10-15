"""Saved API secrets stay bound to their configured API and effective endpoint."""
import socket

import pytest
from fastapi.testclient import TestClient


ORIGIN = 'https://provider.example/v1'


@pytest.fixture(autouse=True)
def _block_socket_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Credential binding tests must never open a network connection')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


def _prepare(monkeypatch, tmp_path, *, backend='api', base_url=ORIGIN):
    from dailydigest import web

    profile = tmp_path / 'profile.yaml'
    profile.write_text('bio: Original reader\nkeywords: [Original topic]\n')
    env = tmp_path / '.env'
    env.write_text('LLM_API_KEY=fixture-origin-key\n')
    monkeypatch.setattr(web, '_get_profile_path', lambda: profile)
    monkeypatch.setattr(web, '_ENV_PATH', env)
    monkeypatch.setattr(web, 'SETTINGS', web.SETTINGS.model_copy(update={
        'llm_backend': backend, 'llm_base_url': base_url, 'llm_api_key': 'fixture-origin-key',
    }))
    return web, profile, env


def _form(web, *, backend='api', base_url=ORIGIN, key='***'):
    return {'_csrf_token': web._CSRF_TOKEN, 'topics': 'Changed topic | 100',
            'bio': 'Changed reader', 'llm_backend': backend, 'llm_base_url': base_url,
            'llm_model': 'fixture-model', 'llm_api_key': key, 'user_tz': 'UTC'}


@pytest.mark.parametrize('saved_backend,selected_backend,selected_url', [
    ('api', 'api', 'https://different-provider.example/v1'),
    ('api', 'api', 'https://provider.example/other-api'),
    ('api', 'api', 'https://provider.example:8443/v1'),
    ('api', 'api', 'http://provider.example/v1'),
    ('api', 'anthropic', ORIGIN),
    ('anthropic', 'api', ORIGIN),
    ('extractive', 'api', ORIGIN),
    ('claude_cli', 'anthropic', ORIGIN),
    ('codex_cli', 'api', ORIGIN),
])
@pytest.mark.parametrize('key', ['***', ''])
def test_rebinding_saved_secret_requires_explicit_key_before_any_write(
    monkeypatch, tmp_path, saved_backend, selected_backend, selected_url, key,
):
    web, profile, env = _prepare(monkeypatch, tmp_path, backend=saved_backend)
    before = profile.read_bytes(), env.read_bytes()
    writes = []
    monkeypatch.setattr(web, '_write_private_yaml', lambda *a: writes.append('profile'))
    monkeypatch.setattr(web, '_write_env_file', lambda *a: writes.append('credentials'))
    with TestClient(web.app) as client:
        response = client.request('POST', '/setup', data=_form(
            web, backend=selected_backend, base_url=selected_url, key=key), follow_redirects=False)
    assert response.status_code == 400
    assert writes == []
    assert 'fixture-origin-key' not in response.text
    assert (profile.read_bytes(), env.read_bytes()) == before


@pytest.mark.parametrize('base_url', [ORIGIN, ORIGIN + '/', 'https://PROVIDER.example:443/v1/'])
@pytest.mark.parametrize('key', ['***', ''])
def test_unchanged_api_endpoint_preserves_saved_secret(monkeypatch, tmp_path, base_url, key):
    web, _, env = _prepare(monkeypatch, tmp_path)
    with TestClient(web.app) as client:
        response = client.request('POST', '/setup', data=_form(web, base_url=base_url, key=key), follow_redirects=False)
    assert response.status_code == 303
    assert web._read_env_file(env)['LLM_API_KEY'] == 'fixture-origin-key'


@pytest.mark.parametrize('backend', ['api', 'anthropic'])
def test_explicit_new_key_allows_provider_change(monkeypatch, tmp_path, backend):
    web, _, env = _prepare(monkeypatch, tmp_path)
    with TestClient(web.app) as client:
        response = client.request('POST', '/setup', data=_form(web, backend=backend,
            base_url='https://different-provider.example/v1', key='fixture-new-key'), follow_redirects=False)
    assert response.status_code == 303
    assert web._read_env_file(env)['LLM_API_KEY'] == 'fixture-new-key'
    assert web._read_env_file(env)['LLM_BASE_URL'] == 'https://different-provider.example/v1'


@pytest.mark.parametrize('backend', ['extractive', 'claude_cli', 'codex_cli'])
def test_leaving_api_then_returning_cannot_implicitly_reuse_key(monkeypatch, tmp_path, backend):
    web, profile, env = _prepare(monkeypatch, tmp_path)
    monkeypatch.setenv('LLM_API_KEY', 'fixture-origin-key')
    with TestClient(web.app) as client:
        first = client.request('POST', '/setup', data=_form(web, backend=backend), follow_redirects=False)
        assert first.status_code == 303
        before = profile.read_bytes(), env.read_bytes()
        second = client.request('POST', '/setup', data=_form(web), follow_redirects=False)
    assert second.status_code == 400
    assert (profile.read_bytes(), env.read_bytes()) == before
