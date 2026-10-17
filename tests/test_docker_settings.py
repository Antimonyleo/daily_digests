"""Compose keeps bootstrap config read-only and durable user saves in data/."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml


def test_compose_mounts_bootstrap_read_only_and_selects_directory_override():
    compose = yaml.safe_load((Path(__file__).resolve().parents[1] / 'docker-compose.yml').read_text())
    service = compose['services']['dailydigest']
    assert './.env:/app/.env:ro' in service['volumes']
    assert service['environment']['DD_ENV_PATH'] == 'data/.env'
    assert not any(key.startswith('LLM_') for key in service['environment'])


def test_explicit_writable_env_layers_after_bootstrap_without_migration(tmp_path, monkeypatch):
    from dailydigest import config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DD_APP_ROOT', str(tmp_path))
    monkeypatch.setenv('DD_ENV_PATH', 'data/.env')
    monkeypatch.delenv('TOP_RESEARCH', raising=False)
    (tmp_path / '.env').write_text('TOP_RESEARCH=3\nDIGEST_HOUR=11\n')
    assert config.load_settings().top_research == 3
    assert not (tmp_path / 'data/.env').exists()
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/.env').write_text('TOP_RESEARCH=8\n')
    settings = config.load_settings()
    assert (settings.top_research, settings.digest_hour) == (8, 11)
    assert (tmp_path / '.env').read_text() == 'TOP_RESEARCH=3\nDIGEST_HOUR=11\n'


def test_docker_save_and_restart_cover_custom_profile_outside_data(tmp_path, monkeypatch):
    from dailydigest import config, web
    from tests.test_web import _request

    data = tmp_path / 'data'
    data.mkdir()
    bootstrap = tmp_path / '.env'
    bootstrap.write_text('PROFILE_PATH=private/profile.yaml\nLLM_BACKEND=extractive\nTOP_RESEARCH=3\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DD_APP_ROOT', str(tmp_path))
    monkeypatch.setenv('DD_ENV_PATH', 'data/.env')
    monkeypatch.delenv('PROFILE_PATH', raising=False)
    monkeypatch.delenv('OPPORTUNITY_PROFILE_PATH', raising=False)
    monkeypatch.delenv('TOP_RESEARCH', raising=False)
    config.get_settings.cache_clear()
    monkeypatch.setattr(web, '_REPO_ROOT', tmp_path)
    monkeypatch.setattr(web, '_ENV_PATH', config.settings_env_path(tmp_path))
    monkeypatch.setattr(web, 'SETTINGS', config.load_settings())
    request = _request('POST', '/setup', form={
        '_csrf_token': web._CSRF_TOKEN, 'bio': 'Changed', 'topics': 'RNA | 10',
        'llm_backend': 'extractive', 'top_research': '8',
    })
    response = asyncio.run(web.setup_post.__wrapped__(request))
    assert response.status_code == 303
    assert (tmp_path / 'private/profile.yaml').is_file()
    assert (data / '.env').is_file()
    assert bootstrap.read_text().endswith('TOP_RESEARCH=3\n')
    assert not (data / '.settings-transaction.json').exists()
    monkeypatch.delenv('TOP_RESEARCH', raising=False)
    config.get_settings.cache_clear()
    assert config.get_settings().top_research == 8


@pytest.mark.parametrize('selector', ['../outside.env', '/outside.env'])
def test_env_override_must_stay_inside_app_root(tmp_path, monkeypatch, selector):
    from dailydigest import config

    monkeypatch.setenv('DD_APP_ROOT', str(tmp_path))
    monkeypatch.setenv('DD_ENV_PATH', selector)
    with pytest.raises(ValueError, match='DD_ENV_PATH'):
        config.load_settings()


def test_directory_volume_journal_recovers_app_relative_targets_before_import(tmp_path):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'data').mkdir()
    (tmp_path / 'private').mkdir()
    (tmp_path / '.env').write_text('PROFILE_PATH=private/profile.yaml\n')
    (tmp_path / 'data/.env').write_bytes(b'TOP_RESEARCH=3\n')
    (tmp_path / 'private/profile.yaml').write_bytes(b'bio: old\nkeywords: [RNA]\n')
    script = r'''
import os
from pathlib import Path
from dailydigest.settings_transaction import settings_transaction, _durable_write
root = Path.cwd()
changes = {root / 'private/profile.yaml': b'bio: changed\nkeywords: [RNA]\n', root / 'data/.env': b'TOP_RESEARCH=8\n'}
with settings_transaction(root, changes, journal_root=root / 'data'):
    for path, value in changes.items():
        _durable_write(path, value)
    os._exit(74)
'''
    env = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'DD_APP_ROOT': str(tmp_path), 'DD_ENV_PATH': 'data/.env'}
    env.pop('TOP_RESEARCH', None)
    crashed = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=env, capture_output=True, timeout=30)
    assert crashed.returncode == 74, crashed.stderr.decode()
    assert (tmp_path / 'data/.settings-transaction.json').exists()
    recovered = subprocess.run([sys.executable, '-c', 'from dailydigest.config import SETTINGS; assert SETTINGS.top_research == 3'],
                               cwd=tmp_path, env=env, capture_output=True, timeout=30)
    assert recovered.returncode == 0, recovered.stderr.decode()
    assert (tmp_path / 'private/profile.yaml').read_bytes() == b'bio: old\nkeywords: [RNA]\n'
    assert not (tmp_path / 'data/.settings-transaction.json').exists()
