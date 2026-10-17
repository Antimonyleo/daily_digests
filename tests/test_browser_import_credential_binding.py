"""Portable imports cannot attach destination credentials to another service."""
import asyncio
import importlib.util
from pathlib import Path
import sqlite3

import pytest


@pytest.fixture
def runtime():
    spec = importlib.util.spec_from_file_location(
        'browser_import_binding_test', Path(__file__).parents[1] / 'browser/runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(root, env):
    (root / 'data').mkdir(parents=True)
    (root / '.env').write_text(env)
    (root / 'data/profile.yaml').write_text('bio: Private destination reader\n')
    with sqlite3.connect(root / 'data/digest.db') as db:
        db.execute('CREATE TABLE fixture_votes (value INTEGER)')
        db.execute('INSERT INTO fixture_votes VALUES (1)')


def _portable(runtime, root, env):
    (root / 'data').mkdir(parents=True)
    (root / '.env').write_text(env)
    (root / 'data/profile.yaml').write_text('bio: Imported reader\n')
    return runtime.export_portable_snapshot(root=root)


def _bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('saved,imported', [
    ('LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/v1\n', 'LLM_BACKEND=api\nLLM_BASE_URL=https://new.example/v1\n'),
    ('LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/v1\n', 'LLM_BACKEND=anthropic\nLLM_BASE_URL=https://old.example/v1\n'),
    ('LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/v1\n', 'LLM_BACKEND=API\nLLM_BASE_URL=https://new.example/v1\n'),
    ('LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/v1\n', 'llm_backend=api\nllm_base_url=https://new.example/v1\n'),
    ('LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/v1\n', 'LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/other\n'),
    ('LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/v1\n', 'LLM_BACKEND=api\nLLM_BASE_URL=https://old.example:8443/v1\n'),
    ('LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/v1\n', 'LLM_BACKEND=api\nLLM_BASE_URL=http://old.example/v1\n'),
    ('LLM_BACKEND=extractive\n', 'LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/v1\n'),
    ('', 'LLM_BACKEND=api\n'),
    ('LLM_BACKEND=anthropic\nLLM_BASE_URL=https://api.anthropic.com/v1\n', 'LLM_BACKEND=anthropic\n'),
    ('LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/v1\n', 'LLM_BACKEND=api\nLLM_BASE_URL=https://user:password@old.example/v1\n'),
    ('IMAP_HOST=mail.old.example\nIMAP_USER=reader\n', 'IMAP_HOST=mail.new.example\nIMAP_USER=reader\n'),
    ('IMAP_HOST=mail.old.example\nIMAP_USER=reader\n', 'imap_host=mail.new.example\nimap_user=reader\n'),
    ('IMAP_HOST=mail.old.example\nIMAP_USER=reader\n', 'IMAP_HOST=mail.old.example\nIMAP_USER=other\n'),
    ('IMAP_HOST=mail.old.example\nIMAP_USER=reader\n', 'IMAP_HOST=mail.old.example\nIMAP_USER=reader\nIMAP_PORT=1993\n'),
    ('IMAP_HOST=mail.old.example\nIMAP_USER=reader\n', 'IMAP_HOST=mail.old.example\nIMAP_USER=reader\nIMAP_PORT=bad\n'),
    ('', 'IMAP_HOST=mail.old.example\nIMAP_USER=reader\n'),
])
def test_rebinding_import_rejects_before_close_commit_or_any_private_mutation(
    runtime, tmp_path, monkeypatch, saved, imported,
):
    destination = tmp_path / 'destination'
    _state(destination, saved + 'LLM_API_KEY=destination-api-key\nIMAP_PASSWORD=destination-mail-password\n')
    payload = _portable(runtime, tmp_path / 'source', imported)
    before = _bytes(destination)
    closed, commits = [], []
    monkeypatch.setattr(runtime, 'close_app', lambda: closed.append(True))

    async def commit(snapshot):
        commits.append(snapshot)

    with pytest.raises(ValueError, match='credential|key|password'):
        asyncio.run(runtime.transactional_import(payload, commit, root=destination,
                                                scratch_root=tmp_path))
    assert _bytes(destination) == before
    assert closed == commits == []


@pytest.mark.parametrize('saved,imported', [
    ('LLM_BACKEND=api\nLLM_BASE_URL=https://provider.example/v1\n', 'LLM_BACKEND=api\nLLM_BASE_URL=https://PROVIDER.example:443/v1/\n'),
    ('LLM_BACKEND=api\n', 'LLM_BACKEND=api\nLLM_BASE_URL=https://api.openai.com/v1\n'),
    ('LLM_BACKEND=anthropic\nLLM_BASE_URL=https://api.anthropic.com/v1\n', 'LLM_BACKEND=ANTHROPIC\nLLM_BASE_URL=https://API.ANTHROPIC.COM:443/v1/\n'),
    ('', 'LLM_BACKEND=extractive\n'),
    ('', 'LLM_BACKEND=claude_cli\n'),
    ('IMAP_HOST=mail.example\nIMAP_USER=reader\n', 'IMAP_HOST=MAIL.EXAMPLE\nIMAP_PORT=993\nIMAP_USER=reader\n'),
    ('IMAP_HOST=mail.example\nIMAP_USER=reader\n', 'IMAP_HOST=mail.new.example\nIMAP_USER=\n'),
])
def test_matching_or_inactive_import_preserves_existing_private_credentials(
    runtime, tmp_path, monkeypatch, saved, imported,
):
    destination = tmp_path / 'destination'
    _state(destination, saved + 'LLM_API_KEY=destination-api-key\nIMAP_PASSWORD=destination-mail-password\n')
    payload = _portable(runtime, tmp_path / 'source', imported)
    monkeypatch.setattr(runtime, 'close_app', lambda: None)
    commits = []

    async def commit(snapshot):
        commits.append(snapshot)

    asyncio.run(runtime.transactional_import(payload, commit, root=destination, scratch_root=tmp_path))
    values = runtime._environment((destination / '.env').read_text())
    assert values['LLM_API_KEY'] == 'destination-api-key'
    assert values['IMAP_PASSWORD'] == 'destination-mail-password'
    assert len(commits) == 1


def test_explicit_lowercase_credentials_are_not_overwritten_by_destination_secrets(runtime, tmp_path):
    import json

    destination = tmp_path / 'destination'
    _state(destination, 'LLM_BACKEND=api\nLLM_BASE_URL=https://old.example/v1\nLLM_API_KEY=old-key\n')
    source = tmp_path / 'source'
    _state(source, 'llm_backend=api\nllm_base_url=https://new.example/v1\nllm_api_key=explicit-new-key\n')
    snapshot = json.loads(runtime.export_snapshot(source))
    # A backup can omit some credentials while explicitly carrying another.
    snapshot['credentialsIncluded'] = False
    merged = runtime.preserve_omitted_credentials(json.dumps(snapshot), root=destination)
    decoded = runtime.decode_snapshot(merged)
    values = runtime._environment(decoded[Path('.env')].decode())
    assert values['llm_api_key'] == 'explicit-new-key'
    assert 'LLM_API_KEY' not in values


def test_matching_credential_import_commit_failure_restores_all_private_bytes(runtime, tmp_path, monkeypatch):
    destination = tmp_path / 'destination'
    environment = ('LLM_BACKEND=api\nLLM_BASE_URL=https://provider.example/v1\n'
                   'LLM_API_KEY=destination-key\nIMAP_PASSWORD=unused-mail-password\n')
    _state(destination, environment)
    payload = _portable(runtime, tmp_path / 'source',
                        'LLM_BACKEND=api\nLLM_BASE_URL=https://provider.example/v1\n')
    before = _bytes(destination)
    monkeypatch.setattr(runtime, 'close_app', lambda: None)

    async def fail_commit(snapshot):
        assert runtime._environment((destination / '.env').read_text())['LLM_API_KEY'] == 'destination-key'
        raise OSError('fixture storage failure')

    with pytest.raises(OSError, match='fixture storage failure'):
        asyncio.run(runtime.transactional_import(payload, fail_commit, root=destination,
                                                scratch_root=tmp_path))
    assert _bytes(destination) == before
