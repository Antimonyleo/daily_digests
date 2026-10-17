"""External imports are bounded; existing browser state remains recoverable."""
import asyncio
import base64
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runtime():
    spec = importlib.util.spec_from_file_location('browser_import_limit_runtime', Path(__file__).parents[1] / 'browser/runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normal_import_rejects_10001_files_before_any_private_mutation(runtime, tmp_path, monkeypatch):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/profile.yaml').write_bytes(b'original profile\x00bytes')
    (tmp_path / '.env').write_bytes(b'LLM_API_KEY="private-fixture"\n')
    previous = runtime.export_snapshot(tmp_path)
    touched = []
    monkeypatch.setattr(runtime, 'close_app', lambda: touched.append('close'))
    async def commit(snapshot):
        touched.append('commit')
    payload = json.dumps({'format': 'dailydigest-browser-v1', 'files': {f'data/f{i}': '' for i in range(10001)}})
    with pytest.raises(ValueError, match='import limit'):
        asyncio.run(runtime.transactional_import(payload, commit, root=tmp_path))
    assert touched == []
    assert runtime.export_snapshot(tmp_path) == previous


def test_import_character_limit_is_checked_before_parsing_or_mutation(runtime, tmp_path, monkeypatch):
    assert runtime.MAX_IMPORT_CHARACTERS == 128 * 1024 * 1024
    monkeypatch.setattr(runtime, 'MAX_IMPORT_CHARACTERS', 128)
    previous = runtime.export_snapshot(tmp_path)
    touched = []
    monkeypatch.setattr(runtime, 'close_app', lambda: touched.append('close'))
    async def commit(snapshot):
        touched.append('commit')
    payload = json.dumps({'format': 'dailydigest-browser-v1', 'files': {'data/a': base64.b64encode(b'x' * 200).decode()}})
    with pytest.raises(ValueError, match='import limit'):
        asyncio.run(runtime.transactional_import(payload, commit, root=tmp_path))
    assert touched == []
    assert runtime.export_snapshot(tmp_path) == previous


def test_trusted_export_restore_and_import_rollback_preserve_state_above_limits(runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'MAX_IMPORT_FILES', 2)
    original = tmp_path / 'original'
    (original / 'data').mkdir(parents=True)
    for i in range(3):
        (original / f'data/f{i}').write_bytes(bytes([i, 255]))
    previous = runtime.export_snapshot(original)
    restored = tmp_path / 'restored'
    runtime.restore_snapshot(previous, root=restored)
    assert runtime.export_snapshot(restored) == previous
    payload = json.dumps({'format': 'dailydigest-browser-v1', 'files': {'data/new': ''}})
    async def failed(snapshot):
        raise RuntimeError('quota')
    with pytest.raises(RuntimeError, match='quota'):
        asyncio.run(runtime.transactional_import(payload, failed, root=restored))
    assert runtime.export_snapshot(restored) == previous
