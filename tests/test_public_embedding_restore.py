"""Only validated public assets may warm the build cache; failure is optional."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    browser = Path(__file__).resolve().parents[1] / 'browser'
    monkeypatch.syspath_prepend(str(browser))
    spec = importlib.util.spec_from_file_location('restore_public_fixture', browser / 'fetch_public_embeddings.py')
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    # Another build test can load the helper via importlib under a different name.
    assert sys.modules['public_embeddings'] is not None
    return loaded


def public_fixture():
    vector = np.zeros((1, 384), dtype='<f4')
    vector[0, 0] = 1
    raw = vector.tobytes()
    manifest = {'version': 1, 'public_only': True, 'signature': 'fixture-model',
                'backend': 'browser-onnx', 'dimension': 384, 'dtype': 'float32-le',
                'hashes': ['a' * 64], 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                'file': f'public-embeddings-{hashlib.sha256(raw).hexdigest()[:20]}.f32'}
    return manifest, raw


def transport(monkeypatch, module, manifest, raw):
    calls = []

    class Reply:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, limit):
            return self.value[:limit]

    def fetch(url, timeout):
        calls.append(url)
        assert timeout == 20
        return Reply(json.dumps(manifest).encode() if url.endswith('.json') else raw)

    monkeypatch.setattr(module, 'urlopen', fetch)
    return calls


def test_valid_public_vectors_restore_without_private_files(module, monkeypatch, tmp_path):
    manifest, raw = public_fixture()
    calls = transport(monkeypatch, module, manifest, raw)
    assert module.restore_public_cache(tmp_path, 'https://public.invalid/site/')
    assert len(calls) == 2
    assert {path.name for path in tmp_path.iterdir()} == {'public-embeddings.json', manifest['file']}


@pytest.mark.parametrize('manifest', [[], None, 'not a manifest'])
def test_malformed_optional_manifest_does_not_block_build_or_replace_cache(module, monkeypatch, tmp_path, manifest):
    previous = tmp_path / 'public-embeddings.json'
    previous.write_text('previous valid manifest')
    calls = transport(monkeypatch, module, manifest, b'')
    assert module.restore_public_cache(tmp_path, 'https://public.invalid/site') is False
    assert previous.read_text() == 'previous valid manifest'
    assert len(calls) == 1


@pytest.mark.parametrize('defect', ['path', 'checksum', 'model', 'vector', 'empty'])
def test_invalid_download_preserves_last_valid_manifest(module, monkeypatch, tmp_path, defect):
    manifest, raw = public_fixture()
    previous = tmp_path / 'public-embeddings.json'
    previous.write_text('previous valid manifest')
    if defect == 'path':
        manifest['file'] = '../../data/profile.yaml'
    elif defect == 'checksum':
        raw = b'damaged'
    elif defect == 'model':
        manifest['backend'] = 'different-backend'
    elif defect == 'vector':
        raw = bytes(len(raw))
        manifest['sha256'] = hashlib.sha256(raw).hexdigest()
        manifest['file'] = f'public-embeddings-{manifest["sha256"][:20]}.f32'
    else:
        raw = b''
        manifest.update(hashes=[], bytes=0, sha256=hashlib.sha256(raw).hexdigest())
        manifest['file'] = f'public-embeddings-{manifest["sha256"][:20]}.f32'
    calls = transport(monkeypatch, module, manifest, raw)
    assert not module.restore_public_cache(tmp_path, 'https://public.invalid/site')
    assert previous.read_text() == 'previous valid manifest'
    if defect == 'path':
        assert len(calls) == 1
