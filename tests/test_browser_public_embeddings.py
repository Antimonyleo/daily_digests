"""Public precomputation reuses exact browser vectors, never native caches."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    'public_embeddings_builder', Path(__file__).resolve().parents[1] / 'browser/public_embeddings.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def public_snapshot():
    return {'public_only': True, 'sources': {
        'public': {'status': 'ok', 'items': [
            {'title': ' First ', 'abstract': ' Abstract ', 'profile': 'PRIVATE_SENTINEL'},
            {'title': 'Second', 'abstract': ''},
            {'title': 'First', 'abstract': 'Abstract'}]},
        'failed': {'status': 'error', 'items': [{'title': 'Never include me'}]}}}


def vectors(texts):
    return np.asarray([np.eye(1, 384, k=len(text) % 384)[0] for text in texts], dtype=np.float32)


def test_public_only_exact_item_text_and_deduplication():
    assert sorted(module.public_texts(public_snapshot())) == ['First. Abstract', 'Second']
    with pytest.raises(ValueError, match='public'):
        module.public_texts({'public_only': False, 'sources': {}})
    assert len(module.public_texts(public_snapshot(), max_documents=1)) == 1


def test_signature_binds_cpu_assets_and_actual_tokenization_not_native_model_name(tmp_path):
    model = tmp_path / 'model'
    (model / 'onnx').mkdir(parents=True)
    for name in module.MODEL_FILES:
        (model / name).write_bytes(name.encode())
    files = []
    for name in ('compute.mjs', 'bert.mjs', 'package-lock.json'):
        file = tmp_path / name
        file.write_text('original')
        files.append(file)
    signature = module.embedding_signature(model, *files)
    (model / 'onnx/model.onnx').write_bytes(b'changed graph')
    assert module.embedding_signature(model, *files) != signature
    (model / 'onnx/model.onnx').write_bytes(b'onnx/model.onnx')
    assert module.embedding_signature(model, *files) == signature
    files[1].write_text('changed tokenizer preparation')
    assert module.embedding_signature(model, *files) != signature


def test_warm_refresh_reuses_compatible_public_vectors_and_keeps_bound(tmp_path):
    first = tmp_path / 'first'
    result = module.publish_vectors(first, public_snapshot(), 'cpu-fixture', vectors)
    assert result['computed'] == 2 and result['bytes'] == 2 * 384 * 4
    manifest = json.loads((first / 'public-embeddings.json').read_text())
    assert manifest['backend'] == 'browser-onnx' and manifest['dtype'] == 'float32-le'
    assert 'PRIVATE_SENTINEL' not in json.dumps(manifest)
    second = tmp_path / 'second'

    def must_not_encode(texts):
        raise AssertionError('Warm build must not load a model')

    report = module.publish_vectors(second, public_snapshot(), 'cpu-fixture', must_not_encode,
                                    previous=first / 'public-embeddings.json')
    assert report['reused'] == 2 and report['computed'] == 0
    assert (first / manifest['file']).read_bytes() == (second / manifest['file']).read_bytes()
    assert report['complete'] is True


def test_mismatched_or_corrupt_vectors_are_recomputed_and_failure_remains_partial(tmp_path):
    first = tmp_path / 'first'
    module.publish_vectors(first, public_snapshot(), 'other-model', vectors)
    report = module.publish_vectors(tmp_path / 'changed', public_snapshot(), 'cpu-fixture', vectors,
                                    previous=first / 'public-embeddings.json')
    assert report['reused'] == 0 and report['computed'] == 2
    manifest = json.loads((first / 'public-embeddings.json').read_text())
    (first / manifest['file']).write_bytes(b'corrupt')
    report = module.publish_vectors(tmp_path / 'corrupt', public_snapshot(), 'other-model', vectors,
                                    previous=first / 'public-embeddings.json')
    assert report['computed'] == 2

    def unavailable(texts):
        raise RuntimeError('No browser accelerator in this build')

    report = module.publish_vectors(tmp_path / 'missing', public_snapshot(), 'new', unavailable)
    assert not report['complete'] and report['computed'] == 0 and report['missing'] == 2
    assert 'No browser' in report['warning']


def test_nonfinite_or_nonunit_vectors_cannot_be_published_as_success(tmp_path):
    report = module.publish_vectors(tmp_path, public_snapshot(), 'cpu-fixture',
                                    lambda texts: np.full((len(texts), 384), np.nan))
    assert not report['complete'] and report['missing'] == 2
    assert json.loads((tmp_path / 'public-embeddings.json').read_text())['hashes'] == []


def test_partial_refresh_keeps_valid_previous_vectors_on_encoder_failure(tmp_path):
    first = tmp_path / 'first'
    module.publish_vectors(first, public_snapshot(), 'cpu-fixture', vectors)
    snapshot = public_snapshot()
    snapshot['sources']['public']['items'].append({'title': 'New public article'})

    def unavailable(texts):
        assert texts == ['New public article']
        raise RuntimeError('Temporary inference failure')

    report = module.publish_vectors(tmp_path / 'partial', snapshot, 'cpu-fixture', unavailable,
                                    previous=first / 'public-embeddings.json')
    assert report['reused'] == 2 and report['missing'] == 1 and not report['complete']
    assert len(module.read_vectors(tmp_path / 'partial/public-embeddings.json', 'cpu-fixture')) == 2


def test_corrupt_untrusted_manifest_cannot_escape_public_cache_directory(tmp_path):
    path = tmp_path / 'public-embeddings.json'
    path.write_text(json.dumps({'version': 1, 'public_only': True, 'signature': 'cpu-fixture',
                               'backend': 'browser-onnx', 'dimension': 384, 'dtype': 'float32-le',
                               'hashes': [], 'bytes': 0, 'file': '../.env'}))
    assert module.read_vectors(path, 'cpu-fixture') == {}


def test_build_reuses_vectors_without_launching_a_browser_and_stamps_worker(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('public_vector_build',
        Path(__file__).resolve().parents[1] / 'browser/build.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    prototype = tmp_path / 'prototype'
    model = prototype / 'assets/models/bge-small'
    (model / 'onnx').mkdir(parents=True)
    for name in module.MODEL_FILES:
        (model / name).write_text('fixture')
    (prototype / 'bge-tokenizer.mjs').write_text('fixture')
    (prototype / 'package-lock.json').write_text('{}')
    monkeypatch.setattr(builder, 'PROTO', prototype)
    signature = module.embedding_signature(builder.PROTO / 'assets/models/bge-small',
        builder.HERE / 'compute.mjs', builder.PROTO / 'bge-tokenizer.mjs', builder.PROTO / 'package-lock.json')
    cache = tmp_path / 'cache'
    module.publish_vectors(cache / 'public-embeddings', public_snapshot(), signature, vectors)
    site = tmp_path / 'site'
    site.mkdir()
    (site / 'worker.mjs').write_text("const signature = '__PUBLIC_EMBEDDING_SIGNATURE__';")

    def no_browser(*args, **kwargs):
        raise AssertionError('A warm hotfix build must not launch Chromium')

    monkeypatch.setattr(builder.subprocess, 'run', no_browser)
    report = builder.publish_public_embeddings(site, public_snapshot(),
        builder.PROTO / 'assets/models/bge-small', 'unused/', 'unused/', cache)
    assert report['reused'] == 2 and report['computed'] == 0
    assert signature in (site / 'worker.mjs').read_text()
    assert '__PUBLIC_EMBEDDING_SIGNATURE__' not in (site / 'worker.mjs').read_text()


def test_public_age_summary_preserves_oldest_original_fetch():
    spec = importlib.util.spec_from_file_location('public_age_build',
        Path(__file__).resolve().parents[1] / 'browser/build.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    summary = builder.public_source_summary({'sources': {
        'new': {'status': 'ok', 'fetched_at': '2026-10-14T08:00:00-05:00'},
        'old': {'status': 'ok', 'fetched_at': '2026-10-14T10:00:00+00:00', 'carried_forward': True},
        'failed': {'status': 'error'}}})
    assert summary == {'oldest_fetched_at': '2026-10-14T10:00:00+00:00',
                       'successful_sources': 2, 'failed_sources': 1, 'carried_forward_sources': 1}


def test_explicit_partial_checkpoint_preserves_only_completed_valid_prefix(tmp_path):
    def interrupted(texts):
        return {'partial': True, 'vectors': vectors(texts[:1]), 'warning': 'Bounded timeout'}

    report = module.publish_vectors(tmp_path, public_snapshot(), 'cpu-fixture', interrupted)
    assert report['computed'] == 1 and report['missing'] == 1 and not report['complete']
    assert report['warning'] == 'Bounded timeout'


@pytest.mark.skipif(os.name == 'nt', reason='POSIX process group fixture; taskkill needs Windows')
def test_precompute_timeout_stops_owned_launcher_and_descendant(tmp_path):
    marker = tmp_path / 'child.pid'
    code = '''import subprocess,sys,time,signal
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
def stop(*args):
 child.wait(timeout=2)
 raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
'''
    with pytest.raises(subprocess.TimeoutExpired):
        module.run_precompute([sys.executable, '-c', code, str(marker)], cwd=tmp_path, timeout=.3)
    pid = int(marker.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        status = Path(f'/proc/{pid}/stat')
        if status.exists() and status.read_text().split()[2] == 'Z':
            break
        time.sleep(.02)
    else:
        pytest.fail('Timed-out public precompute left a running descendant')
