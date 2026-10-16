"""Semantic model publication keeps public assets pinned and embedding caches stable."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

from dailydigest.rank.semantic_assets import SEMANTIC_FILES, SEMANTIC_REVISION


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[1] / 'browser/build.py'
    spec = importlib.util.spec_from_file_location('semantic_asset_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_assets(tmp_path, monkeypatch, builder):
    files = {name: ('public ' + name).encode() for name in SEMANTIC_FILES}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    monkeypatch.setattr(builder, 'SEMANTIC_FILES', hashes)
    cache = tmp_path / 'cache'
    for name, content in files.items():
        path = cache / 'semantic-model' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (cache / 'semantic-model/.env').write_text('PRIVATE_SECRET')
    (cache / 'semantic-model/download_provenance.json').write_text('PRIVATE_LOCAL_PATH')
    monkeypatch.setattr(builder.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Unexpected network request'))
    return cache, files, hashes


def test_semantic_assets_copy_only_pinned_files_and_notice_without_network(tmp_path, builder, fake_assets):
    cache, files, hashes = fake_assets
    output = tmp_path / 'site'
    output.mkdir()
    (output / 'worker.mjs').write_text("const root = './assets/semantic-models/';")
    report = builder.semantic_model_files(output, cache)
    model = output / report['path'] / 'semantic-matcher'
    assert {p.relative_to(model).as_posix() for p in model.rglob('*') if p.is_file()} == set(files) | {'MODEL-NOTICE.txt'}
    assert report['revision'] == SEMANTIC_REVISION
    assert report['sha256'] == hashes
    assert report['bytes'] == sum(p.stat().st_size for p in model.rglob('*') if p.is_file())
    assert report['path'].encode() in (output / 'worker.mjs').read_bytes()
    assert 'Apache-2.0' in (model / 'MODEL-NOTICE.txt').read_text()
    assert 'Xenova' in (model / 'MODEL-NOTICE.txt').read_text()
    assert all((model / name).read_bytes() == content for name, content in files.items())
    assert not any(b'PRIVATE' in p.read_bytes() for p in model.rglob('*') if p.is_file())


def test_corrupt_cached_semantic_asset_stops_build(tmp_path, builder, fake_assets):
    cache, _, _ = fake_assets
    (cache / 'semantic-model/tokenizer.json').write_text('corrupted')
    output = tmp_path / 'site'
    output.mkdir()
    (output / 'worker.mjs').write_text('')
    with pytest.raises(ValueError, match='checksum'):
        builder.semantic_model_files(output, cache)
    assert not (output / 'assets/semantic-models').exists()


def test_semantic_model_urls_remain_immutable_while_engine_module_is_stamped(builder):
    text = "'./semantic.mjs' './assets/semantic-models/abc/semantic-matcher/config.json'"
    stamped = builder.stamp_urls(text, 'release')
    assert "'./semantic.mjs?build=release'" in stamped
    assert "'./assets/semantic-models/abc/semantic-matcher/config.json'" in stamped


def test_separate_semantic_assets_preserve_embedding_bundle_identity(tmp_path, builder, fake_assets):
    cache, _, _ = fake_assets
    def site(name, semantic):
        output = tmp_path / name
        for model in ('bge-small', 'bge-small-webgpu'):
            path = output / 'assets/models' / model / 'onnx/model.onnx'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'embedding graph')
        (output / 'worker.mjs').write_text("'./assets/models/' './assets/semantic-models/'")
        if semantic:
            builder.semantic_model_files(output, cache)
        return builder.relocate_models(output)
    assert site('without', False) == site('with', True)


def test_semantic_identity_changes_with_pinned_bytes_not_release_name(tmp_path, builder, fake_assets, monkeypatch):
    cache, _, hashes = fake_assets
    def publish(name):
        output = tmp_path / name
        output.mkdir()
        (output / 'worker.mjs').write_text("'./assets/semantic-models/'")
        return builder.semantic_model_files(output, cache)
    first = publish('first')
    assert publish('second')['model_id'] == first['model_id']
    changed = b'new pinned graph'
    (cache / 'semantic-model/onnx/model.onnx').write_bytes(changed)
    monkeypatch.setattr(builder, 'SEMANTIC_FILES', {**hashes, 'onnx/model.onnx': hashlib.sha256(changed).hexdigest()})
    assert publish('changed')['model_id'] != first['model_id']


def test_full_build_manifest_identifies_lazy_semantic_assets_and_ships_engine(tmp_path, builder, fake_assets, monkeypatch):
    import json
    import zipfile

    cache, _, hashes = fake_assets
    browser = cache.parent / 'browser'
    browser.mkdir()
    cache.rename(browser / '.build-cache')
    original = builder.HERE
    for name in ('index.html', 'app.mjs', 'reading-view.mjs', 'compute.mjs', 'public-embeddings.mjs', 'semantic.mjs', 'http.mjs', 'storage.mjs',
                 'style.css', 'worker.mjs', 'manifest.webmanifest', 'app-icon.svg', 'sw.js', 'runtime.py',
                 'requirements-browser.txt'):
        (browser / name).write_bytes((original / name).read_bytes())
    proto = tmp_path / 'prototype'
    for name in ('@huggingface/transformers/dist/transformers.min.js',
                 'onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs',
                 'onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm',
                 'pyodide/pyodide.mjs', 'pyodide/pyodide-lock.json'):
        path = proto / 'node_modules' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    (proto / 'package-lock.json').write_text('{}')
    (proto / 'bge-tokenizer.mjs').write_text('// tokenizer')
    model = proto / 'assets/models/bge-small/onnx/model.onnx'
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b'embedding')
    monkeypatch.setattr(builder, 'HERE', browser)
    monkeypatch.setattr(builder, 'PROTO', proto)
    monkeypatch.setattr(builder, 'pure_package_files', lambda cache: [])
    monkeypatch.setattr(builder, 'resolve_pyodide_packages', lambda lock, names: [])
    monkeypatch.setattr(builder, 'gpu_model_files', lambda root, output, cache: output.mkdir(parents=True))
    monkeypatch.setattr(builder, 'publish_public_embeddings', lambda *args: {
        'published': 0, 'complete': False, 'warning': 'No public snapshot in semantic asset fixture'})
    def archive(path, wheels):
        with zipfile.ZipFile(path, 'w') as output:
            output.writestr('public.py', '# public')
        return {'public.py': 'public hash'}
    monkeypatch.setattr(builder, 'source_archive', archive)
    output = tmp_path / 'site'
    report = builder.build(output)
    manifest = json.loads((output / 'assets/build-manifest.json').read_text())
    assert manifest['semantic_model'] == report['semantic_model']
    assert manifest['semantic_model']['sha256'] == hashes
    assert manifest['semantic_model']['lazy'] is True
    assert manifest['private_data_included'] is False
    assert (output / 'semantic.mjs').is_file()
    assert (output / 'public-embeddings.mjs').is_file()
    assert (output / 'reading-view.mjs').is_file()
    assert './reading-view.mjs?build=' + report['build_id'] in (output / 'app.mjs').read_text()
    assert './reading-view.mjs?build=' + report['build_id'] in (output / 'worker.mjs').read_text()
