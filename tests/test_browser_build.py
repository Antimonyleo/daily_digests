"""Static publication must include application code and exclude reader state."""

import importlib.util
from pathlib import Path
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "browser_build", Path(__file__).resolve().parents[1] / "browser/build.py"
)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def test_source_archive_positive_allowlist_excludes_private_state(tmp_path, monkeypatch):
    root = tmp_path / "project"
    paths = {
        "src/dailydigest/core.py": "PUBLIC_CODE = True\n",
        "templates/page.j2": "Public template",
        "templates/FONT-LICENSE.txt": "Public fixture font license",
        "config/sources.yaml": "research: []\n",
        "browser/network.py": "# public network adapter\n",
        "data/profile.yaml": "PRIVATE_PROFILE_SENTINEL",
        ".env": "SECRET_SENTINEL",
        "config/opportunities.yaml": "PRIVATE_ELIGIBILITY_SENTINEL",
        ".full-review/report.json": "PRIVATE_REPORT_SENTINEL",
    }
    for name, content in paths.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(build, "ROOT", root)
    monkeypatch.setattr(build, "HERE", root / "browser")
    archive = tmp_path / "app.zip"
    manifest = build.source_archive(archive, [])
    assert set(manifest) == {"src/dailydigest/core.py", "templates/page.j2",
                             "templates/FONT-LICENSE.txt", "config/sources.yaml", "browser/network.py"}
    with zipfile.ZipFile(archive) as package:
        assert package.read("templates/FONT-LICENSE.txt") == b"Public fixture font license"
        contents = b"".join(package.read(name) for name in package.namelist())
        assert not any(value.encode() in contents for value in (
            "PRIVATE_PROFILE_SENTINEL", "SECRET_SENTINEL", "PRIVATE_ELIGIBILITY_SENTINEL",
            "PRIVATE_REPORT_SENTINEL"))


def test_pyodide_dependencies_resolve_aliases_and_cycles():
    lock = {"packages": {"pydantic": {"depends": ["pydantic_core"]},
                         "pydantic-core": {"depends": ["typing-extensions"]},
                         "typing-extensions": {"depends": ["pydantic"]}}}
    assert build.resolve_pyodide_packages(lock, ["pydantic"]) == [
        "pydantic", "pydantic-core", "typing-extensions"]


def test_build_refuses_to_overwrite_existing_output(tmp_path):
    (tmp_path / "existing.txt").write_text("Keep me")
    with pytest.raises(ValueError, match="new or empty"):
        build.build(tmp_path)
    assert (tmp_path / "existing.txt").read_text() == "Keep me"


def test_gpu_export_is_pinned_and_copies_only_model_configuration(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    for name in names:
        (model / name).write_text('{}')
    (model / '.env').write_text('PRIVATE_SENTINEL')
    calls = []

    def download(url, target, sha256):
        calls.append((url, sha256))
        target.parent.mkdir(parents=True)
        target.write_bytes(b'public model fixture')
        return target

    monkeypatch.setattr(build, 'download', download)
    output = tmp_path / 'output'
    build.gpu_model_files(model, output, tmp_path / 'cache')
    assert calls == [(f'https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/{build.GPU_MODEL_REVISION}/onnx/model.onnx', build.GPU_MODEL_SHA256)]
    assert {str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()} == {*names, 'onnx/model.onnx'}


def test_cached_dependency_checksum_mismatch_is_rejected(tmp_path):
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'damaged cached graph')
    with pytest.raises(ValueError, match='checksum mismatch'):
        build.download('https://unused.invalid/model', path, '0' * 64)


def test_release_urls_share_hash_and_model_bundles_stay_content_addressed():
    text = '''import { x } from './compute.mjs';
const worker = new URL('./worker.mjs?compute=v1', import.meta.url);
const sw = new URL('./sw.js?isolate=1', import.meta.url);
fetch('./assets/app.zip?compute=v1'); fetch('./assets/runtime.py');
fetch('./assets/public-sources.json'); fetch('./assets/models/bge-small/config.json');
<link href="./style.css"><script src="./app.mjs?compute=v1"></script>'''
    stamped = build.stamp_urls(text, 'abc123')
    for path in ('compute.mjs', 'worker.mjs', 'app.mjs', 'style.css', 'assets/app.zip', 'assets/runtime.py'):
        assert f'{path}?build=abc123' in stamped
    assert 'sw.js?isolate=1&build=abc123' in stamped
    assert "'./assets/public-sources.json?build=abc123'" in stamped
    assert "'./assets/models/bge-small/config.json'" in stamped
    assert '?compute=v1' not in stamped


def test_generated_release_detects_mixed_html_and_python_before_opening_state(tmp_path):
    for name, text in {'index.html': '<head></head><span id="download-size">estimate</span>',
                       'app.mjs': 'console.log("boot");', 'compute.mjs': 'export const value = 1;',
                       'worker.mjs': "py.unpackArchive(bundle, 'zip', { extractDir: '/app' });\nawait py.runPythonAsync(source);",
                       'assets/runtime.py': '# runtime', 'assets/dependencies.json': '{}',
                       'assets/models/bge-small/config.json': '{}',
                       'assets/models/bge-small-webgpu/onnx/model.onnx': 'optional GPU'}.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    with zipfile.ZipFile(tmp_path / 'assets/app.zip', 'w') as archive:
        archive.writestr('public.py', 'PUBLIC = True')
    build.stamp_release(tmp_path, 'abc123')
    html = (tmp_path / 'index.html').read_text()
    assert 'name="dailydigest-build" content="abc123"' in html
    assert 'up to' in html and 'optional GPU' not in html
    app = (tmp_path / 'app.mjs').read_text()
    assert '__dd_build' in app and 'location.replace' in app
    assert app.index('location.replace') < app.index('console.log')
    worker = (tmp_path / 'worker.mjs').read_text()
    assert 'Application release changed' in worker
    assert 'Python runtime release changed' in worker
    with zipfile.ZipFile(tmp_path / 'assets/app.zip') as archive:
        assert b'abc123' in archive.read('browser/build.json')


def test_dependencies_have_immutable_path_for_transitive_wasm_and_package_fetches(tmp_path):
    files = {"node_modules/pyodide/pyodide.mjs": "import './pyodide.asm.js';",
             "node_modules/pyodide/pyodide.asm.js": "PUBLIC",
             "node_modules/pyodide/runtime.wasm": "PUBLIC_BINARY",
             "worker.mjs": "import { loadPyodide } from './node_modules/pyodide/pyodide.mjs';\nnew URL('./node_modules/pyodide/', self.location.href);"}
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    relative = build.relocate_runtime(tmp_path, 'deps123')
    assert relative == 'runtime/deps123/node_modules/'
    assert not (tmp_path / 'node_modules').exists()
    assert (tmp_path / relative / 'pyodide/runtime.wasm').is_file()
    worker = (tmp_path / 'worker.mjs').read_text()
    assert './runtime/deps123/node_modules/pyodide/pyodide.mjs' in worker
    assert "new URL('./runtime/deps123/node_modules/pyodide/'" in worker
    assert (tmp_path / relative / 'pyodide/pyodide.mjs').read_text() == "import './pyodide.asm.js';"


def test_mixed_cached_release_guards_execute_before_storage_restore(tmp_path):
    """Execute generated JavaScript, including the one-reload loop guard."""
    import json
    import subprocess

    (tmp_path / 'assets').mkdir()
    (tmp_path / 'index.html').write_text('<head></head>')
    (tmp_path / 'app.mjs').write_text('globalThis.__releaseBooted = true;')
    (tmp_path / 'worker.mjs').write_text("py.unpackArchive(bundle, 'zip', { extractDir: '/app' });\nawait py.runPythonAsync(source);\nrestoreState();")
    (tmp_path / 'assets/runtime.py').write_text('# public runtime')
    with zipfile.ZipFile(tmp_path / 'assets/app.zip', 'w') as archive:
        archive.writestr('public.py', 'PUBLIC = True')
    build.stamp_release(tmp_path, 'release123')
    script = r'''
import assert from 'node:assert/strict';
const [app, worker] = JSON.parse(process.argv[1]);
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
const runApp = new AsyncFunction('document', 'location', app);
function shell(meta, href) {
  const redirects = [];
  const elements = {startup:{dataset:{}}, 'startup-title':{}, 'startup-status':{}, status:{}, retry:{hidden:true}};
  const reloads = [];
  return { document: {querySelector: () => ({content: meta}), getElementById: id => elements[id]},
           location: {href, replace: value => redirects.push(value), reload: () => reloads.push(true)},
           redirects, elements, reloads };
}
let value = shell('release123', 'https://example.test/daily_digests/');
await runApp(value.document, value.location);
assert.equal(globalThis.__releaseBooted, true);
globalThis.__releaseBooted = false;
value = shell('old', 'https://example.test/daily_digests/#/saved');
const stale = runApp(value.document, value.location);
await Promise.race([stale, new Promise(resolve => setTimeout(resolve, 5))]);
assert.equal(globalThis.__releaseBooted, false);
assert.equal(value.redirects.length, 1);
assert.ok(value.redirects[0].includes('__dd_build=release123'));
assert.ok(value.redirects[0].endsWith('#/saved'));
value = shell('old', 'https://example.test/daily_digests/?__dd_build=release123');
await assert.rejects(runApp(value.document, value.location), /Application release changed/);
assert.equal(value.redirects.length, 0);
assert.equal(value.elements.startup.dataset.state, 'error');
assert.match(value.elements['startup-title'].textContent, /Couldn’t finish updating/);
assert.match(value.elements['startup-status'].textContent, /Application release changed/);
assert.equal(value.elements.status.textContent, value.elements['startup-status'].textContent);
assert.equal(value.elements.retry.hidden, false);
value.elements.retry.onclick(); assert.equal(value.reloads.length, 1);
assert.equal(globalThis.__releaseBooted, false);
let restored = false;
const runWorker = new AsyncFunction('py', 'bundle', 'source', 'restoreState', worker);
const py = {unpackArchive() {}, runPython: () => 'old', runPythonAsync: async () => {},
            globals: {get: () => 'release123'}};
await assert.rejects(runWorker(py, null, '', () => {restored = true;}), /Application release changed/);
assert.equal(restored, false);
py.runPython = () => 'release123';
py.globals.get = () => 'old';
await assert.rejects(runWorker(py, null, '', () => {restored = true;}), /Python runtime release changed/);
assert.equal(restored, false);
py.globals.get = () => 'release123';
await runWorker(py, null, '', () => {restored = true;});
assert.equal(restored, true);
'''
    completed = subprocess.run(['node', '--input-type=module', '-e', script,
                                json.dumps([(tmp_path / 'app.mjs').read_text(),
                                            (tmp_path / 'worker.mjs').read_text()])],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_model_content_paths_change_only_when_graph_or_tokenizer_changes(tmp_path):
    def fixture(folder, *, cpu=b"cpu", tokenizer=b"tokenizer"):
        for name, content in {"bge-small/onnx/model.onnx": cpu,
                              "bge-small/tokenizer.json": tokenizer,
                              "bge-small-webgpu/onnx/model.onnx": b"gpu"}.items():
            path = folder / "assets/models" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (folder / "worker.mjs").write_text("env.localModelPath = new URL('./assets/models/', self.location.href).href;")
    first, same, graph_change, tokenizer_change = [tmp_path / name for name in ("first", "same", "graph", "tokenizer")]
    fixture(first)
    fixture(same)
    fixture(graph_change, cpu=b"new cpu")
    fixture(tokenizer_change, tokenizer=b"new tokenizer")
    model_path, model_id = build.relocate_models(first)
    assert build.relocate_models(same) == (model_path, model_id)
    assert build.relocate_models(graph_change)[1] != model_id
    assert build.relocate_models(tokenizer_change)[1] != model_id
    assert model_path == f"assets/model-bundles/{model_id}/"
    assert not (first / "assets/models").exists()
    assert (first / model_path / "bge-small/onnx/model.onnx").read_bytes() == b"cpu"
    assert (first / model_path / "bge-small-webgpu/onnx/model.onnx").read_bytes() == b"gpu"
    assert f"'./{model_path}'" in (first / "worker.mjs").read_text()


def test_build_accepts_explicit_validated_push_snapshot_but_rejects_unmarked(tmp_path):
    import json
    from datetime import datetime, timezone
    snapshot = {'version': 1, 'public_only': True, 'sources': {'research': {
        'kind': 'rss', 'section': 'research', 'url': 'https://fixture.invalid/feed',
        'category': None, 'days': 21, 'status': 'ok', 'items': [],
        'fetched_at': datetime.now(timezone.utc).isoformat(), 'carried_forward': True,
    }}}
    record = snapshot['sources']['research']
    record['latest_attempt'] = {**record, 'carried_forward': False}
    path = tmp_path / 'public-sources.json'
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match='fresh successes'):
        build.validate_public_snapshot(path)
    snapshot['publication'] = {'mode': 'push-carry-forward'}
    path.write_text(json.dumps(snapshot))
    assert build.validate_public_snapshot(path)['carried_forward'] == 1
    record['fetched_at'] = '2020-01-01T00:00:00+00:00'
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match='expired'):
        build.validate_public_snapshot(path)
