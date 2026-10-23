"""Build an allowlisted static website, never copying a reader's private files."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import urllib.request
import uuid
import zipfile
from pathlib import Path

from dailydigest.rank.semantic_assets import (
    MODEL_NOTICE, SEMANTIC_FILES, SEMANTIC_REPOSITORY, SEMANTIC_REVISION,
)

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "browser"
PROTO = ROOT / "prototypes/browser-compute"
GPU_MODEL_REVISION = "ea104dacec62c0de699686887e3f920caeb4f3e3"
GPU_MODEL_SHA256 = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
PYODIDE_PACKAGES = [
    "numpy", "scipy", "scikit-learn", "pydantic", "sqlalchemy", "pyyaml", "jinja2",
    "tzdata", "httpx", "anyio", "httpcore", "certifi", "idna", "python-dateutil",
    "sniffio", "micropip", "h11", "ssl", "lzma",
]


def download(url: str, target: Path, sha256: str | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as f:
            shutil.copyfileobj(response, f)
        temporary.replace(target)
    if sha256 and hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"Dependency checksum mismatch: {target.name}")
    return target


def resolve_pyodide_packages(lock: dict, names: list[str]) -> list[str]:
    packages = lock["packages"]
    # Some lock dependencies use underscore spelling (pydantic_core).
    lookup = {name.replace("_", "-"): name for name in packages}
    resolved: set[str] = set()

    def add(name: str):
        key = lookup[name.replace("_", "-")]
        if key in resolved:
            return
        resolved.add(key)
        for dependency in packages[key].get("depends", []):
            add(dependency)

    for name in names:
        add(name)
    return sorted(resolved)


def pure_package_files(cache: Path):
    for line in (HERE / "requirements-browser.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as r:
            record = json.load(r)
        candidates = [r for r in record["urls"] if r["filename"].endswith("-none-any.whl")]
        if not candidates:
            raise ValueError(f"No pure Python wheel: {name}=={version}")
        wheel = candidates[0]
        yield download(wheel["url"], cache / wheel["filename"], wheel["digests"]["sha256"])


def source_archive(output: Path, wheels: list[Path]) -> dict[str, str]:
    manifest = {}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        # Positive allowlist, not a broad repository copy with private exclusions.
        paths = list((ROOT / "src/dailydigest").rglob("*.py"))
        paths += list((ROOT / "templates").glob("*.j2"))
        paths += [ROOT / "config/sources.yaml", HERE / "network.py", ROOT / "templates/FONT-LICENSE.txt"]
        for path in sorted(paths):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT)
            content = path.read_bytes()
            archive.writestr(str(relative), content)
            manifest[str(relative)] = hashlib.sha256(content).hexdigest()
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as package:
                for name in package.namelist():
                    if name.endswith("/"):
                        continue
                    if name.startswith("/") or ".." in Path(name).parts:
                        raise ValueError("Unsafe package path")
                    archive.writestr("site-packages/" + name, package.read(name))
        # langdetect distributes a pure-Python sdist rather than a wheel.
        import langdetect
        package_root = Path(langdetect.__file__).parent
        for path in package_root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, "site-packages/langdetect/" + str(path.relative_to(package_root)))
        import sgmllib
        archive.write(sgmllib.__file__, "site-packages/sgmllib.py")
    return manifest


def compile_startup_python(output: Path) -> dict:
    """Compile public startup modules using the exact deployed Python runtime."""
    result = subprocess.run(
        ['node', str(HERE / 'compile-python.mjs'), str(output.resolve())],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def gpu_model_files(model_root: Path, output: Path, cache: Path) -> None:
    """Pin the unfused float32 BGE export; the CPU export has GPU-incompatible attention."""
    graph = download(f"https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/{GPU_MODEL_REVISION}/onnx/model.onnx",
                     cache / "gpu-model.onnx", GPU_MODEL_SHA256)
    (output / "onnx").mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(model_root / name, output / name)
    shutil.copyfile(graph, output / "onnx/model.onnx")


def stamp_urls(text: str, build_id: str) -> str:
    """Version application URLs; metadata revalidates independently of releases."""
    def replace(match):
        path, query = match.group(1), match.group(2) or ""
        if path.startswith(("./assets/models/", "./assets/model-bundles/", "./assets/semantic-models/")):
            return match.group(0)
        parts = [part for part in query.lstrip("?").split("&")
                 if part and not part.startswith(("compute=", "build="))]
        parts.append("build=" + build_id)
        return path + "?" + "&".join(parts)
    return re.sub(r"(\./[^\s'\"`<>?]+\.(?:mjs|json|js|css|html|webmanifest|svg|zip|py))(?![a-zA-Z0-9_.])(\?[^\s'\"`<>]*)?", replace, text)


def semantic_model_files(output: Path, cache: Path) -> dict:
    """Publish only pinned public assets, independently of embedding caches."""
    files = {}
    for name, expected in SEMANTIC_FILES.items():
        files[name] = download(
            f"https://huggingface.co/{SEMANTIC_REPOSITORY}/resolve/{SEMANTIC_REVISION}/{name}",
            cache / "semantic-model" / name, expected)
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + bytes.fromhex(SEMANTIC_FILES[name]))
    digest.update(MODEL_NOTICE.encode())
    model_id = digest.hexdigest()[:16]
    relative = f"assets/semantic-models/{model_id}/"
    target = output / relative / "semantic-matcher"
    for name, source in files.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (target / "MODEL-NOTICE.txt").write_text(MODEL_NOTICE)
    worker = output / "worker.mjs"
    worker.write_text(worker.read_text().replace("./assets/semantic-models/", "./" + relative))
    return {"repository": SEMANTIC_REPOSITORY, "revision": SEMANTIC_REVISION,
            "sha256": dict(SEMANTIC_FILES), "path": relative, "model_id": model_id,
            "bytes": sum(path.stat().st_size for path in target.rglob("*") if path.is_file()),
            "dtype": "fp32", "lazy": True, "notice": relative + "semantic-matcher/MODEL-NOTICE.txt"}


def validate_public_snapshot(path: Path) -> dict:
    snapshot = json.loads(path.read_text())
    spec = importlib.util.spec_from_file_location("public_snapshot_validator", HERE / "collect_sources.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    return validator.validate_report(snapshot, allow_carry_forward=True)


def relocate_runtime(output: Path, dependency_id: str) -> str:
    """Content-address transitive JS/WASM/package URLs without duplicating assets."""
    relative = f"runtime/{dependency_id}/node_modules/"
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    (output / "node_modules").rename(target)
    for path in output.glob("*.mjs"):
        path.write_text(path.read_text().replace("./node_modules/", "./" + relative))
    return relative


def public_source_summary(snapshot: dict) -> dict:
    """Public collection age comes from original fetches, never build time."""
    sources = snapshot.get('sources', {}).values()
    healthy = [row for row in sources if row.get('status') == 'ok' and row.get('fetched_at')]
    oldest = min(healthy, key=lambda row: datetime.fromisoformat(row['fetched_at']), default=None)
    return {'oldest_fetched_at': oldest['fetched_at'] if oldest else None,
            'successful_sources': len(healthy),
            'failed_sources': sum(row.get('status') != 'ok' for row in snapshot.get('sources', {}).values()),
            'carried_forward_sources': sum(bool(row.get('carried_forward')) for row in healthy)}


def publish_public_embeddings(output, snapshot, model_root, model_path, runtime_path, cache):
    """Use exact browser CPU inference; reuse a compatible prior public cache.

    Missing Chromium or an interrupted optional precompute retains compatible
    records and leaves other documents to the visitor's normal local encoder.
    """
    spec = importlib.util.spec_from_file_location('public_embedding_builder', HERE / 'public_embeddings.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    signature = helper.embedding_signature(model_root, HERE / 'compute.mjs',
                                           PROTO / 'bge-tokenizer.mjs', PROTO / 'package-lock.json')
    worker = output / 'worker.mjs'
    worker.write_text(worker.read_text().replace('__PUBLIC_EMBEDDING_SIGNATURE__', signature))
    if snapshot is None:
        return {'published': 0, 'complete': False, 'warning': 'No public source snapshot; using local comparisons.'}
    previous = cache / 'public-embeddings/public-embeddings.json'

    def infer(texts):
        job = cache / 'public-embedding-jobs' / uuid.uuid4().hex
        job.mkdir(parents=True)
        plan, result = job / 'plan.json', job / 'vectors.f32'
        plan.write_text(json.dumps({'texts': texts, 'runtimePath': runtime_path, 'modelPath': model_path}))
        import numpy as np
        try:
            completed = helper.run_precompute(['node', str(HERE / 'precompute-public-embeddings.mjs'),
                                               str(output), str(plan), str(result)], cwd=ROOT)
            (job / 'report.json').write_text(completed.stdout)
            (job / 'progress.log').write_text(completed.stderr)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            if not result.exists():
                raise
            raw = result.read_bytes()
            # Only complete float32 rows from completed public chunks are usable.
            complete_bytes = len(raw) // (helper.DIMENSION * 4) * helper.DIMENSION * 4
            values = np.frombuffer(raw[:complete_bytes], dtype='<f4').reshape(-1, helper.DIMENSION)
            return {'partial': True, 'vectors': values,
                    'warning': 'Public CPU precompute was interrupted; missing articles use local comparisons.'}
        return np.frombuffer(result.read_bytes(), dtype='<f4').reshape(-1, helper.DIMENSION)

    report = helper.publish_vectors(output / 'assets', snapshot, signature, infer, previous=previous)
    # Content-addressed binary first, manifest last; incomplete pairs fail closed.
    manifest = json.loads((output / 'assets/public-embeddings.json').read_text())
    previous.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / 'assets' / manifest['file'], previous.parent / manifest['file'])
    temporary = previous.with_suffix('.json.part')
    shutil.copyfile(output / 'assets/public-embeddings.json', temporary)
    temporary.replace(previous)
    print(json.dumps({'public_embeddings': report}))
    return report



def relocate_models(output: Path) -> tuple[str, str]:
    """Give model graphs and tokenizers one content-addressed, reusable root."""
    source = output / "assets/models"
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(source).as_posix().encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    model_id = digest.hexdigest()[:16]
    relative = f"assets/model-bundles/{model_id}/"
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    worker = output / "worker.mjs"
    worker.write_text(worker.read_text().replace("./assets/models/", "./" + relative))
    return relative, model_id

def stamp_release(output: Path, build_id: str) -> None:
    """Guard mixed cached shells/bundles before restoring any private state."""
    for path in output.iterdir():
        if path.is_file() and path.suffix in {".mjs", ".js", ".html", ".css", ".webmanifest"}:
            text = stamp_urls(path.read_text(), build_id)
            if path.name == "sw.js":
                text = text.replace("manifest.webmanifest'", "manifest.webmanifest?build=" + build_id + "'")
                text = text.replace("app-icon.svg'", "app-icon.svg?build=" + build_id + "'")
                text = text.replace(":v1`", ":" + build_id + "`")
            path.write_text(text)
    index = output / "index.html"
    html = index.read_text().replace("<head>", '<head>\n  <meta name="dailydigest-build" content="' + build_id + '">', 1)
    initial_bytes = sum(path.stat().st_size for path in output.rglob("*")
                        if path.is_file() and "bge-small-webgpu" not in path.parts
                        and "semantic-models" not in path.parts)
    html = re.sub(r'(<span id="download-size">)[^<]*(</span>)',
                  lambda match: match[1] + f"up to {math.ceil(initial_bytes / 1_000_000)} MB including the CPU model; cached files are reused" + match[2], html)
    index.write_text(html)
    app = output / "app.mjs"
    guard = """// One cache-busting reload repairs an HTML/application release mismatch.
if (typeof document !== 'undefined' && document.querySelector('meta[name="dailydigest-build"]')?.content !== BUILD_ID) {
  const fresh = new URL(location.href);
  if (fresh.searchParams.get('__dd_build') !== BUILD_ID) {
    fresh.searchParams.set('__dd_build', BUILD_ID);
    location.replace(fresh.href);
    await new Promise(() => {});
  }
  const message = 'Application release changed. Reload this page to finish updating.';
  const startup = document.getElementById?.('startup');
  if (startup) startup.dataset.state = 'error';
  const title = document.getElementById?.('startup-title');
  if (title) title.textContent = 'Couldn’t finish updating your reading desk';
  for (const id of ['startup-status', 'status']) {
    const element = document.getElementById?.(id);
    if (element) element.textContent = message;
  }
  const retry = document.getElementById?.('retry');
  if (retry) { retry.hidden = false; retry.onclick = () => location.reload(); }
  throw new Error(message);
}
""".replace("BUILD_ID", json.dumps(build_id))
    app.write_text(guard + app.read_text())
    with zipfile.ZipFile(output / "assets/app.zip", "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("browser/build.json", json.dumps({"build_id": build_id}))
    runtime = output / "assets/runtime.py"
    runtime.write_text(runtime.read_text() + "\nBROWSER_BUILD_ID = " + repr(build_id) + "\n")
    worker = output / "worker.mjs"
    text = worker.read_text()
    text = text.replace("py.unpackArchive(bundle, 'zip', { extractDir: '/app' });",
        "py.unpackArchive(bundle, 'zip', { extractDir: '/app' });\n"
        + "    if (py.runPython(\"__import__('json').loads(open('/app/browser/build.json').read())['build_id']\") !== "
        + json.dumps(build_id) + ") throw new Error('Application release changed. Reload the page to finish updating.');")
    text = text.replace("await py.runPythonAsync(source);", "await py.runPythonAsync(source);\n"
        + "    if (py.globals.get('BROWSER_BUILD_ID') !== " + json.dumps(build_id)
        + ") throw new Error('Python runtime release changed. Reload the page to finish updating.');")
    worker.write_text(text)


def embed_shared_reading_styles(output: Path) -> None:
    """Inline authored CSS for an opaque, scriptless cached reading frame."""
    templates = ROOT / "templates"
    # Opaque article frames cannot fetch CSS under COEP. Embed the exact shared
    # source at build time, avoiding a network dependency before cached reading.
    reading_path = output / "reading-view.mjs"
    reading_source = reading_path.read_text()
    marker = "export const SHARED_READING_STYLES = null;"
    if reading_source.count(marker) != 1:
        raise ValueError("Shared reading style marker must occur exactly once")
    styles = "\n".join((templates / name).read_text() for name in (
        "_design_tokens.css.j2", "_reading_surface.css.j2"))
    reading_path.write_text(reading_source.replace(
        marker, "export const SHARED_READING_STYLES = " + json.dumps(styles) + ";"))


def publish_shared_reading_assets(output: Path) -> None:
    """Publish only authored shared components; never render a reader profile."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    templates = ROOT / "templates"
    shutil.copyfile(templates / "_pip.css.j2", output / "pip.css")
    shutil.copyfile(templates / "_reading_surface.css.j2", output / "reading-surface.css")
    script = (templates / "_pip.js.j2").read_text()
    (output / "pip.mjs").write_text(script + "\nexport { mountPip };\n")
    environment = Environment(loader=FileSystemLoader(templates), autoescape=True,
                              undefined=StrictUndefined)
    markup = environment.get_template("_pip.html.j2").render(daily_note="")
    (output / "pip.html").write_text(markup)


def build(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new or empty directory; choose --output")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir()
    modules = PROTO / "node_modules"
    runtime = modules / "pyodide"
    if not runtime.is_dir():
        raise ValueError("Install prototype dependencies with npm ci --ignore-scripts first")
    cache = HERE / ".build-cache"
    wheels = list(pure_package_files(cache / "wheels"))
    manifest = source_archive(assets / "app.zip", wheels)
    for name in ("index.html", "app.mjs", "guide.mjs", "linked-backup.mjs", "reading-view.mjs", "tab-handoff.mjs", "compute.mjs", "embedding-memo.mjs", "cpu-startup.mjs", "public-embeddings.mjs", "semantic.mjs", "http.mjs", "storage.mjs", "style.css", "worker.mjs", "manifest.webmanifest",
                 "app-icon.svg", "sw.js"):
        shutil.copyfile(HERE / name, output / name)
    shutil.copyfile(ROOT / "templates/_design_tokens.css.j2", output / "design-tokens.css")
    embed_shared_reading_styles(output)
    publish_shared_reading_assets(output)
    shutil.copyfile(HERE / "runtime.py", assets / "runtime.py")
    shutil.copyfile(PROTO / "bge-tokenizer.mjs", output / "bge-tokenizer.mjs")
    for package, files in {
        "@huggingface/transformers/dist": ["transformers.min.js"],
        # The pinned CPU backend's actual browser trace loads these two files.
        # Other ORT builds are unused and would add over 30 MB to publication.
        "onnxruntime-web/dist": ["ort-wasm-simd-threaded.jsep.mjs",
                                  "ort-wasm-simd-threaded.jsep.wasm"],
        "pyodide": [p.name for p in runtime.iterdir()
                    if p.is_file() and p.suffix in {".mjs", ".js", ".wasm", ".json", ".zip"}],
    }.items():
        target = output / "node_modules" / package
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copyfile(modules / package / name, target / name)
    lock = json.loads((runtime / "pyodide-lock.json").read_text())
    resolved = resolve_pyodide_packages(lock, PYODIDE_PACKAGES)
    for name in resolved:
        record = lock["packages"][name]
        path = download("https://cdn.jsdelivr.net/pyodide/v0.28.3/full/" + record["file_name"],
                        cache / "pyodide" / record["file_name"], record["sha256"])
        shutil.copyfile(path, output / "node_modules/pyodide" / path.name)
    compute_packages = resolve_pyodide_packages(lock, ["scipy", "scikit-learn"])
    deferred = set(compute_packages) - set(resolve_pyodide_packages(
        lock, [name for name in PYODIDE_PACKAGES if name not in {"scipy", "scikit-learn"}]))
    (assets / "dependencies.json").write_text(json.dumps({
        "pyodide": [name for name in resolved if name not in deferred],
        "compute": sorted(deferred), "micropip": []}))
    python_bytecode = compile_startup_python(output)
    model_root = PROTO / "assets/models/bge-small"
    if not model_root.is_dir():
        raise ValueError("Prepare prototype model assets with npm run prepare-fixture first")
    shutil.copytree(model_root, assets / "models/bge-small")
    gpu_model_files(model_root, assets / "models/bge-small-webgpu", cache)
    model_path, model_id = relocate_models(output)
    semantic_model = semantic_model_files(output, cache)
    public_snapshot = cache / "public-sources.json"
    snapshot = None
    if public_snapshot.is_file():
        validate_public_snapshot(public_snapshot)
        shutil.copyfile(public_snapshot, assets / "public-sources.json")
        snapshot = json.loads(public_snapshot.read_text())
    dependency_id = hashlib.sha256((PROTO / "package-lock.json").read_bytes()
                                   + (runtime / "pyodide-lock.json").read_bytes()).hexdigest()[:16]
    runtime_path = relocate_runtime(output, dependency_id)
    public_embeddings = publish_public_embeddings(output, snapshot, model_root, model_path, runtime_path, cache)
    # GitHub Pages must serve the node_modules directory without Jekyll filtering.
    (output / ".nojekyll").write_text("")
    release_inputs = {"source_hashes": manifest,
        "python_compiler": hashlib.sha256((HERE / 'compile-python.mjs').read_bytes()).hexdigest(),
        "python_requirements": hashlib.sha256((HERE / 'requirements-browser.txt').read_bytes()).hexdigest(),
        "builder": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "frontend": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()
        if path.is_file()}, "runtime": hashlib.sha256((assets / "runtime.py").read_bytes()).hexdigest(),
        "dependencies": hashlib.sha256((assets / "dependencies.json").read_bytes()).hexdigest(),
        "semantic_model": semantic_model, "public_embeddings": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(assets.glob('public-embeddings*')) if path.is_file()},
        "browser_lock": hashlib.sha256((PROTO / "package-lock.json").read_bytes()).hexdigest()}
    build_id = hashlib.sha256(json.dumps(release_inputs, sort_keys=True).encode()).hexdigest()[:16]
    stamp_release(output, build_id)
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    report = {"schema_version": 1, "build_id": build_id, "runtime_path": runtime_path,
              "dependency_id": dependency_id, "model_path": model_path, "model_id": model_id,
              "semantic_model": semantic_model,
              "python_bytecode": python_bytecode,
              "public_embeddings": public_embeddings,
              "public_sources": public_source_summary(snapshot or {}),
              "private_data_included": False,
              "source_hashes": manifest, "pyodide": "0.28.3", "bytes": total,
              "gpu_model": {"publisher": "Xenova/bge-small-en-v1.5", "revision": GPU_MODEL_REVISION,
                            "sha256": GPU_MODEL_SHA256, "dtype": "fp32", "graph_optimization": "disabled"},
              "python_dependencies": (HERE / "requirements-browser.txt").read_text().splitlines()}
    (assets / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "dist")
    args = parser.parse_args()
    try:
        report = build(args.output.resolve())
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1
    print(json.dumps({"status": "built", "output": str(args.output), "bytes": report["bytes"],
                      "private_data_included": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
