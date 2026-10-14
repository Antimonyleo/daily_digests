"""Build an allowlisted static website, never copying a reader's private files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

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
        paths += [ROOT / "config/sources.yaml", HERE / "network.py"]
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


def gpu_model_files(model_root: Path, output: Path, cache: Path) -> None:
    """Pin the unfused float32 BGE export; the CPU export has GPU-incompatible attention."""
    graph = download(f"https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/{GPU_MODEL_REVISION}/onnx/model.onnx",
                     cache / "gpu-model.onnx", GPU_MODEL_SHA256)
    (output / "onnx").mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(model_root / name, output / name)
    shutil.copyfile(graph, output / "onnx/model.onnx")


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
    for name in ("index.html", "app.mjs", "compute.mjs", "style.css", "worker.mjs", "manifest.webmanifest",
                 "app-icon.svg", "sw.js"):
        shutil.copyfile(HERE / name, output / name)
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
    (assets / "dependencies.json").write_text(json.dumps({"pyodide": resolved, "micropip": []}))
    model_root = PROTO / "assets/models/bge-small"
    if not model_root.is_dir():
        raise ValueError("Prepare prototype model assets with npm run prepare-fixture first")
    shutil.copytree(model_root, assets / "models/bge-small")
    gpu_model_files(model_root, assets / "models/bge-small-webgpu", cache)
    public_snapshot = cache / "public-sources.json"
    if public_snapshot.is_file():
        snapshot = json.loads(public_snapshot.read_text())
        if snapshot.get("public_only") is not True:
            raise ValueError("Source snapshot must explicitly contain public metadata only")
        shutil.copyfile(public_snapshot, assets / "public-sources.json")
    # GitHub Pages must serve the node_modules directory without Jekyll filtering.
    (output / ".nojekyll").write_text("")
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    report = {"schema_version": 1, "private_data_included": False,
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
