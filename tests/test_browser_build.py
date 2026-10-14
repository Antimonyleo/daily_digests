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
                             "config/sources.yaml", "browser/network.py"}
    with zipfile.ZipFile(archive) as package:
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
