"""Packaged templates/default sources work without a repository working directory."""

import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_declared_package_assets_work_from_an_isolated_installed_layout(tmp_path):
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    wheel = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]
    includes = wheel["force-include"]
    assert includes["templates"] == "dailydigest/templates"
    assert includes["config/sources.yaml"] == "dailydigest/defaults/sources.yaml"
    installed = tmp_path / "installed"
    shutil.copytree(ROOT / "src/dailydigest", installed / "dailydigest",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for source, target in includes.items():
        original, destination = ROOT / source, installed / target
        if original.is_dir():
            shutil.copytree(original, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
    working = tmp_path / "unrelated-working-directory"
    working.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(installed)
    env["SOURCES_PATH"] = "config/sources.yaml"
    result = subprocess.run([sys.executable, "-c", '''
from pathlib import Path
from dailydigest import config, email_render, web
from jinja2 import Environment, FileSystemLoader

assert Path(web.__file__).parent.parent == Path(__import__('os').environ['PYTHONPATH'])
assert web._TEMPLATE_DIR == Path(web.__file__).parent / 'templates'
assert email_render._TEMPLATE_DIR == web._TEMPLATE_DIR
environment = Environment(loader=FileSystemLoader(web._TEMPLATE_DIR), autoescape=True)
for name in environment.list_templates():
    environment.get_template(name)
assert config.load_sources()
print('Packaged assets loaded outside repository')
'''], cwd=working, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_wheel_contains_templates_and_default_sources_when_builder_available(tmp_path, monkeypatch):
    builder = pytest.importorskip("hatchling.build", reason="Hatchling is not a locked dev dependency; no download attempted")
    monkeypatch.chdir(ROOT)
    wheel = tmp_path / builder.build_wheel(str(tmp_path))
    with zipfile.ZipFile(wheel) as archive:
        for template in (ROOT / "templates").glob("*.j2"):
            assert archive.read(f"dailydigest/templates/{template.name}") == template.read_bytes()
        assert archive.read("dailydigest/defaults/sources.yaml") == (ROOT / "config/sources.yaml").read_bytes()
        assert not any(name.endswith(".env") or name.startswith("data/") for name in archive.namelist())
