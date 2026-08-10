from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_windows_launchers_use_frozen_sync_and_localhost():
    start = (ROOT / "scripts" / "start.ps1").read_text()
    install = (ROOT / "scripts" / "install.ps1").read_text()
    batch = (ROOT / "DailyDigest-Windows.bat").read_text()

    assert "sync --frozen" in start
    assert '"127.0.0.1"' in start
    assert '"run", "dd", "start"' in start
    assert "https://astral.sh/uv/install.ps1" in install
    assert "scripts\\install.ps1" in batch


@pytest.mark.skipif(os.name != "nt", reason="PowerShell parser check runs on Windows CI.")
@pytest.mark.parametrize("name", ["start.ps1", "install.ps1"])
def test_powershell_launchers_parse_on_windows(name):
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell is not None
    script = str(ROOT / "scripts" / name).replace("'", "''")
    command = f"[void][ScriptBlock]::Create([IO.File]::ReadAllText('{script}'))"

    result = subprocess.run(
        [shell, "-NoProfile", "-Command", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
