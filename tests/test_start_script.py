from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start.sh"


def test_start_script_fails_clearly_when_uv_missing(tmp_path):
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin"

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "DailyDigest needs uv" in result.stderr


def test_start_script_runs_uv_sync_then_dd_start_with_no_browser(tmp_path):
    log = tmp_path / "uv.log"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$UV_LOG\"\n"
        "exit 0\n"
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:/usr/bin:/bin"
    env["UV_LOG"] = str(log)
    env["NO_BROWSER"] = "1"
    env["HOST"] = "127.0.0.1"
    env["PORT"] = "9999"

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert log.read_text().splitlines() == [
        "sync --frozen",
        "run dd start --host 127.0.0.1 --port 9999 --no-browser",
    ]
