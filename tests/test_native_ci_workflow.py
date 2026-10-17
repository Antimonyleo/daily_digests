"""Native CI must exercise this event's source with offline, read-only jobs."""
from pathlib import Path
import re
import shlex

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_native_ci_covers_linux_and_existing_cross_platform_launcher_modules():
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    assert {"push", "pull_request"} <= workflow["on"].keys()
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    linux = jobs["linux"]
    assert linux["runs-on"] == "ubuntu-latest"
    commands = [step["run"] for step in linux["steps"] if "run" in step]
    assert "uv sync --frozen --group dev" in commands
    assert "uv run --no-sync ruff check --select F,B src tests" in commands
    assert "uv run --no-sync pytest -rs" in commands
    launchers = jobs["native-launchers"]
    assert set(launchers["strategy"]["matrix"]["os"]) == {"windows-latest", "macos-latest"}
    assert launchers["runs-on"] == "${{ matrix.os }}"
    test_commands = [step["run"] for step in launchers["steps"] if "pytest" in step.get("run", "")]
    assert len(test_commands) == 1
    selected = {part for part in shlex.split(test_commands[0]) if part.startswith("tests/")}
    assert {"tests/test_windows_launcher.py", "tests/test_start_script.py",
            "tests/test_cli_process_tree.py", "tests/test_companion_cli.py",
            "tests/test_scheduler_ownership.py", "tests/test_settings_transaction.py",
            "tests/test_settings_config_recovery.py", "tests/test_summary_cancellation.py",
            "tests/test_companion_cancellation.py"} <= selected
    assert all((ROOT / path).is_file() for path in selected)
    for name, job in jobs.items():
        assert "permissions" not in job
        if name in {"linux", "native-launchers"}:
            assert job["env"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "LLM_API_KEY": ""}
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
            if step.get("uses", "").startswith("actions/checkout@"):
                assert "ref" not in step.get("with", {})  # Event commit; never another branch.
