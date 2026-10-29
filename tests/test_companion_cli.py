"""The packaged companion CLI is usable without a repository script path."""
from typer.testing import CliRunner

from dailydigest import companion
from dailydigest.cli import app


def test_companion_cli_forwards_flags_and_its_own_help(monkeypatch):
    calls = []
    monkeypatch.setattr(companion, 'main', lambda argv: calls.append(argv))
    runner = CliRunner()
    result = runner.invoke(app, ['companion', '--port', '9876', '--token-file', 'private/token'])
    assert result.exit_code == 0, result.output
    assert calls[-1] == ['--port', '9876', '--token-file', 'private/token']
    result = runner.invoke(app, ['companion', '--help'])
    assert result.exit_code == 0, result.output
    assert calls[-1] == ['--help']


def test_browser_schedule_command_is_accepted_by_companion(tmp_path, monkeypatch):
    """Copy/paste from the website reaches the real parser without starting a server."""
    import json
    import shlex
    import shutil
    import subprocess
    from pathlib import Path

    import pytest
    import uvicorn

    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to exercise the browser-generated command')
    root = Path(__file__).resolve().parents[1]
    module = (root / 'browser/app.mjs').as_uri()
    script = (
        f'import {{companionStartCommand}} from {json.dumps(module)};'
        'console.log(JSON.stringify(["https://reader.example", "http://127.0.0.1:8766"]'
        '.map(origin => companionStartCommand(origin, true))));'
    )
    commands = json.loads(subprocess.check_output(
        [node, '--input-type=module', '-e', script], text=True, timeout=15))
    calls, servers = [], []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(companion, 'create_app', lambda **kwargs: calls.append(kwargs) or 'test-app')
    monkeypatch.setattr(uvicorn, 'run', lambda app, **kwargs: servers.append((app, kwargs)))
    for index, command in enumerate(commands):
        args = shlex.split(command)
        assert args[:4] == ['uv', 'run', 'dd', 'companion']
        companion.main(args[4:] + ['--token-file', str(tmp_path / f'token-{index}')])
        assert calls[-1]['schedule'] is True
        assert calls[-1]['state_dir'] == Path('data/companion-state')
        assert servers[-1][1]['host'] == '127.0.0.1'
        assert servers[-1][1]['port'] == 8768
    ignored = subprocess.check_output(
        ['git', '-C', str(root), 'check-ignore', 'data/companion-state/snapshot.json'], text=True)
    assert ignored.strip() == 'data/companion-state/snapshot.json'
