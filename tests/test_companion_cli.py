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
