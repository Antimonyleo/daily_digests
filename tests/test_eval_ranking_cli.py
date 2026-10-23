"""The documented evaluation command and its legacy spelling remain usable."""

import json

import pytest
from typer.testing import CliRunner

from dailydigest.cli import app


@pytest.mark.parametrize("command", ["eval-ranking", "eval"])
def test_ranking_evaluation_command_on_empty_isolated_database(command):
    result = CliRunner().invoke(app, [command, "--section", "research", "--k", "11", "--json"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["section"] == "research"
    assert report["k"] == 11
    assert report["digests_total"] == 0
    assert report["judged_coverage_at_k"] is None
