"""Interrupted private writes must never enter the container image context."""
from pathlib import Path


def test_container_context_excludes_private_recovery_artifacts():
    patterns = (Path(__file__).parents[1] / '.dockerignore').read_text().splitlines()
    assert '.settings-transaction*' in patterns
    assert '.env.*' in patterns
    assert '!.env.example' in patterns
    assert '.lavish' in patterns
