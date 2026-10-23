"""Interrupted private writes must never enter the container image context."""
from pathlib import Path

import pytest


@pytest.mark.parametrize('filename', ['.gitignore', '.dockerignore'])
def test_reader_study_exports_are_excluded_from_release_contexts(filename):
    patterns = (Path(__file__).parents[1] / filename).read_text().splitlines()
    assert 'dailydigest-reader-response*.json' in patterns


def test_container_context_excludes_private_recovery_artifacts():
    patterns = (Path(__file__).parents[1] / '.dockerignore').read_text().splitlines()
    assert '.settings-transaction*' in patterns
    assert '.env.*' in patterns
    assert '!.env.example' in patterns
    assert '.lavish' in patterns
