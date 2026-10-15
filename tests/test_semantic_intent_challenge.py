"""Integrity checks for a scoring-blind, provisional reader-task challenge.

These validate provenance and evaluation boundaries, not ranking performance.
"""

import hashlib
import json
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "semantic_intent_challenge.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text())
REAL_CASES = [case for case in FIXTURE["cases"] if case["record_kind"] == "captured_real"]


def test_challenge_keeps_quality_and_alignment_separate():
    cases = FIXTURE["cases"]
    profiles = {profile["id"] for profile in FIXTURE["profiles"]}
    assert 20 <= len(cases) <= 40
    assert len({case["id"] for case in cases}) == len(cases)
    assert set(FIXTURE["label_policy"]) == {"core", "context", "offtopic", "uncertain"}
    assert FIXTURE["provenance"]["label_status"] == "provisional_not_expert_gold"
    assert FIXTURE["provenance"]["training_use"].startswith("forbidden")
    for case in cases:
        assert case["profile_id"] in profiles
        assert case["label"] in FIXTURE["label_policy"]
        assert case["rationale"]
        assert case["quality"] == "not_assessed"
        assert case["split"] == "challenge"
        if case["record_kind"] == "controlled_fictional":
            assert case["article"]["title"].startswith("Fictional contrast:")
            assert "url" not in case["article"]
            assert case["contrast_family"]


@pytest.mark.parametrize("case", REAL_CASES, ids=lambda case: case["id"])
def test_real_records_preserve_captured_metadata(case):
    source = FIXTURE_PATH.parents[2] / case["source_fixture"]
    captured = json.loads(source.read_text())
    profiles = {profile["id"]: profile for profile in captured["profiles"]}
    article = profiles[case["source_profile"]]["pool"][case["source_index"]]
    assert case["article"] == {key: article[key] for key in case["article"]}


def test_challenge_contains_reader_intent_reversals_and_metadata_uncertainty():
    labels_by_record = {}
    for case in REAL_CASES:
        identity = case["article"]["url"]
        labels_by_record.setdefault(identity, set()).add(case["label"])
    assert sum({"core", "context"} <= labels for labels in labels_by_record.values()) >= 3
    missing = [case for case in REAL_CASES if not case["article"]["abstract"]]
    assert missing
    assert all(case["label"] == "uncertain" for case in missing)
    assert any(case["challenge"] == "alias" for case in REAL_CASES)
    assert any(case["challenge"] == "spelling" for case in REAL_CASES)


def test_frozen_protocol_hash_matches_when_local_artifact_is_available():
    protocol_path = (
        FIXTURE_PATH.parents[2]
        / ".full-review/semantic-bridge-2026-10-15/challenge-protocol.json"
    )
    if not protocol_path.exists():
        pytest.skip("Local review artifact is intentionally outside tracked test fixtures")
    protocol = json.loads(protocol_path.read_text())
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == protocol["fixture_sha256"]
