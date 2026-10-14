"""Research identity must survive every quality gate used by a real brew."""

import json

import pytest

from dailydigest.pipeline import _quality_gate
from dailydigest.store import ItemRow


def _row(title, metadata=None):
    return ItemRow(
        id=1, source="Nature", section="research", external_id="identity-test",
        url="https://example.test/identity-test", title=title,
        abstract="We introduce a measured method that corrects assembly errors and improves yield.",
        metadata_json=json.dumps(metadata or {}),
    )


@pytest.mark.parametrize("title", [
    "Correction of sequence errors in programmable DNA assemblies",
    "Correction of protein design biases using experimental measurements",
])
def test_full_quality_gate_keeps_original_error_correction_research(title):
    row = _row(title, {"publication_types": ["Journal Article"]})
    kept, dropped = _quality_gate([row])
    assert kept == [row]
    assert dropped == []


@pytest.mark.parametrize("title,metadata", [
    ("Correction: Programmable DNA assemblies", {}),
    ("Correction to programmable DNA assemblies", {}),
    ("Author Correction: Programmable DNA assemblies", {}),
    ("Publisher Correction: Programmable DNA assemblies", {}),
    ("Erratum: Programmable DNA assemblies", {}),
    ("Retraction: Programmable DNA assemblies", {}),
    ("Correction of sequence errors in programmable DNA assemblies", {"is_retracted": True}),
    ("Correction of sequence errors in programmable DNA assemblies", {"publication_types": ["Correction"]}),
    ("Correction of sequence errors in programmable DNA assemblies", {"publication_types": ["Published Erratum"]}),
])
def test_full_quality_gate_preserves_confirmed_notice_exclusions(title, metadata):
    kept, dropped = _quality_gate([_row(title, metadata)])
    assert kept == []
    assert len(dropped) == 1


def test_original_record_with_erratum_in_relationship_is_not_a_notice():
    row = _row("Correction of sequence errors in programmable DNA assemblies", {
        "publication_types": ["Journal Article"],
        "comments_corrections": [{"ref_type": "ErratumIn", "pmid": "1234"}],
    })
    assert _quality_gate([row])[0] == [row]
