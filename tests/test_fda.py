from __future__ import annotations

from dailydigest.ingest.fda import FDASource
from dailydigest.models import SourceSpec


def test_fda_custom_query_preserves_default_date_window(monkeypatch):
    captured: dict[str, str] = {}

    def fake_get_json(_url: str, params: dict[str, str]) -> dict:
        captured.update(params)
        return {"results": []}

    monkeypatch.setattr("dailydigest.ingest.fda._get_json", fake_get_json)

    spec = SourceSpec(
        name="openFDA Drug Approvals",
        kind="fda_api",
        section="regulatory",
        endpoint="drug/drugsfda.json",
        query="submissions.submission_status:AP",
    )

    FDASource().fetch(spec)

    search = captured["search"]
    assert "submissions.submission_status_date:[" in search
    assert "submissions.submission_status:AP" in search
    assert " AND " in search


def test_fda_custom_query_with_explicit_date_is_not_rewritten(monkeypatch):
    captured: dict[str, str] = {}

    def fake_get_json(_url: str, params: dict[str, str]) -> dict:
        captured.update(params)
        return {"results": []}

    monkeypatch.setattr("dailydigest.ingest.fda._get_json", fake_get_json)

    custom = "submissions.submission_status_date:[20260501 TO 20260510]"
    spec = SourceSpec(
        name="openFDA Drug Approvals",
        kind="fda_api",
        section="regulatory",
        endpoint="drug/drugsfda.json",
        query=custom,
    )

    FDASource().fetch(spec)

    assert captured["search"] == custom
