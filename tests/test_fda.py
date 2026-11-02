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


def _mock_client(monkeypatch, handler):
    import httpx

    calls = []

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def get(self, url, params=None):
            request = httpx.Request("GET", url, params=params)
            calls.append(request)
            response = handler(request)
            response.request = request
            return response

    monkeypatch.setattr("dailydigest.ingest.fda.httpx.Client", Client)
    return calls


def test_openfda_no_matches_is_an_empty_result_not_a_failure(monkeypatch):
    import httpx

    calls = _mock_client(monkeypatch, lambda request: httpx.Response(
        404, json={"error": {"code": "NOT_FOUND", "message": "No matches found!"}}))
    spec = SourceSpec(name="openFDA", kind="fda_api", section="regulatory")
    assert FDASource().fetch(spec) == []
    assert len(calls) == 1


def test_openfda_permanent_client_error_fails_without_retries(monkeypatch):
    import httpx
    import pytest

    calls = _mock_client(monkeypatch, lambda request: httpx.Response(400, json={"error": {"code": "BAD_REQUEST"}}))
    spec = SourceSpec(name="openFDA", kind="fda_api", section="regulatory")
    with pytest.raises(RuntimeError, match="openFDA fetch failed"):
        FDASource().fetch(spec)
    assert len(calls) == 1
