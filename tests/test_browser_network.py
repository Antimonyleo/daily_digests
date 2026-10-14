"""Fixture-only transport checks: no public or configured services contacted."""

import importlib.util
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import subprocess
import sys

import httpx
import pytest


spec = importlib.util.spec_from_file_location(
    "browser_network", Path(__file__).resolve().parents[1] / "browser" / "network.py"
)
network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network)


class XHR:
    status = 200
    responseURL = "https://example.test/api"

    def __init__(self):
        self.headers = []

    def open(self, *args):
        self.opened = args

    def setRequestHeader(self, name, value):
        self.headers.append((name, value))

    def send(self, body):
        self.body = body

    def getAllResponseHeaders(self):
        return "Content-Type: application/octet-stream\r\nX-Test: yes\r\n"


def client(xhr):
    return httpx.Client(transport=network.BrowserTransport(
        xhr_factory=lambda: xhr, response_bytes=lambda _: b"\xff\x00body",
        request_body=bytes,
    ))


def test_method_binary_headers_status_and_timeout():
    xhr = XHR()
    xhr.status = 429
    with client(xhr) as http:
        response = http.request("POST", "https://example.test/api", content=b"\xff\x00", timeout=3,
                             headers={"User-Agent": "private", "Origin": "fake",
                                      "Referer": "fake", "Sec-Secret": "fake",
                                      "X-Test": "ok", "Content-Type": "application/json"})
    assert xhr.opened == ("POST", "https://example.test/api", False)
    assert xhr.body == b"\xff\x00"
    assert xhr.timeout == 3000
    assert not xhr.withCredentials
    assert dict(xhr.headers)["x-test"] == "ok"
    assert all(key.lower() not in network._FORBIDDEN for key, _ in xhr.headers)
    assert all(not key.lower().startswith("sec-") for key, _ in xhr.headers)
    assert response.content == b"\xff\x00body"
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.extensions["browser_final_url"] == xhr.responseURL
    with pytest.raises(httpx.HTTPStatusError):
        response.raise_for_status()


@pytest.mark.parametrize("message,error", [("NetworkError: CORS", httpx.ConnectError),
                                         ("TimeoutError", httpx.ReadTimeout)])
def test_failed_requests_are_httpx_errors(message, error):
    xhr = XHR()

    def fail(body):
        raise RuntimeError(message)

    xhr.send = fail
    with client(xhr) as http, pytest.raises(error) as caught:
        http.request("GET", "https://example.test/api")
    assert str(caught.value.request.url) == "https://example.test/api"


def test_unreadable_response_fails_instead_of_succeeding():
    xhr = XHR()
    xhr.status = 0
    with client(xhr) as http, pytest.raises(httpx.ConnectError):
        http.request("GET", "https://example.test/api")


def test_install_preserves_explicit_transport_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(httpx.Client, "_init_transport", httpx.Client._init_transport)
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    network.install_httpx_transport()
    patched = httpx.Client._init_transport
    network.install_httpx_transport()
    assert httpx.Client._init_transport is patched
    with httpx.Client(trust_env=False) as http:
        assert isinstance(http._transport, network.BrowserTransport)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(204))) as http:
        assert http.request("GET", "https://example.test/api").status_code == 204


def test_unwanted_browser_redirect_fails_explicitly(monkeypatch):
    monkeypatch.setattr(httpx.Client, "_init_transport", httpx.Client._init_transport)
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    network.install_httpx_transport()
    xhr = XHR()
    xhr.responseURL = "https://example.test/final"
    with client(xhr) as http:
        with pytest.raises(httpx.UnsupportedProtocol, match="follow_redirects=False"):
            http.request("GET", "https://example.test/api")
        response = http.request("GET", "https://example.test/api", follow_redirects=True)
        assert response.extensions["browser_final_url"] == xhr.responseURL


@pytest.fixture
def snapshot(monkeypatch):
    from dailydigest.ingest.arxiv import ArxivSource
    from dailydigest.ingest.rss import RSSSource
    from dailydigest.models import SourceSpec

    instant = datetime(2026, 10, 13, tzinfo=timezone.utc)
    source_spec = SourceSpec(name="Public feed", kind="rss", url="https://example.test/rss")
    row = {"kind": "rss", "section": "research", "url": source_spec.url,
           "category": None, "days": 7, "status": "ok", "fetched_at": instant.isoformat(),
           "items": [{"source": source_spec.name, "section": "research", "external_id": "1",
                      "url": "https://example.test/article", "title": "Public article"}]}
    manifest = {"version": 1, "public_only": True, "sources": {source_spec.name: row}}

    def failed(self, spec, days=2):
        raise RuntimeError("CORS blocked")

    monkeypatch.setattr(RSSSource, "fetch", failed)
    monkeypatch.setattr(ArxivSource, "fetch", failed)
    loads = []

    def loader():
        loads.append(True)
        return manifest

    network.install_public_source_fallback("/assets/public-sources.json", loader=loader,
                                           now=lambda: instant)
    return RSSSource(), source_spec, manifest, instant, loads


def test_public_snapshot_restores_items_without_attempting_direct_fetch(snapshot):
    source, spec, manifest, instant, loads = snapshot
    items = source.fetch(spec, days=7)
    assert items[0].title == "Public article"
    assert "direct_error" not in source.last_fetch_diagnostics
    assert source.last_fetch_diagnostics["transport"] == "public_snapshot"
    assert source.last_fetch_diagnostics["snapshot_fetched_at"] == instant.isoformat()
    source.fetch(spec)
    assert len(loads) == 1


@pytest.fixture
def refreshable_snapshot(snapshot):
    source, spec, manifest, instant, loads = snapshot
    clock = {"now": instant}
    response = {"value": manifest}
    calls = []

    def loader():
        calls.append(clock["now"])
        value = response["value"]
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)

    network.install_public_source_fallback("/assets/public-sources.json", loader=loader,
                                           now=lambda: clock["now"])
    return source, spec, response, clock, calls


def test_snapshot_short_reuse_then_daily_refresh_gets_new_articles(refreshable_snapshot):
    source, spec, response, clock, calls = refreshable_snapshot
    assert source.fetch(spec)[0].title == "Public article"
    clock["now"] += timedelta(minutes=4)
    assert source.fetch(spec)[0].title == "Public article"
    assert len(calls) == 1
    clock["now"] += timedelta(hours=12)
    updated = response["value"]["sources"][spec.name]
    updated["fetched_at"] = clock["now"].isoformat()
    updated["items"][0]["title"] = "New daily article"
    assert source.fetch(spec)[0].title == "New daily article"
    assert len(calls) == 2
    assert source.last_fetch_diagnostics["snapshot_fetched_at"] == clock["now"].isoformat()


@pytest.mark.parametrize("refresh", ["failed", "invalid"])
def test_failed_refresh_retains_only_fresh_cache_and_records_error(refreshable_snapshot, refresh):
    source, spec, response, clock, calls = refreshable_snapshot
    source.fetch(spec)
    clock["now"] += timedelta(hours=12)
    response["value"] = RuntimeError("Snapshot refresh unavailable") if refresh == "failed" else {}
    assert source.fetch(spec)[0].title == "Public article"
    assert source.last_fetch_diagnostics["snapshot_refresh_error"]
    assert source.last_fetch_diagnostics["transport"] == "public_snapshot"
    source.fetch(spec)
    assert len(calls) == 2  # a failed refresh must not download once per source


@pytest.mark.parametrize("refresh", ["failed", "stale"])
def test_daily_refresh_cannot_bless_a_stale_snapshot(refreshable_snapshot, refresh):
    source, spec, response, clock, calls = refreshable_snapshot
    source.fetch(spec)
    clock["now"] += timedelta(hours=25)
    if refresh == "failed":
        response["value"] = RuntimeError("Snapshot refresh unavailable")
    with pytest.raises(RuntimeError, match="stale or future-dated"):
        source.fetch(spec)
    assert source.last_fetch_diagnostics["direct_error"] == "RuntimeError: CORS blocked"
    assert len(calls) == 2


def test_arxiv_uses_the_same_validated_public_snapshot(snapshot):
    from dailydigest.ingest.arxiv import ArxivSource

    source, spec, manifest, instant, loads = snapshot
    spec.kind = "arxiv"
    spec.url = None
    spec.category = "q-bio.QM"
    row = manifest["sources"][spec.name]
    row.update(kind=spec.kind, url=spec.url, category=spec.category)
    source = ArxivSource()
    assert source.fetch(spec)[0].title == "Public article"
    assert source.last_fetch_diagnostics["transport"] == "public_snapshot"


@pytest.mark.parametrize("days", [21, 90])
def test_rss_snapshot_ignores_days_and_keeps_native_feed_dates(snapshot, days):
    source, spec, manifest, instant, loads = snapshot
    manifest["sources"][spec.name]["items"][0]["published_at"] = "2000-01-01T00:00:00Z"
    items = source.fetch(spec, days=days)
    assert items[0].title == "Public article"
    assert items[0].published_at.year == 2000
    assert source.last_fetch_diagnostics["transport"] == "public_snapshot"


@pytest.fixture
def arxiv_snapshot(snapshot):
    from dailydigest.ingest.arxiv import ArxivSource

    source, spec, manifest, instant, loads = snapshot
    spec.kind, spec.url, spec.category = "arxiv", None, "q-bio.QM"
    row = manifest["sources"][spec.name]
    row.update(kind=spec.kind, url=spec.url, category=spec.category, days=21)
    original = row["items"][0]
    row["items"] = [{**original, "external_id": str(index), "title": f"Paper {index}",
                     "published_at": "2000-01-01T00:00:00Z"} for index in range(400)]
    return ArxivSource(), spec, row


def test_arxiv_long_snapshot_preserves_native_two_day_50_item_prefix(arxiv_snapshot):
    source, spec, row = arxiv_snapshot
    items = source.fetch(spec, days=2)
    assert [item.title for item in items] == [f"Paper {index}" for index in range(50)]
    assert all(item.published_at.year == 2000 for item in items)


def test_arxiv_21_day_snapshot_covers_90_day_native_400_item_cap(arxiv_snapshot):
    source, spec, row = arxiv_snapshot
    assert len(source.fetch(spec, days=90)) == 400
    assert source.last_fetch_diagnostics["transport"] == "public_snapshot"


def test_arxiv_short_collected_cap_falls_back_and_reports_real_failure(arxiv_snapshot):
    source, spec, row = arxiv_snapshot
    row["days"] = 7
    row["items"] = row["items"][:200]
    with pytest.raises(RuntimeError, match="snapshot window too short"):
        source.fetch(spec, days=21)
    assert source.last_fetch_diagnostics["direct_error"] == "RuntimeError: CORS blocked"


def test_successful_direct_fetch_when_snapshot_is_unavailable(monkeypatch):
    from dailydigest.ingest.arxiv import ArxivSource
    from dailydigest.ingest.rss import RSSSource
    from dailydigest.models import SourceSpec

    monkeypatch.setattr(RSSSource, "fetch", lambda self, spec, days: [])
    monkeypatch.setattr(ArxivSource, "fetch", ArxivSource.fetch)

    attempts = []
    clock = {"now": datetime(2026, 10, 13, tzinfo=timezone.utc)}

    def unavailable_snapshot():
        attempts.append(True)
        raise RuntimeError("Snapshot asset not found")

    network.install_public_source_fallback("/assets/public-sources.json", loader=unavailable_snapshot,
                                           now=lambda: clock["now"])
    source = RSSSource()
    spec = SourceSpec(name="Public feed", kind="rss")
    assert source.fetch(spec) == []
    assert source.last_fetch_diagnostics["transport"] == "direct"
    assert "Snapshot asset not found" in source.last_fetch_diagnostics["snapshot_error"]
    clock["now"] += timedelta(minutes=4)
    source.fetch(spec)
    assert len(attempts) == 1
    clock["now"] += timedelta(minutes=1)
    source.fetch(spec)
    assert len(attempts) == 2


def test_genuine_empty_snapshot_does_not_attempt_direct_fetch(snapshot):
    source, spec, manifest, instant, loads = snapshot
    manifest["sources"][spec.name]["items"] = []
    assert source.fetch(spec) == []
    assert source.last_fetch_diagnostics["transport"] == "public_snapshot"


@pytest.mark.parametrize("failure", ["missing", "failed", "stale", "future", "naive",
                                     "kind", "identity"])
def test_unusable_public_snapshot_is_a_real_source_failure(snapshot, failure):
    source, spec, manifest, instant, loads = snapshot
    row = manifest["sources"][spec.name]
    if failure == "missing":
        manifest["sources"].clear()
    elif failure == "failed":
        row["status"] = "error"
    elif failure == "stale":
        row["fetched_at"] = (instant - timedelta(hours=25)).isoformat()
    elif failure == "future":
        row["fetched_at"] = (instant + timedelta(hours=1)).isoformat()
    elif failure == "naive":
        row["fetched_at"] = instant.replace(tzinfo=None).isoformat()
    elif failure == "kind":
        row["kind"] = "arxiv"
    else:
        row["items"][0]["source"] = "another feed"
    with pytest.raises(RuntimeError, match="direct fetch and public snapshot failed"):
        source.fetch(spec, days=7)
    assert "snapshot_error" in source.last_fetch_diagnostics


def test_profile_driven_source_never_uses_public_snapshot(snapshot):
    source, spec, manifest, instant, loads = snapshot
    spec.profile_driven = True
    with pytest.raises(RuntimeError, match="CORS blocked"):
        source.fetch(spec)
    assert loads == []


def test_public_collector_filters_private_queries_and_reports_failure(tmp_path):
    collector_spec = importlib.util.spec_from_file_location(
        "public_collector", Path(__file__).resolve().parents[1] / "browser" / "collect_sources.py"
    )
    collector = importlib.util.module_from_spec(collector_spec)
    collector_spec.loader.exec_module(collector)
    config = tmp_path / "sources.yaml"
    config.write_text("research:\n  - {name: feed, kind: rss, url: 'https://example.test/rss'}\n"
                      "  - {name: private, kind: rss, profile_driven: true}\n"
                      "  - {name: query, kind: pubmed, profile_driven: true}\n")
    specs = collector.public_specs(config)
    assert [spec.name for spec in specs] == ["feed"]

    class FailedAdapter:
        def fetch(self, spec, days):
            raise RuntimeError("Publisher HTTP 403")

    report = collector.collect(specs, adapters={"rss": FailedAdapter})
    assert report["public_only"]
    assert report["days"] == 21
    row = report["sources"]["feed"]
    assert row["status"] == "error"
    assert row["fetched_at"] is None
    assert row["items"] == []
    assert "HTTP 403" in row["error"]


def test_collector_cli_ignores_private_cwd_dotenv_and_environment(tmp_path):
    collector = Path(__file__).resolve().parents[1] / "browser" / "collect_sources.py"
    (tmp_path / ".env").write_text("TOP_RESEARCH=0\nLLM_API_KEY=private-dotenv-sentinel\n")
    (tmp_path / "sources.yaml").write_text("research: []\n")
    environment = {**os.environ, "TMPDIR": str(tmp_path), "TOP_RESEARCH": "0",
                   "LLM_API_KEY": "private-environment-sentinel"}
    completed = subprocess.run(
        [sys.executable, str(collector), "--sources", "sources.yaml", "--output", "public.json"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=True,
    )
    text = (tmp_path / "public.json").read_text()
    assert json.loads(text)["sources"] == {}
    assert json.loads(text)["days"] == 21
    assert "private-" not in text + completed.stdout + completed.stderr
    assert not (tmp_path / "data").exists()
    scratch = next(tmp_path.glob("dailydigest-public-sources-*"))
    assert not (scratch / "unused.db").exists()
    assert not (scratch / "unused-profile.yaml").exists()
