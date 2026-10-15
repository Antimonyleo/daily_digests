"""Permanent source refusals should fail once; temporary failures can recover."""
import httpx
import pytest
from contextlib import contextmanager
from tenacity import RetryError, wait_none


def _client(respond):
    class FixtureClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            request = httpx.Request('GET', url)
            response = respond(request)
            response.request = request
            return response

        @contextmanager
        def stream(self, method, url, **kwargs):
            yield self.get(url)

    return FixtureClient


@pytest.mark.parametrize("adapter", ["rss", "arxiv"])
@pytest.mark.parametrize("status,attempts", [(403, 1), (404, 1), (410, 1), (429, 3), (503, 3)])
def test_source_retry_distinguishes_permanent_status(monkeypatch, adapter, status, attempts):
    from dailydigest.ingest import arxiv, rss

    module = rss if adapter == "rss" else arxiv
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(status)

    monkeypatch.setattr(httpx, "Client", _client(respond))
    monkeypatch.setattr(arxiv, "_arxiv_polite_wait", lambda: None)
    function = module._http_get_bytes if adapter == "rss" else module._http_get_text
    function = function.retry_with(wait=wait_none(), sleep=lambda _: None)
    with pytest.raises((httpx.HTTPStatusError, RetryError)):
        function("https://fixture.invalid/feed", *([] if adapter == "rss" else [{}]))
    assert len(calls) == attempts


def test_transient_network_failure_can_recover(monkeypatch):
    from dailydigest.ingest import rss

    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("fixture timeout", request=request)
        return httpx.Response(200, content=b"feed")

    monkeypatch.setattr(httpx, "Client", _client(respond))
    assert rss._http_get_bytes.retry_with(wait=wait_none(), sleep=lambda _: None)("https://fixture.invalid/feed") == b"feed"
    assert len(calls) == 2
