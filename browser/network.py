"""HTTPX transport for synchronous requests inside a browser Web Worker.

XHR enforces the browser's CORS policy. It follows redirects itself and exposes
only CORS-visible response headers; neither restriction can be bypassed here.
Install only in the Pyodide worker, before importing the application.
"""

from functools import wraps
from datetime import datetime, timedelta, timezone
import json
import math
import time

import httpx


_FORBIDDEN = {
    "accept-charset", "accept-encoding", "access-control-request-headers",
    "access-control-request-method", "connection", "content-length", "cookie",
    "cookie2", "date", "dnt", "expect", "host", "keep-alive", "origin",
    "permissions-policy", "referer", "te", "trailer", "transfer-encoding",
    "upgrade", "user-agent", "via",
}


def _xhr_factory():
    from js import XMLHttpRequest

    return XMLHttpRequest.new()


def _response_bytes(xhr):
    from js import Uint8Array

    return bytes(Uint8Array.new(xhr.response).to_py()) if xhr.response is not None else b""


def _request_body(content):
    from js import Uint8Array

    # A typed array preserves arbitrary bytes, including non-UTF-8 content.
    array = Uint8Array.new(len(content))
    array.assign(content)
    return array


class BrowserTransport(httpx.BaseTransport):
    def __init__(self, xhr_factory=None, response_bytes=None, request_body=None):
        self._factory = xhr_factory or _xhr_factory
        self._bytes = response_bytes or _response_bytes
        self._body = request_body or _request_body

    def handle_request(self, request):
        xhr = self._factory()
        timeout = request.extensions.get("timeout", {}).get("read")
        started = time.monotonic()
        try:
            xhr.open(request.method, str(request.url), False)
            xhr.responseType = "arraybuffer"
            xhr.withCredentials = False
            if timeout is not None:
                xhr.timeout = max(1, int(timeout * 1000))
            for name, value in request.headers.multi_items():
                lower = name.lower()
                if lower in _FORBIDDEN or lower.startswith(("sec-", "proxy-")):
                    continue
                if lower in {"x-http-method", "x-http-method-override", "x-method-override"}:
                    if any(v.strip().upper() in {"CONNECT", "TRACE", "TRACK"} for v in value.split(",")):
                        continue
                xhr.setRequestHeader(name, value)
            content = request.read()
            xhr.send(self._body(content) if content else None)
        except Exception as exc:
            elapsed = time.monotonic() - started
            if "TimeoutError" in str(exc) or (timeout is not None and elapsed >= timeout):
                raise httpx.ReadTimeout("Browser request timed out", request=request) from exc
            raise httpx.ConnectError(
                f"Browser request failed (CORS or network): {exc}", request=request
            ) from exc
        if not xhr.status:
            raise httpx.ConnectError("Browser returned no readable HTTP response", request=request)
        final_url = str(xhr.responseURL)
        if (final_url and final_url != str(request.url)
                and request.extensions.get("browser_follow_redirects") is False):
            raise httpx.UnsupportedProtocol(
                "Browser followed a redirect despite follow_redirects=False; "
                "XHR cannot expose the original redirect response", request=request
            )
        headers = []
        for line in str(xhr.getAllResponseHeaders()).splitlines():
            if ":" in line:
                name, value = line.split(":", 1)
                headers.append((name.strip(), value.strip()))
        return httpx.Response(
            int(xhr.status), headers=headers, content=self._bytes(xhr),
            extensions={"browser_final_url": final_url},
        )


def install_httpx_transport():
    """Use XHR for ordinary HTTPX clients; retain explicit test transports.

    Idempotent. HTTPX timeout/read semantics remain usable, but XHR cannot expose
    intermediate redirects, prevent redirects, or provide separate connect and
    write timeout phases. No proxy or opaque-response fallback is installed.
    """
    original = httpx.Client._init_transport
    if getattr(original, "_browser_transport", False):
        return

    @wraps(original)
    def initialize(self, *args, **kwargs):
        transport = kwargs.get("transport")
        return transport if transport is not None else BrowserTransport()

    initialize._browser_transport = True
    httpx.Client._init_transport = initialize
    original_send = httpx.Client.send

    @wraps(original_send)
    def send(self, request, **kwargs):
        follow = kwargs.get("follow_redirects", httpx.USE_CLIENT_DEFAULT)
        if follow is httpx.USE_CLIENT_DEFAULT:
            follow = self.follow_redirects
        request.extensions["browser_follow_redirects"] = follow
        return original_send(self, request, **kwargs)

    httpx.Client.send = send


def install_public_source_fallback(snapshot_url, *, loader=None, now=None):
    """Use fresh public metadata for RSS/arXiv, then try direct fetching.

    The worker supplies a same-origin asset URL; no reader information is sent.
    Missing, failed, mismatched, insufficient-capacity, or stale snapshots fail the
    source normally so existing pipeline coverage gates remain authoritative.
    """
    from dailydigest.ingest.arxiv import ArxivSource
    from dailydigest.ingest.rss import RSSSource
    from dailydigest.models import Item

    if loader is None:
        def loader():
            from js import location
            from urllib.parse import urljoin, urlparse

            target = urljoin(str(location.href), snapshot_url)
            parsed, current = urlparse(target), urlparse(str(location.href))
            if (parsed.scheme, parsed.netloc) != (current.scheme, current.netloc):
                raise RuntimeError("Public source snapshot must have the worker's origin")
            with httpx.Client(follow_redirects=False, timeout=20) as client:
                response = client.get(target)
                response.raise_for_status()
                return json.loads(response.content)

    clock = now or (lambda: datetime.now(timezone.utc))
    cached = None

    def snapshot_items(spec, days):
        nonlocal cached
        if cached is None:
            cached = loader()
        if cached.get("version") != 1 or cached.get("public_only") is not True:
            raise RuntimeError("Invalid public source snapshot manifest")
        record = cached.get("sources", {}).get(spec.name)
        if not record or record.get("status") != "ok":
            raise RuntimeError(f"Public source snapshot missing or failed for {spec.name}")
        for field in ("kind", "section", "url", "category"):
            if record.get(field) != getattr(spec, field):
                raise RuntimeError(f"Public source snapshot configuration mismatch for {spec.name}")
        fetched = datetime.fromisoformat(record["fetched_at"])
        if fetched.tzinfo is None:
            raise RuntimeError("Public source snapshot timestamp has no timezone")
        age = clock() - fetched
        if age > timedelta(hours=24) or age < -timedelta(minutes=5):
            raise RuntimeError(f"Public source snapshot stale or future-dated for {spec.name}")
        requested_cap = None
        if spec.kind == "arxiv":
            # Native arXiv expands a result count, not a publication-date
            # filter. At 400 results, longer catch-up windows have equal scope.
            def cap(window):
                return min(ArxivSource.MAX_RESULTS * max(1, math.ceil(max(1, window) / 2)), 400)

            requested_cap = cap(days)
            if cap(record["days"]) < requested_cap:
                raise RuntimeError(f"Public source snapshot window too short for {spec.name}")
        items = [Item.model_validate(row) for row in record["items"]]
        if any(item.source != spec.name or item.section != spec.section for item in items):
            raise RuntimeError(f"Public source snapshot item identity mismatch for {spec.name}")
        # RSS deliberately ignores days in the native adapter. Preserve its
        # current feed and dates; ranking handles the reader's recency window.
        return items[:requested_cap] if requested_cap is not None else items, record

    def wrap(original):
        @wraps(original)
        def fetch(self, spec, days=2):
            self.last_fetch_diagnostics = {}
            if spec.profile_driven:
                return original(self, spec, days=days)
            try:
                items, record = snapshot_items(spec, days)
                self.last_fetch_diagnostics = {
                    "transport": "public_snapshot", "snapshot_url": snapshot_url,
                    "snapshot_fetched_at": record["fetched_at"],
                    "snapshot_generated_at": cached.get("generated_at"),
                    "snapshot_window_days": record["days"],
                }
                return items
            except Exception as snapshot_error:
                self.last_fetch_diagnostics = {"snapshot_error": str(snapshot_error)[:500],
                                               "transport": "direct"}
                try:
                    return original(self, spec, days=days)
                except Exception as direct_error:
                    self.last_fetch_diagnostics["direct_error"] = (
                        f"{type(direct_error).__name__}: {direct_error}"[:500]
                    )
                    raise RuntimeError(
                        f"{spec.name} direct fetch and public snapshot failed: {snapshot_error}"
                    ) from direct_error
        fetch._public_snapshot_original = original
        return fetch

    # Reinstalling updates the URL/cache without stacking wrappers.
    for source in (RSSSource, ArxivSource):
        original = getattr(source.fetch, "_public_snapshot_original", source.fetch)
        source.fetch = wrap(original)
