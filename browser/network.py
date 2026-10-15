"""HTTPX transport for synchronous requests inside a browser Web Worker.

The asynchronous fetch bridge suspends Python through JSPI so Stop remains
responsive. Both fetch and the legacy XHR fixture fallback enforce CORS.
Install only in the Pyodide worker, before importing the application.
"""

from functools import wraps
from datetime import datetime, timedelta, timezone
import base64
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


def _bridge_request(payload):
    try:
        from js import browserHttpRequest
    except (ImportError, ModuleNotFoundError):
        return None
    from pyodide.ffi import run_sync

    return json.loads(run_sync(browserHttpRequest(json.dumps(payload))))


def _cooperative_sleep(seconds):
    from js import browserHttpWait
    from pyodide.ffi import run_sync

    run_sync(browserHttpWait(max(0, seconds * 1000)))


class BrowserRequestUnavailable(RuntimeError):
    """Opaque CORS/network denial; HTTP retry cannot make it readable."""


class BrowserTransport(httpx.BaseTransport):
    def __init__(self, xhr_factory=None, response_bytes=None, request_body=None, bridge_request=None):
        self._factory = xhr_factory or _xhr_factory
        self._bytes = response_bytes or _response_bytes
        self._body = request_body or _request_body
        self._bridge = bridge_request if bridge_request is not None else (None if xhr_factory else _bridge_request)

    def handle_request(self, request):
        # Match Anthropic's explicit browser opt-in only for authenticated
        # Messages calls. Native HTTP clients never acquire this header.
        if (request.method == 'POST' and request.url.scheme == 'https'
                and request.url.host == 'api.anthropic.com'
                and request.url.path == '/v1/messages'
                and request.headers.get('x-api-key')
                and request.headers.get('anthropic-version')):
            request.headers['anthropic-dangerous-direct-browser-access'] = 'true'
        if self._bridge is not None:
            headers = []
            for name, value in request.headers.multi_items():
                lower = name.lower()
                if lower in _FORBIDDEN or lower.startswith(("sec-", "proxy-")):
                    continue
                if lower in {"x-http-method", "x-http-method-override", "x-method-override"}:
                    if any(v.strip().upper() in {"CONNECT", "TRACE", "TRACK"} for v in value.split(",")):
                        continue
                headers.append((name, value))
            timeout = request.extensions.get("timeout", {}).get("read")
            try:
                result = self._bridge({"method": request.method, "url": str(request.url),
                    "headers": headers, "body": base64.b64encode(request.read()).decode(),
                    "timeoutMs": None if timeout is None else max(1, int(timeout * 1000)),
                    "followRedirects": request.extensions.get("browser_follow_redirects", True)})
            except Exception as exc:
                if "cancelled" in str(exc).lower():
                    from dailydigest.pipeline import BrewCancelled
                    raise BrewCancelled("Brew cancelled during source request; saved digest unchanged.") from exc
                if "TimeoutError" in str(exc):
                    raise httpx.ReadTimeout("Browser request timed out", request=request) from exc
                # Opaque fetch failures (including CORS) are not retriable HTTP
                # statuses. Avoid three attempts at every blocked publisher.
                raise BrowserRequestUnavailable(f"Browser CORS or network request unavailable: {exc}") from exc
            if result is not None:
                return httpx.Response(result["status"], content=base64.b64decode(result["body"]),
                    headers=[(name, value) for name, value in result["headers"]
                             if name.lower() not in {"content-encoding", "content-length", "transfer-encoding"}],
                    extensions={"browser_final_url": result["url"]})
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
                # XHR has already decoded compression and transfer framing.
                # Forwarding wire encoding headers makes HTTPX decode twice;
                # its response constructor should compute the decoded length.
                if name.strip().lower() in {"content-encoding", "content-length", "transfer-encoding"}:
                    continue
                headers.append((name.strip(), value.strip()))
        return httpx.Response(
            int(xhr.status), headers=headers, content=self._bytes(xhr),
            extensions={"browser_final_url": final_url},
        )


def install_httpx_transport():
    """Use abortable browser fetch for HTTPX; retain explicit test transports.

    Idempotent. Fetch cannot expose intermediate redirects or provide separate
    connect/write timeout phases. Legacy fixture-only XHR remains supported.
    No proxy or opaque-response fallback is installed.
    """
    # Native retry/polite waits must yield while Python is suspended through
    # JSPI. Install only when the browser provides the asynchronous bridge.
    try:
        from js import browserHttpWait  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        time.sleep = _cooperative_sleep
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
    """Use public metadata up to 48 hours old, then try direct fetching.

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
                response = client.get(target, headers={"Cache-Control": "no-cache"})
                response.raise_for_status()
                return json.loads(response.content)

    clock = now or (lambda: datetime.now(timezone.utc))
    cached = None
    cache_attempted_at = None
    refresh_error = None

    def snapshot_items(spec, days):
        nonlocal cached, cache_attempted_at, refresh_error
        instant = clock()
        if (cache_attempted_at is None or instant < cache_attempted_at
                or instant - cache_attempted_at >= timedelta(minutes=5)):
            # Bound both successful and failed refresh attempts across the
            # entire source list. A failed refresh can reuse a still-fresh
            # snapshot, but never changes the collection timestamp below.
            cache_attempted_at = instant
            try:
                refreshed = loader()
                if (not isinstance(refreshed, dict) or refreshed.get("version") != 1
                        or refreshed.get("public_only") is not True
                        or not isinstance(refreshed.get("sources"), dict)):
                    raise RuntimeError("Invalid public source snapshot manifest")
                cached, refresh_error = refreshed, None
            except Exception as exc:
                refresh_error = f"{type(exc).__name__}: {exc}"[:500]
        if cached is None:
            raise RuntimeError(f"Public source snapshot unavailable: {refresh_error}")
        record = cached.get("sources", {}).get(spec.name)
        if not record or record.get("status") != "ok":
            raise RuntimeError(f"Public source snapshot missing or failed for {spec.name}")
        for field in ("kind", "section", "url", "category"):
            if record.get(field) != getattr(spec, field):
                raise RuntimeError(f"Public source snapshot configuration mismatch for {spec.name}")
        fetched = datetime.fromisoformat(record["fetched_at"])
        if fetched.tzinfo is None:
            raise RuntimeError("Public source snapshot timestamp has no timezone")
        age = instant - fetched
        if age > timedelta(hours=48) or age < -timedelta(minutes=5):
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
                    "snapshot_age_hours": max(0, (clock() - datetime.fromisoformat(record["fetched_at"])).total_seconds() / 3600),
                    "snapshot_stale_warning": clock() - datetime.fromisoformat(record["fetched_at"]) > timedelta(hours=24),
                    "snapshot_carried_forward": bool(record.get("carried_forward")),
                }
                if record.get("latest_attempt"):
                    self.last_fetch_diagnostics["snapshot_latest_attempt"] = record["latest_attempt"].get("attempted_at") or record["latest_attempt"].get("fetched_at")
                    self.last_fetch_diagnostics["snapshot_latest_attempt_failed"] = record["latest_attempt"].get("status") != "ok"
                if refresh_error:
                    self.last_fetch_diagnostics["snapshot_refresh_error"] = refresh_error
                return items
            except Exception as snapshot_error:
                self.last_fetch_diagnostics = {"snapshot_error": str(snapshot_error)[:500],
                                               "transport": "direct"}
                if refresh_error:
                    self.last_fetch_diagnostics["snapshot_refresh_error"] = refresh_error
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
