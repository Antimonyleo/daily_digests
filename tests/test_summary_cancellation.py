"""Stop between summary batches without contacting real providers."""
import pytest

from dailydigest import summarize
from dailydigest.brew import BrewCancelled
from dailydigest.config import Settings
from dailydigest.store import ItemRow


def _items(count=15):
    return [ItemRow(id=i, title=f'Fixture paper {i}', abstract='A measured result.',
                    source='Nature', section='research') for i in range(1, count + 1)]


@pytest.mark.parametrize('backend,method', [
    ('api', '_call_llm'), ('anthropic', '_call_anthropic'),
    ('claude_cli', '_call_cli'), ('codex_cli', '_call_cli'),
])
def test_stop_after_first_batch_prevents_more_provider_calls(monkeypatch, backend, method):
    cancelled, calls = [False], []

    def generate(batch, *args, **kwargs):
        calls.append([row.id for row in batch])
        cancelled[0] = True
        return {row.id: 'Fixture summary' for row in batch}

    monkeypatch.setattr(summarize, method, generate)
    with pytest.raises(BrewCancelled):
        summarize.summarize_items_with_provenance(
            _items(), settings=Settings(_env_file=None, llm_backend=backend, llm_api_key='fixture'),
            should_cancel=lambda: cancelled[0],
        )
    assert calls == [[1, 2, 3, 4, 5, 6]]


def test_cancelled_provider_request_does_not_become_extractive_fallback(monkeypatch):
    def cancelled(*args, **kwargs):
        raise BrewCancelled('Fixture stop')

    monkeypatch.setattr(summarize, '_call_llm', cancelled)
    monkeypatch.setattr(summarize, '_extractive', lambda row: pytest.fail('Stop became fallback'))
    with pytest.raises(BrewCancelled, match='Fixture stop'):
        summarize.summarize_items_with_provenance(
            _items(), settings=Settings(_env_file=None, llm_backend='api', llm_api_key='fixture'),
        )


@pytest.mark.parametrize('backend', ['extractive', 'api'])
def test_stop_before_local_fallback_prevents_additional_extraction(monkeypatch, backend):
    cancelled, extracted = [False], []

    def extract(row):
        extracted.append(row.id)
        cancelled[0] = True
        return 'Fixture extractive summary'

    monkeypatch.setattr(summarize, '_extractive', extract)
    monkeypatch.setattr(summarize, '_call_llm', lambda *args, **kwargs: {})
    with pytest.raises(BrewCancelled):
        summarize.summarize_items(
            _items(3), settings=Settings(_env_file=None, llm_backend=backend, llm_api_key='fixture'),
            should_cancel=lambda: cancelled[0],
        )
    assert extracted == [1]


@pytest.mark.parametrize('backend', ['api', 'anthropic'])
@pytest.mark.parametrize('response_started', [False, True])
def test_stop_closes_active_http_transport_before_timeout(monkeypatch, backend, response_started):
    """A local socket peer proves cancellation closes headers/body waits."""
    import socketserver
    from threading import Event, Thread
    import time

    # Bypass the generic no-network guard only for this explicit loopback fixture.
    monkeypatch.setattr(summarize.httpx.Client, "post",
                        lambda self, url, **kw: self.request("POST", url, **kw))

    async def local_post(self, url, **kw):
        return await self.request("POST", url, **kw)

    monkeypatch.setattr(summarize.httpx.AsyncClient, "post", local_post)
    received, disconnected, cancel = Event(), Event(), Event()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(5)
            length = 0
            self.rfile.readline()
            while line := self.rfile.readline():
                if line == b'\r\n':
                    break
                if line.lower().startswith(b'content-length:'):
                    length = int(line.split(b':', 1)[1])
            self.rfile.read(length)
            if response_started:
                self.wfile.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 99999\r\n\r\n{')
                self.wfile.flush()
            received.set()
            if self.rfile.read(1) == b'':
                disconnected.set()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server(('127.0.0.1', 0), Handler) as server:
        serving = Thread(target=server.serve_forever, daemon=True)
        serving.start()

        def stop():
            if received.wait(5):
                cancel.set()

        stopper = Thread(target=stop, daemon=True)
        stopper.start()
        try:
            started = time.monotonic()
            with pytest.raises(BrewCancelled):
                summarize.summarize_items_with_provenance(
                    _items(1), should_cancel=cancel.is_set,
                    settings=Settings(_env_file=None, llm_backend=backend,
                                      llm_api_key='fixture-only',
                                      llm_base_url=f'http://127.0.0.1:{server.server_address[1]}',
                                      llm_summary_max_failures=1),
                )
            assert time.monotonic() - started < 3, 'Stop waited for the network timeout'
            assert disconnected.wait(1), 'Cancelled request kept its provider socket alive'
        finally:
            cancel.set()
            server.shutdown()
            serving.join(2)
            stopper.join(2)


def test_stop_during_dns_does_not_wait_for_resolver_executor(monkeypatch, tmp_path):
    """A private transport process must own resolver threads as well as sockets."""
    import socket
    import time
    from threading import Event

    ready = tmp_path / 'fixture-dns-entered'
    parent_dns = Event()

    def blocked_parent_dns(*args, **kwargs):
        parent_dns.set()
        ready.write_text('resolving')
        time.sleep(3)
        raise socket.gaierror('Fixture resolver stopped')

    async def local_post(self, url, **kwargs):
        return await self.request('POST', url, **kwargs)

    monkeypatch.setattr(socket, 'getaddrinfo', blocked_parent_dns)
    monkeypatch.setattr(summarize.httpx.AsyncClient, 'post', local_post)
    bootstrap = getattr(summarize, '_HTTP_POST_BOOTSTRAP', '')
    injected = ("import socket,time; from pathlib import Path\n"
                "def fixture_dns(*args, **kwargs):\n"
                f" Path({str(ready)!r}).write_text('resolving')\n"
                " time.sleep(30)\n"
                " raise socket.gaierror('Fixture resolver stopped')\n"
                "socket.getaddrinfo=fixture_dns\n")
    monkeypatch.setattr(summarize, '_HTTP_POST_BOOTSTRAP', injected + bootstrap, raising=False)
    started = time.monotonic()
    with pytest.raises(BrewCancelled):
        summarize._http_post('http://localhost:32123/fixture-only',
                             headers={}, body={'fixture': 'selected metadata'}, timeout=30,
                             should_cancel=ready.exists)
    assert time.monotonic() - started < 2, 'Stop waited for blocked DNS executor shutdown'
    assert not parent_dns.is_set(), 'Provider resolver escaped transport process ownership'


def test_browser_direct_api_retains_cancellable_http_bridge_without_native_process(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(summarize, 'sys', SimpleNamespace(platform='emscripten'))
    monkeypatch.setattr(summarize, '_run_cli_process',
                        lambda *args, **kwargs: pytest.fail('Browser tried to launch native transport'))
    stopped = [False]
    calls = []

    def browser_post(self, url, **kwargs):
        calls.append(url)
        return summarize.httpx.Response(200, json={'fixture': 'selected summary'},
                                        request=summarize.httpx.Request('POST', url))

    monkeypatch.setattr(summarize.httpx.Client, 'post', browser_post)
    result = summarize._http_post('https://fixture.invalid/explicit-provider', headers={},
                                  body={}, timeout=5, should_cancel=lambda: stopped[0])
    assert result == {'fixture': 'selected summary'}
    stopped[0] = True
    with pytest.raises(BrewCancelled):
        summarize._http_post('https://fixture.invalid/explicit-provider', headers={},
                             body={}, timeout=5, should_cancel=lambda: stopped[0])
    assert len(calls) == 1
