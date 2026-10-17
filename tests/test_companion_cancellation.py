"""Disconnect a real loopback Companion from slow, fixture-only providers."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import socket
import socketserver
from threading import Event, Thread
import time

import httpx
import pytest
import uvicorn

from dailydigest import companion
from dailydigest.config import Settings


@pytest.mark.parametrize('backend', ['api', 'anthropic', 'claude_cli'])
@pytest.mark.parametrize('endpoint', ['/summarize', '/text'])
def test_browser_disconnect_closes_companion_provider_transport(monkeypatch, tmp_path, backend, endpoint):
    received, disconnected = Event(), Event()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(5)
            self.rfile.readline()
            length = 0
            while line := self.rfile.readline():
                if line == b'\r\n':
                    break
                if line.lower().startswith(b'content-length:'):
                    length = int(line.split(b':', 1)[1])
            self.rfile.read(length)
            received.set()
            with suppress(TimeoutError):
                try:
                    closed = self.rfile.read(1) == b''
                except ConnectionResetError:
                    closed = True
                if closed:
                    disconnected.set()

    class Provider(socketserver.ThreadingTCPServer):
        daemon_threads = True

    async def loopback_post(self, url, **kwargs):
        assert str(url).startswith('http://127.0.0.1:')
        return await self.request('POST', url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, 'post', loopback_post)
    if backend == 'claude_cli':
        import sys
        from dailydigest import summarize as native

        ready = tmp_path / 'cli-ready'
        script = "import sys,time; from pathlib import Path; Path(sys.argv[1]).write_text('ready'); time.sleep(30)"
        monkeypatch.setattr(companion.shutil, 'which', lambda command: sys.executable)

        def fixture_cli(system, user, backend, *, should_cancel, **kwargs):
            def stop():
                if ready.exists():
                    received.set()
                return should_cancel()

            try:
                native._run_cli_process([sys.executable, '-u', '-c', script, str(ready)],
                                        'Fixture selected metadata', str(tmp_path), 30,
                                        should_cancel=stop)
            finally:
                disconnected.set()

        monkeypatch.setattr(native, '_cli_text', fixture_cli)
    with Provider(('127.0.0.1', 0), Handler) as provider, socket.socket() as listener:
        serving = Thread(target=provider.serve_forever, daemon=True)
        serving.start()
        listener.bind(('127.0.0.1', 0))
        token = 'fixture-private-token-' + 'x' * 32
        origin = 'https://example.org'
        settings = Settings(_env_file=None, llm_backend=backend, llm_api_key='fixture',
                            llm_base_url=f'http://127.0.0.1:{provider.server_address[1]}')
        app = companion.create_app(token=token, settings=settings, allowed_origin=origin)
        server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='off',
                                               timeout_graceful_shutdown=2))
        running = Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        running.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and time.monotonic() < deadline:
                time.sleep(.01)
            assert server.started

            async def disconnect():
                async with httpx.AsyncClient(timeout=5) as browser:
                    payload = ({'backend': backend, 'items': [{'id': 1, 'title': 'Fictional paper'}]}
                               if endpoint == '/summarize' else
                               {'backend': backend, 'system': 'Fixture system', 'user': 'Fictional title'})
                    request = asyncio.create_task(browser.request(
                        'POST', f'http://127.0.0.1:{listener.getsockname()[1]}{endpoint}',
                        json=payload, headers={'Origin': origin, 'X-DailyDigest-Token': token},
                    ))
                    try:
                        assert await asyncio.to_thread(received.wait, 3), 'Provider did not receive fixture request'
                    finally:
                        request.cancel()
                        with suppress(asyncio.CancelledError):
                            await request
                assert await asyncio.to_thread(disconnected.wait, 3), 'Companion kept provider work alive after browser disconnected'

            asyncio.run(disconnect())
        finally:
            server.should_exit = True
            running.join(4)
            provider.shutdown()
            serving.join(2)
        assert not running.is_alive(), 'Companion shutdown left a request waiting for provider timeout'


def test_disconnected_provider_does_not_wait_for_busy_scheduler_lock():
    from dailydigest.brew import BrewCancelled
    from threading import RLock

    entered = Event()
    lock = RLock()

    class BusyLock:
        def acquire(self, **kwargs):
            entered.set()
            return lock.acquire(**kwargs)

        def release(self):
            lock.release()

    class Request:
        async def receive(self):
            assert await asyncio.to_thread(entered.wait, 2)
            return {'type': 'http.disconnect'}

    def provider(should_cancel):
        with companion._provider_lock(BusyLock(), should_cancel):
            pytest.fail('Disconnected queued provider acquired busy scheduler lock')

    started = time.monotonic()
    with lock:
        with pytest.raises(BrewCancelled):
            asyncio.run(companion._run_provider_request(Request(), provider))
    assert time.monotonic() - started < 1
