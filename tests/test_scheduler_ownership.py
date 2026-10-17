"""Scheduled filesystem jobs remain owned when the service stops."""
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from dailydigest import companion
from tests.test_browser_companion import settings as settings
from tests.test_browser_companion import scheduler_snapshot


def test_stop_during_export_does_not_publish_snapshot(settings, tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from tests.test_browser_companion import scheduler_snapshot

    store = companion._state_store(tmp_path / 'store')
    previous = scheduler_snapshot()
    companion._write_private(store / 'snapshot.json', previous)
    stop = threading.Event()

    def finished(command, *, cwd, **kwargs):
        stop.set()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(companion, '_run_scheduled_process', finished)
    status = {}
    companion._schedule_once(store, settings, companion.RLock(), status, tmp_path / 'sources.yaml',
                             now=datetime(2026, 10, 17, 8, tzinfo=timezone.utc), stop=stop)
    assert status['last_status'] != 'success'
    assert (store / 'snapshot.json').read_text() == previous


def test_stop_during_snapshot_fsync_does_not_replace_previous_snapshot(tmp_path, monkeypatch):
    path = tmp_path / 'snapshot.json'
    path.write_text('previous private snapshot')
    stop = threading.Event()
    original_fsync = companion.os.fsync

    def fsync_and_stop(fd):
        original_fsync(fd)
        stop.set()

    monkeypatch.setattr(companion.os, 'fsync', fsync_and_stop)
    with pytest.raises(RuntimeError, match='stopped before publication'):
        companion._write_private(path, 'replacement', should_cancel=stop.is_set)
    assert path.read_text() == 'previous private snapshot'
    assert not list(tmp_path.glob('.snapshot-*'))


def test_coordinator_returns_pending_status_without_running_native_brew(settings, tmp_path, monkeypatch):
    host = tmp_path / 'host-private-root'
    host.mkdir()
    journal = host / '.settings-transaction.json'
    journal.write_text('damaged host journal must remain untouched')
    monkeypatch.setenv('DD_ENV_PATH', 'data/.env')
    monkeypatch.setenv('DD_APP_ROOT', str(host))
    store = companion._state_store(tmp_path / 'store')
    status = {}
    companion._run_scheduled_job(store, settings, companion.RLock(), status,
                                 tmp_path / 'sources.yaml', stop=threading.Event())
    assert status['last_status'] == 'pending_uploaded_state'
    assert not list(store.glob('native-reader-*'))
    assert journal.read_text() == 'damaged host journal must remain untouched'


def test_coordinator_launch_failure_is_reported_without_stopping_scheduler(settings, tmp_path, monkeypatch):
    def cannot_start(*args, **kwargs):
        raise OSError('fixture private details')

    monkeypatch.setattr(companion.subprocess, 'Popen', cannot_start)
    status = {}
    companion._run_scheduled_job(tmp_path, settings, companion.RLock(), status,
                                 tmp_path / 'sources.yaml', stop=threading.Event())
    assert status['last_status'] == 'failed'
    assert status['error_type'] == 'OSError'
    assert 'fixture private details' not in json.dumps(status)


def test_queued_scheduler_can_stop_without_waiting_for_another_request(settings, tmp_path, monkeypatch):
    lock = companion.RLock()
    stop = threading.Event()
    entered = threading.Event()
    monkeypatch.setattr(companion.subprocess, 'Popen',
                        lambda *args, **kwargs: pytest.fail('Stopped queued job launched'))

    def run():
        entered.set()
        companion._run_scheduled_job(tmp_path, settings, lock, {}, tmp_path / 'sources.yaml', stop=stop)

    lock.acquire()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert entered.wait(1)
        stop.set()
        thread.join(1)
        assert not thread.is_alive(), 'Queued scheduler ignored Stop while another request held the lock'
    finally:
        lock.release()
        thread.join(2)


@pytest.mark.parametrize('model_location', ['host', 'data'])
def test_uploaded_semantic_model_directory_stays_inside_private_workspace(
    settings, tmp_path, monkeypatch, model_location,
):
    host = tmp_path / 'private-host-model'
    host.mkdir()
    marker = host / 'untouched.txt'
    marker.write_text('private host model must not be used')
    configured_path = str(host) if model_location == 'host' else 'data/semantic-verifier'
    snapshot = json.loads(scheduler_snapshot())
    uploaded_env = base64.b64decode(snapshot['files']['.env'])
    snapshot['files']['.env'] = base64.b64encode(
        uploaded_env + f'SEMANTIC_MATCHING=true\nSEMANTIC_MODEL_DIR={configured_path}\n'.encode()
    ).decode()
    snapshot['files']['data/semantic-verifier/config.json'] = base64.b64encode(b'{}').decode()
    previous = json.dumps(snapshot)
    store = companion._state_store(tmp_path / 'store')
    companion._write_private(store / 'snapshot.json', previous)
    calls = []

    def native_cli(command, *, cwd, env, **kwargs):
        calls.append(command)
        assert model_location == 'data', 'Host model path reached native subprocess'
        assert env['SEMANTIC_MODEL_DIR'] == str(Path(cwd) / 'data/semantic-verifier')
        assert (Path(env['SEMANTIC_MODEL_DIR']) / 'config.json').read_bytes() == b'{}'
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(companion, '_run_scheduled_process', native_cli)
    status = {}
    companion._schedule_once(store, settings, companion.RLock(), status, tmp_path / 'sources.yaml',
                             now=datetime(2026, 10, 17, 8, tzinfo=timezone.utc))
    if model_location == 'host':
        assert status['last_status'] == 'failed'
        assert status['error_type'] == 'ValueError'
        assert calls == []
        assert (store / 'snapshot.json').read_text() == previous
    else:
        assert status['last_status'] == 'success'
        assert len(calls) == 1
        assert json.loads((store / 'snapshot.json').read_text()) == snapshot
    assert marker.read_text() == 'private host model must not be used'


@pytest.mark.parametrize('reads_configuration', [True, False])
def test_coordinator_stops_while_filesystem_job_is_blocked(settings, tmp_path, reads_configuration):
    """An uncooperative job cannot keep the service's scheduling thread alive."""
    marker = tmp_path / 'entered'
    script = (
        'import signal,time\n'
        'from pathlib import Path\n'
        'from dailydigest import companion\n'
        'def blocked(*args, **kwargs):\n'
        ' signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        f' Path({str(marker)!r}).write_text("entered")\n'
        ' time.sleep(60)\n'
        'companion._schedule_once=blocked\n'
        'companion._scheduled_job_main()\n'
    )
    if not reads_configuration:
        # More than an anonymous pipe buffer: on Windows communicate's stdin
        # write would otherwise block the supervisor before it checked Stop.
        settings.llm_model = 'x' * (1024 * 1024)
        script = (
            'import signal,time\nfrom pathlib import Path\n'
            'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
            f'Path({str(marker)!r}).write_text("entered")\ntime.sleep(60)\n'
        )
    stop = threading.Event()
    status = {}
    failure = []

    def run():
        try:
            companion._run_scheduled_job(tmp_path, settings, companion.RLock(), status,
                                         tmp_path / 'sources.yaml', stop=stop,
                                         command=[sys.executable, '-c', script])
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists(), failure
        began = time.monotonic()
        stop.set()
        thread.join(12)
        assert not thread.is_alive(), 'Scheduling thread retained uncooperative filesystem work'
        assert time.monotonic() - began < 12
        assert not failure, failure
        assert status['last_status'] == 'stopped'
        assert status['workspace_retained'] is True
    finally:
        stop.set()
        thread.join(12)


@pytest.mark.skipif(os.name != 'nt', reason='Real Windows process tree ownership')
def test_windows_scheduled_timeout_reaps_actual_descendant_tree(tmp_path):
    import ctypes

    pids = tmp_path / 'fixture-pids.json'
    code = (
        'import json,os,subprocess,sys,time;from pathlib import Path;'
        'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);'
        f'Path({str(pids)!r}).write_text(json.dumps([os.getpid(),child.pid]));time.sleep(60)'
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            companion._run_scheduled_process([sys.executable, '-c', code],
                                              cwd=tmp_path, env=dict(os.environ), timeout=3)
        assert pids.is_file()
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        for pid in json.loads(pids.read_text()):
            handle = kernel.OpenProcess(0x101000, False, pid)
            if not handle:
                assert ctypes.get_last_error() == 87, 'Could not inspect fixture process'
                continue
            try:
                assert kernel.WaitForSingleObject(handle, 5000) == 0, f'Process {pid} did not exit'
                result = ctypes.c_ulong()
                assert kernel.GetExitCodeProcess(handle, ctypes.byref(result))
                assert result.value != 259, f'Process {pid} remains active after timeout'
            finally:
                kernel.CloseHandle(handle)
    finally:
        if pids.exists():
            for pid in json.loads(pids.read_text()):
                subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
