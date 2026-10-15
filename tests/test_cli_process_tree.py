"""Exercise CLI timeout containment with Python fixtures, never model providers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from dailydigest import summarize


def _running(pid: int) -> bool:
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            if ctypes.get_last_error() in (87, 1168):  # missing process
                return False
            raise OSError(ctypes.get_last_error(), 'Cannot inspect fixture descendant')
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError(ctypes.get_last_error(), 'Cannot inspect fixture descendant exit')
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed orphan can briefly remain a zombie until its OS reaper runs;
    # it no longer executes or holds the inherited output pipe open.
    if sys.platform.startswith('linux'):
        try:
            return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            return False
    status = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True,
                            text=True, timeout=2, check=False)
    return bool(status.stdout.strip()) and not status.stdout.lstrip().startswith('Z')


@pytest.mark.parametrize('redirect_child', [False, True])
def test_cli_timeout_terminates_real_descendant_holding_output_pipe(tmp_path, monkeypatch, redirect_child):
    pid_file = tmp_path / 'fixture-pids.json'
    child = "import os,sys,time; from pathlib import Path; print('descendant-ready',flush=True); Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    if redirect_child:
        # Windows virtualenv executables may be redirectors: Popen.pid belongs
        # to a launcher, while os.getpid() belongs to the Python descendant.
        child = "import subprocess,sys; sys.exit(subprocess.call([sys.executable,'-u','-c'," + repr(child) + ",sys.argv[1]]))"
    parent = "import json,os,subprocess,sys,time; from pathlib import Path; child=subprocess.Popen([sys.executable,'-u','-c',sys.argv[2],sys.argv[3]]); Path(sys.argv[1]).write_text(json.dumps({'parent':os.getpid(),'child':child.pid})); time.sleep(30)"
    ready_file = tmp_path / 'descendant-ready'
    parents = []
    real_popen = subprocess.Popen

    def record_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        parents.append(process)
        return process

    monkeypatch.setattr(summarize.subprocess, 'Popen', record_process)
    child_pid = None
    pids = {}
    started = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            summarize._run_cli_process([sys.executable, '-u', '-c', parent,
                                        str(pid_file), child, str(ready_file)],
                                       'fixture input', str(tmp_path), timeout=3)
        assert pid_file.exists() and ready_file.exists(), 'Fixture descendant did not start before timeout'
        pids = json.loads(pid_file.read_text())
        child_pid = int(ready_file.read_text())
        process_ids = {child_pid, pids['child'], pids['parent']}
        # Windows communicate() times out before its reader thread publishes
        # partial output. The readiness file is written after stdout flush;
        # verify pipe closure directly rather than assuming exception.output.
        assert parents[0].stdout.closed
        deadline = time.monotonic() + 2
        while any(_running(pid) for pid in process_ids) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(_running(pid) for pid in process_ids), 'Timeout left a launcher or output-pipe-holding descendant running'
        assert parents[0].poll() is not None, 'Timeout did not reap its direct child'
        assert time.monotonic() - started < 9, 'Process-tree timeout cleanup took too long'
    finally:
        if pid_file.exists():
            pids = json.loads(pid_file.read_text())
        if child_pid is None and ready_file.exists():
            child_pid = int(ready_file.read_text())
        if sys.platform == 'win32':
            for pid in {parents[0].pid if parents else None, child_pid, pids.get('parent'), pids.get('child')}:
                if pid is not None and _running(pid):
                    subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=5, check=False)
        elif parents:
            try:
                os.killpg(parents[0].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if parents:
            parents[0].wait(timeout=5)
