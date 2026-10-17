"""Settings saves recover an interrupted group before any configuration is read."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('after_write', [1, 2, 3])
@pytest.mark.parametrize('existing', [True, False])
def test_interrupted_settings_save_recovers_before_config_import(tmp_path, after_write, existing):
    root = Path(__file__).resolve().parents[1]
    old = {
        'profile.yaml': b'bio: Original\nkeywords: [RNA]\n',
        'opportunities.yaml': b'description: Original details\n',
        '.env': b'LLM_BACKEND=extractive\nTOP_RESEARCH=3\n',
    }
    for name, content in old.items():
        if existing:
            (tmp_path / name).write_bytes(content)
    script = r'''
import asyncio, os
from pathlib import Path
from dailydigest import web
from tests.test_web import _request
root = Path.cwd()
web._get_profile_path = lambda: root / 'profile.yaml'
web._get_opportunity_profile_path = lambda: root / 'opportunities.yaml'
web._ENV_PATH = root / '.env'
original = web._atomic_private_write
count = 0
def interrupted(path, content):
    global count
    original(path, content)
    count += 1
    if count == int(os.environ['AFTER_WRITE']):
        os._exit(74)
web._atomic_private_write = interrupted
request = _request('POST', '/setup', form={
    '_csrf_token': web._CSRF_TOKEN, 'bio': 'Changed', 'topics': 'RNA | 10',
    'llm_backend': 'extractive', 'top_research': '8', 'include_opportunities': 'true',
    'opportunity_career_stage': 'assistant professor',
    'opportunity_institution_type': 'nonprofit university',
    'opportunity_country': 'United States', 'opportunity_applicant_role': 'principal investigator',
})
asyncio.run(web.setup_post.__wrapped__(request))
'''
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(root / 'src'), str(root)]),
           'DD_APP_ROOT': str(tmp_path), 'AFTER_WRITE': str(after_write)}
    crashed = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=env, capture_output=True, timeout=30)
    assert crashed.returncode == 74, crashed.stderr.decode()
    expected = 3 if existing else 12
    env.pop('TOP_RESEARCH', None)
    recovery = subprocess.run([sys.executable, '-c',
                               f'from dailydigest.config import SETTINGS; assert SETTINGS.top_research == {expected}'],
                              cwd=tmp_path, env=env, capture_output=True, timeout=30)
    assert recovery.returncode == 0, recovery.stderr.decode()
    if existing:
        assert {name: (tmp_path / name).read_bytes() for name in old} == old
    else:
        assert not any((tmp_path / name).exists() for name in old)


def test_recovery_preserves_external_edit_and_blocks_use(tmp_path):
    from dailydigest.settings_transaction import SettingsRecoveryError, recover_settings_transaction, settings_transaction

    path = tmp_path / 'profile.yaml'
    path.write_bytes(b'old')
    with pytest.raises(KeyboardInterrupt):
        with settings_transaction(tmp_path, {path: b'new'}):
            path.write_bytes(b'new')
            raise KeyboardInterrupt
    # Simulate a prepared journal left by abrupt process death instead below.
    import dailydigest.settings_transaction as transaction
    remove = transaction._remove_journal
    transaction._remove_journal = lambda *_: None
    try:
        with settings_transaction(tmp_path, {path: b'new'}):
            transaction._durable_write(path, b'new')
    finally:
        transaction._remove_journal = remove
    path.write_bytes(b'external edit')
    with pytest.raises(SettingsRecoveryError, match='changed outside'):
        recover_settings_transaction(tmp_path)
    assert path.read_bytes() == b'external edit'
    assert (tmp_path / '.settings-transaction.json').exists()


def test_committed_cleanup_failure_keeps_new_group_and_retries(tmp_path, monkeypatch):
    from dailydigest import settings_transaction as transaction

    path = tmp_path / 'profile.yaml'
    path.write_bytes(b'old')
    remove = transaction._remove_journal
    monkeypatch.setattr(transaction, '_remove_journal', lambda *_: (_ for _ in ()).throw(OSError('cleanup')))
    with transaction.settings_transaction(tmp_path, {path: b'new'}):
        transaction._durable_write(path, b'new')
    assert path.read_bytes() == b'new'
    assert (tmp_path / '.settings-transaction.json').stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(transaction, '_remove_journal', remove)
    transaction.recover_settings_transaction(tmp_path)
    assert path.read_bytes() == b'new'
    assert not (tmp_path / '.settings-transaction.json').exists()


def test_commit_directory_sync_failure_rolls_back_coherently(tmp_path, monkeypatch):
    from dailydigest import settings_transaction as transaction

    path = tmp_path / 'profile.yaml'
    path.write_bytes(b'old')
    sync = transaction._sync_directory
    failed = False

    def fail_commit(directory):
        nonlocal failed
        journal = tmp_path / '.settings-transaction.json'
        if not failed and journal.exists() and b'"committed"' in journal.read_bytes():
            failed = True
            raise OSError('directory sync failed')
        sync(directory)

    monkeypatch.setattr(transaction, '_sync_directory', fail_commit)
    with pytest.raises(OSError, match='directory sync'):
        with transaction.settings_transaction(tmp_path, {path: b'new'}):
            transaction._durable_write(path, b'new')
    assert failed
    assert path.read_bytes() == b'old'
    assert not (tmp_path / '.settings-transaction.json').exists()


@pytest.mark.parametrize('journal', [b'{', b'[]', b'{"version":1,"state":"prepared","files":{"../outside":{"old":null,"new":"eA=="}}}'])
def test_bad_journal_blocks_without_touching_settings(tmp_path, journal):
    from dailydigest.settings_transaction import SettingsRecoveryError, recover_settings_transaction

    path = tmp_path / 'profile.yaml'
    path.write_bytes(b'private original')
    (tmp_path / '.settings-transaction.json').write_bytes(journal)
    with pytest.raises(SettingsRecoveryError):
        recover_settings_transaction(tmp_path)
    assert path.read_bytes() == b'private original'
    assert (tmp_path / '.settings-transaction.json').read_bytes() == journal


@pytest.mark.skipif(os.name != 'posix', reason='POSIX SIGKILL regression; Windows forced exit covered separately')
def test_sigkill_after_member_rename_recovers(tmp_path):
    from dailydigest.settings_transaction import recover_settings_transaction

    root = Path(__file__).resolve().parents[1]
    path = tmp_path / 'profile.yaml'
    path.write_bytes(b'old')
    script = r'''
import os, signal
from pathlib import Path
from dailydigest.settings_transaction import settings_transaction, _durable_write
root = Path.cwd()
with settings_transaction(root, {root / 'profile.yaml': b'new'}):
    _durable_write(root / 'profile.yaml', b'new')
    os.kill(os.getpid(), signal.SIGKILL)
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': str(root / 'src')}, timeout=30)
    assert result.returncode == -9
    recover_settings_transaction(tmp_path)
    assert path.read_bytes() == b'old'


def test_reload_inside_save_fails_without_deadlock_and_restores(tmp_path):
    from dailydigest.settings_transaction import SettingsRecoveryError, _durable_write, recover_settings_transaction, settings_transaction

    path = tmp_path / 'profile.yaml'
    path.write_bytes(b'old')
    with pytest.raises(SettingsRecoveryError, match='being saved'):
        with settings_transaction(tmp_path, {path: b'new'}):
            _durable_write(path, b'new')
            recover_settings_transaction(tmp_path)
    assert path.read_bytes() == b'old'


def test_busy_mounted_file_refuses_transaction_before_in_place_write(tmp_path, monkeypatch):
    import errno
    from dailydigest import web
    from dailydigest.settings_transaction import settings_transaction

    path = tmp_path / 'profile.yaml'
    path.write_bytes(b'old')
    replace = web.os.replace

    def busy(source, destination):
        if destination == path:
            raise OSError(errno.EBUSY, 'mounted file')
        replace(source, destination)

    monkeypatch.setattr(web.os, 'replace', busy)
    with pytest.raises(OSError):
        with settings_transaction(tmp_path, {path: b'new'}, writer=web._atomic_private_write):
            web._atomic_private_write(path, b'new')
    assert path.read_bytes() == b'old'
    assert not (tmp_path / '.settings-transaction.json').exists()


def test_windows_journal_rename_requests_write_through(tmp_path, monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from dailydigest import settings_transaction as transaction

    calls = []

    class Move:
        def __call__(self, source, destination, flags):
            calls.append((source, destination, flags))
            return True

    monkeypatch.setattr(ctypes, 'WinDLL', lambda *args, **kwargs: SimpleNamespace(MoveFileExW=Move()), raising=False)
    # Patch just the dispatch module name after creating POSIX Paths.
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    monkeypatch.setattr(transaction, 'os', SimpleNamespace(name='nt'))
    transaction._replace_durable(source, destination)
    assert calls == [(str(source), str(destination), 0x9)]


def test_windows_removal_uses_durable_tombstone_before_unlink(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from dailydigest import settings_transaction as transaction

    path = tmp_path / 'new-profile.yaml'
    path.write_bytes(b'private')
    calls = []

    def replace(source, destination):
        calls.append((source, destination))
        os.replace(source, destination)

    monkeypatch.setattr(transaction, '_replace_durable', replace)
    monkeypatch.setattr(transaction, 'os', SimpleNamespace(name='nt', close=os.close))
    transaction._durable_remove(path)
    assert len(calls) == 1 and calls[0][0] == path
    assert calls[0][1].name.startswith('.new-profile.yaml.removed.')
    assert not list(tmp_path.iterdir())


def test_recovery_interrupted_mid_rollback_repeats_safely(tmp_path):
    from dailydigest.settings_transaction import recover_settings_transaction

    root = Path(__file__).resolve().parents[1]
    for name in ['profile.yaml', '.env']:
        (tmp_path / name).write_bytes(b'old')
    env = {**os.environ, 'PYTHONPATH': str(root / 'src')}
    prepared = r'''
import os
from pathlib import Path
from dailydigest.settings_transaction import settings_transaction, _durable_write
root = Path.cwd()
with settings_transaction(root, {root / 'profile.yaml': b'new', root / '.env': b'new'}):
    _durable_write(root / 'profile.yaml', b'new')
    _durable_write(root / '.env', b'new')
    os._exit(74)
'''
    assert subprocess.run([sys.executable, '-c', prepared], cwd=tmp_path, env=env, timeout=30).returncode == 74
    interrupted = r'''
import os
from pathlib import Path
from dailydigest.settings_transaction import _recover, _durable_write
root = Path.cwd()
def writer(path, content):
    _durable_write(path, content)
    os._exit(75)
_recover(root, writer=writer)
'''
    assert subprocess.run([sys.executable, '-c', interrupted], cwd=tmp_path, env=env, timeout=30).returncode == 75
    assert recover_settings_transaction(tmp_path) is True
    assert recover_settings_transaction(tmp_path) is False
    assert [(tmp_path / name).read_bytes() for name in ['profile.yaml', '.env']] == [b'old', b'old']


def test_member_flush_uses_write_access_required_by_windows(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from dailydigest import settings_transaction as transaction

    path = tmp_path / 'profile.yaml'
    path.write_bytes(b'old')
    open_path, sync = Path.open, transaction.os.fsync
    handles = {}

    @contextmanager
    def tracked_open(self, *args, **kwargs):
        with open_path(self, *args, **kwargs) as handle:
            handles[handle.fileno()] = handle
            try:
                yield handle
            finally:
                handles.pop(handle.fileno(), None)

    def windows_sync(descriptor):
        if descriptor in handles and not handles[descriptor].writable():
            raise OSError('Windows flush requires write access')
        sync(descriptor)

    monkeypatch.setattr(Path, 'open', tracked_open)
    monkeypatch.setattr(transaction.os, 'fsync', windows_sync)
    with transaction.settings_transaction(tmp_path, {path: b'new'}):
        transaction._durable_write(path, b'new')
    assert path.read_bytes() == b'new'
