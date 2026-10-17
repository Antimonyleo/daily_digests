"""Durable recovery for the small group of private Settings files.

A prepared journal restores the old group; a committed journal verifies the new
one. Unknown bytes block configuration loading rather than overwrite an external
edit. POSIX directory fsync makes rename/unlink boundaries durable where the
filesystem supports it. Windows uses write-through journal publication;
removal first renames to a write-through tombstone. Hardware must honor flushes.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import os
import sys
from pathlib import Path
import tempfile
import threading

logger = logging.getLogger(__name__)

_JOURNAL = '.settings-transaction.json'
_LOCK = '.settings-transaction.lock'
_THREAD_LOCK = threading.RLock()
_ACTIVE = ContextVar('settings_transaction_active', default=False)


class SettingsRecoveryError(RuntimeError):
    """Private configuration needs recovery before it is safe to use."""


def settings_transaction_active() -> bool:
    return _ACTIVE.get()


def _sync_directory(path: Path) -> None:
    if os.name != 'posix' or sys.platform == 'emscripten':
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_durable(source: Path, destination: Path) -> None:
    if os.name != 'nt':
        os.replace(source, destination)
        return
    # https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw
    import ctypes
    from ctypes import wintypes

    move = ctypes.WinDLL('kernel32', use_last_error=True).MoveFileExW
    move.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    move.restype = wintypes.BOOL
    if not move(str(source), str(destination), 0x1 | 0x8):
        raise ctypes.WinError()


def _durable_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as handle:
            temporary = Path(handle.name)
            if os.name == 'posix':
                os.fchmod(handle.fileno(), 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_durable(temporary, path)
        temporary = None
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _locked(root: Path):
    with _THREAD_LOCK:
        path = root / _LOCK
        if path.is_symlink():
            raise SettingsRecoveryError('Settings recovery lock must not be a symbolic link.')
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == 'posix' and sys.platform != 'emscripten':
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            elif os.name == 'nt':
                import msvcrt
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b'\0')
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            yield
        finally:
            os.close(descriptor)


def _target(root: Path, name: str) -> Path:
    path = root / name
    if not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise SettingsRecoveryError('Settings recovery journal contains an unsafe path.')
    # Resolving must not follow links, including a symlinked parent directory.
    if path.absolute() != path.resolve() or path == root / _JOURNAL or path == root / _LOCK:
        raise SettingsRecoveryError('Settings recovery target must be a regular private file.')
    return path


def _read(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _encode(value: bytes | None):
    return None if value is None else base64.b64encode(value).decode('ascii')


def _decode(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('invalid encoded file')
    return base64.b64decode(value, validate=True)


def _write_journal(root: Path, document: dict) -> None:
    _durable_write(root / _JOURNAL, json.dumps(document, separators=(',', ':')).encode())


def _durable_remove(path: Path) -> None:
    if not path.exists():
        return
    if os.name == 'nt':
        # Python cannot fsync Windows directories. A write-through rename makes
        # the original pathname disappear durably before journal cleanup. A
        # crash may retain a private tombstone, never a live mixed config file.
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.removed.')
        os.close(descriptor)
        tombstone = Path(name)
        try:
            _replace_durable(path, tombstone)
        finally:
            tombstone.unlink(missing_ok=True)
    else:
        path.unlink()
        _sync_directory(path.parent)


def _remove_journal(root: Path) -> None:
    _durable_remove(root / _JOURNAL)


def _recover(root: Path, *, writer=_durable_write, journal_root: Path | None = None) -> None:
    journal_root = journal_root or root
    journal = journal_root / _JOURNAL
    if not journal.exists():
        return
    try:
        if journal.is_symlink():
            raise ValueError('linked journal')
        document = json.loads(journal.read_bytes())
        if document.get('version') != 1 or document.get('state') not in {'prepared', 'committed'}:
            raise ValueError('invalid journal format')
        entries = document['files']
        if not isinstance(entries, dict) or not entries:
            raise ValueError('invalid journal files')
        decoded = []
        for name, values in entries.items():
            path = _target(root, name)
            if path in {journal_root / _JOURNAL, journal_root / _LOCK}:
                raise ValueError('journal cannot overwrite its own recovery files')
            old, new = _decode(values['old']), _decode(values['new'])
            if new is None:
                raise ValueError('invalid new file')
            current = _read(path)
            if current not in (old, new):
                raise SettingsRecoveryError(
                    'Settings changed outside an interrupted save. Configuration loading is blocked; '
                    'preserve the recovery journal and resolve those edits before continuing.')
            decoded.append((path, old, new, current))
    except SettingsRecoveryError:
        raise
    except Exception as exc:
        raise SettingsRecoveryError('Settings recovery journal could not be verified. Configuration loading is blocked.') from exc
    committed = document['state'] == 'committed'
    failures = []
    for path, old, new, current in reversed(decoded):
        desired = new if committed else old
        try:
            if current == desired:
                continue
            if desired is None:
                _durable_remove(path)
            else:
                writer(path, desired)
                _sync_directory(path.parent)
        except Exception as exc:
            failures.append(exc)
    if failures:
        raise SettingsRecoveryError('Some settings could not be restored. Configuration loading is blocked; keep the recovery journal.') from failures[0]
    _remove_journal(journal_root)


def recover_settings_transaction(root: Path, *, journal_root: Path | None = None) -> bool:
    root = Path(root).resolve()
    journal_root = Path(journal_root).resolve() if journal_root is not None else root
    # Normal startup remains read-only when no interrupted save exists.
    if not (journal_root / _JOURNAL).exists():
        return False
    if settings_transaction_active():
        raise SettingsRecoveryError('Settings are being saved; configuration cannot be reloaded before the transaction finishes.')
    with _locked(journal_root):
        if not (journal_root / _JOURNAL).exists():
            return False
        _recover(root, journal_root=journal_root)
        return True


@contextmanager
def settings_transaction(root: Path, changes: dict[Path, bytes], *, writer=_durable_write, journal_root: Path | None = None):
    root = Path(root).resolve()
    journal_root = Path(journal_root).resolve() if journal_root is not None else root
    missing_directories = []
    current = journal_root
    while not current.exists():
        missing_directories.append(current)
        current = current.parent
    journal_root.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing_directories):
        _sync_directory(created.parent)
    with _locked(journal_root):
        _recover(root, writer=writer, journal_root=journal_root)
        document = {'version': 1, 'state': 'prepared', 'files': {}}
        for path, new in changes.items():
            name = str(Path(path).absolute().relative_to(root))
            target = _target(root, name)
            if target in {journal_root / _JOURNAL, journal_root / _LOCK}:
                raise SettingsRecoveryError('Settings paths cannot overwrite recovery files.')
            document['files'][name] = {'old': _encode(_read(target)), 'new': _encode(new)}
        _write_journal(journal_root, document)
        token = _ACTIVE.set(True)
        try:
            yield
            # No commit marker may reach disk before every member and its
            # containing directory. Writers may have been injected by a caller.
            for name, values in document['files'].items():
                path = _target(root, name)
                if _read(path) != _decode(values['new']):
                    raise OSError('Settings write did not produce the expected bytes.')
                # Windows FlushFileBuffers requires a handle opened for writing.
                with path.open('r+b') as handle:
                    os.fsync(handle.fileno())
                _sync_directory(path.parent)
            document['state'] = 'committed'
            _write_journal(journal_root, document)
        except BaseException:
            # A commit-marker fsync may fail after rename. Reset the intent
            # durably before undoing members, so another crash during rollback
            # cannot interpret restored old members as a committed new group.
            document['state'] = 'prepared'
            try:
                _write_journal(journal_root, document)
            except Exception as exc:
                raise SettingsRecoveryError('Settings recovery could not be made durable. Configuration loading is blocked; keep the recovery journal.') from exc
            _recover(root, writer=writer, journal_root=journal_root)
            raise
        finally:
            _ACTIVE.reset(token)
        try:
            _remove_journal(journal_root)
        except OSError:
            # The new group and commit marker are already durable. Keeping a
            # verified committed journal is safe and recoverable at next startup.
            logger.warning('Settings saved; recovery journal cleanup will be retried at startup.')
