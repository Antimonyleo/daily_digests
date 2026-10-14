"""Browser adapters around the installed DailyDigest application.

Only execution and transport change: routes, persistence, ranking and pipeline
remain the packaged Python application's implementations.
"""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
import sqlite3
import tempfile
from collections import deque
from concurrent.futures import Future
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path('/app')
_JOBS = deque()
_REQUESTS = {}
_WEB = None


async def _inline(function, *args, **kwargs):
    return function(*args, **kwargs)


async def _iterate_inline(iterator):
    for value in iterator:
        yield value


class BrowserThread:
    """Queue existing background targets for a worker callPromising boundary."""
    def __init__(self, target, args=(), kwargs=None, daemon=None):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        _JOBS.append((self.target, self.args, self.kwargs))


class BrowserExecutor:
    """Retain native provider groups and futures without unsupported threads."""
    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


def has_jobs():
    return bool(_JOBS)


def run_job():
    target, args, kwargs = _JOBS.popleft()
    return target(*args, **kwargs)


def _snapshot_paths(root):
    data = root / 'data'
    if data.exists():
        yield from (path for path in data.rglob('*') if path.is_file())
    if (root / '.env').is_file():
        yield root / '.env'


def export_snapshot(root=ROOT):
    """Copy only virtual private state; no deployed code or model artifacts."""
    return json.dumps({'format': 'dailydigest-browser-v1', 'files': {
        str(path.relative_to(root)): base64.b64encode(path.read_bytes()).decode('ascii')
        for path in _snapshot_paths(root)
        if path.name != '.compute.lock'
    }})


def decode_snapshot(payload):
    snapshot = json.loads(payload)
    if not isinstance(snapshot, dict) or snapshot.get('format') != 'dailydigest-browser-v1':
        raise ValueError('Unsupported DailyDigest backup format')
    files = snapshot.get('files')
    if not isinstance(files, dict):
        raise ValueError('Backup files must be an object')
    decoded = {}
    for name, encoded in files.items():
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or
                name != '.env' and (len(path.parts) < 2 or path.parts[0] != 'data')):
            raise ValueError('Backup contains a path outside private browser state')
        if path.name == '.compute.lock' or path in decoded:
            raise ValueError('Backup contains a duplicate or transient lock path')
        decoded[path] = base64.b64decode(encoded, validate=True)
    if any(parent in decoded for path in decoded for parent in path.parents):
        raise ValueError('Backup contains conflicting file and directory paths')
    return decoded


def restore_snapshot(payload, *, overwrite=False, root=ROOT):
    """Validate the complete snapshot before touching browser state."""
    decoded = decode_snapshot(payload)
    if not overwrite and any(_snapshot_paths(root)):
        raise ValueError('Existing browser state requires explicit overwrite')
    if overwrite:
        for path in _snapshot_paths(root):
            path.unlink()
        # Empty private directories must not prevent restoring a former file.
        data = root / 'data'
        if data.exists():
            for path in sorted(data.rglob('*'), key=lambda path: len(path.parts), reverse=True):
                if path.is_dir():
                    path.rmdir()
    for path, content in decoded.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def validate_snapshot_sqlite(decoded, *, scratch_root=None):
    """Check copied databases and their WAL files without opening live state."""
    candidates = [path for path, content in decoded.items()
                  if path.suffix.lower() in {'.db', '.sqlite', '.sqlite3'}
                  or content.startswith(b'SQLite format 3\0')]
    if not candidates:
        return
    # Browser scratch is its virtual /tmp. Tests supply an isolated directory;
    # retain scratch for inspection rather than deleting it automatically.
    stage = Path(tempfile.mkdtemp(prefix='dailydigest-import-', dir=scratch_root))
    for path in candidates:
        content = decoded[path]
        if not content.startswith(b'SQLite format 3\0'):
            raise ValueError(f'Backup database {path} is not a SQLite database')
        target = stage / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        for suffix in ('-wal', '-shm'):
            sidecar = Path(str(path) + suffix)
            if sidecar in decoded:
                Path(str(target) + suffix).write_bytes(decoded[sidecar])
        try:
            with sqlite3.connect(target) as connection:
                result = connection.execute('PRAGMA integrity_check').fetchall()
        except sqlite3.DatabaseError as exc:
            raise ValueError(f'Backup database {path} is damaged: {exc}') from exc
        if result != [('ok',)]:
            raise ValueError(f'Backup database {path} failed SQLite integrity validation')


async def transactional_import(payload, commit, *, root=ROOT, scratch_root=None):
    """Apply an explicit import only when browser storage commits successfully."""
    validate_snapshot_sqlite(decode_snapshot(payload), scratch_root=scratch_root)
    close_app()
    previous = export_snapshot(root)
    try:
        restore_snapshot(payload, overwrite=True, root=root)
        await commit(export_snapshot(root))
    except Exception:
        restore_snapshot(previous, overwrite=True, root=root)
        raise


def handle_import(payload, commit):
    from pyodide.ffi import run_sync

    async def persist_snapshot(snapshot):
        await commit(snapshot)

    run_sync(transactional_import(payload, persist_snapshot))


async def transactional_inbound(payload, commit, *, record_vote=None, root=ROOT):
    """Resolve reply labels against browser state and commit before mailbox ACK."""
    if not isinstance(payload, dict) or payload.get('ok') is not True:
        raise ValueError('Companion did not confirm inbound reply retrieval')
    records = payload.get('votes', [])
    if not isinstance(records, list):
        raise ValueError('Inbound replies must be a list')
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get('mail_id'), str)
                or not record['mail_id'] or not isinstance(record.get('line'), str)
                or not isinstance(record.get('digest_id'), str)):
            raise ValueError('Inbound reply requires a stable mail ID, vote line and digest date')
        date.fromisoformat(record['digest_id'])
    ledger_path = root / 'data/companion-inbound.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'version': 1, 'seen': {}}
    if (not isinstance(ledger, dict) or ledger.get('version') != 1
            or not isinstance(ledger.get('seen'), dict)):
        raise ValueError('Inbound deduplication state needs recovery')
    if record_vote is None:
        from dailydigest.votes import record_votes
        record_vote = record_votes
    result = {'ok': True, 'messages_received': len(records), 'messages_applied': 0,
              'votes_up': 0, 'votes_down': 0, 'votes_unknown': 0, 'already_seen': 0,
              'mail_ids': [], 'unresolved_messages': 0, 'errors': list(payload.get('errors', [])),
              'skipped_unauthorized': payload.get('skipped_unauthorized', 0),
              'skipped_missing_digest': payload.get('skipped_missing_digest', 0)}
    close_app()
    previous = export_snapshot(root)
    try:
        for record in records:
            mail_id = record['mail_id']
            if mail_id in ledger['seen']:
                result['already_seen'] += 1
            else:
                counts = record_vote(record['line'], digest_id=record['digest_id'])
                result['votes_up'] += int(counts.get('up', 0))
                result['votes_down'] += int(counts.get('down', 0))
                result['votes_unknown'] += int(counts.get('unknown', 0))
                if counts.get('up', 0) or counts.get('down', 0):
                    result['messages_applied'] += 1
                ledger['seen'][mail_id] = {'digest_id': record['digest_id'], 'counts': counts}
            recorded_counts = ledger['seen'][mail_id]['counts']
            if (not recorded_counts.get('unknown', 0)
                    and (recorded_counts.get('up', 0) or recorded_counts.get('down', 0))):
                if mail_id not in result['mail_ids']:
                    result['mail_ids'].append(mail_id)
            else:
                result['unresolved_messages'] += 1
                result['errors'].append(
                    f"Reply for digest {record['digest_id']} has unresolved votes; mailbox message left unread."
                )
        if records:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_path.write_text(json.dumps(ledger, sort_keys=True))
            await commit(export_snapshot(root))
    except Exception:
        close_app()
        restore_snapshot(previous, overwrite=True, root=root)
        raise
    return result


def handle_inbound(payload, commit):
    from pyodide.ffi import run_sync

    async def persist_snapshot(snapshot):
        await commit(snapshot)

    return json.dumps(run_sync(transactional_inbound(json.loads(payload), persist_snapshot)))


def browser_provider_error(backend, companion_configured=False):
    if not companion_configured and str(backend).strip().lower() in {'claude_cli', 'codex_cli'}:
        return ('Claude Code and Codex CLI summaries require a local process and cannot '
                'run in this browser. Choose Extractive, or explicitly configure an API provider.')
    return None


def companion_summary_payload(items, backend, bio='', keywords=()):
    """Transfer selected metadata and a short profile after explicit consent."""
    return {'backend': backend, 'items': [{
        'id': int(row.id), 'title': (row.title or '').strip(),
        'abstract': (row.abstract or '').strip()[:1500],
        'authors': getattr(row, 'authors', '') or '', 'source': row.source or '',
        'section': row.section or '', 'url': getattr(row, 'url', '') or '',
        'published_at': row.published_at.isoformat() if row.published_at else None,
    } for row in items if row.id is not None],
        'profile': {'bio': str(bio)[:400], 'keywords': list(keywords)[:20]}}


def companion_text_payload(system, user, backend, timeout=120.0):
    if backend not in {'claude_cli', 'codex_cli'}:
        raise ValueError('Companion text requests require a configured CLI backend')
    if len(system) > 8000 or len(user) > 32000:
        raise ValueError('Selected text exceeds the companion request limit')
    return {'backend': backend, 'system': system, 'user': user,
            'timeout': max(1.0, min(float(timeout), 120.0))}


def companion_email_payload():
    from dailydigest.config import get_settings
    from dailydigest.store import DigestRow, session_scope
    from sqlalchemy import select

    previews = sorted(Path(get_settings().db_path).parent.glob('digest-*.html'))
    if not previews:
        raise ValueError('Brew a digest before sending it through the companion')
    with session_scope() as session:
        digest_id = session.scalar(select(DigestRow.id).order_by(DigestRow.created_at.desc()))
    if not digest_id:
        raise ValueError('There is no published digest to email')
    return json.dumps({'html': previews[-1].read_text(),
                       'subject': f'DailyDigest {digest_id}', 'digest_id': digest_id})


def companion_email_sent(digest_id):
    from dailydigest.store import mark_sent

    mark_sent(digest_id)


def install_browser():
    global _WEB
    import threading
    import fastapi.dependencies.utils
    import fastapi.routing
    import starlette.concurrency
    import starlette.routing
    import starlette.responses
    from filelock import SoftFileLock
    from pyodide.ffi import run_sync, to_js
    from js import browserEmbed, browserYield, browserCompanion, browserCompanionConfigured
    import numpy as np
    from dailydigest import job_lock
    from dailydigest.rank import embed

    # Neither native file locks nor Python threads exist in the WASM process.
    # navigator.locks excludes other browser tabs before this app is installed.
    def acquire_browser_lock(db_path, *, timeout=0.0):
        lock = SoftFileLock(str(Path(db_path).resolve().parent / '.compute.lock'))
        lock.acquire(timeout=timeout)
        return lock

    job_lock.acquire_compute_lock = acquire_browser_lock
    for module in (fastapi.dependencies.utils, fastapi.routing,
                   starlette.concurrency, starlette.routing):
        module.run_in_threadpool = _inline
    for module in (starlette.concurrency, starlette.responses, fastapi.routing):
        module.iterate_in_threadpool = _iterate_inline

    def build_encoder(_pref, model, _device, _batch, _threads):
        if model != 'BAAI/bge-small-en-v1.5':
            raise ValueError('The browser bundle contains only BAAI/bge-small-en-v1.5')

        def encoder(texts):
            result = run_sync(browserEmbed(to_js(texts)))
            return np.asarray(result.to_py(), dtype=np.float32)

        encoder.embedding_backend = 'browser-onnx'
        return encoder

    embed._build_encoder = build_encoder
    # Keep browser vectors distinct from imported native cache backends.
    embed._resolved_backend_name = lambda: 'browser-onnx'

    from dailydigest import pipeline, web
    pipeline.ThreadPoolExecutor = BrowserExecutor
    _WEB = web
    web.threading = SimpleNamespace(Thread=BrowserThread, Lock=threading.Lock, Event=threading.Event)
    original_validate = web._validate_setup

    def validate_browser_setup(form):
        errors = original_validate(form)
        provider_error = browser_provider_error(form.get('llm_backend', 'extractive'), browserCompanionConfigured())
        if provider_error:
            errors.append(provider_error)
        return errors

    web._validate_setup = validate_browser_setup
    from dailydigest import summarize
    original_provider = summarize._summarize_via_provider_with_provenance
    companion_provenance = {}

    def companion_cli(batch, backend, *, timeout=180.0, settings=None):
        if not browserCompanionConfigured():
            raise RuntimeError(browser_provider_error(backend))
        payload = companion_summary_payload(
            batch, backend, summarize._READER_CONTEXT, summarize._READER_KEYWORDS,
        )
        result = json.loads(run_sync(browserCompanion('/summarize', json.dumps(payload),
                                                     'POST', timeout)))
        allowed = {int(row.id) for row in batch if row.id is not None}
        generated = {int(key): str(value) for key, value in result.get('summaries', {}).items()
                     if int(key) in allowed and isinstance(value, str) and value.strip()}
        companion_provenance.update({int(key): value
                                     for key, value in result.get('provenance', {}).items()
                                     if int(key) in generated})
        return generated

    def companion_provider(items, backend, *, settings=None, diagnostics=None):
        companion_provenance.clear()
        generated, provenance = original_provider(items, backend, settings=settings,
                                                   diagnostics=diagnostics)
        if backend in {'claude_cli', 'codex_cli'}:
            provenance.update(companion_provenance)
            if diagnostics is not None:
                diagnostics['fallback_items'] = sum(value == 'extractive_fallback'
                                                    for value in provenance.values())
        return generated, provenance

    def companion_text(system, user, backend, *, timeout=180.0, settings=None):
        if not browserCompanionConfigured():
            raise RuntimeError(browser_provider_error(backend))
        payload = companion_text_payload(system, user, backend, timeout)
        result = json.loads(run_sync(browserCompanion('/text', json.dumps(payload),
                                                     'POST', payload['timeout'] + 5)))
        text = result.get('text')
        if not isinstance(text, str):
            raise ValueError('Companion returned no text response')
        return text

    summarize._cli_text = companion_text
    summarize._call_cli = companion_cli
    summarize._summarize_via_provider_with_provenance = companion_provider
    original_run = web.run_all

    def browser_run(*args, **kwargs):
        provider_error = browser_provider_error(web.get_settings().llm_backend, browserCompanionConfigured())
        if provider_error:
            raise ValueError(provider_error)
        callback = kwargs.get('progress_callback')

        def progress(stage, payload):
            if callback:
                callback(stage, payload)
            # Cooperative safe point: flush browser progress/cancel requests;
            # all native publication and cancellation checks remain in run_all.
            run_sync(browserYield())

        kwargs['progress_callback'] = progress
        return original_run(*args, **kwargs)

    web.run_all = browser_run
    return web.app


async def asgi_request(message, emit):
    """Invoke FastAPI in memory using a trusted synthetic loopback origin."""
    request_id = str(message['id'])
    parsed = urlsplit(message.get('path', '/'))
    if parsed.scheme or parsed.netloc or not parsed.path.startswith('/'):
        raise ValueError('Request path must be local to DailyDigest')
    body = message.get('body') or ''
    if not isinstance(body, str):
        body = json.dumps(body)
    body = body.encode('utf-8')
    headers = {str(k).lower(): str(v) for k, v in message.get('headers', {}).items()}
    headers.update({'host': 'localhost', 'origin': 'http://localhost'})
    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
             'http_version': '1.1', 'method': message.get('method', 'GET').upper(),
             'scheme': 'http', 'path': parsed.path, 'raw_path': parsed.path.encode(),
             'query_string': parsed.query.encode(), 'root_path': '',
             'headers': [(k.encode(), v.encode()) for k, v in headers.items()],
             'client': ('127.0.0.1', 1234), 'server': ('localhost', 80)}
    disconnected = asyncio.Event()
    _REQUESTS[request_id] = disconnected
    delivered = False
    status, response_headers, chunks = 500, {}, []
    streaming = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type': 'http.request', 'body': body, 'more_body': False}
        await disconnected.wait()
        return {'type': 'http.disconnect'}

    async def send(event):
        nonlocal status, response_headers, streaming
        if disconnected.is_set():
            raise OSError('Browser client disconnected')
        if event['type'] == 'http.response.start':
            status = event['status']
            response_headers = {k.decode(): v.decode() for k, v in event['headers']}
            streaming = response_headers.get('content-type', '').startswith('text/event-stream')
        elif event['type'] == 'http.response.body':
            text = event.get('body', b'').decode('utf-8')
            if streaming:
                emit(json.dumps({'id': message['id'], 'type': 'stream', 'body': text}))
            else:
                chunks.append(text)

    try:
        await _WEB.app(scope, receive, send)
        return json.dumps({'id': message['id'], 'type': 'response', 'status': status,
                           'headers': response_headers, 'body': ''.join(chunks)})
    finally:
        _REQUESTS.pop(request_id, None)


def handle_request(payload, emit):
    from pyodide.ffi import run_sync
    return run_sync(asgi_request(json.loads(payload), emit))


def cancel_request(request_id):
    event = _REQUESTS.get(str(request_id))
    if event:
        event.set()


def close_app():
    """Release SQLite handles before explicit replacement and worker restart."""
    if _WEB is not None:
        from dailydigest import store
        for value in vars(store).values():
            if hasattr(value, 'dispose') and not inspect.isclass(value):
                value.dispose()
