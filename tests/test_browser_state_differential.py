"""Real application routes must agree with unconditional private checkpoints.

The app runs through the browser ASGI adapter, with browser-style inline tasks.
Node runs the actual snapshot writer, not a Python imitation. Pyodide execution
and IndexedDB durability are separately checked by the browser acceptance suite.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
from sqlalchemy import event, text


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def tracked_app(tmp_path, monkeypatch):
    from dailydigest import config, store, web

    root = tmp_path / 'browser-app'
    (root / 'data').mkdir(parents=True)
    profile = root / 'data/profile.yaml'
    profile.write_text('name: Ada\nbio: Researcher.\nkeywords: [protein design]\ndownweight: []\n')
    monkeypatch.setenv('DB_PATH', str(root / 'data/digest.db'))
    monkeypatch.setenv('PROFILE_PATH', str(profile))
    monkeypatch.setenv('OPPORTUNITY_PROFILE_PATH', str(root / 'data/opportunities.yaml'))
    config.reload_settings()
    monkeypatch.setattr(store, 'SETTINGS', config.SETTINGS)
    monkeypatch.setattr(web, 'SETTINGS', config.SETTINGS)
    monkeypatch.setattr(web, '_REPO_ROOT', root)
    monkeypatch.setattr(web, '_ENV_PATH', root / '.env')
    monkeypatch.setattr(web, '_digest_id', lambda: '2026-05-05')

    spec = importlib.util.spec_from_file_location('differential_browser_runtime', REPO / 'browser/runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.ROOT = root
    runtime._WEB = web
    # Audit hooks cannot be removed. Keep this one inert after this fixture;
    # never observe another test's or the user's private state.
    audit_enabled = True
    original_audit = runtime.private_write_audit

    def private_audit(event, args):
        if audit_enabled:
            original_audit(event, args, root=root)

    monkeypatch.setattr(runtime, 'private_write_audit', private_audit)
    listeners = []
    listen = event.listen

    def remember_listener(target, name, callback, *args, **kwargs):
        listen(target, name, callback, *args, **kwargs)
        listeners.append((target, name, callback))

    with monkeypatch.context() as installation:
        installation.setattr(event, 'listen', remember_listener)
        runtime.install_state_tracking()
    import fastapi.dependencies.utils
    import fastapi.routing
    import starlette.concurrency
    import starlette.responses
    import starlette.routing
    import threading

    for module in (fastapi.dependencies.utils, fastapi.routing,
                   starlette.concurrency, starlette.routing):
        monkeypatch.setattr(module, 'run_in_threadpool', runtime._inline)
    for module in (starlette.concurrency, starlette.responses, fastapi.routing):
        monkeypatch.setattr(module, 'iterate_in_threadpool', runtime._iterate_inline)
    monkeypatch.setattr(web, 'threading', SimpleNamespace(
        Thread=runtime.BrowserThread, Lock=threading.Lock, Event=threading.Event,
    ))
    monkeypatch.setattr(web, '_RUN_STARTED', set())
    monkeypatch.setattr(web, '_RUN_QUEUES', {})
    monkeypatch.setattr(web, '_RUN_CANCEL', {})
    monkeypatch.setattr(web, '_RUN_PUBLISHED', set())
    monkeypatch.setattr(web, '_BREW_JOB', {'running': False, 'run_id': None})
    store.init_db()
    with store.session_scope() as session:
        session.add(store.DigestRow(id='2026-05-05', item_count=1))
        item = store.ItemRow(source='Nature', section='research', external_id='differential',
                             url='https://fixture.invalid/paper', title='Designed protein assembly',
                             abstract='A designed protein assembly with experimental validation.',
                             digest_id='2026-05-05', item_label='R1')
        session.add(item)
        session.flush()
        item_id = int(item.id)
    store.write_impressions('2026-05-05', [('research', item_id, 0, 0.9)], model_version='fixture')
    try:
        yield SimpleNamespace(runtime=runtime, root=root, web=web, store=store, item_id=item_id)
    finally:
        audit_enabled = False
        for target, name, callback in listeners:
            event.remove(target, name, callback)


def _request(app, method, path, *, form=None, body=None):
    headers = {'X-CSRF-Token': app.web._CSRF_TOKEN}
    if form is not None:
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        body = urlencode({'_csrf_token': app.web._CSRF_TOKEN, **form})
    elif isinstance(body, dict):
        headers['Content-Type'] = 'application/json'
        body = json.dumps(body)
    return json.loads(asyncio.run(app.runtime.asgi_request({
        'id': 'differential', 'method': method, 'path': path,
        'headers': headers, 'body': body or '',
    }, lambda _message: None)))


def _assert_real_writer_matches(checkpoints):
    node = shutil.which('node')
    if not node:
        pytest.fail('Node is required for the real browser snapshot writer differential test')
    script = r'''
      import { createSnapshotWriter, capturePrivateSnapshot } from './browser/storage.mjs';
      const checkpoints = JSON.parse(await new Promise(resolve => {
        let text = ''; process.stdin.setEncoding('utf8');
        process.stdin.on('data', chunk => text += chunk);
        process.stdin.on('end', () => resolve(text));
      }));
      let checkpoint, stored = null; const cache = new Map();
      const writer = createSnapshotWriter({
        generation: () => checkpoint.generation,
        ready: () => checkpoint.ready,
        capture: () => {
          const files = JSON.parse(checkpoint.snapshot).files;
          return capturePrivateSnapshot(checkpoint.names,
            name => new Uint8Array(Buffer.from(files[name], 'base64')), {cache, versions:checkpoint.versions || {}}).payload;
        },
        save: async snapshot => { stored = snapshot; },
      });
      for (checkpoint of checkpoints) {
        const flushed = await writer.flush({force: true});
        if (!flushed || stored !== checkpoint.snapshot)
          throw Error(`Tracked snapshot differs byte-for-byte at ${checkpoint.label}: ` +
            JSON.stringify(Object.entries(JSON.parse(checkpoint.snapshot).files).filter(
              ([path, bytes]) => JSON.parse(stored || '{"files":{}}').files[path] !== bytes
            ).map(([path]) => path)));
      }
      console.log(JSON.stringify({checkpoints: checkpoints.map(value => value.label)}));
    '''
    result = subprocess.run([node, '--input-type=module', '-e', script], input=json.dumps(checkpoints),
                            cwd=REPO, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)['checkpoints']) == len(checkpoints)


def test_real_mutating_routes_match_full_private_export_after_writer_flush(tracked_app, monkeypatch):
    app = tracked_app
    checkpoints = []

    def checkpoint(label):
        checkpoints.append({'label': label, 'generation': app.runtime.state_generation(),
                            'ready': app.runtime.snapshot_ready(),
                            'names': json.loads(app.runtime.snapshot_file_names(app.root)),
                            'versions': json.loads(app.runtime.snapshot_file_versions(app.root)),
                            'snapshot': app.runtime.export_snapshot(app.root)})

    checkpoint('initial committed state')
    for path in ('/healthz', '/setup', '/saved', '/ranking/status', '/reading/study'):
        before = app.runtime.state_generation()
        response = _request(app, 'GET', path)
        assert response['status'] == 200, (path, response)
        assert app.runtime.state_generation() == before, f'Read-only GET dirtied state: {path}'
        checkpoint(f'read-only {path}')

    def mutation(label, method, path, **kwargs):
        before = app.runtime.state_generation()
        response = _request(app, method, path, **kwargs)
        assert response['status'] in (200, 303), (label, response)
        assert app.runtime.state_generation() > before, f'Untracked real route: {label}'
        checkpoint(label)

    item_id = app.item_id
    mutation('vote', 'POST', f'/vote/{item_id}/70')
    mutation('save', 'POST', f'/bookmark/{item_id}')
    mutation('unsave', 'DELETE', f'/bookmark/{item_id}')
    mutation('known', 'POST', f'/known/{item_id}')
    mutation('unknown', 'DELETE', f'/known/{item_id}')
    mutation('negative rating', 'POST', f'/vote/{item_id}/10')
    mutation('rating reason', 'POST', f'/vote/{item_id}/reason/off_topic')
    mutation('remove rating reason', 'DELETE', f'/vote/{item_id}/reason/off_topic')
    mutation('clear vote', 'DELETE', f'/vote/{item_id}')
    mutation('setup profile and settings', 'POST', '/setup', form={
        'name': 'Ada', 'bio': 'Researcher in protein design.', 'topics': 'protein design | 100',
        'downweight': '', 'llm_backend': 'extractive', 'user_tz': 'UTC',
    })
    mutation('digest view writes impressions', 'GET', '/')
    with app.store.session_scope() as session:
        assert all(row.viewed for row in session.query(app.store.ImpressionRow).all())

    def mocked_brew(**_kwargs):
        # Actual queued route/job execution; collection, models and providers
        # are intentionally replaced by a small committed publication.
        with app.store.session_scope() as session:
            session.execute(text('UPDATE items SET summary=:summary WHERE id=:item_id'),
                            {'summary': 'Mocked brew published summary', 'item_id': item_id})
        (app.root / 'data/mock-brew.json').write_text('{"published":true}')

    monkeypatch.setattr(app.web, 'run_all', mocked_brew)
    response = _request(app, 'POST', '/run/start', body={'run_id': 'd1ff12345678'})
    assert response['status'] == 200
    assert app.runtime.has_jobs()
    before = app.runtime.state_generation()
    app.runtime.run_job()
    assert app.runtime.state_generation() > before
    checkpoint('mocked queued brew commits')
    assert not app.runtime.has_jobs()
    for row in checkpoints:
        assert row['ready'], row['label']
        app.runtime.validate_snapshot_sqlite(app.runtime.decode_snapshot(row['snapshot']),
                                             scratch_root=app.root.parent)
    _assert_real_writer_matches(checkpoints)


def test_mid_brew_uncommitted_transaction_defers_snapshot_and_restores_after_commit(tracked_app):
    app = tracked_app
    # A browser job can yield for HTTP/embeddings with a SQLite transaction open.
    # A checkpoint may only acknowledge durability after that transaction ends.
    assert app.runtime.snapshot_ready()
    with app.store._ENGINE.connect() as connection:
        transaction = connection.begin()
        connection.execute(text('UPDATE items SET summary=:summary WHERE id=:item_id'),
                           {'summary': 'not committed yet', 'item_id': app.item_id})
        assert not app.runtime.snapshot_ready(), 'Checkpoint must defer an active SQLite writer'
        with pytest.raises(RuntimeError, match='transaction is still active'):
            app.runtime.export_snapshot(app.root)
        with pytest.raises(RuntimeError, match='transaction is still active'):
            app.runtime.snapshot_file_names(app.root)
        transaction.commit()
    assert app.runtime.snapshot_ready()
    snapshot = app.runtime.export_snapshot(app.root)
    decoded = app.runtime.decode_snapshot(snapshot)
    app.runtime.validate_snapshot_sqlite(decoded, scratch_root=app.root.parent)
    restored = app.root.parent / 'restored-checkpoint'
    app.runtime.restore_snapshot(snapshot, root=restored)
    import sqlite3

    with sqlite3.connect(restored / 'data/digest.db') as connection:
        assert connection.execute('SELECT summary FROM items WHERE id=?', (app.item_id,)).fetchone() == (
            'not committed yet',
        )


def test_actual_legacy_migration_marks_c_level_backup_write(tracked_app, monkeypatch):
    import sqlite3

    from dailydigest import config

    app = tracked_app
    legacy = app.root / 'data/legacy.db'
    with sqlite3.connect(legacy) as connection:
        connection.executescript('''
            CREATE TABLE items (
                id INTEGER PRIMARY KEY, source VARCHAR NOT NULL,
                section VARCHAR NOT NULL, external_id VARCHAR NOT NULL,
                url VARCHAR NOT NULL, title TEXT NOT NULL
            );
            CREATE TABLE votes (
                id INTEGER PRIMARY KEY, item_id INTEGER NOT NULL,
                value INTEGER NOT NULL, created_at DATETIME
            );
            INSERT INTO items VALUES
                (1, 'Fixture', 'research', 'one', 'https://fixture.invalid/one', 'One');
            INSERT INTO votes VALUES
                (1, 1, 1, '2026-01-01 10:00:00'),
                (2, 1, -1, '2026-01-01 11:00:00');
        ''')
    app.store._ENGINE.dispose()
    monkeypatch.setenv('DB_PATH', str(legacy))
    config.reload_settings()
    monkeypatch.setattr(app.store, 'SETTINGS', config.SETTINGS)
    monkeypatch.setattr(app.store, '_ENGINE', None)
    monkeypatch.setattr(app.store, '_SessionLocal', None)
    monkeypatch.setattr(app.store, '_INITIALIZED', False)
    notifications = []
    mark_changed = app.store.mark_private_state_changed

    def record_backup_mark(path):
        before = app.runtime.state_generation()
        mark_changed(path)
        notifications.append((Path(path), before, app.runtime.state_generation()))

    monkeypatch.setattr(app.store, 'mark_private_state_changed', record_backup_mark)
    app.store.init_db()
    backup = app.root / 'data/backups/digest-pre-v1.db'
    expected = {Path(str(database) + suffix) for database in (legacy, backup)
                for suffix in ('', '-wal', '-journal')}
    assert len(notifications) == len(expected)
    assert {path for path, _before, _after in notifications} == expected
    assert all(after > before for _path, before, after in notifications)
    with sqlite3.connect(backup) as connection:
        assert connection.execute('SELECT value FROM votes ORDER BY id').fetchall() == [(1,), (-1,)]
    checkpoint = {'label': 'C-level migration backup', 'generation': app.runtime.state_generation(),
                  'ready': app.runtime.snapshot_ready(),
                  'names': json.loads(app.runtime.snapshot_file_names(app.root)),
                            'versions': json.loads(app.runtime.snapshot_file_versions(app.root)),
                  'snapshot': app.runtime.export_snapshot(app.root)}
    assert 'data/backups/digest-pre-v1.db' in json.loads(checkpoint['snapshot'])['files']
    app.runtime.validate_snapshot_sqlite(app.runtime.decode_snapshot(checkpoint['snapshot']),
                                         scratch_root=app.root.parent)
    _assert_real_writer_matches([checkpoint])
