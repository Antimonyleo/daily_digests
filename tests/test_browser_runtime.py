"""Browser host adapters preserve private-state and route contracts."""
import asyncio
import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runtime():
    spec = importlib.util.spec_from_file_location(
        'browser_runtime_test', Path(__file__).parents[1] / 'browser/runtime.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_roundtrip_keeps_private_files_only(runtime, tmp_path):
    source = tmp_path / 'source'
    (source / 'data').mkdir(parents=True)
    (source / 'data/digest.db').write_bytes(b'\0sqlite fixture')
    (source / '.env').write_text('PROFILE_PATH=data/profile.yaml\n')
    (source / 'app.py').write_text('not private state')
    (source / 'data/.compute.lock').write_text('ephemeral')
    snapshot = runtime.export_snapshot(source)
    assert set(json.loads(snapshot)['files']) == {'data/digest.db', '.env'}
    target = tmp_path / 'target'
    runtime.restore_snapshot(snapshot, root=target)
    assert (target / 'data/digest.db').read_bytes() == b'\0sqlite fixture'
    assert (target / '.env').read_text() == 'PROFILE_PATH=data/profile.yaml\n'
    with pytest.raises(ValueError, match='explicit overwrite'):
        runtime.restore_snapshot(snapshot, root=target)


def test_startup_restore_decodes_once_and_checks_sqlite_before_writing(runtime, tmp_path, monkeypatch):
    payload = json.dumps({'format': 'dailydigest-browser-v1', 'files': {
        'data/profile.yaml': base64.b64encode(b'keywords: [astronomy]').decode()}})
    calls = []
    original = runtime.decode_snapshot

    def decode(value, **kwargs):
        calls.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(runtime, 'decode_snapshot', decode)
    runtime.restore_startup_snapshot(payload, root=tmp_path, scratch_root=tmp_path)
    assert calls == [payload]
    assert (tmp_path / 'data/profile.yaml').read_bytes() == b'keywords: [astronomy]'
    invalid = json.dumps({'format': 'dailydigest-browser-v1', 'files': {
        'data/broken.sqlite': base64.b64encode(b'broken').decode()}})
    other = tmp_path / 'bad'
    with pytest.raises(ValueError, match='not a SQLite database'):
        runtime.restore_startup_snapshot(invalid, root=other, scratch_root=tmp_path)
    assert not (other / 'data').exists()


def test_fast_browser_checkpoint_equals_full_export_byte_for_byte(runtime, tmp_path):
    import shutil
    import subprocess

    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the actual browser checkpoint encoder')
    (tmp_path / 'data' / 'nested').mkdir(parents=True)
    (tmp_path / 'data' / 'nested' / 'unicode-\U0001f9ec-\u00e9.bin').write_bytes(bytes(range(256)) * 16384)
    (tmp_path / 'data' / 'empty.bin').write_bytes(b'')
    (tmp_path / 'data' / 'payload.bin').write_bytes(b'\x00\xff\x01')
    (tmp_path / 'data' / '.compute.lock').write_text('excluded')
    (tmp_path / '.env').write_text('LLM_BACKEND=extractive\n')
    names = runtime.snapshot_file_names(tmp_path)
    script = '''import {readFileSync} from 'node:fs';
import {capturePrivateSnapshot} from './browser/storage.mjs';
const names=JSON.parse(process.argv[2]);
const result=capturePrivateSnapshot(names,name=>new Uint8Array(readFileSync(process.argv[1]+'/'+name)));
process.stdout.write(result.payload);
'''
    result = subprocess.run([node, '--input-type=module', '-e', script, str(tmp_path), names],
                            cwd=Path(__file__).parents[1], text=True, capture_output=True, check=True)
    assert result.stdout == runtime.export_snapshot(tmp_path)


@pytest.mark.parametrize('path', ['/data/foo', 'data/../../app.py', 'app.py', '.env/child'])
def test_snapshot_rejects_escape_before_deleting_state(runtime, tmp_path, path):
    (tmp_path / 'data').mkdir()
    original = tmp_path / 'data/profile.yaml'
    original.write_text('private state')
    payload = json.dumps({'format': 'dailydigest-browser-v1', 'files': {
        path: base64.b64encode(b'bad').decode()
    }})
    with pytest.raises(ValueError):
        runtime.restore_snapshot(payload, overwrite=True, root=tmp_path)
    assert original.read_text() == 'private state'


def test_threads_queue_native_target_until_host_starts_job(runtime):
    calls = []
    runtime.BrowserThread(target=calls.append, args=('trained',), daemon=True).start()
    assert calls == []
    assert runtime.has_jobs()
    runtime.run_job()
    assert calls == ['trained']
    assert not runtime.has_jobs()


def test_browser_cancel_signal_tracks_only_the_active_brew(runtime):
    import threading

    event = threading.Event()
    assert runtime.browser_cancelled() is False
    runtime._WEB = SimpleNamespace(_BREW_JOB={'run_id': 'active'}, _RUN_CANCEL={'active': event})
    assert runtime.browser_cancelled() is False
    event.set()
    assert runtime.browser_cancelled() is True
    runtime._WEB._BREW_JOB['run_id'] = 'another'
    assert runtime.browser_cancelled() is False


def test_asgi_preserves_query_body_and_streams_without_buffering(runtime):
    scopes, emitted = [], []

    async def app(scope, receive, send):
        scopes.append(scope)
        assert (await receive())['body'] == b'{"vote":1}'
        await send({'type': 'http.response.start', 'status': 200,
                    'headers': [(b'content-type', b'text/event-stream')]})
        await send({'type': 'http.response.body', 'body': b'data: {"stage":"done"}\n\n'})

    runtime._WEB = SimpleNamespace(app=app)
    result = json.loads(asyncio.run(runtime.asgi_request({
        'id': 1, 'method': 'POST', 'path': '/run/stream?run_id=abc',
        'body': '{"vote":1}', 'headers': {'X-CSRF-Token': 'secret'},
    }, emitted.append)))
    assert scopes[0]['query_string'] == b'run_id=abc'
    assert (b'host', b'localhost') in scopes[0]['headers']
    assert (b'x-csrf-token', b'secret') in scopes[0]['headers']
    assert result['body'] == ''
    assert json.loads(emitted[0])['type'] == 'stream'
    assert json.loads(emitted[0])['body'] == 'data: {"stage":"done"}\n\n'
    assert runtime._REQUESTS == {}


def test_browser_executor_preserves_future_results_and_failures(runtime):
    from concurrent.futures import as_completed

    def failure():
        raise ValueError('provider unavailable')

    with runtime.BrowserExecutor(max_workers=4) as executor:
        successful = executor.submit(lambda value: value * 2, 3)
        failed = executor.submit(failure)
        assert set(as_completed([successful, failed])) == {successful, failed}
        assert successful.result() == 6
        with pytest.raises(ValueError, match='provider unavailable'):
            failed.result()


@pytest.mark.parametrize('backend', ['claude_cli', 'codex_cli'])
def test_browser_rejects_local_cli_providers_explicitly(runtime, backend):
    assert 'cannot run in this browser' in runtime.browser_provider_error(backend)


@pytest.mark.parametrize('backend', ['extractive', 'api', 'anthropic'])
def test_browser_retains_supported_explicit_provider_configuration(runtime, backend):
    assert runtime.browser_provider_error(backend) is None


def test_invalid_sqlite_import_preserves_open_app_and_private_state(runtime, tmp_path, monkeypatch):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/profile.yaml').write_text('existing profile')
    closed, commits = [], []
    monkeypatch.setattr(runtime, 'close_app', lambda: closed.append(True))
    invalid = json.dumps({'format': 'dailydigest-browser-v1', 'files': {
        'data/digest.db': base64.b64encode(b'not a SQLite database').decode(),
        'data/profile.yaml': base64.b64encode(b'imported profile').decode(),
    }})

    async def commit(snapshot):
        commits.append(snapshot)

    with pytest.raises(ValueError, match='not a SQLite database'):
        asyncio.run(runtime.transactional_import(invalid, commit, root=tmp_path,
                                                scratch_root=tmp_path))
    assert (tmp_path / 'data/profile.yaml').read_text() == 'existing profile'
    assert not (tmp_path / 'data/digest.db').exists()
    assert closed == commits == []


def test_import_storage_failure_rolls_back_all_private_files(runtime, tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/profile.yaml').write_text('existing profile')
    (tmp_path / 'data/state').write_text('file before import')
    (tmp_path / '.env').write_text('LLM_BACKEND=extractive\n')
    before = json.loads(runtime.export_snapshot(tmp_path))
    imported = json.dumps({'format': 'dailydigest-browser-v1', 'files': {
        'data/profile.yaml': base64.b64encode(b'imported profile').decode(),
        'data/state/new.txt': base64.b64encode(b'new nested path').decode(),
        '.env': base64.b64encode(b'LLM_BACKEND=api\n').decode(),
    }})

    async def commit(_snapshot):
        assert (tmp_path / 'data/profile.yaml').read_text() == 'imported profile'
        raise OSError('IndexedDB quota exceeded')

    with pytest.raises(OSError, match='IndexedDB quota exceeded'):
        asyncio.run(runtime.transactional_import(imported, commit, root=tmp_path))
    assert json.loads(runtime.export_snapshot(tmp_path)) == before


def test_valid_sqlite_import_commits_snapshot_after_validation(runtime, tmp_path):
    import sqlite3

    source = tmp_path / 'source'
    (source / 'data').mkdir(parents=True)
    with sqlite3.connect(source / 'data/digest.db') as connection:
        connection.execute('CREATE TABLE votes (item INTEGER, vote INTEGER)')
        connection.execute('INSERT INTO votes VALUES (7, 1)')
    payload = runtime.export_snapshot(source)
    commits = []
    target = tmp_path / 'target'

    async def commit(snapshot):
        commits.append(snapshot)

    asyncio.run(runtime.transactional_import(payload, commit, root=target,
                                            scratch_root=tmp_path))
    assert len(commits) == 1
    assert json.loads(commits[0]) == json.loads(payload)
    with sqlite3.connect(target / 'data/digest.db') as connection:
        assert connection.execute('SELECT * FROM votes').fetchall() == [(7, 1)]


def test_import_write_failure_rolls_back_before_commit(runtime, tmp_path, monkeypatch):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/profile.yaml').write_text('existing profile')
    original_restore = runtime.restore_snapshot
    calls = []

    def interrupted_restore(payload, **kwargs):
        calls.append(payload)
        if len(calls) == 1:
            (tmp_path / 'data/profile.yaml').write_text('interrupted write')
            raise OSError('write interrupted')
        return original_restore(payload, **kwargs)

    monkeypatch.setattr(runtime, 'restore_snapshot', interrupted_restore)

    async def commit(_snapshot):
        raise AssertionError('interrupted import must never commit')

    payload = json.dumps({'format': 'dailydigest-browser-v1', 'files': {}})
    with pytest.raises(OSError, match='write interrupted'):
        asyncio.run(runtime.transactional_import(payload, commit, root=tmp_path))
    assert (tmp_path / 'data/profile.yaml').read_text() == 'existing profile'
    assert len(calls) == 2


def test_explicit_companion_restores_cli_provider_choice(runtime):
    assert runtime.browser_provider_error('claude_cli', companion_configured=True) is None
    assert runtime.browser_provider_error('codex_cli', companion_configured=True) is None


def test_companion_summary_transfers_only_selected_metadata_and_short_profile(runtime):
    from datetime import datetime, timezone

    selected = SimpleNamespace(id=7, title='Paper', abstract='a' * 2000, authors='Author',
                               source='PubMed', section='research', url='https://example.org/paper',
                               published_at=datetime(2026, 10, 13, tzinfo=timezone.utc),
                               private_note='Never transfer this note', embedding=[1, 2, 3])
    payload = runtime.companion_summary_payload([selected], 'claude_cli', 'bio' * 300,
                                              [f'interest {index}' for index in range(30)])
    assert payload['backend'] == 'claude_cli'
    assert payload['items'][0]['id'] == 7
    assert len(payload['items'][0]['abstract']) == 1500
    assert len(payload['profile']['bio']) == 400
    assert len(payload['profile']['keywords']) == 20
    assert 'private_note' not in payload['items'][0]
    assert 'embedding' not in payload['items'][0]
    assert payload['items'][0]['published_at'] == '2026-10-13T00:00:00+00:00'


def test_companion_text_limits_and_timeout_match_service_contract(runtime):
    payload = runtime.companion_text_payload('system', 'selected text', 'claude_cli', 180)
    assert payload['timeout'] == 120
    with pytest.raises(ValueError, match='request limit'):
        runtime.companion_text_payload('system', 'x' * 32001, 'codex_cli')
    with pytest.raises(ValueError, match='configured online'):
        runtime.companion_text_payload('system', 'text', 'extractive')


def test_worker_first_companion_connection_and_url_boundaries():
    import shutil
    import subprocess

    if not shutil.which('node'):
        pytest.skip('Node is required to exercise the worker host adapter')
    worker = Path(__file__).parents[1] / 'browser/worker.mjs'
    script = r'''
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const posted = [], fetched = [], saved = [];
const context = vm.createContext({
  URL, setTimeout, AbortSignal,
  self: { postMessage: message => posted.push(message) },
  fetch: async (url, options) => {
    fetched.push({url, options});
    return {ok: true, text: async () => JSON.stringify({ok: true, capabilities: {email: true}})};
  },
});
const source = fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*$/gm, '');
vm.runInContext(source, context);
context.saved = saved;
vm.runInContext(`
  boot = Promise.resolve({});
  state = {py: {globals: {get: name => () => false}}};
  database = {transaction: () => {
    const transaction = {objectStore: () => ({put: (value, key) => {
      saved.push({value, key}); setTimeout(() => transaction.oncomplete(), 0);
    }})};
    return transaction;
  }};
`, context);
(async () => {
  await context.self.onmessage({data: {id: 1, type: 'configure-companion', endpoint: 'https://companion.example', token: 'explicit-secret'}});
  assert.equal(posted[0].status, 200);
  assert.equal(fetched[0].url, 'https://companion.example/health');
  assert.equal(fetched[0].options.headers['X-DailyDigest-Token'], 'explicit-secret');
  assert.equal(saved[0].key, 'companion');
  assert.equal(saved[0].value.token, 'explicit-secret');
  await context.self.onmessage({data: {id: 2, type: 'companion-status'}});
  assert.equal(JSON.parse(posted[1].body).configured, true);
  assert.ok(!posted[1].body.includes('explicit-secret'));
  assert.equal(fetched.length, 1);
  assert.equal(context.companionEndpoint('http://127.0.0.1:8768'), 'http://127.0.0.1:8768');
  assert.throws(() => context.companionEndpoint('http://remote.example'), /require HTTPS/);
  assert.throws(() => context.companionEndpoint('https://secret@remote.example'), /embedded credentials/);
  assert.throws(() => context.companionEndpoint('https://remote.example?token=secret'), /query or fragment/);
  await context.self.onmessage({data: {id: 3, type: 'configure-companion', endpoint: '', token: ''}});
  assert.equal(fetched.length, 1);
  assert.equal(saved[1].value, null);
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run(['node', '-e', script, str(worker)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_inbound_resolves_reply_against_browser_named_digest_and_deduplicates(
    runtime, tmp_path, monkeypatch,
):
    from dailydigest import config, store, votes
    from sqlalchemy import select

    (tmp_path / 'data').mkdir()
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'data/digest.db'))
    config.get_settings.cache_clear()
    monkeypatch.setattr(votes, '_update_rocchio', lambda *args, **kwargs: None)
    runtime._WEB = SimpleNamespace()  # Enable native connection disposal around snapshots.
    store.init_db()
    with store.session_scope() as session:
        old = store.ItemRow(source='fixture', section='research', external_id='old',
                            url='https://example.org/old', title='Earlier paper',
                            digest_id='2026-10-12', item_label='R1')
        recent = store.ItemRow(source='fixture', section='research', external_id='recent',
                               url='https://example.org/recent', title='Recent paper',
                               digest_id='2026-10-13', item_label='R1')
        session.add_all([old, recent])
        session.flush()
        old_id, recent_id = old.id, recent.id
    store.write_digest('2026-10-12', [('R1', old_id)])
    store.mark_sent('2026-10-12', run_id='a' * 32, labeled_items=[('R1', old_id)])
    commits = []

    async def commit(snapshot):
        commits.append(snapshot)

    payload = {'ok': True, 'votes': [{'mail_id': 'stable-mail', 'line': '+R1 -R99',
                                      'digest_id': '2026-10-12', 'run_id': 'a' * 32}]}
    result = asyncio.run(runtime.transactional_inbound(payload, commit, root=tmp_path))
    assert result['messages_applied'] == result['votes_up'] == 1
    assert result['votes_down'] == 0
    assert result['votes_unknown'] == 1
    with store.session_scope() as session:
        assert session.scalar(select(store.VoteRow.value).where(store.VoteRow.item_id == old_id)) == 1
        assert session.scalar(select(store.VoteRow.value).where(store.VoteRow.item_id == recent_id)) is None
    repeated = asyncio.run(runtime.transactional_inbound(payload, commit, root=tmp_path))
    assert repeated['already_seen'] == 1
    assert repeated['messages_applied'] == repeated['votes_up'] == repeated['votes_down'] == 0
    assert result['mail_ids'] == repeated['mail_ids'] == []
    assert result['unresolved_messages'] == repeated['unresolved_messages'] == 1
    assert len(commits) == 2


def test_inbound_storage_failure_restores_votes_and_dedup_state(runtime, tmp_path):
    (tmp_path / 'data').mkdir()
    state = tmp_path / 'data/fake-votes.json'
    state.write_text('{"existing":1}')
    before = json.loads(runtime.export_snapshot(tmp_path))

    def record_vote(line, digest_id, *, run_id, sent_only):
        assert (line, digest_id) == ('+R1', '2026-10-13')
        assert run_id is None and sent_only
        state.write_text('{"existing":1,"new":1}')
        return {'up': 1, 'down': 0, 'unknown': 0}

    async def commit(_snapshot):
        assert (tmp_path / 'data/companion-inbound.json').exists()
        raise OSError('IndexedDB write failed')

    payload = {'ok': True, 'votes': [{'mail_id': 'new-mail', 'line': '+R1',
                                      'digest_id': '2026-10-13'}]}
    with pytest.raises(OSError, match='IndexedDB write failed'):
        asyncio.run(runtime.transactional_inbound(payload, commit, record_vote=record_vote,
                                                 root=tmp_path))
    assert json.loads(runtime.export_snapshot(tmp_path)) == before


def test_inbound_requires_digest_date_before_applying_any_votes(runtime, tmp_path):
    calls = []

    async def commit(_snapshot):
        raise AssertionError('invalid reply must not commit')

    payload = {'ok': True, 'votes': [{'mail_id': 'new-mail', 'line': '+R1', 'digest_id': None}]}
    with pytest.raises(ValueError, match='digest date'):
        asyncio.run(runtime.transactional_inbound(payload, commit,
                                                 record_vote=lambda *args: calls.append(args),
                                                 root=tmp_path))
    assert calls == []


def test_worker_acknowledges_mail_only_after_browser_commit():
    import shutil
    import subprocess

    if not shutil.which('node'):
        pytest.skip('Node is required to exercise the worker host adapter')
    worker = Path(__file__).parents[1] / 'browser/worker.mjs'
    script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const events = [], posted = [];
let failCommit = false;
const context = vm.createContext({URL, setTimeout, clearTimeout, AbortSignal,
  self: {postMessage: message => posted.push(message)},
  fetch: async url => {
    events.push(url.endsWith('/ack') ? 'ack' : 'fetch');
    return {ok: true, text: async () => JSON.stringify({ok: true, votes: []})};
  },
});
vm.runInContext(fs.readFileSync(process.argv[1].replace('worker.mjs', 'storage.mjs'), 'utf8').replace(/^export /gm, ''), context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*$/gm, ''), context);
context.apply = {callPromising: async (_payload, commit) => {
  events.push('apply'); await commit('saved snapshot');
  events.push('saved');
  return JSON.stringify({ok: true, mail_ids: ['stable-mail'], messages_applied: 1, errors: []});
}};
context.transaction = () => {
  const transaction = {error: new Error('IndexedDB failed'), objectStore: () => ({delete() {}, put: () => {
    setTimeout(() => { events.push(failCommit ? 'abort' : 'commit');
      if (failCommit) transaction.onabort(); else transaction.oncomplete(); }, 0);
  }})};
  return transaction;
};
vm.runInContext(`
  boot = Promise.resolve({}); companion = {endpoint: 'https://companion.example', token: 'explicit'};
  state = {py: {globals: {get: name => name === 'handle_inbound' ? apply : () => false}}};
  database = {transaction};
  storageWriter = createSnapshotWriter({generation: () => Number(state.py.globals.get('state_generation')()), capture: () => 'existing digest snapshot', save: saveSnapshot});
`, context);
(async () => {
  await context.self.onmessage({data: {id: 1, type: 'companion-inbound'}});
  assert.deepEqual(events, ['fetch', 'apply', 'commit', 'saved', 'ack']);
  assert.equal(posted[0].status, 200);
  assert.equal(JSON.parse(posted[0].body).mailbox_acknowledged, true);
  events.length = 0; failCommit = true;
  await context.self.onmessage({data: {id: 2, type: 'companion-inbound'}});
  assert.deepEqual(events, ['fetch', 'apply', 'abort']);
  assert.equal(posted[1].status, 500);
  assert.match(posted[1].body, /IndexedDB failed/);
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
    result = subprocess.run(['node', '-e', script, str(worker)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_companion_email_captures_rendered_labels_before_preview_changes(runtime, monkeypatch):
    from dailydigest import store, votes

    store.init_db()
    with store.session_scope() as session:
        original = store.ItemRow(source='fixture', section='research', external_id='email-original',
                                 title='Original emailed paper', url='https://example.test/original')
        preview = store.ItemRow(source='fixture', section='research', external_id='email-preview',
                                title='Later preview paper', url='https://example.test/preview')
        session.add_all([original, preview])
        session.flush()
        original_id, preview_id = int(original.id), int(preview.id)
    store.write_digest('2026-10-16', [('R3', original_id)])
    payload = runtime.companion_email_payload()
    captured = json.loads(payload)
    assert 'Original emailed paper' in captured['html']
    assert 'Later preview paper' not in captured['html']
    assert captured['labeled_items'] == [['R3', original_id]]
    assert f"[run:{captured['run_id']}]" in captured['subject']
    store.write_digest('2026-10-16', [('R3', preview_id)])
    runtime.companion_email_sent(payload)
    monkeypatch.setattr(votes, '_update_rocchio', lambda: None)
    assert votes.record_votes('+R3', '2026-10-16', run_id=captured['run_id'], sent_only=True)['up'] == 1
    assert votes.get_vote_value(original_id) == 1
    assert votes.get_vote_value(preview_id) is None


def test_worker_confirms_the_captured_email_payload_after_delivery():
    import shutil
    import subprocess

    if not shutil.which('node'):
        pytest.skip('Node is required to exercise the worker host adapter')
    worker = Path(__file__).parents[1] / 'browser/worker.mjs'
    script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const posted = [], confirmed = [];
const payload = JSON.stringify({html:'<p>Original R3</p>',subject:'DailyDigest 2026-10-16 [run:abc]',
  digest_id:'2026-10-16',run_id:'abc',labeled_items:[['R3',1]]});
const context = vm.createContext({URL,setTimeout,clearTimeout,AbortSignal,
  self:{postMessage:message=>posted.push(message)},
  fetch:async (_url, options)=>{
    const sent=JSON.parse(options.body);
    assert.deepEqual(Object.keys(sent).sort(),['digest_id','html','subject']);
    return {ok:true,text:async()=>JSON.stringify({ok:true,sent:true})};
  },payload,confirmed});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*$/gm,''),context);
vm.runInContext(`boot=Promise.resolve({});companion={endpoint:'https://companion.example',token:'explicit'};
state={py:{globals:{get:name=>name==='companion_email_payload'?()=>payload:
  name==='companion_email_sent'?value=>confirmed.push(value):()=>false}}};
storageWriter={flush:async()=>true};`,context);
(async()=>{await context.self.onmessage({data:{id:1,type:'companion-email'}});
  assert.deepEqual(confirmed,[payload]);assert.equal(posted[0].status,200);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run(['node', '-e', script, str(worker)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def companion_provider_module(monkeypatch):
    from dailydigest import summarize

    for name in ('effective_summary_backend', '_call_llm', '_call_anthropic', '_call_cli',
                 '_summarize_via_provider_with_provenance', '_llm_text', 'summary_signature'):
        monkeypatch.setattr(summarize, name, getattr(summarize, name))
    monkeypatch.setattr(summarize, '_extractive', lambda row: f'Local summary {row.id}')
    return summarize


def _provider_rows(count):
    from dailydigest.store import ItemRow

    return [ItemRow(id=index, title=f'Paper {index}', abstract='selected article ' * 150,
                    source='fixture', section='research', authors='Public Author',
                    url=f'https://example.org/paper/{index}') for index in range(1, count + 1)]


@pytest.mark.parametrize('backend', ['api', 'anthropic'])
def test_advertised_companion_api_uses_server_credentials_and_actual_provenance(
    runtime, companion_provider_module, backend,
):
    from dailydigest.config import get_settings

    module, calls = companion_provider_module, []
    settings = get_settings().model_copy(update={'llm_backend': backend, 'llm_api_key': ''})

    def call(path, payload, timeout):
        calls.append((path, payload, timeout))
        return {'summaries': {'1': 'Provider summary', '2': 'Local summary 2',
                              '999': 'Unrequested article'},
                'provenance': {'1': backend, '2': 'extractive_fallback'},
                'diagnostics': {'provider_calls': 1, 'fallback_items': 1}}

    runtime.install_companion_providers(module, lambda chosen: chosen == backend, call)
    profile = SimpleNamespace(bio='reader biography ' * 100,
                              keywords=[f'topic {index}' for index in range(30)],
                              canonical_facets={})
    diagnostics = {}
    generated, provenance = module.summarize_items_with_provenance(
        _provider_rows(2), profile, settings=settings, diagnostics=diagnostics,
    )
    assert module.effective_summary_backend(settings=settings) == backend
    assert generated == {1: 'Provider summary', 2: 'Local summary 2'}
    assert provenance == {1: backend, 2: 'extractive_fallback'}
    assert diagnostics['fallback_items'] == 1
    assert calls[0][0] == '/summarize'
    payload = calls[0][1]
    assert payload['backend'] == backend
    assert len(payload['items'][0]['abstract']) == 1500
    assert len(payload['profile']['bio']) == 400
    assert len(payload['profile']['keywords']) == 20
    assert set(payload) == {'backend', 'items', 'profile', 'timeout'}
    assert 'llm_api_key' not in json.dumps(payload)
    assert 'token' not in payload


def test_companion_failure_retains_native_api_budget_and_extractive_fallback(
    runtime, companion_provider_module, monkeypatch,
):
    from dailydigest.config import get_settings

    module, clock, calls = companion_provider_module, [0.0], []
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    settings = get_settings().model_copy(update={
        'llm_backend': 'api', 'llm_api_key': '', 'llm_summary_budget_seconds': 1.0,
        'llm_summary_max_failures': 2,
    })

    def unavailable(_path, payload, timeout):
        calls.append((payload, timeout))
        clock[0] = 2.0
        raise OSError('companion provider unavailable')

    runtime.install_companion_providers(module, lambda backend: backend == 'api', unavailable)
    diagnostics = {}
    generated, provenance = module.summarize_items_with_provenance(
        _provider_rows(7), settings=settings, diagnostics=diagnostics,
    )
    assert len(calls) == 1
    assert calls[0][1] == 1.0
    assert set(provenance.values()) == {'extractive_fallback'}
    assert len(generated) == 7
    assert diagnostics['provider_calls'] == 1
    assert diagnostics['provider_stopped'] is True
    assert diagnostics['fallback_items'] == 7


def test_nonadvertised_backend_preserves_direct_provider_and_key_requirement(
    runtime, companion_provider_module, monkeypatch,
):
    from dailydigest.config import get_settings

    module, direct = companion_provider_module, []

    def direct_api(batch, *, settings, timeout):
        direct.append(settings.llm_api_key)
        return {row.id: 'Direct provider summary' for row in batch}

    monkeypatch.setattr(module, '_call_llm', direct_api)

    def unauthorized_companion(*_args):
        raise AssertionError('nonadvertised backend must not contact companion')

    runtime.install_companion_providers(module, lambda backend: backend == 'anthropic',
                                       unauthorized_companion)
    settings = get_settings().model_copy(update={'llm_backend': 'api', 'llm_api_key': ''})
    generated, provenance = module.summarize_items_with_provenance(_provider_rows(1), settings=settings)
    assert provenance == {1: 'extractive'}
    assert generated == {1: 'Local summary 1'}
    assert direct == []
    settings.llm_api_key = 'browser-only-secret'
    generated, provenance = module.summarize_items_with_provenance(_provider_rows(1), settings=settings)
    assert generated == {1: 'Direct provider summary'}
    assert provenance == {1: 'api'}
    assert direct == ['browser-only-secret']


def test_setup_waives_only_api_key_for_explicit_advertised_backend(runtime):
    errors = ['API backend requires an API key.', 'API backend requires a model name.',
              'API key cannot contain line breaks.', 'Add at least one research interest.']
    validate = lambda _form: list(errors)
    assert runtime.validate_companion_setup(validate, {'llm_backend': 'api'},
                                            lambda backend: backend == 'api') == errors[1:]
    assert runtime.validate_companion_setup(validate, {'llm_backend': 'api'},
                                            lambda backend: backend == 'anthropic') == errors


@pytest.mark.parametrize('backend,timeout,expected', [
    ('api', None, 60.0), ('anthropic', 0.25, 0.25), ('claude_cli', None, 120.0),
])
def test_advertised_companion_text_uses_native_backend_timeout_without_browser_key(
    runtime, companion_provider_module, backend, timeout, expected,
):
    from dailydigest.config import get_settings

    module, calls = companion_provider_module, []
    settings = get_settings().model_copy(update={'llm_backend': backend, 'llm_api_key': ''})

    def call(path, payload, request_timeout):
        calls.append((path, payload, request_timeout))
        return {'text': 'Companion-generated catch-up'}

    runtime.install_companion_providers(module, lambda chosen: chosen == backend, call)
    assert module._llm_text('system', 'selected article titles', settings=settings,
                            timeout=timeout) == 'Companion-generated catch-up'
    assert calls[0][0] == '/text'
    assert calls[0][1] == {'backend': backend, 'system': 'system',
                          'user': 'selected article titles', 'timeout': expected}
    assert calls[0][2] == expected + 5


def test_unadvertised_text_preserves_native_direct_path(runtime, companion_provider_module, monkeypatch):
    from dailydigest.config import get_settings

    module, direct = companion_provider_module, []

    def direct_text(system, user, *, settings, timeout):
        direct.append((system, user, settings.llm_api_key, timeout))
        return 'Direct completion'

    monkeypatch.setattr(module, '_llm_text', direct_text)
    runtime.install_companion_providers(module, lambda _backend: False,
                                       lambda *_args: pytest.fail('unexpected companion call'))
    settings = get_settings().model_copy(update={'llm_backend': 'api', 'llm_api_key': 'direct-key'})
    assert module._llm_text('system', 'user', settings=settings, timeout=0.25) == 'Direct completion'
    assert direct == [('system', 'user', 'direct-key', 0.25)]


def test_pipeline_aliases_use_companion_credentials_and_identity_for_cached_summaries(
    runtime, companion_provider_module, monkeypatch,
):
    from dailydigest import pipeline
    from dailydigest.config import get_settings

    module, identity, calls = companion_provider_module, ['endpoint|model-one'], []
    for name in ('effective_summary_backend', 'summary_signature'):
        monkeypatch.setattr(pipeline, name, getattr(pipeline, name))

    def call(path, payload, _timeout):
        calls.append((path, payload))
        return {'summaries': {'1': 'Companion summary'}, 'provenance': {'1': 'api'}}

    runtime.install_companion_providers(module, lambda backend: backend == 'api', call,
                                       lambda _backend: identity[0], pipeline=pipeline)
    settings = get_settings().model_copy(update={'llm_backend': 'api', 'llm_api_key': ''})
    rows = _provider_rows(1)
    assert pipeline.effective_summary_backend is module.effective_summary_backend
    assert pipeline.summary_signature is module.summary_signature
    backend = pipeline.effective_summary_backend(settings=settings)
    assert backend == 'api'
    signature = pipeline.summary_signature(rows[0], backend=backend, settings=settings)
    generated, provenance = pipeline.summarize_items_with_provenance(rows, settings=settings)
    assert generated == {1: 'Companion summary'}
    assert provenance == {1: 'api'}
    assert calls[0][0] == '/summarize'
    assert pipeline.summary_signature(rows[0], backend=backend, settings=settings) == signature
    identity[0] = 'endpoint|model-two'
    assert pipeline.summary_signature(rows[0], backend=backend, settings=settings) != signature


def test_worker_refreshes_provider_identity_before_brew_and_preserves_digest_on_health_failure():
    import shutil
    import subprocess

    if not shutil.which('node'):
        pytest.skip('Node is required to exercise the worker host adapter')
    worker = Path(__file__).parents[1] / 'browser/worker.mjs'
    script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const events = [], posted = [];
let failHealth = false;
const context = vm.createContext({URL, setTimeout, clearTimeout, AbortSignal,
  self: {postMessage: message => posted.push(message)},
  fetch: async url => {
    assert.ok(url.endsWith('/health')); events.push('health');
    if (failHealth) throw new Error('companion health unavailable');
    return {ok: true, text: async () => JSON.stringify({ok: true,
      capabilities: {summarize: ['api']}, provider_identities: {api: 'new-model-fingerprint'}})};
  },
});
vm.runInContext(fs.readFileSync(process.argv[1].replace('worker.mjs', 'storage.mjs'), 'utf8').replace(/^export /gm, ''), context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*$/gm, ''), context);
context.request = {callPromising: async message => {
  events.push('native-request');
  assert.equal(vm.runInContext('companion.provider_identities.api', context), 'new-model-fingerprint');
  return JSON.stringify({id: JSON.parse(message).id, type: 'response', status: 200, headers: {}, body: '{}'});
}};
context.transaction = () => {
  const transaction = {objectStore: () => ({delete() {}, put: (_value, key) => {
    events.push(`save-${key}`); setTimeout(() => transaction.oncomplete(), 0);
  }})};
  return transaction;
};
vm.runInContext(`
  boot = Promise.resolve({}); companion = {endpoint: 'https://companion.example', token: 'explicit',
    capabilities: {summarize: ['api']}, provider_identities: {api: 'old-model-fingerprint'}};
  globalThis.browserCompanionSupports = backend => companion.capabilities.summarize.includes(backend);
  state = {py: {globals: {get: name => {
    if (name === 'selected_browser_backend') return () => 'api';
    if (name === 'handle_request') return request;
    if (name === 'export_snapshot') return () => 'existing digest snapshot';
    if (name === 'state_generation') return () => 1;
    return () => false;
  }}}};
  database = {transaction};
  storageWriter = createSnapshotWriter({generation: () => Number(state.py.globals.get('state_generation')()), capture: () => 'existing digest snapshot', save: saveSnapshot});
`, context);
(async () => {
  await context.self.onmessage({data: {id: 1, type: 'request', method: 'POST', path: '/run/start', body: '{}'}});
  assert.deepEqual(events, ['health', 'save-companion', 'native-request']);
  assert.equal(await vm.runInContext('storageWriter.flush({force:true})', context), true);
  assert.deepEqual(events, ['health', 'save-companion', 'native-request', 'save-snapshot']);
  assert.equal(posted.find(row => row.type === 'response' && row.id === 1).status, 200);
  events.length = 0; failHealth = true;
  await context.self.onmessage({data: {id: 2, type: 'request', method: 'POST', path: '/run/start', body: '{}'}});
  assert.deepEqual(events, ['health']);
  assert.equal(posted.find(row => row.type === 'response' && row.id === 2).status, 500);
  assert.match(posted.find(row => row.type === 'response' && row.id === 2).body, /companion health unavailable/);
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
    result = subprocess.run(['node', '-e', script, str(worker)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_portable_snapshot_redacts_credentials_and_preserves_nonsecret_settings(runtime, tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/profile.yaml').write_text('keywords: [protein degradation]\n')
    (tmp_path / '.env').write_text('LLM_BACKEND=api\nLLM_API_KEY="private\\nkey"\nIMAP_PASSWORD=mail-secret\nDIGEST_HOUR=9\n')
    internal = json.loads(runtime.export_snapshot(tmp_path))
    assert b'private' in base64.b64decode(internal['files']['.env'])
    portable = json.loads(runtime.export_portable_snapshot(root=tmp_path))
    environment = base64.b64decode(portable['files']['.env']).decode()
    assert 'private' not in environment and 'mail-secret' not in environment
    assert 'LLM_BACKEND="api"' in environment and 'DIGEST_HOUR="9"' in environment
    assert portable['credentialsIncluded'] is False
    assert set(portable['omittedCredentialKeys']) == {'LLM_API_KEY', 'IMAP_PASSWORD'}
    assert json.loads(runtime.export_portable_snapshot(True, root=tmp_path)) == internal


def test_portable_import_preserves_only_existing_destination_credentials(runtime, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / '.env').write_text('LLM_BACKEND=extractive\nLLM_API_KEY=source-secret\nDIGEST_HOUR=9\n')
    portable = runtime.export_portable_snapshot(root=source)
    destination = tmp_path / 'destination'
    destination.mkdir()
    (destination / '.env').write_text('LLM_API_KEY=destination-secret\nDIGEST_HOUR=6\n')
    result = runtime.preserve_omitted_credentials(portable, root=destination)
    values = runtime._environment(base64.b64decode(json.loads(result)['files']['.env']).decode())
    assert values['LLM_API_KEY'] == 'destination-secret'
    assert values['DIGEST_HOUR'] == '9'
    assert values['LLM_BACKEND'] == 'extractive'
    assert 'source-secret' not in result


def test_private_mutation_tracking_observes_file_writes_but_not_reads(runtime, tmp_path):
    import os

    before = runtime.state_generation()
    runtime.private_write_audit('open', (str(tmp_path / 'data/profile.yaml'), 'r', os.O_RDONLY), root=tmp_path)
    runtime.private_write_audit('open', (str(tmp_path / 'public.py'), 'w', os.O_WRONLY), root=tmp_path)
    runtime.private_write_audit('open', (str(tmp_path / 'data/.compute.lock'), 'w', os.O_WRONLY), root=tmp_path)
    assert runtime.state_generation() == before
    runtime.private_write_audit('open', (str(tmp_path / 'data/profile.yaml'), 'w', os.O_WRONLY), root=tmp_path)
    runtime.private_write_audit('os.rename', (str(tmp_path / 'data/new'), str(tmp_path / 'data/profile.yaml')), root=tmp_path)
    assert runtime.state_generation() == before + 2


def test_sql_mutation_tracking_ignores_read_transactions_and_noop_writes():
    import subprocess
    import sys

    runtime_path = Path(__file__).parents[1] / 'browser/runtime.py'
    script = r'''
import importlib.util, sys
from sqlalchemy import create_engine, text
spec = importlib.util.spec_from_file_location('runtime_tracking_test', sys.argv[1])
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
runtime.install_state_tracking()
engine = create_engine('sqlite://')
with engine.begin() as connection:
    connection.execute(text('CREATE TABLE tracking (id INTEGER PRIMARY KEY)'))
before = runtime.state_generation()
with engine.connect() as connection:
    assert connection.execute(text('SELECT COUNT(*) FROM tracking')).scalar() == 0
assert runtime.state_generation() == before
with engine.begin() as connection:
    assert connection.execute(text('INSERT INTO tracking VALUES (1) RETURNING id')).scalar() == 1
assert runtime.state_generation() > before
before = runtime.state_generation()
with engine.begin() as connection:
    connection.execute(text('INSERT OR IGNORE INTO tracking VALUES (1)'))
    connection.execute(text('UPDATE tracking SET id=id WHERE id=999'))
assert runtime.state_generation() == before
'''
    result = subprocess.run([sys.executable, '-c', script, str(runtime_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cold_wal_connection_lifecycle_invalidates_cached_browser_database(tmp_path):
    import shutil
    import sqlite3
    import subprocess
    import sys

    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the actual cached browser checkpoint encoder')
    source = tmp_path / 'source.db'
    root = tmp_path / 'restored'
    (root / 'data').mkdir(parents=True)
    database = root / 'data/digest.db'
    connection = sqlite3.connect(source)
    try:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE fixture (value TEXT)')
        connection.execute("INSERT INTO fixture VALUES ('committed WAL data')")
        connection.commit()
        shutil.copyfile(source, database)
        shutil.copyfile(str(source) + '-wal', str(database) + '-wal')
    finally:
        connection.close()
    script = r'''
import importlib.util, json, subprocess, sys
from pathlib import Path
from sqlalchemy import create_engine, text
spec = importlib.util.spec_from_file_location('runtime_wal_tracking_test', sys.argv[1])
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
root = Path(sys.argv[2])
runtime.ROOT = root
runtime.install_state_tracking()
before = json.loads(runtime.export_snapshot(root))
versions = json.loads(runtime.snapshot_file_versions(root))
cache = {name: {'version':versions[name], 'encoded':encoded} for name, encoded in before['files'].items()}
engine = create_engine('sqlite:///' + str(root / 'data/digest.db'))
with engine.connect() as connection:
    assert connection.execute(text('SELECT value FROM fixture')).scalar() == 'committed WAL data'
read_generation = runtime.state_generation()
with engine.connect() as connection:
    assert connection.execute(text('SELECT value FROM fixture')).scalar() == 'committed WAL data'
assert runtime.state_generation() == read_generation, 'Reused read-only connections must stay clean'
engine.dispose()
after = json.loads(runtime.export_snapshot(root))
assert before['files']['data/digest.db'] != after['files']['data/digest.db']
assert 'data/digest.db-wal' not in after['files'], 'The last close must checkpoint the cold WAL'
encoder = r"""import {readFileSync} from 'node:fs';
import {capturePrivateSnapshot} from './browser/storage.mjs';
const [root,names,versions,previous]=process.argv.slice(1);
const result=capturePrivateSnapshot(JSON.parse(names),name=>new Uint8Array(readFileSync(root+'/'+name)),
  {cache:new Map(Object.entries(JSON.parse(previous))),versions:JSON.parse(versions)});
process.stdout.write(result.payload);"""
encoded = subprocess.run([sys.argv[3], '--input-type=module', '-e', encoder, str(root),
    runtime.snapshot_file_names(root), runtime.snapshot_file_versions(root), json.dumps(cache)],
    check=True, text=True, capture_output=True).stdout
assert encoded == runtime.export_snapshot(root), 'Cached checkpoint lost committed WAL data after last close'
assert runtime.state_generation() > read_generation, 'Closing private SQLite connections must invalidate file versions'
public = create_engine('sqlite:///' + str(root.parent / 'unrelated.db'))
generation = runtime.state_generation()
with public.connect() as connection:
    connection.execute(text('SELECT 1'))
public.dispose()
assert runtime.state_generation() == generation, 'Unrelated SQLite files are not browser private state'
'''
    runtime_path = Path(__file__).parents[1] / 'browser/runtime.py'
    result = subprocess.run([sys.executable, '-c', script, str(runtime_path), str(root), node],
                            cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_failed_startup_recovery_quarantines_original_atomically():
    import shutil
    import subprocess

    if not shutil.which('node'):
        pytest.skip('Node is required to exercise the worker recovery adapter')
    worker = Path(__file__).parents[1] / 'browser/worker.mjs'
    script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const saved = new Map([['snapshot', 'damaged original']]); let fail = false;
const database = {transaction: (_store, mode) => {
  const staged = new Map(saved);
  let pending = 0, ended = false;
  const finish = () => {
    if (mode !== 'readwrite' || pending || ended) return;
    ended = true;
    if (fail) {transaction.error = new Error('quota exceeded'); transaction.onabort();}
    else {saved.clear(); for (const [key, value] of staged) saved.set(key, value); transaction.oncomplete();}
  };
  const read = value => {
    const request = {}; pending++;
    setTimeout(() => {request.result=value();request.onsuccess();pending--;setTimeout(finish,0);},0);
    return request;
  };
  const transaction = {objectStore: () => ({
    get(key) { return read(() => staged.get(key)); },
    getAllKeys() { return read(() => [...staged.keys()]); },
    put(value, key) { staged.set(key, value); }, delete(key) { staged.delete(key); }
  })};
  setTimeout(finish,0);
  return transaction;
}};
const context = vm.createContext({URL, setTimeout, clearTimeout, databaseFixture: database, self: {postMessage() {}}});
vm.runInContext(fs.readFileSync(process.argv[1].replace('worker.mjs', 'storage.mjs'), 'utf8').replace(/^export /gm, ''), context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*$/gm, ''), context);
vm.runInContext('startupFailed = true; ownership = Promise.resolve(); database = databaseFixture;', context);
(async () => {
  await assert.rejects(context.recoveryRequest({type: 'recovery-reset'}), /confirmation/);
  fail = true;
  await assert.rejects(context.recoveryRequest({type: 'recovery-reset', overwrite: true}), /quota/);
  assert.deepEqual([...saved], [['snapshot', 'damaged original']]);
  fail = false;
  const result = await context.recoveryRequest({type: 'recovery-reset', overwrite: true});
  assert.equal(result.quarantined, true);
  assert.equal(saved.has('snapshot'), false);
  assert.equal(saved.get(result.recoveryKey), 'damaged original');
  assert.equal(await context.recoveryRequest({type: 'recovery-export'}), 'damaged original');
  assert.equal(await context.recoveryRequest({type: 'recovery-export', key: result.recoveryKey}), 'damaged original');
  const status = await context.recoveryRequest({type: 'recovery-status'});
  assert.equal(status.available, true);
  assert.equal(status.quarantined.length, 1);
  // Ordinary successful imports retain the previous committed snapshot too,
  // and bounded recovery history does not evict it before older copies.
  // Historical timestamps ahead of the current clock must not evict the new
  // original or make recovery-export pick an older desk (also crosses width).
  for (let i=0;i<4;i++) saved.set(`recovery:${9999999999996+i}:fixture`,'older desk');
  saved.set('snapshot','working desk');
  const key = await context.replaceSnapshot('restored desk');
  assert.equal(saved.get('snapshot'),'restored desk');
  assert.equal(saved.get(key),'working desk');
  assert.equal([...saved.keys()].filter(key=>key.startsWith('recovery:')).length,3);
  assert.equal(await context.recoveryRequest({type:'recovery-export',key}),'working desk');
  const current = await context.recoveryRequest({type:'recovery-status'});
  assert.equal(current.quarantined.at(-1),key);
  await assert.rejects(context.recoveryRequest({type: 'recovery-export', key: 'companion'}), /Invalid/);
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
    result = subprocess.run(['node', '-e', script, str(worker)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_worker_quota_failure_keeps_valid_http_responses_and_reports_unsaved_state():
    import shutil
    import subprocess

    if not shutil.which('node'):
        pytest.skip('Node is required to exercise worker storage failures')
    worker = Path(__file__).parents[1] / 'browser/worker.mjs'
    script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const posted = []; let fail = true, generation = 1;
const context = vm.createContext({URL, setTimeout, clearTimeout, self: {postMessage: value => posted.push(value)}});
vm.runInContext(fs.readFileSync(process.argv[1].replace('worker.mjs', 'storage.mjs'), 'utf8').replace(/^export /gm, ''), context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*$/gm, ''), context);
context.request = {callPromising: async raw => {
  const data = JSON.parse(raw); if (data.method === 'POST') generation++;
  return JSON.stringify({id: data.id, type: 'response', status: 200, headers: {}, body: '{"ok":true}'});
}};
context.generation = () => generation;
context.transaction = () => {
  const transaction = {error: new Error('quota exceeded'), objectStore: () => ({delete() {}, put() {
    setTimeout(() => fail ? transaction.onabort() : transaction.oncomplete(), 0);
  }})}; return transaction;
};
vm.runInContext(`
  boot = Promise.resolve({}); database = {transaction};
  state = {py: {globals: {get: name => name === 'handle_request' ? request : () => false}}};
  storageWriter = createSnapshotWriter({generation, capture: () => 'private state', save: saveSnapshot,
    warning: message => self.postMessage({type: 'storage-warning', message})});
`, context);
(async () => {
  for (const [id, method] of [[1, 'GET'], [2, 'POST']]) {
    await context.self.onmessage({data: {id, type: 'request', method, path: '/fixture'}});
    const response = posted.find(value => value.type === 'response' && value.id === id);
    assert.equal(response.status, 200);
    assert.equal(response.headers['x-dailydigest-storage'], 'pending');
    assert.equal(await vm.runInContext('storageWriter.flush({force:true})', context), false);
    assert.equal(response.body, '{"ok":true}');
  }
  assert.ok(posted.some(value => value.type === 'storage-warning' && value.message.includes('quota')));
  await context.self.onmessage({data: {id: 3, type: 'configure-compute', mode: 'cpu'}});
  const refused = posted.find(value => value.type === 'response' && value.id === 3);
  assert.equal(refused.status, 500);
  assert.match(refused.body, /reading state could not be saved/);
  fail = false; assert.equal(await vm.runInContext('storageWriter.flush({force:true})', context), true);
  assert.ok(posted.some(value => value.type === 'storage-warning' && value.message === null));
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
    result = subprocess.run(['node', '-e', script, str(worker)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_explicit_native_sqlite_write_hook_is_observed(runtime, tmp_path):
    from dailydigest.store import mark_private_state_changed
    import sys

    sys.addaudithook(lambda event, args: runtime.private_write_audit(event, args, root=tmp_path))
    before = runtime.state_generation()
    mark_private_state_changed(tmp_path / 'data/backups/digest-pre-v1.db')
    assert runtime.state_generation() > before


def test_only_sqlite_shared_memory_is_excluded_from_private_export(runtime, tmp_path):
    import sqlite3
    data = tmp_path / 'data'
    data.mkdir()
    database = data / 'digest.db'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE fixture (id INTEGER)')
    (data / 'digest.db-shm').write_bytes(b'ephemeral SQLite bookkeeping')
    (data / 'notes-shm').write_bytes(b'private reader document')
    files = json.loads(runtime.export_snapshot(tmp_path))['files']
    assert 'data/digest.db-shm' not in files
    assert 'data/notes-shm' in files


def test_startup_requires_one_canonical_private_checkpoint(runtime):
    # Restoring a legacy portable snapshot can change JSON ordering/metadata.
    # Capture once after startup; subsequent read-only routes remain clean.
    assert runtime.state_generation() > 0
