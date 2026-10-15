import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { localPath, routeFromHash, renderedDocument, WorkerClient, browserSupportError, friendlyLoadingStatus, publicRoute, announcementKey } from './app.mjs';

test('modified internal links retain the public project path and deep route', () => {
  assert.equal(publicRoute('/saved?section=research', 'https://example.test/daily_digests/#/setup'),
    'https://example.test/daily_digests/#/saved?section=research');
  const html = renderedDocument('<html><head></head><body><a href="/saved">Saved</a></body></html>', 'token', '/', 'https://example.test/daily_digests/');
  assert.match(html, /href="https:\/\/example.test\/daily_digests\/#\/saved"/);
  assert.throws(() => publicRoute('//another.test/'), /local DailyDigest/);
});

test('screen reader progress announces milestones, not each clock tick', () => {
  assert.equal(announcementKey({ phase: 'encoding', completed: 4, total: 100, elapsed_ms: 1000 }),
    announcementKey({ phase: 'encoding', completed: 5, total: 100, elapsed_ms: 2000 }));
  assert.notEqual(announcementKey({ phase: 'encoding', completed: 4, total: 100 }),
    announcementKey({ phase: 'encoding', completed: 14, total: 100 }));
});

test('waiting messages describe real stages without a made-up countdown', () => {
  assert.match(friendlyLoadingStatus('Loading Python'), /kettle/);
  assert.match(friendlyLoadingStatus('Computing embeddings', '4 / 20'), /Steeping.*4 \/ 20/);
  assert.match(friendlyLoadingStatus('Loading embedding model'), /First-time/);
  assert.equal(friendlyLoadingStatus('Other stage', 'actual detail'), 'Other stage · actual detail');
});

test('startup identifies missing browser capabilities before downloading the worker', () => {
  const supported = { WebAssembly: { Suspending() {}, promising() {} }, navigator: { locks: {} } };
  assert.equal(browserSupportError(supported), null);
  assert.match(browserSupportError({}), /current desktop Chrome/);
  assert.ok(browserSupportError({ ...supported, navigator: {} }));
  assert.ok(browserSupportError({ ...supported, WebAssembly: { promising() {} } }));
});

class FakeWorker extends EventTarget {
  sent = [];
  postMessage(data) { this.sent.push(data); }
  respond(data) { this.dispatchEvent(new MessageEvent('message', { data })); }
}

function bridgedFrame() {
  const messages = [];
  const listeners = {};
  const parent = { postMessage: data => messages.push(data) };
  const events = [];
  const window = { addEventListener: (name, handler) => { listeners[name] = handler; }, dispatchEvent: event => events.push(event) };
  const document = { addEventListener() {}, getElementById() {} };
  const html = renderedDocument('<html><head></head></html>', 'token', '/run');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  runInNewContext(script, { parent, window, document, URL, URLSearchParams, FormData, Headers, Request, Response, EventTarget, Event, MessageEvent, CustomEvent });
  return { window, messages, events, deliver: data => listeners.message({ source: parent, data: { ...data, token: 'token' } }) };
}

test('live compute progress reaches the brew panel independently of stream polling', () => {
  const frame = bridgedFrame();
  frame.deliver({ type: 'compute-progress', compute: { completed: 4, total: 2340, message: 'Comparing research text' } });
  assert.equal(frame.events.length, 1);
  assert.equal(frame.events[0].type, 'dd-compute-progress');
  assert.equal(frame.events[0].detail.completed, 4);
});

test('frame fetch uses worker responses and blocks arbitrary external requests', async () => {
  const frame = bridgedFrame();
  const request = frame.window.fetch('/vote/7/2', { method: 'POST', headers: { 'X-CSRF-Token': 'csrf' }, body: '{}' });
  assert.equal(frame.messages[0].path, '/vote/7/2');
  assert.equal(frame.messages[0].headers['x-csrf-token'], 'csrf');
  frame.deliver({ id: 1, status: 200, headers: { 'content-type': 'application/json' }, body: '{"ok":true}' });
  assert.deepEqual(await (await request).json(), { ok: true });
  await assert.rejects(frame.window.fetch('https://example.org/secret'), /explicitly configured/);
  assert.equal(frame.messages.length, 1);
});

test('frame EventSource handles split SSE chunks and disconnects on close', () => {
  const frame = bridgedFrame();
  const stream = new frame.window.EventSource('/run/stream?run_id=123');
  const events = [];
  stream.onmessage = event => events.push(event.data);
  frame.deliver({ id: 1, type: 'stream', body: 'data: {"stage":' });
  frame.deliver({ id: 1, type: 'stream', body: '"ranking"}\n\n: heartbeat\n\ndata: done\n\n' });
  assert.deepEqual(events, ['{"stage":"ranking"}', 'done']);
  assert.equal(stream.readyState, frame.window.EventSource.OPEN);
  stream.close();
  assert.equal(stream.readyState, frame.window.EventSource.CLOSED);
  assert.equal(frame.messages.at(-1).type, 'cancel');
});

test('Pages hashes retain deep app routes without accepting another origin', () => {
  assert.equal(routeFromHash('#/saved?section=research'), '/saved?section=research');
  assert.equal(routeFromHash('#/run?resume=true'), '/run?resume=true');
  assert.equal(routeFromHash('#https://example.org'), '/');
  assert.equal(routeFromHash('#//example.org/path'), '/');
  assert.throws(() => localPath('https://example.org/run'), /local DailyDigest/);
  assert.equal(localPath('?date=2026-10-13', '/saved'), '/saved?date=2026-10-13');
});

test('trusted rendered pages route template navigation and suppress localhost service worker', () => {
  const source = '<html><head><link rel="manifest" href="/manifest.webmanifest"></head><body><script>window.location.href = `/done?digest_id=${id}`; window.location.reload(); window.history.replaceState(null,"",url); const base=window.location.href; navigator.serviceWorker.register("/sw.js").catch(() => {});</script></body></html>';
  const html = renderedDocument(source, 'test-token', '/run');
  assert.match(html, /window\.ddNavigate\(`\/done\?digest_id=\$\{id\}`\)/);
  assert.match(html, /window\.ddReload\(\)/);
  assert.match(html, /window\.ddReplaceState\(null/);
  assert.match(html, /const base=window\.ddLocation/);
  assert.doesNotMatch(html, /register\("\/sw\.js"\)/);
  assert.doesNotMatch(html, /href="\/manifest/);
  assert.match(html, /Content-Security-Policy/);
  assert.match(html, /source !== parent/);
});

test('worker responses correlate concurrent requests and deliver stream progress', async () => {
  const worker = new FakeWorker(); const client = new WorkerClient(worker); const chunks = [];
  const first = client.call({ type: 'request', path: '/run/stream' }, chunk => chunks.push(chunk));
  const second = client.call({ type: 'request', path: '/saved' });
  worker.respond({ id: 1, type: 'response-chunk', body: 'data: started\n\n' });
  worker.respond({ id: 2, type: 'response', status: 200, body: 'saved' });
  assert.equal((await second).body, 'saved');
  assert.deepEqual(chunks, ['data: started\n\n']);
  assert.equal(client.pending.size, 1);
  worker.respond({ id: 1, type: 'response', status: 200, body: '' });
  await first; assert.equal(client.pending.size, 0);
});

test('worker failure rejects outstanding calls and disconnect cancels the matching stream', async () => {
  const worker = new FakeWorker(); const client = new WorkerClient(worker);
  const first = client.call({ type: 'request', path: '/run/stream' });
  first.cancel();
  await assert.rejects(first, /disconnected/);
  assert.deepEqual(worker.sent[1], { id: 1, type: 'cancel-request' });
  const second = client.call({ type: 'request', path: '/saved' });
  worker.dispatchEvent(new Event('error'));
  await assert.rejects(second, /worker stopped/);
  assert.equal(client.pending.size, 0);
});

test('worker request errors are visible and release pending calls', async () => {
  const worker = new FakeWorker(); const client = new WorkerClient(worker);
  const request = client.call({ type: 'request', path: '/setup' });
  worker.respond({ id: 1, type: 'error', message: 'Profile write failed' });
  await assert.rejects(request, /Profile write failed/);
  assert.equal(client.pending.size, 0);
});
