import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createBrowserHttpBridge } from './http.mjs';

const input = (extra = {}) => JSON.stringify({ method: 'GET', url: 'https://example.test/feed',
  headers: [['Accept', '*/*']], timeoutMs: 1000, followRedirects: false, ...extra });

test('async HTTP bridge preserves bytes and uses explicit credential/redirect policy', async () => {
  let options;
  const bridge = createBrowserHttpBridge({ fetch: async (url, opts) => {
    options = opts;
    return new Response(Uint8Array.of(255, 0, 17), { headers: { 'X-Fixture': 'yes' }, status: 200 });
  } });
  const result = JSON.parse(await bridge.request(input({ method: 'POST', body: '/wAR' })));
  assert.equal(result.body, '/wAR');
  assert.equal(result.status, 200);
  assert.equal(options.credentials, 'omit');
  assert.equal(options.redirect, 'error');
  assert.deepEqual([...options.body], [255, 0, 17]);
});

test('Stop aborts a pending network request promptly', async () => {
  let cancel = false;
  const bridge = createBrowserHttpBridge({ shouldCancel: () => cancel, pollMs: 5,
    fetch: (url, { signal }) => new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => reject(new Error('aborted')));
    }) });
  const pending = bridge.request(input());
  setTimeout(() => { cancel = true; }, 10);
  await assert.rejects(pending, /cancelled/);
});

test('read timeout aborts fetch', async () => {
  const bridge = createBrowserHttpBridge({ fetch: (url, { signal }) => new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => reject(new Error('aborted')));
  }) });
  await assert.rejects(bridge.request(input({ timeoutMs: 5 })), /TimeoutError/);
});

test('repeated network failure is skipped per URL, expires, and does not block other URLs', async () => {
  let attempts = 0, time = 100;
  const bridge = createBrowserHttpBridge({ now: () => time, fetch: async () => {
    attempts++; throw new TypeError('Failed to fetch');
  } });
  await assert.rejects(bridge.request(input()), /Failed to fetch/);
  await assert.rejects(bridge.request(input()), /skipping repeated/);
  assert.equal(attempts, 1);
  await assert.rejects(bridge.request(input({ url: 'https://example.test/working-other-path' })));
  assert.equal(attempts, 2);
  time += 300001;
  await assert.rejects(bridge.request(input()), /Failed to fetch/);
  assert.equal(attempts, 3);
});

test('HTTP errors are returned and do not poison the URL failure cache', async () => {
  let attempts = 0;
  const bridge = createBrowserHttpBridge({ fetch: async () => { attempts++; return new Response('busy', { status: 503 }); } });
  assert.equal(JSON.parse(await bridge.request(input())).status, 503);
  assert.equal(JSON.parse(await bridge.request(input())).status, 503);
  assert.equal(attempts, 2);
});

test('retry sleep yields and can be cancelled', async () => {
  let cancel = false;
  const bridge = createBrowserHttpBridge({ shouldCancel: () => cancel, pollMs: 5 });
  const pending = bridge.wait(1000);
  setTimeout(() => { cancel = true; }, 10);
  await assert.rejects(pending, /cancelled/);
});
