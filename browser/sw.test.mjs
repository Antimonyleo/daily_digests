import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { runInNewContext } from 'node:vm';

const source = await readFile(new URL('./sw.js', import.meta.url), 'utf8');

function harness(isolate) {
  const handlers = {};
  const self = { location: { href: `https://example.test/daily_digests/sw.js${isolate ? '?isolate=1' : ''}` },
    registration: { scope: 'https://example.test/daily_digests/' },
    addEventListener: (event, handler) => { handlers[event] = handler; } };
  runInNewContext(source, { self, URL, Response, Headers,
    fetch: async () => new Response('body', { headers: { 'content-type': 'text/javascript' } }) });
  return async url => {
    let response;
    handlers.fetch({ request: new Request(url), respondWith: promise => { response = promise; } });
    return response ? await response : null;
  };
}

test('isolation is opt-in and scoped to this Pages project', async () => {
  const fetch = harness(true);
  const response = await fetch('https://example.test/daily_digests/worker.mjs');
  assert.equal(response.headers.get('cross-origin-opener-policy'), 'same-origin');
  assert.equal(response.headers.get('cross-origin-embedder-policy'), 'require-corp');
  assert.equal(await response.text(), 'body');
  assert.equal(await fetch('https://example.test/another-project/'), null);
  assert.equal(await fetch('https://source.example/feed'), null);
  assert.equal(await harness(false)('https://example.test/daily_digests/worker.mjs'), null);
});
