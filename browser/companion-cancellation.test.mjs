import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('./worker.mjs', import.meta.url), 'utf8');
const call = source.match(/async function companionCall\([\s\S]*?\n\}/)[0];

for (const phase of ['headers', 'body']) {
  test(`Stop aborts active companion ${phase} and releases its timer`, async () => {
    let cancelled = false, timer, cleared = false, aborted = false, signal;
    const context = vm.createContext({
      companion: { endpoint: 'http://localhost:9999', token: 'fixture' },
      state: { py: { globals: { get: () => () => cancelled } } },
      AbortController, AbortSignal, DOMException,
      setInterval: callback => { timer = callback; return 1; },
      clearInterval: () => { cleared = true; },
      setTimeout, clearTimeout,
      fetch: async (_url, options) => {
        signal = options.signal;
        const blocked = () => new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => { aborted = true; reject(signal.reason); });
        });
        if (phase === 'headers') return blocked();
        return { ok: true, text: blocked };
      },
    });
    vm.runInContext(call, context);
    const pending = context.companionCall('/summarize', '{}');
    const rejection = assert.rejects(pending, /stopped/i);
    await new Promise(resolve => setImmediate(resolve));
    cancelled = true;
    assert.equal(typeof timer, 'function', 'Active Companion request has no Stop monitor');
    timer();
    await rejection;
    assert.equal(aborted, true);
    assert.equal(cleared, true);
  });
}

test('Stop already requested prevents sending another companion summary', async () => {
  let calls = 0;
  const context = vm.createContext({
    companion: { endpoint: 'http://localhost:9999', token: 'fixture' },
    state: { py: { globals: { get: () => () => true } } },
    AbortController, AbortSignal, DOMException, setTimeout, clearTimeout, setInterval, clearInterval,
    fetch: async () => { calls++; return { ok: true, text: async () => '{}' }; },
  });
  vm.runInContext(call, context);
  await assert.rejects(context.companionCall('/text', '{}'), /stopped/i);
  assert.equal(calls, 0);
});
