import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {test} from 'node:test';

const worker = await readFile(new URL('./worker.mjs', import.meta.url), 'utf8');
const source = worker.slice(worker.indexOf('function preloadStartupBytecode('),
  worker.indexOf('async function loadPythonPackages'));

function fixture(fetch, options = {}) {
  const context = vm.createContext({fetch, AbortController, options});
  vm.runInContext(source, context);
  return vm.runInContext('preloadStartupBytecode(options)', context);
}

test('a measured slow connection does not spend bandwidth on optional bytecode', () => {
  const loader = fixture(() => assert.fail('Optional cache competes with required downloads'),
    {applicationBytes: 2_700_000, applicationMs: 2200});
  loader.install({unpackArchive() {assert.fail('No bytecode was requested');}});
  loader.cancel();
});

test('ready bytecode is installed, and invalid archives fall back to source', async () => {
  const bytes = new Uint8Array([1, 2, 3]).buffer;
  const loader = fixture(async () => ({ok: true, arrayBuffer: async () => bytes}),
    {applicationBytes: 2_700_000, applicationMs: 50});
  await new Promise(resolve => setImmediate(resolve));
  let count = 0;
  loader.install({unpackArchive(data, type, options) {
    assert.equal(data, bytes); assert.equal(type, 'zip'); assert.equal(options.extractDir, '/');
    count++; throw new Error('Invalid archive');
  }});
  assert.equal(count, 1);
});

test('a stalled optional download cannot stall Python startup', async () => {
  let signal;
  const loader = fixture((_url, options) => {
    signal = options.signal;
    return new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(Error('aborted'))));
  });
  loader.install({unpackArchive() { assert.fail('No bytes have arrived'); }});
  assert.equal(signal.aborted, true);
  await new Promise(resolve => setImmediate(resolve));
});

test('failed startup can cancel prefetch before installing Python', () => {
  let signal;
  const loader = fixture((_url, options) => { signal = options.signal; return new Promise(() => {}); });
  loader.cancel();
  assert.equal(signal.aborted, true);
});
