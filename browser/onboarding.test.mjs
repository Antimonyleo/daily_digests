import test from 'node:test';
import assert from 'node:assert/strict';
import {companionStartCommand, browserSupportError} from './app.mjs';

test('scheduling requires an explicit command choice and a separate state directory', () => {
  const origin = 'https://reader.example';
  assert.equal(companionStartCommand(origin), 'uv run dd companion --allowed-origin https://reader.example');
  assert.equal(companionStartCommand(origin, true), 'uv run dd companion --allowed-origin https://reader.example --state-dir ./data/companion-state --schedule');
  assert.throws(() => companionStartCommand('https://reader.example/path'));
  assert.throws(() => companionStartCommand('file:///tmp'));
  assert.throws(() => companionStartCommand('https://x;whoami'));
  assert.throws(() => companionStartCommand('https://x$(whoami).example'));
});

test('unsupported devices receive an actionable fallback without claiming shortcut compatibility', () => {
  for (const environment of [{}, {WebAssembly:{Suspending(){}, promising(){}}}, {navigator:{locks:{}}}]) {
    assert.match(browserSupportError(environment), /current desktop Chrome/);
    assert.match(browserSupportError(environment), /shortcut does not add/);
  }
  assert.equal(browserSupportError({WebAssembly:{Suspending(){},promising(){}},navigator:{locks:{}}}), null);
});

test('preview commands opt in to local HTTP without allowing arbitrary insecure origins', () => {
  for (const origin of ['http://localhost:8766', 'http://127.0.0.1:8766', 'http://[::1]:8766']) {
    assert.match(companionStartCommand(origin), / --allow-local-preview$/);
  }
  assert.throws(() => companionStartCommand('http://reader.example'));
});
