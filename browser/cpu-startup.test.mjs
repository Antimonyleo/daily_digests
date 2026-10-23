import assert from 'node:assert/strict';
import {test} from 'node:test';
import {defaultComputeMode, prepareDefaultCPU} from './cpu-startup.mjs';

const storage = () => {
  const values = new Map();
  return {getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value)};
};
function fixture() {
  const worker = Object.assign(new EventTarget(), {state: 'activated'});
  const serviceWorker = Object.assign(new EventTarget(), {controller: {scriptURL: 'https://fixture.test/sw.js?isolate=1'}});
  return {navigator: {hardwareConcurrency: 8, deviceMemory: 8, serviceWorker}, storage: storage(), session: storage(),
    key: 'cpu', isolated: false, secure: true, readChoice: async () => undefined,
    register: async () => ({active: worker}), timeoutMs: 20};
}

test('CPU defaults use available cores while preserving every explicit choice', () => {
  assert.equal(defaultComputeMode(undefined, {multicore: true, threads: 4}), 'multicore');
  assert.equal(defaultComputeMode(undefined, {multicore: false}), 'cpu');
  assert.equal(defaultComputeMode(undefined, {multicore: true, threads: 2}), 'cpu');
  assert.equal(defaultComputeMode(undefined, {multicore: true, threads: 4}, 2), 'cpu');
  for (const saved of ['cpu', 'gpu', 'multicore']) {
    assert.equal(defaultComputeMode(saved, {multicore: true}), saved);
    assert.equal(defaultComputeMode(saved, {multicore: false}), saved);
  }
});

test('new desktop enables isolation with at most one preparation reload', async () => {
  const options = fixture();
  assert.deepEqual(await prepareDefaultCPU(options), {isolate: true, reload: true});
  assert.equal(options.storage.getItem('cpu'), 'true');
  assert.deepEqual(await prepareDefaultCPU(options), {isolate: true, reload: false});
});

test('explicit compute choices and low-memory or unavailable environments stay unchanged', async () => {
  for (const change of [o => {o.readChoice = async () => 'cpu';}, o => {o.readChoice = async () => 'gpu';},
    o => {o.storage.setItem('cpu', 'false');}, o => {o.navigator.deviceMemory = 2;},
    o => {o.navigator.hardwareConcurrency = 2;}, o => {o.navigator.hardwareConcurrency = undefined;},
    o => {o.secure = false;}, o => {o.navigator.serviceWorker = null;}]) {
    const options = fixture(); change(options);
    options.register = () => assert.fail('An explicit or constrained device must not be changed');
    assert.deepEqual(await prepareDefaultCPU(options), {isolate: false, reload: false});
  }
});

test('blocked storage, registration and activation cannot prevent startup or cause reload loops', async () => {
  for (const change of [o => {o.readChoice = async () => {throw Error('storage blocked');};},
    o => {o.register = async () => {throw Error('service workers blocked');};},
    o => {o.register = () => new Promise(() => {});},
    o => {o.register = async () => ({installing: Object.assign(new EventTarget(), {state: 'installing'})});}]) {
    const options = fixture(); change(options);
    assert.equal((await prepareDefaultCPU(options)).reload, false);
    assert.equal((await prepareDefaultCPU(options)).reload, false);
  }
});
