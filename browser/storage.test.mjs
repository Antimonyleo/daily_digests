import test from 'node:test';
import assert from 'node:assert/strict';
import { createSnapshotWriter, validateRecoverySnapshot } from './storage.mjs';

test('read-only requests do not capture state; concurrent mutations save in order', async () => {
  let generation = 0, captures = 0;
  const saves = [], releases = [];
  const writer = createSnapshotWriter({ generation: () => generation,
    capture: () => { captures++; return generation; },
    save: value => { saves.push(value); return new Promise(resolve => releases.push(resolve)); } });
  assert.equal(await writer.flush(), true);
  assert.equal(captures, 0);
  generation = 1;
  const first = writer.flush();
  await Promise.resolve();
  generation = 2;
  assert.equal(writer.flush(), first);
  releases.shift()(); await Promise.resolve(); await Promise.resolve();
  assert.deepEqual(saves, [1, 2]);
  releases.shift()(); await first;
  assert.equal(writer.status().durable, true);
  await writer.flush(); assert.equal(captures, 2);
});

test('quota failure remains recoverable and warns without rejecting a valid request', async () => {
  let generation = 1, captures = 0, fail = true;
  const warnings = [];
  const writer = createSnapshotWriter({ generation: () => generation,
    capture: () => { captures++; return 'private-state'; },
    save: async () => { if (fail) throw new Error('QuotaExceededError'); }, warning: value => warnings.push(value) });
  assert.equal(await writer.flush(), false);
  assert.equal(writer.status().dirty, true);
  assert.equal(writer.status().saving, false);
  assert.match(warnings[0], /Export a backup/);
  await writer.flush(); assert.equal(captures, 1); // retries are coalesced/backed off
  fail = false; generation++;
  assert.equal(await writer.flush({ force: true }), true);
  assert.equal(writer.status().durable, true);
  assert.equal(warnings.at(-1), null);
});

test('capture failure does not leave a permanently pending checkpoint', async () => {
  let fail = true;
  const writer = createSnapshotWriter({ generation: () => 1,
    capture: () => { if (fail) throw new Error('capture failed'); return 'restored'; }, save: async () => {} });
  assert.equal(await writer.flush(), false);
  assert.equal(writer.status().saving, false);
  fail = false;
  assert.equal(await writer.flush({ force: true }), true);
});

test('startup recovery validates paths and encoded content without Python', () => {
  const payload = files => JSON.stringify({ format: 'dailydigest-browser-v1', files });
  assert.equal(validateRecoverySnapshot(payload({ 'data/profile.yaml': 'Zml4dHVyZQ==' })).format, 'dailydigest-browser-v1');
  for (const files of [{ '../private': '' }, { 'data/../bad': '' }, { 'data/bad': 'bad!' },
    { 'data/parent': '', 'data/parent/child': '' }, { '.env/key': '' }, { 'data/.compute.lock': '' }]) {
    assert.throws(() => validateRecoverySnapshot(payload(files)));
  }
});

test('a suspended SQL writer defers capture until its transaction commits', async () => {
  let committed = false, captures = 0, saved = null;
  const writer = createSnapshotWriter({generation: () => 1, ready: () => committed,
    capture: () => { captures++; return 'committed snapshot'; }, save: async value => { saved = value; }});
  assert.equal(await writer.flush({force: true}), false);
  assert.equal(captures, 0);
  assert.equal(writer.status().durable, false);
  committed = true;
  assert.equal(await writer.flush({force: true}), true);
  assert.equal(saved, 'committed snapshot');
});
