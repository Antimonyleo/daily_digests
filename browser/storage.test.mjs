import test from 'node:test';
import assert from 'node:assert/strict';
import { createSnapshotWriter, validateRecoverySnapshot, validateStoredSnapshot, capturePrivateSnapshot } from './storage.mjs';

test('browser checkpoint validates private paths before reading any file', () => {
  let reads = 0;
  for (const names of [['data/good', '../escape'], ['.env/child'], ['data/a', 'data/a/child'], ['data/.compute.lock']]) {
    assert.throws(() => capturePrivateSnapshot(names, () => { reads++; return new Uint8Array(); }));
  }
  assert.equal(reads, 0);
});

test('checkpoint fallback encodes large binary files without overflowing the call stack', () => {
  const descriptor = Object.getOwnPropertyDescriptor(Uint8Array.prototype, 'toBase64');
  Object.defineProperty(Uint8Array.prototype, 'toBase64', {value:undefined, configurable:true});
  try {
    const bytes = Uint8Array.from({length:4*1024*1024}, (_, index) => index % 256);
    const result = capturePrivateSnapshot(['data/vectors.bin'], () => bytes);
    assert.equal(result.warning, null);
    assert.deepEqual(Buffer.from(JSON.parse(result.payload).files['data/vectors.bin'], 'base64'), Buffer.from(bytes));
  } finally {
    if (descriptor) Object.defineProperty(Uint8Array.prototype, 'toBase64', descriptor);
    else delete Uint8Array.prototype.toBase64;
  }
});

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

test('trusted stored state above import limits restores with a warning and stays exportable', async () => {
  const { validateStoredSnapshot } = await import('./storage.mjs');
  const files = Object.fromEntries(Array.from({ length: 10001 }, (_, i) => [`data/f${i}`, '']));
  const payload = JSON.stringify({ format: 'dailydigest-browser-v1', files });
  assert.throws(() => validateRecoverySnapshot(payload), /recovery limit/);
  const trusted = validateStoredSnapshot(payload);
  assert.equal(Object.keys(trusted.snapshot.files).length, 10001);
  assert.match(trusted.warning, /preserved.*import/i);
  assert.equal(JSON.stringify(trusted.snapshot), payload);
  assert.equal(validateStoredSnapshot(JSON.stringify({ format: 'dailydigest-browser-v1', files: {} })).warning, null);
  assert.throws(() => validateStoredSnapshot(JSON.stringify({ format: 'dailydigest-browser-v1', files: { 'data/../escape': '' } })), /invalid/);
});


for (const [kind, validate] of [['import', payload => validateRecoverySnapshot(payload)],
  ['stored startup', payload => validateStoredSnapshot(payload).snapshot]]) {
  test(`multi-megabyte ${kind} snapshots retain every database byte without stack overflow`, () => {
    const bytes = Buffer.alloc(4 * 1024 * 1024);
    for (let i = 0; i < bytes.length; i++) bytes[i] = i % 256;
    bytes.write('SQLite format 3\0', 0, 'ascii');
    const encoded = bytes.toString('base64');
    const payload = JSON.stringify({ format: 'dailydigest-browser-v1',
      files: { 'data/digest.db': encoded, 'data/profile.yaml': 'Zml4dHVyZQ==' } });
    const snapshot = validate(payload);
    assert.equal(snapshot.files['data/digest.db'], encoded);
    assert.ok(Buffer.from(snapshot.files['data/digest.db'], 'base64').equals(bytes));
    assert.equal(snapshot.files['data/profile.yaml'], 'Zml4dHVyZQ==');
  });
}

test('base64 alphabet and terminal padding rules hold for imported and stored snapshots', () => {
  const payload = encoded => JSON.stringify({ format: 'dailydigest-browser-v1', files: { 'data/file': encoded } });
  for (const validate of [validateRecoverySnapshot, value => validateStoredSnapshot(value).snapshot]) {
    for (const encoded of ['', 'AA==', 'AAA=', 'AAAA', 'AB==', 'AAB=', 'AA+/', '////', '++++']) {
      assert.equal(validate(payload(encoded)).files['data/file'], encoded);
    }
    for (const encoded of ['A', 'AA', 'AAA', 'AAAAA', '====', 'A===', '=AAA', 'AA=A',
      'A=AA', 'AA==AAAA', 'AAA=AAAA', 'AAAA\n', 'AAA ', 'AA-=', 'AA_=', 'éAAA', '💡AA']) {
      assert.throws(() => validate(payload(encoded)), /invalid encoded data/, JSON.stringify(encoded));
    }
    const largeInvalid = 'A'.repeat(4 * 1024 * 1024 - 1) + '!';
    assert.throws(() => validate(payload(largeInvalid)), /invalid encoded data/);
  }
});

test('linear validation preserves conflicting path and import count protections', () => {
  const payload = files => JSON.stringify({ format: 'dailydigest-browser-v1', files });
  const files = { 'data/parent': '', 'data/parent-child': '', 'data/parent/child': '' };
  assert.throws(() => validateRecoverySnapshot(payload(files)), /conflicting paths/);
  assert.throws(() => validateStoredSnapshot(payload(files)), /conflicting paths/);
  assert.equal(Object.keys(validateRecoverySnapshot(payload({ 'data/parent': '', 'data/parent-child': '' })).files).length, 2);
  const many = Object.fromEntries(Array.from({ length: 10001 }, (_, i) => [`data/file${i}`, 'AA==']));
  assert.throws(() => validateRecoverySnapshot(payload(many)), /recovery limit/);
  assert.equal(Object.keys(validateStoredSnapshot(payload(many)).snapshot.files).length, 10001);
});

test('internal settings recovery journals survive checkpoints and remain forbidden in portable imports', () => {
  const bytes = new TextEncoder().encode('{"private":"pending credential bytes"}');
  const {payload} = capturePrivateSnapshot(['.settings-transaction.json'], () => bytes);
  const {snapshot} = validateStoredSnapshot(payload);
  assert.equal(snapshot.files['.settings-transaction.json'], Buffer.from(bytes).toString('base64'));
  assert.throws(() => validateRecoverySnapshot(payload), /invalid private path/);
  assert.throws(() => capturePrivateSnapshot(['.settings-transaction.json/escape'], () => bytes), /invalid private path/);
});

test('a backup from a newer release asks for a reload instead of a vague format error', async () => {
  const { SNAPSHOT_FORMAT, READABLE_SNAPSHOT_FORMATS } = await import('./storage.mjs');
  assert.ok(READABLE_SNAPSHOT_FORMATS.has(SNAPSHOT_FORMAT));
  assert.throws(() => validateRecoverySnapshot(JSON.stringify({ format: 'dailydigest-browser-v9', files: {} })), /newer version of DailyDigest.*Reload/);
  assert.throws(() => validateStoredSnapshot(JSON.stringify({ format: 'dailydigest-browser-v9', files: {} })), /newer version/);
  assert.throws(() => validateRecoverySnapshot(JSON.stringify({ format: 'other', files: {} })), /Unsupported DailyDigest backup format/);
});
