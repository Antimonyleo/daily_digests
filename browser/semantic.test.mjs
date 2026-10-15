import test from 'node:test';
import assert from 'node:assert/strict';
import { prepareSemanticRecords, createSemanticEngine } from './semantic.mjs';

const tokenizer = { cls_token_id: 101, sep_token_id: 102, pad_token_id: 0,
  encode: text => text.trim() ? text.trim().split(/\s+/).map(Number) : [] };
const tokens = (length, offset = 1000) => Array.from({ length }, (_, i) => i + offset).join(' ');
const record = fields => ({ query: '201 202', title: '301', abstract: '401 402', ...fields });

test('BERT windows preserve the complete query, title, article tail, and segment IDs', () => {
  const prepared = prepareSemanticRecords(tokenizer, [record({ abstract: tokens(900) })]);
  assert.deepEqual(prepared.records, [{ windows: 2, complete: true }]);
  assert.deepEqual(prepared.windows[0].input_ids.slice(0, 6), [101, 201, 202, 102, 301, 1000]);
  assert.equal(prepared.windows[0].input_ids.length, 512);
  assert.equal(prepared.windows.at(-1).input_ids.at(-2), 1899);
  assert.equal(prepared.windows.at(-1).input_ids.at(-1), 102);
  assert.deepEqual(prepared.windows[0].token_type_ids.slice(0, 6), [0, 0, 0, 0, 1, 1]);
  assert.ok(prepared.windows.every(row => row.attention_mask.every(value => value === 1)));
});

test('bounded windows select first, middle, last and disclose incomplete coverage', () => {
  const prepared = prepareSemanticRecords(tokenizer, [record({ abstract: tokens(2500) })]);
  assert.deepEqual(prepared.records, [{ windows: 3, complete: false }]);
  assert.deepEqual(prepared.windows.map(row => row.input_ids[5]), [1000, 2422, 3370]);
  assert.equal(prepared.windows.at(-1).input_ids.at(-2), 3499);
});

test('long queries are bounded explicitly; a clipped title is reported; empty abstracts retain the title', () => {
  const bounded = prepareSemanticRecords(tokenizer, [record({ query: tokens(300), topic: '501 502' })]);
  assert.deepEqual(bounded.records, [{ windows: 1, complete: false, query_truncated: true }]);
  assert.deepEqual(bounded.windows[0].input_ids.slice(1, 5), [501, 502, 1000, 1001]);
  assert.equal(bounded.windows[0].input_ids[193], 102);
  const fallback = prepareSemanticRecords(tokenizer, [record({ query: tokens(193) })]);
  assert.equal(fallback.records[0].query_truncated, true);
  assert.deepEqual(fallback.windows[0].input_ids.slice(1, 193), Array.from({ length: 192 }, (_, i) => i + 1000));
  const prepared = prepareSemanticRecords(tokenizer, [record({ title: tokens(140), abstract: '' })]);
  assert.deepEqual(prepared.records, [{ windows: 1, complete: false }]);
  assert.equal(prepared.windows[0].input_ids.at(-2), 1127);
  assert.throws(() => prepareSemanticRecords(tokenizer, [record({ abstract: 42 })]), /strings/);
  assert.throws(() => prepareSemanticRecords(tokenizer, [record({ query: ' ' })]), /query/);
});

test('a long topic leaves half the query budget for explicit description', () => {
  const prepared = prepareSemanticRecords(tokenizer, [record({ query: tokens(300), topic: tokens(120, 2000) })]);
  const query = prepared.windows[0].input_ids.slice(1, 193);
  assert.deepEqual(query.slice(0, 96), Array.from({ length: 96 }, (_, i) => i + 2000));
  assert.deepEqual(query.slice(96), Array.from({ length: 96 }, (_, i) => i + 1000));
  assert.deepEqual(prepared.records, [{ windows: 1, complete: false, query_truncated: true }]);
});

function harness(overrides = {}) {
  const loads = [], inputs = [], outputs = [], progress = [];
  const engine = createSemanticEngine({
    loadTokenizer: async () => { loads.push('tokenizer'); return tokenizer; },
    loadModel: async options => { loads.push(options); return async feed => {
      inputs.push(feed);
      const size = feed.input_ids.dims[0];
      const tensor = { dims: [size, 1], data: Float32Array.from({ length: size }, (_, i) => inputs.length * 10 + i),
        dispose() { this.disposed = true; } };
      outputs.push(tensor);
      return { logits: tensor };
    }; },
    makeTensor: (data, dims) => ({ data, dims, dispose() { this.disposed = true; } }),
    progress: event => progress.push(event), yieldTask: async () => {}, ...overrides,
  });
  return { engine, loads, inputs, outputs, progress };
}

test('lazy CPU scoring batches four windows, pads, aggregates maximum, and preserves record order', async () => {
  const h = harness();
  assert.deepEqual(await h.engine.score([]), []);
  assert.equal(h.loads.length, 0);
  const result = await h.engine.score([record({ abstract: tokens(900) }), record(), record(), record()]);
  assert.deepEqual(result, [{ logit: 11, windows: 2, complete: true }, { logit: 12, windows: 1, complete: true },
    { logit: 13, windows: 1, complete: true }, { logit: 20, windows: 1, complete: true }]);
  assert.deepEqual(h.loads, ['tokenizer', { device: 'wasm', dtype: 'fp32' }]);
  assert.deepEqual(h.inputs.map(feed => feed.input_ids.dims[0]), [4, 1]);
  assert.equal(h.inputs[0].attention_mask.data[512 * 3 - 1], 0n);
  assert.ok(h.outputs.every(output => output.disposed));
  assert.ok(h.inputs.every(feed => Object.values(feed).every(tensor => tensor.disposed)));
  await h.engine.score([record()]);
  assert.equal(h.loads.length, 2);
});

test('invalid scalar output and model failures reject the complete call', async () => {
  for (const logits of [{ dims: [1, 2], data: [1, 2] }, { dims: [1, 1], data: [NaN] }, { dims: [1, 1], data: [Infinity] }]) {
    const h = harness({ loadModel: async () => async () => ({ logits }) });
    await assert.rejects(h.engine.score([record()]), /logit/);
  }
  const h = harness({ loadModel: async () => async () => { throw new Error('inference failed'); } });
  await assert.rejects(h.engine.score([record()]), /inference failed/);
});

test('cancellation after a batch rejects without returning partial scores', async () => {
  let cancelled = false;
  const h = harness({ shouldCancel: () => cancelled, yieldTask: async () => { cancelled = true; } });
  await assert.rejects(h.engine.score(Array.from({ length: 5 }, () => record())), /stopped/);
  assert.equal(h.inputs.length, 1);
  assert.equal(h.progress.some(event => event.phase === 'complete'), false);
});

test('three overlapping windows cover the abstract and the maximum query remains intact', () => {
  const query = tokens(192, 2000);
  const prepared = prepareSemanticRecords(tokenizer, [record({ query, title: tokens(128, 3000), abstract: tokens(500, 4000) })]);
  assert.deepEqual(prepared.records, [{ windows: 3, complete: true }]);
  for (const window of prepared.windows) {
    assert.deepEqual(window.input_ids.slice(1, 193), query.split(' ').map(Number));
    assert.equal(window.input_ids[193], 102);
    assert.equal(window.input_ids.at(-1), 102);
    assert.ok(window.input_ids.length <= 512);
  }
  assert.equal(prepared.windows.at(-1).input_ids.at(-2), 4499);
});

test('assets failures propagate and a later call can retry loading', async () => {
  let attempts = 0;
  const h = harness({ loadTokenizer: async () => {
    if (++attempts === 1) throw new Error('Missing local tokenizer');
    return tokenizer;
  } });
  await assert.rejects(h.engine.score([record()]), /Missing local tokenizer/);
  assert.equal(h.inputs.length, 0);
  assert.equal((await h.engine.score([record()]))[0].logit, 10);
});
