import test from 'node:test';
import assert from 'node:assert/strict';
import { computeCapabilities, computeSelection, createEmbeddingEngine, formatComputeProgress, guardGPUModel } from './compute.mjs';
import { withEmbeddingMemo } from './embedding-memo.mjs';
import { prepareBertBatch } from '../prototypes/browser-compute/bge-tokenizer.mjs';

test('compute choices depend on browser capabilities across operating systems', () => {
  const caps = computeCapabilities({ navigator: { gpu: {}, hardwareConcurrency: 12 }, crossOriginIsolated: true, SharedArrayBuffer });
  assert.deepEqual(caps, { gpu: true, multicore: true, threads: 4 });
  assert.equal(computeCapabilities({ navigator: { hardwareConcurrency: 12 } }).multicore, false);
  assert.equal(computeCapabilities({ navigator: {} }).gpu, false);
  assert.deepEqual(computeSelection('gpu', caps), { device: 'webgpu', threads: 1, fallback: '' });
  assert.equal(computeSelection('multicore', caps).threads, 4);
  assert.equal(computeSelection('gpu', { gpu: false }).device, 'wasm');
  assert.throws(() => computeSelection('invalid', caps), /Unknown compute/);
});

test('GPU loss rejects a hung inference and skips broken-session disposal', async () => {
  let loseDevice;
  const model = () => new Promise(() => {});
  model.dispose = () => { throw new Error('Broken disposal must not block recovery'); };
  const guarded = guardGPUModel(model, { lost: new Promise(resolve => { loseDevice = resolve; }) });
  const pending = guarded({});
  loseDevice({ reason: 'destroyed' });
  await assert.rejects(pending, /GPU device lost/);
  await guarded.dispose();
  await assert.rejects(guarded({}), /GPU device lost/);
});

function harness(mode = 'cpu', overrides = {}) {
  const progress = [], loads = [], outputs = [];
  let clock = 0;
  const options = {
    mode, capabilities: { gpu: true, multicore: true, threads: 4 }, env: { wasm: {} }, prepareBatch: prepareBertBatch,
    progress: row => progress.push(row), now: () => clock,
    loadTokenizer: async () => async texts => ({ input_ids: texts.map(text => [101, text.length, 102]), attention_mask: texts.map(() => [1, 1, 1]) }),
    loadModel: async ({ device, dtype }) => {
      loads.push({ device, dtype });
      return async inputs => {
        clock += 2000;
        const rows = inputs.input_ids.dims[0], length = inputs.input_ids.dims[1];
        const data = new Float32Array(rows * length * 384);
        for (let row = 0; row < rows; row++) data[row * length * 384 + Number(inputs.input_ids.data[row * length + 1])] = 1;
        const tensor = { dims: [rows, length, 384], data, disposed: false, dispose() { this.disposed = true; } };
        outputs.push(tensor);
        return { last_hidden_state: tensor };
      };
    },
    makeTensor: (data, dims) => ({ data, dims, dispose() {} }),
    ...overrides,
  };
  return { engine: withEmbeddingMemo(createEmbeddingEngine(options), {shouldCancel: options.shouldCancel, yieldTask: options.yieldTask}), options, progress, loads, outputs };
}

test('counts report completed batches, preserve input order, and reuse the loaded model', async () => {
  const h = harness();
  const vectors = await h.engine.embed(['longest text', 'x', 'medium', 'four', 'xx']);
  assert.deepEqual(h.progress.filter(row => row.phase === 'encoding').map(row => row.completed), [0, 4, 5]);
  assert.equal(h.progress.find(row => row.completed === 4).elapsed_ms, 2000);
  assert.equal(vectors[0][12], 1);
  assert.equal(vectors[1][1], 1);
  assert.ok(h.outputs.every(row => row.disposed));
  await h.engine.embed(['x']);
  assert.equal(h.loads.length, 1);
});

test('GPU is explicitly selected with full precision and falls back when loading fails', async () => {
  const baseline = harness();
  const h = harness('gpu', { loadModel: async config => {
    if (config.device === 'webgpu') throw new Error('No GPU adapter');
    return baseline.options.loadModel(config);
  } });
  await h.engine.embed(['x']);
  assert.equal(h.engine.device, 'wasm');
  assert.ok(h.progress.some(row => row.phase === 'fallback'));
  assert.deepEqual(baseline.loads, [{ device: 'wasm', dtype: 'fp32' }]);
});

test('an inference failure restarts the entire embedding call on CPU', async () => {
  const baseline = harness();
  const h = harness('gpu', { loadModel: async config => config.device === 'webgpu'
    ? async () => { throw new Error('GPU device lost'); }
    : baseline.options.loadModel(config) });
  const result = await h.engine.embed(['abc', 'x']);
  assert.equal(result[0][3], 1);
  assert.equal(h.engine.device, 'wasm');
  assert.equal(h.engine.stats.cpuFallbacks, 1);
  assert.equal(h.engine.stats.fallbackReason, 'device-lost');
  assert.match(h.progress.find(row => row.phase === 'fallback').message, /device was lost/);
});

test('a broken GPU session can fail during disposal without preventing CPU fallback', async () => {
  const baseline = harness();
  const failed = async () => { throw new Error('GPU device lost'); };
  failed.dispose = async () => { throw new Error('GPU cleanup failed'); };
  const h = harness('gpu', { loadModel: async config => config.device === 'webgpu' ? failed : baseline.options.loadModel(config) });
  const result = await h.engine.embed(['x']);
  assert.equal(result[0][1], 1);
  assert.equal(h.engine.device, 'wasm');
});

test('fallback preserves the export after returning profile vectors on GPU', async () => {
  const baseline = harness(), configs = [];
  let calls = 0;
  const h = harness('gpu', { loadModel: async config => {
    configs.push(config);
    const run = await baseline.options.loadModel(config);
    return async inputs => {
      if (config.device === 'webgpu' && ++calls > 1) throw new Error('GPU lost after profile encoding');
      return run(inputs);
    };
  } });
  await h.engine.embed(['profile']);
  const signature = h.engine.cacheBackend;
  assert.notEqual(signature, baseline.engine.cacheBackend);
  await h.engine.embed(['document']);
  assert.equal(h.engine.device, 'wasm');
  assert.equal(h.engine.cacheBackend, signature);
  assert.equal(configs.at(-1).exportKind, 'gpu');
});

test('cached vectors pin the export even before the first inference succeeds', async () => {
  const baseline = harness(), configs = [];
  const h = harness('gpu', { loadModel: async config => {
    configs.push(config);
    if (config.device === 'webgpu') throw new Error('GPU initialization failed');
    return baseline.options.loadModel(config);
  } });
  const cachedSignature = h.engine.cacheBackend;
  await h.engine.embed(['new profile']);
  assert.equal(h.engine.cacheBackend, cachedSignature);
  assert.equal(configs.at(-1).exportKind, 'gpu');
});

test('an unavailable GPU uses the original CPU export from the start', async () => {
  const h = harness('gpu', { capabilities: { gpu: false, multicore: false, threads: 1 } });
  await h.engine.embed(['profile']);
  assert.equal(h.engine.cacheBackend, harness().engine.cacheBackend);
  assert.deepEqual(h.loads, [{ device: 'wasm', dtype: 'fp32' }]);
});

test('CPU failures are reported rather than inventing successful vectors', async () => {
  const h = harness('cpu', { loadModel: async () => async () => { throw new Error('Inference failed'); } });
  await assert.rejects(h.engine.embed(['x']), /Inference failed/);
});

test('a cancelled long call stops before starting the next batch', async () => {
  let cancelled = false;
  const h = harness('cpu', { shouldCancel: () => cancelled, progress: row => { if (row.completed === 4) setTimeout(() => { cancelled = true; }, 0); } });
  await assert.rejects(h.engine.embed(['a','b','c','d','e','f']), /stopped before publication/);
  assert.equal(h.outputs.length, 1);
  assert.equal(h.engine.stats.cpuFallbacks, 0);
});

test('multicore requires isolation and never silently claims multiple cores', async () => {
  const supported = harness('multicore');
  await supported.engine.embed(['x']);
  assert.equal(supported.options.env.wasm.numThreads, 4);
  const unsupported = harness('multicore', { capabilities: { multicore: false, gpu: false, threads: 1 } });
  await unsupported.engine.embed(['x']);
  assert.equal(unsupported.options.env.wasm.numThreads, 1);
  assert.ok(unsupported.progress.some(row => row.phase === 'fallback'));
});

test('visible progress separates downloading, processing and completion with approximate timing', () => {
  assert.match(formatComputeProgress({ phase: 'loading', file: 'model.onnx', percent: 42 }), /model.onnx.*42%/);
  const message = formatComputeProgress({ phase: 'encoding', completed: 800, total: 2340, elapsed_ms: 400000, remaining_ms: 770000, backend: 'GPU' });
  assert.match(message, /800 \/ 2340 completed/);
  assert.match(message, /6m 40s elapsed/);
  assert.match(message, /about 13m remaining/);
  assert.match(formatComputeProgress({ phase: 'complete', completed: 2340, total: 2340, elapsed_ms: 1200 }), /completed/);
});

test('repeat profile and negative-interest text reuses exact vectors without inference', async () => {
  const h = harness();
  const original = await h.engine.embed(['query: protein design', 'query: avoid clinical trials']);
  const calls = h.outputs.length;
  const repeated = await h.engine.embed(['query: avoid clinical trials', 'query: protein design']);
  assert.equal(h.outputs.length, calls, 'unchanged interests must not run the model again');
  assert.deepEqual(repeated, [original[1], original[0]]);
  repeated[0][0] = 42;
  assert.deepEqual(await h.engine.embed(['query: avoid clinical trials']), [original[1]], 'callers cannot mutate cached vectors');
});

test('edited text and query/document prefixes remain distinct while duplicates share work', async () => {
  const h = harness();
  await h.engine.embed(['query: protein', 'protein', 'query: protein']);
  assert.equal(h.engine.stats.completedInputs, 2);
  const before = h.engine.stats.completedInputs;
  await h.engine.embed(['query: protein', 'query: protein design']);
  assert.equal(h.engine.stats.completedInputs - before, 1);
});

test('cached-only and empty requests respect cancellation without loading a model', async () => {
  let cancelled = false;
  const h = harness('cpu', {shouldCancel: () => cancelled});
  assert.deepEqual(await h.engine.embed([]), []);
  assert.equal(h.loads.length, 0);
  await h.engine.embed(['cached']);
  cancelled = true;
  await assert.rejects(h.engine.embed(['cached']), /stopped before publication/);
});

test('memo is bounded and a failed call cannot supply partial cached vectors', async () => {
  const h = harness();
  await h.engine.embed(Array.from({length:257}, (_,i) => `text-${i}`));
  const count = h.engine.stats.completedInputs;
  await h.engine.embed(['text-0']);
  assert.equal(h.engine.stats.completedInputs, count + 1, 'least recent entry is evicted');
  let fail = true;
  const base = harness();
  const failed = harness('cpu', {loadModel: async config => {
    const run = await base.options.loadModel(config);let batches=0;
    return async inputs => {if (fail && ++batches === 2) throw new Error('Inference failed');return run(inputs);};
  }});
  await assert.rejects(failed.engine.embed(['a','bb','ccc','dddd','eeeee']), /Inference failed/);
  fail = false;
  const outputs = base.outputs.length;
  await failed.engine.embed(['a']);
  assert.equal(base.outputs.length, outputs + 1);
});

test('successful memo hits remain cancellable at the worker event-loop boundary', async () => {
  let cancelled = false, cancelOnYield = false;
  const h = harness('cpu', {shouldCancel: () => cancelled, yieldTask: async () => {if (cancelOnYield) cancelled = true;}});
  await h.engine.embed(['profile']);cancelOnYield = true;
  await assert.rejects(h.engine.embed(['profile']), /stopped before publication/);
});

test('large text does not remain in the bounded memo and model instances do not share it', async () => {
  const text = 'x'.repeat(8193);
  // The fake model uses string length as its feature index; use a bounded output
  // for this retention test while still passing the real large input to embed.
  const loadTokenizer = async () => async texts => ({input_ids:texts.map(()=>[101,1,102]),attention_mask:texts.map(()=>[1,1,1])});
  const large = harness('cpu', {loadTokenizer});
  await large.engine.embed([text]);await large.engine.embed([text]);
  assert.equal(large.engine.stats.completedInputs, 2);
  const first = harness(), second = harness('gpu');
  await first.engine.embed(['profile']);await second.engine.embed(['profile']);
  assert.equal(second.engine.stats.completedInputs, 1);
  assert.notEqual(first.engine.cacheBackend, second.engine.cacheBackend);
});
