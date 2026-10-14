import { AutoTokenizer, AutoModel, Tensor, env } from './node_modules/@huggingface/transformers/dist/transformers.min.js';
import { loadPyodide } from './node_modules/pyodide/pyodide.mjs';
import { compareVectors, compareRankings, makeVerdict } from './parity.mjs';
import { prepareBertBatch } from './bge-tokenizer.mjs';

let runtime = null;
const progress = (stage, detail) => self.postMessage({ type: 'progress', stage, detail });

async function initialize() {
  if (runtime) return runtime;
  runtime = (async () => {
    const started = performance.now();
    progress('Loading fixture', 'Fictional papers and public example interests');
    const [fixture, bundle, reference, kernel, cases] = await Promise.all([
      fetch('./assets/fixture.json').then(r => r.json()),
      fetch('./assets/bundle.json').then(r => r.json()),
      fetch('./assets/reference.json').then(r => r.json()),
      fetch('./assets/kernel.py').then(r => r.text()),
      fetch('./assets/policy-cases.json').then(r => r.json()),
    ]);
    progress('Loading Python', 'WebAssembly runtime and NumPy; calculation stays in this browser');
    const py = await loadPyodide({ indexURL: new URL('./node_modules/pyodide/', self.location.href).href });
    await py.loadPackage('numpy');
    await py.runPythonAsync(kernel);
    py.globals.set('bundle_json', JSON.stringify(bundle));
    py.runPython('modules = install_bundle(json.loads(bundle_json))');
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.useBrowserCache = false; // Prototype state stays in memory; no persistent model-cache surprises.
    env.localModelPath = new URL('./assets/models/', self.location.href).href;
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.wasmPaths = new URL('./node_modules/onnxruntime-web/dist/', self.location.href).href;
    // No extra WASM proxy worker: this module is already a dedicated worker.
    env.backends.onnx.wasm.proxy = false;
    progress('Loading embedding model', 'Same native reference graph; CPU WebAssembly, no inference API');
    const tokenizer = await AutoTokenizer.from_pretrained('bge-small', { local_files_only: true });
    const model = await AutoModel.from_pretrained('bge-small', { device: 'wasm', dtype: 'fp32', local_files_only: true });
    return { fixture, bundle, reference, py, tokenizer, model, cases, setupMs: performance.now() - started };
  })();
  try { return await runtime; }
  catch (error) { runtime = null; throw error; }
}

async function embed(state, texts, name) {
  const ordered = texts.map((text, index) => ({ text, index })).sort((a, b) => Array.from(a.text).length - Array.from(b.text).length);
  const results = Array(texts.length);
  for (let start = 0; start < ordered.length; start += 4) {
    const batch = ordered.slice(start, start + 4);
    progress('Computing embeddings', `${name}: ${Math.min(start + 4, ordered.length)} / ${ordered.length}`);
    const raw = await state.tokenizer(batch.map(row => row.text), { padding:false, truncation:false, return_tensor:false });
    const prepared=prepareBertBatch(raw,{padId:state.tokenizer.pad_token_id,sepId:state.tokenizer.sep_token_id});
    const inputs=Object.fromEntries(Object.entries(prepared.encoded).map(([key,rows])=>
      [key,new Tensor('int64',BigInt64Array.from(rows.flat().map(BigInt)),prepared.dims)]));
    const output = await state.model(inputs);
    const hidden = output.last_hidden_state;
    if (!hidden || hidden.dims.length !== 3 || hidden.dims[2] !== 384) throw new Error('Unexpected model output; no substitute embeddings used');
    const [, sequence, dimension] = hidden.dims;
    for (let row = 0; row < batch.length; row++) {
      const offset = row * sequence * dimension;
      const vector = Array.from(hidden.data.slice(offset, offset + dimension)); // CLS pooling, matching FastEmbed.
      const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
      if (!Number.isFinite(norm) || !norm) throw new Error('Invalid model embedding');
      results[batch[row].index] = vector.map(value => value / norm);
    }
    for (const tensor of Object.values(output)) tensor?.dispose?.();
  }
  return results;
}

function rank(state, vectors, fixture=state.fixture) {
  state.py.globals.set('fixture_json', JSON.stringify(fixture));
  state.py.globals.set('vectors_json', JSON.stringify(vectors));
  return JSON.parse(state.py.runPython('json.dumps(rank_fixture(json.loads(fixture_json), json.loads(vectors_json), modules), allow_nan=False)'));
}

self.onmessage = async ({ data }) => {
  if (data.type !== 'brew') return;
  try {
    const state = await initialize();
    const started = performance.now();
    const policyRanking = rank(state, state.reference.vectors);
    const policy = compareRankings(policyRanking, state.reference.ranking);
    const stressCases = state.cases.map(test => {
      const comparison = compareRankings(rank(state, test.vectors, test.fixture), test.reference);
      return { name:test.name, ...comparison, passed:comparison.selectionMatches && !comparison.decisionMismatches.length && comparison.maxScoreDifference<=1e-5 };
    });
    const queries = state.fixture.queries.map(text => state.fixture.query_prefix + text);
    const documents = state.fixture.papers.map(row => row.title.trim() + (row.abstract.trim() ? '. ' + row.abstract.trim() : ''));
    const embeddingStart = performance.now();
    const vectors = { documents: await embed(state, documents, 'Papers'), queries: await embed(state, queries, 'Interests') };
    const inferenceMs = performance.now() - embeddingStart;
    progress('Ranking and selecting', 'Existing Python gates, venue weights, identity deduplication and balanced picker');
    const rankingStart = performance.now();
    const ranking = rank(state, vectors);
    const rankingMs = performance.now() - rankingStart;
    const browser = { documents: compareVectors(vectors.documents, state.reference.vectors.documents),
      queries: compareVectors(vectors.queries, state.reference.vectors.queries), ranking: compareRankings(ranking, state.reference.ranking) };
    const verdict=makeVerdict(browser,policy);
    verdict.stressPass=stressCases.every(test=>test.passed);
    verdict.passed=verdict.passed&&verdict.stressPass;
    self.postMessage({ type: 'result', fixture: state.fixture, ranking,
      report: { schema_version: 1, status: 'browser_fixture_experiment', browser, policy,
        verdict, stressCases, timings: { setupMs: state.setupMs, inferenceMs, rankingMs, brewMs: performance.now() - started },
        model: state.reference.manifest, userAgent: navigator.userAgent, backend: 'wasm-cpu',
        paperCount: documents.length, queryCount: queries.length, privateUserDataUsed: false,
        limits: ['Fictional fixture parity only; not scientific quality or reader usefulness.',
          'No live retrieval, persistence, synchronization, learned feedback model, attribution bonuses, enrichment, history or scheduling.',
          'Desktop CPU/WASM measurement; mobile hardware and WebGPU are not established.'] } });
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message || String(error) });
  }
};
self.postMessage({ type: 'ready' });
