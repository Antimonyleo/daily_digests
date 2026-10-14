import { loadPyodide } from './node_modules/pyodide/pyodide.mjs';
import { AutoTokenizer, AutoModel, Tensor, env } from './node_modules/@huggingface/transformers/dist/transformers.min.js';
import { prepareBertBatch } from './bge-tokenizer.mjs';
import { computeCapabilities, computeSelection, createEmbeddingEngine, formatComputeProgress } from './compute.mjs';

let companion = null;
let boot, state, database, jobRunning = false, stateReplacing = false, computeMode = 'cpu';
const progress = (stage, detail) => self.postMessage({ type: 'progress', stage, detail });
const emit = text => self.postMessage(JSON.parse(text));
const response = (id, body, status = 200) => ({ id, type: 'response', status, headers: { 'content-type': 'application/json' }, body });

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(`dailydigest:${new URL('.', self.location.href).pathname}`, 1);
    request.onupgradeneeded = () => request.result.createObjectStore('state');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
function readSnapshot(key = 'snapshot') {
  return new Promise((resolve, reject) => {
    const request = database.transaction('state').objectStore('state').get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
function saveSnapshot(snapshot, key = 'snapshot') {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction('state', 'readwrite');
    transaction.objectStore('state').put(snapshot, key);
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error || new Error('Browser storage transaction aborted'));
  });
}
async function ownStorage() {
  if (!navigator.locks) throw new Error('This browser does not provide Web Locks required to protect saved reading state.');
  await new Promise((resolve, reject) => {
    navigator.locks.request(`dailydigest:${new URL('.', self.location.href).pathname}`, { ifAvailable: true }, async lock => {
      if (!lock) { reject(new Error('DailyDigest is open in another tab. Close that tab before opening this one.')); return; }
      resolve();
      await new Promise(() => {}); // Tab lifetime ownership; browser releases it on termination.
    }).catch(reject);
  });
}
async function fetchAsset(path, kind) {
  const result = await fetch(path);
  if (!result.ok) throw new Error(`Could not load ${path}: HTTP ${result.status}`);
  return result[kind]();
}
function companionEndpoint(value) {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('Companion endpoint must be an HTTP or HTTPS URL without embedded credentials, query or fragment.');
  }
  if (url.protocol === 'http:' && !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) {
    throw new Error('Remote companion endpoints require HTTPS. HTTP is allowed only for localhost, 127.0.0.1 or [::1].');
  }
  return url.href.replace(/\/$/, '');
}
async function companionCall(path, payload = null, method = 'POST', timeout = 180, configuration = companion) {
  if (!configuration?.endpoint) throw new Error('Configure the optional local companion explicitly before using this action.');
  const result = await fetch(configuration.endpoint + path, {
    method, credentials: 'omit', mode: 'cors', signal: AbortSignal.timeout(Math.ceil(timeout * 1000)),
    headers: { 'X-DailyDigest-Token': configuration.token, ...(payload !== null ? { 'Content-Type': 'application/json' } : {}) },
    ...(payload !== null ? { body: payload } : {}),
  });
  const body = await result.text();
  if (!result.ok) throw new Error(`Companion request failed (HTTP ${result.status}): ${body}`);
  return body;
}
function idleForCompanion() {
  if (jobRunning || state.py.globals.get('has_jobs')()) throw new Error('Wait for the current brew or ranking job before using this companion action.');
}
async function replaceState(payload) {
  stateReplacing = true;
  try {
    await state.py.globals.get('handle_import').callPromising(payload, saveSnapshot);
  } catch (error) {
    stateReplacing = false;
    throw error;
  }
  // Remain closed to requests until UI reloads the successful replacement.
}
async function initialize() {
  if (boot) return boot;
  boot = (async () => {
    if (!WebAssembly.Suspending || !WebAssembly.promising) {
      throw new Error('DailyDigest requires WebAssembly JS Promise Integration to run the complete Python pipeline. Use a current Chromium browser. This browser cannot run the deployed application.');
    }
    await ownStorage();
    database = await openDatabase();
    progress('Loading Python', 'The full DailyDigest application runs in this browser.');
    const [bundle, source, dependencies, snapshot, savedCompanion, savedCompute] = await Promise.all([
      fetchAsset('./assets/app.zip?compute=v1', 'arrayBuffer'), fetchAsset('./assets/runtime.py?compute=v1', 'text'),
      fetchAsset('./assets/dependencies.json', 'json'), readSnapshot(), readSnapshot('companion'), readSnapshot('compute'),
    ]);
    const py = await loadPyodide({ indexURL: new URL('./node_modules/pyodide/', self.location.href).href });
    await py.loadPackage(['micropip', ...dependencies.pyodide]);
    py.globals.set('browser_wheels', JSON.stringify(dependencies.micropip.map(path => new URL(path, self.location.href).href)));
    await py.runPythonAsync('import json, micropip\nawait micropip.install(json.loads(browser_wheels), deps=False)');
    py.unpackArchive(bundle, 'zip', { extractDir: '/app' });
    py.runPython("import os, sys\nos.chdir('/app')\nsys.path.insert(0, '/app/src')\nsys.path.insert(0, '/app/browser')\nsys.path.insert(0, '/app/site-packages')");
    // Install browser HTTP transport before importing web/pipeline.
    py.globals.set('public_source_url', new URL('./assets/public-sources.json', self.location.href).href);
    await py.runPythonAsync('import network\nnetwork.install_httpx_transport()');
    await py.runPythonAsync(source);
    if (snapshot) py.globals.get('restore_snapshot')(snapshot);
    // Native ingest imports bind config.SETTINGS: restore .env before them.
    await py.runPythonAsync('network.install_public_source_fallback(public_source_url)');
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = new URL('./assets/models/', self.location.href).href;
    env.backends.onnx.wasm.wasmPaths = new URL('./node_modules/onnxruntime-web/dist/', self.location.href).href;
    env.backends.onnx.wasm.proxy = false;
    computeMode = ['cpu', 'gpu', 'multicore'].includes(savedCompute) ? savedCompute : 'cpu';
    const engine = createEmbeddingEngine({ mode: computeMode, capabilities: computeCapabilities(), env: env.backends.onnx,
      loadTokenizer: options => AutoTokenizer.from_pretrained('bge-small', { local_files_only: true, ...options }),
      loadModel: async options => {
        if (options.device === 'webgpu') {
          const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
          if (!adapter || adapter.info?.isFallbackAdapter) throw new Error('Hardware GPU acceleration is unavailable');
          env.backends.onnx.webgpu.adapter = adapter;
        }
        return AutoModel.from_pretrained(options.device === 'webgpu' ? 'bge-small-webgpu' : 'bge-small', {
          local_files_only: true, ...options,
          ...(options.device === 'webgpu' ? { session_options: { graphOptimizationLevel: 'disabled' } } : {}),
        });
      },
      prepareBatch: prepareBertBatch, makeTensor: (data, dims) => new Tensor('int64', data, dims),
      shouldCancel: () => Boolean(state.py.globals.get('browser_cancelled')()),
      progress: info => self.postMessage({ type: 'progress', stage: info.phase === 'loading' ? 'Loading embedding model'
        : info.phase === 'complete' ? 'Embeddings complete' : 'Computing embeddings', message: formatComputeProgress(info), compute: info }),
    });
    state = { py, engine };
    globalThis.browserYield = () => new Promise(resolve => setTimeout(resolve, 0));
    globalThis.browserEmbed = texts => engine.embed(texts);
    globalThis.browserEmbeddingBackend = () => engine.device === 'webgpu' ? 'browser-onnx-webgpu-v1' : 'browser-onnx';
    companion = savedCompanion || null;
    globalThis.browserCompanionSupports = backend => Boolean(companion?.endpoint && ['api', 'anthropic', 'claude_cli', 'codex_cli'].includes(backend) && companion.capabilities?.summarize?.includes(backend));
    globalThis.browserCompanionIdentity = backend => globalThis.browserCompanionSupports(backend)
      ? JSON.stringify([companion.endpoint, companion.provider_identities?.[backend] || 'legacy-provider']) : '';
    globalThis.browserCompanion = companionCall;
    py.globals.get('install_browser')();
    progress('Ready', 'Your profile, votes and reading history stay in browser storage.');
    self.postMessage({ type: 'ready' });
    return state;
  })();
  return boot;
}
async function persist() {
  await saveSnapshot(state.py.globals.get('export_snapshot')());
}
async function pumpJobs() {
  if (jobRunning) return;
  jobRunning = true;
  try {
    while (state.py.globals.get('has_jobs')()) {
      await state.py.globals.get('run_job').callPromising();
      await persist();
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message || String(error) });
  } finally { jobRunning = false; }
}
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') { await initialize(); return; }
    await initialize();
    if (data.type === 'cancel-request') { state.py.globals.get('cancel_request')(data.id); return; }
    if (stateReplacing) {
      self.postMessage(response(data.id, JSON.stringify({ detail: 'Backup import is replacing browser state. Wait for the page to reload.' }), 409));
      return;
    }
    if (data.type === 'compute-status') {
      self.postMessage(response(data.id, JSON.stringify({ mode: computeMode, capabilities: computeCapabilities(),
        device: state.engine.device, threads: state.engine.threads })));
      return;
    }
    if (data.type === 'configure-compute') {
      if (jobRunning || state.py.globals.get('has_jobs')()) throw new Error('Wait for the current brew before changing compute settings.');
      const selection = computeSelection(data.mode, computeCapabilities());
      if (selection.fallback) throw new Error(selection.fallback);
      await saveSnapshot(data.mode, 'compute');
      self.postMessage(response(data.id, JSON.stringify({ ok: true, reload: true })));
      return;
    }
    if (data.type === 'companion-status') {
      self.postMessage(response(data.id, JSON.stringify({ configured: Boolean(companion?.endpoint), endpoint: companion?.endpoint || '', capabilities: companion?.capabilities || {} })));
      return;
    }
    if (data.type === 'configure-companion') {
      idleForCompanion();
      if (!data.endpoint && !data.token) {
        await saveSnapshot(null, 'companion'); companion = null;
      } else {
        if (!data.token) throw new Error('The companion requires its explicitly configured authentication token.');
        const next = { endpoint: companionEndpoint(data.endpoint), token: data.token };
        const health = JSON.parse(await companionCall('/health', null, 'GET', 20, next));
        if (health.ok !== true) throw new Error('Companion health check did not confirm availability.');
        next.capabilities = health.capabilities || {};
        next.provider_identities = health.provider_identities || {};
        await saveSnapshot(next, 'companion');
        companion = next;
      }
      self.postMessage(response(data.id, JSON.stringify({ ok: true, configured: Boolean(companion), capabilities: companion?.capabilities || {} })));
      return;
    }
    if (data.type === 'companion-inbound') {
      idleForCompanion();
      const replies = await companionCall('/inbound', '{}');
      idleForCompanion();
      stateReplacing = true;
      try {
        const applied = JSON.parse(await state.py.globals.get('handle_inbound').callPromising(replies, saveSnapshot));
        if (applied.mail_ids.length) {
          try {
            const acknowledged = JSON.parse(await companionCall('/inbound/ack', JSON.stringify({ mail_ids: applied.mail_ids })));
            applied.mailbox_acknowledged = acknowledged.ok === true;
            if (acknowledged.errors?.length) applied.errors.push(...acknowledged.errors);
          } catch (error) {
            applied.mailbox_acknowledged = false;
            applied.errors.push(`Browser votes are saved, but mailbox acknowledgement failed: ${error.message}`);
          }
        }
        delete applied.mail_ids;
        self.postMessage(response(data.id, JSON.stringify(applied)));
      } finally { stateReplacing = false; }
      return;
    }
    if (data.type === 'companion-email' || data.type === 'companion-push') {
      idleForCompanion();
      const payload = data.type === 'companion-email' ? state.py.globals.get('companion_email_payload')()
        : data.type === 'companion-push' ? JSON.stringify({ snapshot: state.py.globals.get('export_snapshot')() }) : '{}';
      const path = data.type === 'companion-email' ? '/email' : data.type === 'companion-push' ? '/state' : '/inbound';
      const body = await companionCall(path, payload, data.type === 'companion-push' ? 'PUT' : 'POST');
      if (data.type === 'companion-email' && JSON.parse(body).sent === true) {
        state.py.globals.get('companion_email_sent')(JSON.parse(payload).digest_id);
        try { await persist(); }
        catch (error) { throw new Error(`The companion sent the email, but saving the browser delivery confirmation failed: ${error.message}`); }
      }
      self.postMessage(response(data.id, body)); return;
    }
    if (data.type === 'companion-pull') {
      idleForCompanion();
      if (data.overwrite !== true) throw new Error('Pulling companion state requires explicit browser-state overwrite confirmation.');
      stateReplacing = true;
      try {
        const remote = JSON.parse(await companionCall('/state', null, 'GET'));
        await replaceState(remote.snapshot);
      } catch (error) { stateReplacing = false; throw error; }
      self.postMessage(response(data.id, JSON.stringify({ ok: true, reload: true }))); return;
    }
    if (data.type === 'export') {
      if (jobRunning || state.py.globals.get('has_jobs')()) throw new Error('Wait for the current brew or ranking job before exporting.');
      self.postMessage(response(data.id, state.py.globals.get('export_snapshot')()));
      return;
    }
    if (data.type === 'import') {
      if (jobRunning || state.py.globals.get('has_jobs')()) throw new Error('Wait for the current brew or ranking job before importing.');
      if (data.overwrite !== true) throw new Error('Import requires explicit confirmation to overwrite browser state.');
      await replaceState(data.body);
      self.postMessage(response(data.id, JSON.stringify({ ok: true, reload: true })));
      return;
    }
    if (data.type !== 'request') throw new Error('Unknown browser request type.');
    if (data.method?.toUpperCase() === 'POST' && ['/run/start', '/refresh'].includes(data.path.split('?')[0])) {
      const backend = state.py.globals.get('selected_browser_backend')();
      if (globalThis.browserCompanionSupports(backend)) {
        const health = JSON.parse(await companionCall('/health', null, 'GET', 20));
        if (health.ok !== true || !health.capabilities?.summarize?.includes(backend)) {
          throw new Error(`The companion no longer advertises ${backend}. Reconnect or choose an available summary provider before brewing.`);
        }
        if (typeof health.provider_identities?.[backend] !== 'string' || !health.provider_identities[backend]) {
          throw new Error('The companion must provide summary model identity before brewing. Update and reconnect the companion.');
        }
        const next = { ...companion, capabilities: health.capabilities, provider_identities: health.provider_identities };
        await saveSnapshot(next, 'companion'); companion = next;
      }
    }
    const result = await state.py.globals.get('handle_request').callPromising(JSON.stringify(data), emit);
    if (!jobRunning && data.path.split('?')[0] !== '/run/stream') await persist();
    emit(result);
    // Start native background targets only after the start response reaches UI.
    setTimeout(() => { void pumpJobs(); }, 0);
  } catch (error) {
    if (data.id != null) self.postMessage(response(data.id, JSON.stringify({ detail: error.message || String(error) }), 500));
    else self.postMessage({ type: 'error', message: error.message || String(error) });
  }
};
