import { loadPyodide } from './node_modules/pyodide/pyodide.mjs';
import { AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Tensor, env } from './node_modules/@huggingface/transformers/dist/transformers.min.js';
import { prepareBertBatch } from './bge-tokenizer.mjs';
import { computeCapabilities, computeSelection, createEmbeddingEngine, formatComputeProgress, guardGPUModel } from './compute.mjs';

import { createSnapshotWriter, validateRecoverySnapshot, validateStoredSnapshot, capturePrivateSnapshot } from './storage.mjs';
import { createBrowserHttpBridge } from './http.mjs';
import { createSemanticEngine } from './semantic.mjs';
import { createPublicEmbeddingCache } from './public-embeddings.mjs';
import { validateReadingView } from './reading-view.mjs';

let companion = null;
let boot, state, database, storageWriter, ownership, recoveryPy, startupFailed = false, startupReady = false, jobRunning = false, stateReplacing = false, computeMode = 'cpu';
let configurationChanging = false, reloadPending = false;
const progress = (stage, detail) => self.postMessage({ type: 'progress', stage, detail });
let gpuProbe = null;

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
    if (key === 'snapshot') transaction.objectStore('state').delete('reading-view');
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error || new Error('Browser storage transaction aborted'));
  });
}
async function ownStorage({ timeoutMs = 60000 } = {}) {
  if (ownership) return ownership;
  if (!navigator.locks) throw new Error('This browser does not provide Web Locks required to protect saved reading state.');
  const controller = new AbortController();
  const waiting = setTimeout(() => {
    progress('Waiting for another tab', 'DailyDigest is open in another tab. Switch to it, or choose Use this tab instead.');
    self.postMessage({type:'storage-waiting'});
  }, 250);
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  ownership = new Promise((resolve, reject) => {
    navigator.locks.request(`dailydigest:${new URL('.', self.location.href).pathname}`, { signal: controller.signal }, async () => {
      clearTimeout(waiting); clearTimeout(timeout);
      resolve();
      await new Promise(() => {}); // Tab lifetime ownership; browser releases it on termination.
    }).catch(error => {
      clearTimeout(waiting); clearTimeout(timeout); ownership = null;
      reject(controller.signal.aborted
        ? new Error('DailyDigest is open in another tab. Switch to it, choose Use this tab instead, or close it and retry.') : error);
    });
  });
  return ownership;
}
async function probeGPU() {
  if (!navigator.gpu) return { available: false, reason: 'no-api', message: 'This browser does not expose WebGPU. Try a browser with WebGPU enabled.' };
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { available: false, reason: 'no-adapter', message: 'WebGPU is exposed, but the browser could not find a usable GPU adapter. Check browser GPU settings and graphics drivers.' };
    const info = adapter.info || {};
    if (info.isFallbackAdapter) return { available: false, reason: 'software-adapter', message: 'Only a software GPU adapter is available; using CPU instead.' };
    return { available: true, reason: 'available', message: 'A hardware GPU adapter is available. Model compatibility is checked when used.',
      info: { vendor: info.vendor || 'not disclosed', architecture: info.architecture || 'not disclosed' }, adapter };
  } catch {
    return { available: false, reason: 'probe-failed', message: 'The browser could not check the GPU adapter. Check browser GPU settings and graphics drivers.' };
  }
}
async function fetchAsset(path, kind) {
  const result = await fetch(path);
  if (!result.ok) throw new Error(`Could not load ${path}: HTTP ${result.status}`);
  return result[kind]();
}
async function loadPythonPackages(py, dependencies) {
  const wheels = dependencies.micropip || [];
  const packages = (dependencies.pyodide || []).filter(name => name !== 'micropip');
  if (wheels.length) packages.push('micropip');
  await py.loadPackage(packages);
  // Pure Python wheels are already unpacked in app.zip in current builds.
  // Importing micropip and asking it to install [] costs startup time for no work.
  if (wheels.length) {
    py.globals.set('browser_wheels', JSON.stringify(wheels.map(path => new URL(path, self.location.href).href)));
    await py.runPythonAsync('import json, micropip\nawait micropip.install(json.loads(browser_wheels), deps=False)');
  }
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
  if (stateReplacing || configurationChanging || reloadPending || jobRunning || state.py.globals.get('has_jobs')()) {
    const error = new Error('Wait for the current brew or state change before using this action.');
    error.status = 409;
    throw error;
  }
}
async function replaceState(payload) {
  try { validateRecoverySnapshot(payload); }
  catch (error) {
    // Rejected input must leave even the previous IndexedDB checkpoint intact.
    error.skipCheckpoint = true;
    throw error;
  }
  stateReplacing = true;
  try {
    await persist({ force: true });
    await state.py.globals.get('handle_import').callPromising(payload, saveSnapshot);
  } catch (error) {
    stateReplacing = false;
    throw error;
  }
  // Remain closed to requests until UI reloads the successful replacement.
}
function recoveryKeys() {
  return new Promise((resolve, reject) => {
    const request = database.transaction('state').objectStore('state').getAllKeys();
    request.onsuccess = () => resolve(request.result.filter(key => typeof key === 'string' && key.startsWith('recovery:')).sort());
    request.onerror = () => reject(request.error);
  });
}
async function recoveryRequest(data) {
  await ownStorage();
  database ||= await openDatabase();
  if (data.type === 'recovery-status') {
    const snapshot = await readSnapshot();
    const quarantined = await recoveryKeys();
    return { available: snapshot != null || quarantined.length > 0, failed: startupFailed, ready: startupReady,
      bytes: typeof snapshot === 'string' ? snapshot.length : 0, quarantined };
  }
  if (data.type === 'recovery-export') {
    if (data.key && !String(data.key).startsWith('recovery:')) throw new Error('Invalid recovery key.');
    if (data.key) return (await readSnapshot(data.key)) ?? '';
    return (await readSnapshot()) ?? (await readSnapshot((await recoveryKeys()).at(-1))) ?? '';
  }
  if (!startupFailed || startupReady) throw new Error('Recovery replacement is available only after startup fails.');
  if (data.overwrite !== true) throw new Error('Recovery requires explicit confirmation to replace browser state.');
  const next = data.type === 'recovery-reset' ? null : data.body;
  if (next !== null) {
    validateRecoverySnapshot(next);
    if (recoveryPy) {
      recoveryPy.globals.set('recovery_payload', next);
      await recoveryPy.runPythonAsync('validate_snapshot_sqlite(decode_snapshot(recovery_payload))');
    }
  }
  const previous = await readSnapshot();
  const key = `recovery:${Date.now()}:${Math.random().toString(36).slice(2)}`;
  await new Promise((resolve, reject) => {
    const transaction = database.transaction('state', 'readwrite');
    const store = transaction.objectStore('state');
    if (previous != null) store.put(previous, key);
    if (next === null) store.delete('snapshot'); else store.put(next, 'snapshot');
    store.delete('reading-view');
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error || new Error('Recovery was not saved; original state is unchanged.'));
  });
  return { ok: true, reload: true, quarantined: previous != null, recoveryKey: previous != null ? key : null,
    credentialsOmitted: next !== null && JSON.parse(next).credentialsIncluded === false };
}

async function initialize() {
  if (boot) return boot;
  boot = (async () => {
    if (!WebAssembly.Suspending || !WebAssembly.promising) {
      throw new Error('DailyDigest requires WebAssembly JS Promise Integration to run the complete Python pipeline. Use a current Chromium browser. This browser cannot run the deployed application.');
    }
    const startupStarted = performance.now(), timings = [];
    let stageStarted = startupStarted, previousStage = 'storage-ownership';
    const stage = (name, label, detail) => {
      const now = performance.now(); timings.push({ stage: previousStage, durationMs: now - stageStarted });
      previousStage = name; stageStarted = now; progress(label, detail);
    };
    await ownStorage();
    database = await openDatabase();
    stage('application-assets', 'Loading application', 'Opening the local application…');
    const [bundle, source, dependencies, snapshot, savedCompanion, savedCompute] = await Promise.all([
      fetchAsset('./assets/app.zip?compute=v1', 'arrayBuffer'), fetchAsset('./assets/runtime.py?compute=v1', 'text'),
      fetchAsset('./assets/dependencies.json', 'json'), readSnapshot(), readSnapshot('companion'), readSnapshot('compute'),
    ]);
    stage('python-runtime', 'Loading Python', 'Starting the Python runtime…');
    const py = await loadPyodide({ indexURL: new URL('./node_modules/pyodide/', self.location.href).href });
    stage('python-packages', 'Loading libraries', 'Preparing application libraries…');
    await loadPythonPackages(py, dependencies);
    stage('application-import', 'Opening application', 'Preparing your reading desk…');
    py.unpackArchive(bundle, 'zip', { extractDir: '/app' });
    py.runPython("import os, sys\nos.chdir('/app')\nsys.path.insert(0, '/app/src')\nsys.path.insert(0, '/app/browser')\nsys.path.insert(0, '/app/site-packages')");
    // Install browser HTTP transport before importing web/pipeline.
    const httpBridge = createBrowserHttpBridge({ shouldCancel: () => Boolean(state?.py.globals.get('browser_cancelled')?.()) });
    globalThis.browserHttpRequest = httpBridge.request;
    globalThis.browserHttpWait = httpBridge.wait;
    py.globals.set('public_source_url', new URL('./assets/public-sources.json', self.location.href).href);
    await py.runPythonAsync('import network\nnetwork.install_httpx_transport()');
    await py.runPythonAsync(source);
    recoveryPy = py;
    stage('snapshot-restore', 'Restoring your data', snapshot ? 'Restoring your saved reading desk…' : 'Preparing a new reading desk…');
    let snapshotLimitWarning = null;
    if (snapshot) {
      snapshotLimitWarning = validateStoredSnapshot(snapshot).warning;
      py.globals.set('startup_snapshot', snapshot);
      await py.runPythonAsync('restore_startup_snapshot(startup_snapshot)');
      py.runPython('del startup_snapshot');
    }
    stage('application-setup', 'Preparing reading desk', 'Connecting the local application…');
    // Native ingest imports bind config.SETTINGS: restore .env before them.
    await py.runPythonAsync('network.install_public_source_fallback(public_source_url)');
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = new URL('./assets/models/', self.location.href).href;
    env.backends.onnx.wasm.wasmPaths = new URL('./node_modules/onnxruntime-web/dist/', self.location.href).href;
    env.backends.onnx.wasm.proxy = false;
    computeMode = ['cpu', 'gpu', 'multicore'].includes(savedCompute) ? savedCompute : 'cpu';
    const capabilities = computeCapabilities();
    if (computeMode === 'gpu' && capabilities.gpu) {
      progress('Checking GPU', 'Checking hardware acceleration on this device…');
      gpuProbe = await probeGPU();
      capabilities.gpu = gpuProbe.available;
      if (gpuProbe.available) env.backends.onnx.webgpu.adapter = gpuProbe.adapter;
    }
    const engine = createEmbeddingEngine({ mode: computeMode, capabilities, env: env.backends.onnx,
      loadTokenizer: options => AutoTokenizer.from_pretrained('bge-small', { local_files_only: true, ...options }),
      loadModel: async ({ exportKind, ...options }) => {
        const model = await AutoModel.from_pretrained(exportKind === 'gpu' ? 'bge-small-webgpu' : 'bge-small', {
          local_files_only: true, ...options,
          ...(exportKind === 'gpu' ? { session_options: { graphOptimizationLevel: 'disabled' } } : {}),
        });
        return options.device === 'webgpu' ? guardGPUModel(model, env.backends.onnx.webgpu.device) : model;
      },
      prepareBatch: prepareBertBatch, makeTensor: (data, dims) => new Tensor('int64', data, dims),
      shouldCancel: () => Boolean(state.py.globals.get('browser_cancelled')()),
      progress: info => self.postMessage({ type: 'progress', stage: info.phase === 'loading' ? 'Loading embedding model'
        : info.phase === 'complete' ? 'Embeddings complete' : 'Computing embeddings', message: formatComputeProgress(info), compute: info }),
    });
    // Absolute same-origin paths bypass env.localModelPath without enabling
    // remote models or racing embedding loads by changing their global root.
    const semanticPath = new URL('semantic-matcher', new URL('./assets/semantic-models/', self.location.href)).pathname;
    let semanticStarted = 0;
    const semanticEngine = createSemanticEngine({
      loadTokenizer: () => AutoTokenizer.from_pretrained(semanticPath, { local_files_only: true }),
      loadModel: options => AutoModelForSequenceClassification.from_pretrained(semanticPath, {
        ...options, local_files_only: true,
      }),
      makeTensor: (data, dims) => new Tensor('int64', data, dims),
      shouldCancel: () => Boolean(state.py.globals.get('browser_cancelled')()),
      progress: info => {
        const compute = { ...info, phase: info.phase === 'scoring' ? 'encoding' : info.phase,
          elapsed_ms: performance.now() - semanticStarted, backend: 'CPU · local topic matching' };
        self.postMessage({ type: 'progress', stage: info.phase === 'loading' ? 'Loading topic matcher'
          : info.phase === 'complete' ? 'Topic matching complete' : 'Matching research topics',
        message: formatComputeProgress(compute), compute });
      },
    });
    const publicEmbeddings = createPublicEmbeddingCache({ signature: '__PUBLIC_EMBEDDING_SIGNATURE__',
      backend: () => engine.cacheBackend, embed: texts => engine.embed(texts),
      shouldCancel: () => Boolean(state.py.globals.get('browser_cancelled')()),
      url: new URL('./assets/public-embeddings.json', self.location.href) });
    state = { py, engine, semanticEngine, capabilities, publicEmbeddings };
    globalThis.browserYield = () => new Promise(resolve => setTimeout(resolve, 0));
    globalThis.browserEmbed = texts => publicEmbeddings.embed(texts);
    globalThis.browserSemanticScore = async payload => {
      semanticStarted = performance.now();
      return JSON.stringify(await semanticEngine.score(JSON.parse(payload)));
    };
    globalThis.browserEmbeddingBackend = () => engine.cacheBackend;
    companion = savedCompanion || null;
    globalThis.browserCompanionSupports = backend => Boolean(companion?.endpoint && ['api', 'anthropic', 'claude_cli', 'codex_cli'].includes(backend) && companion.capabilities?.summarize?.includes(backend));
    globalThis.browserCompanionIdentity = backend => globalThis.browserCompanionSupports(backend)
      ? JSON.stringify([companion.endpoint, companion.provider_identities?.[backend] || 'legacy-provider']) : '';
    globalThis.browserCompanion = companionCall;
    py.globals.get('install_browser')();
    const encodedFiles = new Map();
    storageWriter = createSnapshotWriter({
      generation: () => Number(py.globals.get('state_generation')()),
      ready: () => Boolean(py.globals.get('snapshot_ready')()),
      capture: () => {
        const names = JSON.parse(py.globals.get('snapshot_file_names')());
        const checkpoint = capturePrivateSnapshot(names, name => py.FS.readFile('/app/' + name), {cache:encodedFiles, versions:JSON.parse(py.globals.get('snapshot_file_versions')())});
        snapshotLimitWarning = checkpoint.warning;
        return checkpoint.payload;
      }, save: saveSnapshot,
      warning: message => {
        self.postMessage({ type: 'storage-warning', message: message || snapshotLimitWarning });
        if (!message) self.postMessage({type:'storage-checkpoint', generation:Number(py.globals.get('state_generation')())});
      },
    });
    if (snapshotLimitWarning) self.postMessage({ type: 'storage-warning', message: snapshotLimitWarning });
    const interruptedBrew = await readSnapshot('brew-in-progress');
    if (interruptedBrew) {
      self.postMessage({ type: 'brew-interrupted', message: 'The previous brew was interrupted when its tab closed or reloaded. Your last saved digest and feedback are kept. Start a new brew when ready; it will reuse saved calculations.' });
      await saveSnapshot(null, 'brew-in-progress');
    }
    stage('ready', 'Ready', 'Your reading desk is ready.');
    self.postMessage({ type: 'startup-timings', totalMs: performance.now() - startupStarted, stages: timings });
    startupReady = true;
    progress('Ready', 'Your profile, votes and reading history stay in browser storage.');
    self.postMessage({ type: 'ready' });
    return state;
  })().catch(error => { startupFailed = true; throw error; });
  return boot;
}
async function persist(options = {}) {
  return storageWriter ? storageWriter.flush(options) : true;
}
async function pumpJobs() {
  if (jobRunning || !state.py.globals.get('has_jobs')()) return;
  jobRunning = true;
  try { await saveSnapshot({ startedAt: new Date().toISOString() }, 'brew-in-progress'); }
  catch { self.postMessage({ type: 'storage-warning', message: 'Browser storage could not record brew recovery status. Keep this tab open and export a backup before reloading.' }); }
  self.postMessage({ type: 'compute-state', busy: true });
  try {
    while (state.py.globals.get('has_jobs')()) {
      try { await state.py.globals.get('run_job').callPromising(); }
      finally { await persist({ force: true }); }
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message || String(error) });
  } finally {
    jobRunning = false;
    await persist({ force: true });
    try { await saveSnapshot(null, 'brew-in-progress'); } catch { /* A stale marker only shows a cautious interruption notice. */ }
    self.postMessage({ type: 'compute-state', busy: false });
  }
}
self.onmessage = async ({ data }) => {
  try {
    if (['recovery-status', 'recovery-export', 'recovery-import', 'recovery-reset'].includes(data.type)) {
      const result = await recoveryRequest(data);
      self.postMessage(response(data.id, typeof result === 'string' ? result : JSON.stringify(result)));
      return;
    }
    if (data.type === 'init') { await initialize(); return; }
    await initialize();
    if (data.type === 'cancel-request') { state.py.globals.get('cancel_request')(data.id); return; }
    if (data.type === 'cancel-reload') {
      if (stateReplacing || configurationChanging) {
        self.postMessage(response(data.id, JSON.stringify({ detail: 'A state change is still being saved. Reload preparation cannot be cancelled yet.' }), 409));
        return;
      }
      reloadPending = false;
      self.postMessage(response(data.id, JSON.stringify({ ok: true })));
      return;
    }
    if (stateReplacing || configurationChanging || reloadPending) {
      self.postMessage(response(data.id, JSON.stringify({ detail: reloadPending ? 'The application is about to reload. Wait for the page to reopen.' : stateReplacing
        ? 'Backup import is replacing browser state. Wait for the page to reload.'
        : 'Browser settings are being saved. Retry this action after they finish.' }), 409));
      return;
    }
    if (data.type === 'cache-reading-view') {
      const view = validateReadingView(data.view);
      const current = Number(state.py.globals.get('state_generation')());
      if (view && data.generation === current && storageWriter.status().durable && !jobRunning) {
        await saveSnapshot(view, 'reading-view');
        self.postMessage(response(data.id, JSON.stringify({cached:true})));
      } else self.postMessage(response(data.id, JSON.stringify({cached:false})));
      return;
    }
    if (data.type === 'flush-state') {
      const saved = await persist({force:true});
      self.postMessage(response(data.id, JSON.stringify({saved}))); return;
    }
    if (data.type === 'storage-status') {
      self.postMessage(response(data.id, JSON.stringify(storageWriter?.status() || {})));
      return;
    }
    if (data.type === 'compute-status') {
      self.postMessage(response(data.id, JSON.stringify({ mode: computeMode, capabilities: state.capabilities,
        gpu: gpuProbe && { available: gpuProbe.available, reason: gpuProbe.reason, message: gpuProbe.message, info: gpuProbe.info },
        device: state.engine.device, threads: state.engine.threads, stats: state.engine.stats, publicCache: state.publicEmbeddings?.stats,
        busy: jobRunning || Boolean(state.py.globals.get('has_jobs')()) })));
      return;
    }
    if (data.type === 'probe-gpu') {
      const result = await probeGPU();
      self.postMessage(response(data.id, JSON.stringify({ available: result.available, reason: result.reason, message: result.message, info: result.info })));
      return;
    }
    if (data.type === 'prepare-reload') {
      idleForCompanion();
      if (!await persist({ force: true })) throw new Error('Your latest changes could not be saved. Export a backup before reloading.');
      idleForCompanion();
      reloadPending = true;
      self.postMessage(response(data.id, JSON.stringify({ ok: true })));
      return;
    }
    if (data.type === 'configure-compute') {
      idleForCompanion();
      if (!await persist({ force: true })) {
        throw new Error('Compute settings were not changed because your reading state could not be saved. Export a backup before reloading.');
      }
      const selection = computeSelection(data.mode, computeCapabilities());
      if (selection.fallback && !(data.mode === 'multicore' && data.allowIsolationReload === true)) throw new Error(selection.fallback);
      idleForCompanion();
      configurationChanging = true;
      try { await saveSnapshot(data.mode, 'compute'); }
      finally { configurationChanging = false; }
      reloadPending = true;
      self.postMessage(response(data.id, JSON.stringify({ ok: true, reload: true })));
      return;
    }
    if (data.type === 'companion-status') {
      self.postMessage(response(data.id, JSON.stringify({ configured: Boolean(companion?.endpoint), endpoint: companion?.endpoint || '', capabilities: companion?.capabilities || {} })));
      return;
    }
    if (data.type === 'configure-companion') {
      idleForCompanion();
      let next = null;
      if (!data.endpoint && !data.token) {
        next = null;
      } else {
        if (!data.token) throw new Error('The companion requires its explicitly configured authentication token.');
        next = { endpoint: companionEndpoint(data.endpoint), token: data.token };
        const health = JSON.parse(await companionCall('/health', null, 'GET', 20, next));
        if (health.ok !== true) throw new Error('Companion health check did not confirm availability.');
        next.capabilities = health.capabilities || {};
        next.provider_identities = health.provider_identities || {};
      }
      idleForCompanion();
      configurationChanging = true;
      try {
        await saveSnapshot(next, 'companion');
        companion = next;
      } finally { configurationChanging = false; }
      self.postMessage(response(data.id, JSON.stringify({ ok: true, configured: Boolean(companion), capabilities: companion?.capabilities || {} })));
      return;
    }
    if (data.type === 'companion-inbound') {
      idleForCompanion();
      const replies = await companionCall('/inbound', '{}');
      idleForCompanion();
      stateReplacing = true;
      try {
        await persist({ force: true });
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
        : data.type === 'companion-push' ? JSON.stringify({ snapshot: state.py.globals.get('export_portable_snapshot')(data.includeCredentials === true) }) : '{}';
      const path = data.type === 'companion-email' ? '/email' : data.type === 'companion-push' ? '/state' : '/inbound';
      const body = await companionCall(path, payload, data.type === 'companion-push' ? 'PUT' : 'POST');
      if (data.type === 'companion-email' && JSON.parse(body).sent === true) {
        state.py.globals.get('companion_email_sent')(JSON.parse(payload).digest_id);
        if (!await persist({ force: true })) throw new Error('The companion sent the email, but saving the browser delivery confirmation failed. Keep this tab open and export a backup.');
      }
      self.postMessage(response(data.id, body)); return;
    }
    if (data.type === 'companion-pull') {
      idleForCompanion();
      if (data.overwrite !== true) throw new Error('Pulling companion state requires explicit browser-state overwrite confirmation.');
      stateReplacing = true;
      let credentialsOmitted = false;
      try {
        const remote = JSON.parse(await companionCall('/state', null, 'GET'));
        await replaceState(remote.snapshot);
        credentialsOmitted = JSON.parse(remote.snapshot).credentialsIncluded === false;
      } catch (error) { stateReplacing = false; throw error; }
      self.postMessage(response(data.id, JSON.stringify({ ok: true, reload: true, credentialsOmitted }))); return;
    }
    if (data.type === 'export') {
      await persist({force:true});
      if (jobRunning || state.py.globals.get('has_jobs')()) throw new Error('Wait for the current brew or ranking job before exporting.');
      self.postMessage(response(data.id, state.py.globals.get('export_portable_snapshot')(data.includeCredentials === true)));
      return;
    }
    if (data.type === 'import') {
      if (jobRunning || state.py.globals.get('has_jobs')()) throw new Error('Wait for the current brew or ranking job before importing.');
      if (data.overwrite !== true) throw new Error('Import requires explicit confirmation to overwrite browser state.');
      await replaceState(data.body);
      const imported = JSON.parse(data.body);
      self.postMessage(response(data.id, JSON.stringify({ ok: true, reload: true,
        credentialsOmitted: imported.credentialsIncluded === false, omittedCredentialKeys: imported.omittedCredentialKeys || [] })));
      return;
    }
    if (data.type !== 'request') throw new Error('Unknown browser request type.');
    if (data.method?.toUpperCase() === 'POST' && ['/run/start', '/refresh'].includes(data.path.split('?')[0])) {
      idleForCompanion();
      const backend = state.py.globals.get('selected_browser_backend')();
      if (globalThis.browserCompanionSupports(backend)) {
        const checkedCompanion = companion;
        const health = JSON.parse(await companionCall('/health', null, 'GET', 20, checkedCompanion));
        idleForCompanion();
        if (companion !== checkedCompanion) {
          const error = new Error('The companion changed during its health check. Retry the brew with the current settings.');
          error.status = 409;
          throw error;
        }
        if (health.ok !== true || !health.capabilities?.summarize?.includes(backend)) {
          throw new Error(`The companion no longer advertises ${backend}. Reconnect or choose an available summary provider before brewing.`);
        }
        if (typeof health.provider_identities?.[backend] !== 'string' || !health.provider_identities[backend]) {
          throw new Error('The companion must provide summary model identity before brewing. Update and reconnect the companion.');
        }
        const next = { ...companion, capabilities: health.capabilities, provider_identities: health.provider_identities };
        configurationChanging = true;
        try { await saveSnapshot(next, 'companion'); companion = next; }
        finally { configurationChanging = false; }
      }
    }
    const result = await state.py.globals.get('handle_request').callPromising(JSON.stringify(data), emit);
    // A committed route response reaches the UI before an idle checkpoint.
    // Forced flushes still fence reload, imports, exports and tab handoff.
    const delivered = JSON.parse(result);
    delivered.headers = { ...delivered.headers,
      'x-dailydigest-generation': String(state.py.globals.get('state_generation')()),
      'x-dailydigest-storage': storageWriter.status().durable ? 'saved' : 'pending' };
    if (data.method && data.method.toUpperCase() !== 'GET' && delivered.status < 400 && storageWriter.status().dirty) self.postMessage({type:'storage-dirty'});
    emit(JSON.stringify(delivered));
    storageWriter.schedule();
    // Start native background targets only after the start response reaches UI.
    setTimeout(() => { void pumpJobs(); }, 0);
  } catch (error) {
    if (state && !stateReplacing && !error.skipCheckpoint) await persist({ force: true });
    if (data.id != null) self.postMessage(response(data.id, JSON.stringify({ detail: error.message || String(error) }), error.status || 500));
    else self.postMessage({ type: 'error', message: error.message || String(error) });
  }
};
