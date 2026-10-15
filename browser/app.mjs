const LOCAL_ORIGIN = 'http://127.0.0.1:8765';
import { formatComputeProgress } from './compute.mjs';

export function computeStatusText(result) {
  const completed = result.stats?.completedInputs || 0;
  const backend = result.device === 'webgpu' ? 'GPU' : result.threads > 1 ? `CPU · up to ${result.threads} cores` : 'CPU · one core';
  const actual = result.mode === 'gpu' && result.device !== 'webgpu'
    ? 'GPU requested; using CPU fallback.'
    : completed ? `Used ${backend} for ${completed} new text comparisons this session.`
      : `${backend} selected; no new text comparisons completed this session.`;
  return actual + (result.busy ? ' A brew is running. Stop it or wait for it to finish before changing Compute.' : ' Changes apply to the next brew.');
}

export async function requestStorageProtection(storage) {
  if (!storage?.persist) return { granted: false, message: 'This browser cannot request storage protection. Export backups regularly.' };
  const granted = await storage.persist();
  return { granted, message: granted
    ? 'Persistent storage granted. Keep backups for device changes or manual clearing.'
    : 'Persistent storage was not granted. Export backups regularly; this browser may clear site data.' };
}

// Only app-local routes can enter the Python request bridge.
export function localPath(value, base = '/') {
  const url = new URL(value, LOCAL_ORIGIN + base);
  if (url.origin !== LOCAL_ORIGIN) throw new Error('Only local DailyDigest routes are supported.');
  return url.pathname + url.search;
}

export function routeFromHash(hash) {
  try { return localPath(hash.startsWith('#/') ? hash.slice(1) : '/'); }
  catch { return '/'; }
}

export function publicRoute(path, base = 'https://example.invalid/') {
  const url = new URL(base);
  url.hash = localPath(path);
  return url.href;
}

export function announcementKey(info) {
  if (info.phase !== 'encoding') return info.phase;
  return `encoding:${Math.floor(10 * (info.completed || 0) / Math.max(1, info.total || 1))}`;
}

export function browserSupportError(environment = globalThis) {
  const missing = [];
  if (!environment.WebAssembly?.Suspending || !environment.WebAssembly?.promising) missing.push('WebAssembly Promise Integration');
  if (!environment.navigator?.locks) missing.push('Web Locks');
  if (missing.length) {
    return `Local brewing is unavailable because this browser lacks ${missing.join(' and ')}. Try a current desktop Chrome browser. Installing the shortcut does not add these capabilities.`;
  }
  return null;
}

export function friendlyLoadingStatus(stage, detail = '') {
  const messages = {
    'Loading Python': 'Putting the kettle on… Getting your reading desk ready.',
    'Loading embedding model': 'Preparing the tea leaves… First-time setup takes a little longer.',
    'Computing embeddings': 'Steeping your research interests…',
    'Embeddings complete': 'Tea leaves ready. Choosing your best matches…',
    'Ready': 'Your reading desk is ready.',
  };
  return messages[stage] ? messages[stage] + (stage === 'Computing embeddings' && detail ? ` ${detail}` : '') : [stage, detail].filter(Boolean).join(' · ');
}

export class WorkerClient {
  constructor(worker) {
    this.worker = worker;
    this.pending = new Map();
    this.sequence = 0;
    worker.addEventListener('message', ({ data }) => {
      const pending = this.pending.get(data.id);
      if (!pending) return;
      if (data.type === 'stream' || data.type === 'response-chunk') {
        pending.chunk?.(data.body ?? data.chunk ?? '');
      } else if (data.type === 'response') {
        this.pending.delete(data.id);
        pending.resolve(data);
      } else if (data.type === 'error') {
        this.pending.delete(data.id);
        pending.reject(new Error(data.message || data.error || 'Python request failed'));
      }
    });
    worker.addEventListener('error', () => this.fail(new Error('Python worker stopped. Reload to reconnect.')));
  }
  call(payload, chunk) {
    const id = ++this.sequence;
    const promise = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, chunk });
      this.worker.postMessage({ ...payload, id });
    });
    promise.cancel = () => {
      this.worker.postMessage({ id, type: 'cancel-request' });
      const pending = this.pending.get(id);
      this.pending.delete(id);
      pending?.reject(new Error('Request disconnected'));
    };
    return promise;
  }
  fail(error) {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
}

// This function is serialized into each trusted, Python-rendered document.
function frameBridge(token, path, origin, publicBase) {
  let sequence = 0;
  const pending = new Map();
  const notify = (payload) => parent.postMessage({ ...payload, token }, '*');
  const route = (value) => {
    if (publicBase && String(value).startsWith(publicBase + '#/')) value = String(value).slice(publicBase.length + 1);
    const url = new URL(value, origin + path);
    if (url.origin !== origin) throw new Error('External fetch requires an explicitly configured provider.');
    return url.pathname + url.search;
  };
  window.ddLocation = origin + path;
  window.ddNavigate = (value) => notify({ type: 'navigate', path: route(value) });
  window.ddReload = () => window.ddNavigate(path);
  window.ddReplaceState = (_state, _title, value) => {
    path = route(value);
    window.ddLocation = origin + path;
    notify({ type: 'replace-route', path });
  };
  window.addEventListener('message', ({ source, data }) => {
    if (source !== parent || data?.token !== token) return;
    if (data.type === 'compute-progress') {
      window.dispatchEvent(new CustomEvent('dd-compute-progress', { detail: data.compute }));
      return;
    }
    if (data.type === 'compute-config') {
      const notice = document.getElementById('browser-compute-notice');
      if (notice) { notice.hidden = false; notice.textContent = data.message; }
      return;
    }
    if (data.type === 'companion-capabilities') {
      window.dispatchEvent(new CustomEvent('dd-companion-capabilities', {
        detail: { backends: Array.isArray(data.backends) ? data.backends : [] },
      }));
      return;
    }
    const request = pending.get(data.id);
    if (!request) return;
    if (data.type === 'stream') request.chunk?.(data.body);
    else {
      pending.delete(data.id);
      if (data.error) request.reject(new Error(data.error));
      else request.resolve(data);
    }
  });
  const request = (payload, chunk) => {
    const id = ++sequence;
    const promise = new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject, chunk });
      notify({ ...payload, id });
    });
    promise.cancel = () => { pending.delete(id); notify({ type: 'cancel', id }); };
    return promise;
  };
  window.fetch = async (input, options = {}) => {
    const headers = Object.fromEntries(new Headers(options.headers || (input instanceof Request ? input.headers : {})));
    let body = options.body;
    if (body instanceof FormData) { body = new URLSearchParams(body).toString(); headers['content-type'] = 'application/x-www-form-urlencoded'; }
    if (body instanceof URLSearchParams) { body = body.toString(); headers['content-type'] = 'application/x-www-form-urlencoded'; }
    const response = await request({ type: 'request', method: options.method || input.method || 'GET', path: route(input.url || input), headers, body });
    return new Response([204, 205, 304].includes(response.status) ? null : response.body, { status: response.status, headers: response.headers });
  };
  class LocalEventSource extends EventTarget {
    static CONNECTING = 0; static OPEN = 1; static CLOSED = 2;
    constructor(url) {
      super(); this.readyState = 0; let buffer = '';
      this.request = request({ type: 'request', path: route(url), method: 'GET', headers: { accept: 'text/event-stream' } }, (chunk) => {
        if (this.readyState === 2) return;
        if (this.readyState === 0) { this.readyState = 1; this.emit('open', new Event('open')); }
        buffer += chunk.replace(/\r\n/g, '\n');
        let boundary;
        while ((boundary = buffer.indexOf('\n\n')) >= 0) {
          const block = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
          const lines = block.split('\n').filter(line => line.startsWith('data:'));
          if (lines.length) this.emit('message', new MessageEvent('message', { data: lines.map(line => line.slice(5).trimStart()).join('\n') }));
        }
      });
      this.request.then((response) => {
        if (this.readyState !== 2) { this.readyState = 2; this.emit('error', new Event('error')); }
      }).catch(() => { if (this.readyState !== 2) { this.readyState = 2; this.emit('error', new Event('error')); } });
    }
    emit(type, event) { this.dispatchEvent(event); this['on' + type]?.(event); }
    close() { this.readyState = 2; this.request.cancel(); }
  }
  window.EventSource = LocalEventSource;
  document.addEventListener('DOMContentLoaded', () => {
    const field = document.getElementById('llm_api_key');
    if (field) {
      const notice = document.createElement('p');
      notice.className = 'hint';
      notice.textContent = 'API keys entered here are stored in this browser. A Companion can keep provider credentials on your own computer instead. Portable backups exclude credentials unless you choose to include them.';
      field.parentElement.append(notice);
    }
  });
  if (publicBase && typeof MutationObserver !== 'undefined') {
    const rewrite = node => {
      for (const link of [node, ...node.querySelectorAll?.('a[href]') || []]) {
        if (link.tagName !== 'A') continue;
        const href = link.getAttribute('href') || '';
        if (href.startsWith('/') && !href.startsWith('//')) link.setAttribute('href', publicBase + '#' + href);
      }
    };
    new MutationObserver(records => {
      for (const record of records) {
        if (record.type === 'attributes') rewrite(record.target);
        else for (const node of record.addedNodes) rewrite(node);
      }
    }).observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ['href'] });
  }
  window.addEventListener('dd-compute-stop', () => notify({ type: 'compute-stop' }));
  document.addEventListener('click', (event) => {
    const anchor = event.target.closest?.('a[href]');
    if (!anchor || event.defaultPrevented || event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    const href = anchor.getAttribute('href');
    if (href.startsWith('#')) return;
    if (publicBase && href.startsWith(publicBase + '#/')) {
      event.preventDefault(); window.ddNavigate(href); return;
    }
    const url = new URL(href, origin + path);
    if (url.origin !== origin) { anchor.target = '_blank'; anchor.rel = 'noopener noreferrer'; return; }
    event.preventDefault(); window.ddNavigate(href);
  });
  document.addEventListener('submit', async (event) => {
    if (event.defaultPrevented) return;
    const form = event.target;
    event.preventDefault();
    const data = new URLSearchParams(new FormData(form, event.submitter));
    const method = (form.getAttribute('method') || 'GET').toUpperCase();
    let target = route(form.getAttribute('action') || path);
    if (method === 'GET') { const url = new URL(target, origin); for (const [key, value] of data) url.searchParams.append(key, value); window.ddNavigate(url.pathname + url.search); }
    else notify({ type: 'submit', path: target, method, body: data.toString(), headers: { 'content-type': 'application/x-www-form-urlencoded' } });
  });
}

export function renderedDocument(html, token, path, publicURL) {
  // Templates use location for a small number of navigation operations. Route
  // these through the outer hash router; the Pages subpath never becomes an API.
  html = html.replace(/window\.location\.href\s*=\s*([^;]+);/g, 'window.ddNavigate($1);')
    .replace(/window\.location\.reload\(\)/g, 'window.ddReload()')
    .replace(/window\.location\.href/g, 'window.ddLocation')
    .replace(/window\.history\.replaceState/g, 'window.ddReplaceState')
    .replace(/navigator\.serviceWorker\.register\("\/sw\.js"\)/g, 'Promise.reject(new Error("Browser worker owns runtime"))')
    .replace(/<link[^>]+href="\/(?:manifest\.webmanifest|app-icon\.svg)"[^>]*>/g, '');
  let publicBase = '';
  if (publicURL) {
    const url = new URL(publicURL); url.hash = ''; publicBase = url.href;
    html = html.replace(/(<a\b[^>]*\bhref=["'])(\/(?!\/)[^"']*)(["'])/gi,
      (_match, before, href, quote) => before + publicRoute(href, publicBase) + quote);
  }
  const args = [token, path, LOCAL_ORIGIN, publicBase].map(value => JSON.stringify(value).replace(/</g, '\\u003c')).join(',');
  const bridge = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; form-action 'none'; base-uri 'none'"><script>(${frameBridge.toString()})(${args});</script>`;
  return html.replace(/<head[^>]*>/i, match => match + bridge);
}

async function boot() {
  const supportError = browserSupportError();
  if (supportError) throw new Error(supportError);
  const frame = document.getElementById('app');
  try {
    document.documentElement.dataset.theme = localStorage.getItem('dailydigest-theme')
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  } catch { document.documentElement.dataset.theme = 'light'; }
  const startup = document.getElementById('startup');
  const status = document.getElementById('status');
  const startupStatus = document.getElementById('startup-status');
  const announcement = document.getElementById('announcements');
  const importNoticeKey = `dailydigest:import-notice:${location.pathname}`;
  const backupKey = `dailydigest:last-backup:${location.pathname}`;
  const showLastBackup = value => {
    const timestamp = new Date(value);
    if (Number.isFinite(timestamp.getTime())) document.getElementById('backup-status').textContent = `Last backup export started: ${timestamp.toLocaleString()}. Keep the downloaded file private.`;
  };
  try { const value = localStorage.getItem(backupKey); if (value) showLastBackup(value); } catch { /* optional history */ }
  navigator.storage?.estimate?.().then(({ usage, quota }) => {
    if (Number.isFinite(usage) && Number.isFinite(quota)) document.getElementById('storage-usage').textContent =
      `Site storage: about ${(usage / 1048576).toFixed(1)} MB used of ${(quota / 1048576).toFixed(0)} MB available to this origin.`;
  }).catch(() => {});
  const rememberImportNotice = result => {
    if (!result.credentialsOmitted) return;
    try { sessionStorage.setItem(importNoticeKey, 'This portable backup excluded credentials. Check Settings and configure your chosen provider if credentials are missing.'); } catch { /* browser storage may be disabled */ }
  };
  const transferNotice = document.getElementById('transfer-notice');
  try {
    const message = sessionStorage.getItem(importNoticeKey);
    if (message) { document.getElementById('transfer-message').textContent = message; transferNotice.hidden = false; sessionStorage.removeItem(importNoticeKey); }
  } catch { /* no notice persisted */ }
  document.getElementById('dismiss-transfer').onclick = () => { transferNotice.hidden = true; };
  let storageWarning = '';
  const setStatus = (message, announce = true) => {
    if (storageWarning && message.includes('saved in this browser')) message = 'Your reading desk · changes need a backup';
    status.textContent = message; status.title = message;
    if (!startup.hidden) startupStatus.textContent = message;
    if (announce && announcement.textContent !== message) announcement.textContent = message;
  };
  const toolbar = document.querySelector('.browser-bar');
  new ResizeObserver(() => document.documentElement.style.setProperty('--browser-bar-height', `${toolbar.getBoundingClientRect().height}px`)).observe(toolbar);
  let installPrompt;
  const installButton = document.getElementById('install');
  window.addEventListener('beforeinstallprompt', event => {
    event.preventDefault();
    installPrompt = event;
    installButton.hidden = false;
  });
  installButton.onclick = async () => {
    if (!installPrompt) return;
    await installPrompt.prompt();
    await installPrompt.userChoice;
    installPrompt = null;
    installButton.hidden = true;
  };
  window.addEventListener('appinstalled', () => { installPrompt = null; installButton.hidden = true; });
  const isolationKey = `dailydigest:isolation:${location.pathname}`;
  let isolationRequested = false;
  try { isolationRequested = localStorage.getItem(isolationKey) === 'true'; } catch { /* storage can be disabled */ }
  const registerWorker = async isolate => {
    const url = new URL('./sw.js', import.meta.url);
    if (isolate) url.searchParams.set('isolate', '1');
    return navigator.serviceWorker.register(url, { scope: new URL('.', import.meta.url).pathname });
  };
  if ('serviceWorker' in navigator) registerWorker(isolationRequested)
    .catch(error => { setStatus(`App shortcut unavailable: ${error.message}`); });
  let token = crypto.randomUUID();
  const worker = new Worker(new URL('./worker.mjs?compute=v1', import.meta.url), { type: 'module' });
  const client = new WorkerClient(worker);
  const streams = new Map();
  let ready = false, companionBackends = [], themeObserver;
  const sendCompanionCapabilities = () => frame.contentWindow.postMessage({ type: 'companion-capabilities', token, backends: companionBackends }, '*');
  let navigation = 0;
  let activePath = '/';
  let computeTimer, latestCompute, computeReceivedAt, lastAnnouncement;
  const computeNotice = 'Brewing uses this device’s processor and memory, and can increase heat, fan noise and battery use. Thousands of new articles may take 20 minutes or more on one CPU core. Choose optional acceleration in Compute. Keep this tab open until the brew finishes to save its calculations.';
  frame.addEventListener('load', () => {
    frame.contentWindow.postMessage({ type: 'compute-config', token, message: computeNotice }, '*');
    sendCompanionCapabilities();
    themeObserver?.disconnect();
    const root = frame.contentDocument?.documentElement;
    if (root) {
      const syncTheme = () => { document.documentElement.dataset.theme = root.dataset.theme || 'light'; };
      syncTheme();
      themeObserver = new MutationObserver(syncTheme);
      themeObserver.observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    }
    const heading = frame.contentDocument?.querySelector('h1, main, [role="main"]');
    if (heading) { heading.setAttribute('tabindex', '-1'); heading.focus({ preventScroll: true }); }
    frame.contentDocument?.addEventListener('click', () => {
      for (const menu of document.querySelectorAll('.browser-bar > details[open]')) menu.open = false;
    });
  });
  const displayCompute = () => {
    if (!latestCompute) return;
    const info = { ...latestCompute };
    if (info.phase === 'encoding') info.elapsed_ms += performance.now() - computeReceivedAt;
    info.message = formatComputeProgress(info);
    const milestone = announcementKey(info);
    setStatus(info.message, milestone !== lastAnnouncement);
    lastAnnouncement = milestone;
    frame.contentWindow.postMessage({ type: 'compute-progress', token, compute: info }, '*');
  };
  const errorPanel = document.getElementById('app-error');
  let errorFocus;
  const dismissError = () => { errorPanel.hidden = true; errorFocus?.focus(); };
  document.getElementById('reload-app').onclick = () => location.reload();
  document.getElementById('dismiss-error').onclick = dismissError;
  errorPanel.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); dismissError(); }
  });
  const report = (error) => {
    const message = error.message || String(error);
    if (!ready || frame.hidden) {
      startup.dataset.state = 'error'; document.getElementById('startup-title').textContent = 'Couldn’t open your reading desk';
      setStatus(message); document.getElementById('retry').hidden = false;
      document.getElementById('recovery-tools').hidden = false;
      document.getElementById('recovery-reset').hidden = ready;
      document.querySelector('.recovery-import-label').hidden = ready;
      return;
    }
    setStatus('Action needs attention');
    if (errorPanel.hidden) errorFocus = document.activeElement;
    document.getElementById('app-error-message').textContent = message;
    errorPanel.hidden = false;
    document.getElementById('dismiss-error').focus();
  };
  const companionStatus = document.getElementById('companion-status');
  document.querySelector('.companion-command').textContent = `dd companion --allowed-origin ${location.origin}`;
  const companionEndpoint = document.getElementById('companion-endpoint');
  const companionToken = document.getElementById('companion-token');
  const browserMenus = [...document.querySelectorAll('.browser-bar > details')];
  document.addEventListener('click', event => {
    for (const menu of browserMenus) if (menu.open && !menu.contains(event.target)) menu.open = false;
  });
  for (const menu of browserMenus) {
    menu.addEventListener('toggle', () => {
      if (menu.open) for (const other of browserMenus) if (other !== menu) other.open = false;
    });
    menu.addEventListener('keydown', event => {
      if (event.key === 'Escape' && menu.open) {
        menu.open = false;
        menu.querySelector('summary').focus();
      }
    });
  }
  async function companionCall(payload) {
    const response = await client.call(payload);
    let result;
    try { result = JSON.parse(response.body); } catch { throw new Error('The application returned an unreadable response.'); }
    if (response.status >= 400) throw new Error(result.detail || result.message || `Request HTTP ${response.status}`);
    return result;
  }
  async function refreshCompanion() {
    try {
      const result = await companionCall({ type: 'companion-status' });
      companionBackends = result.capabilities?.summarize || [];
      sendCompanionCapabilities();
      const configured = Boolean(result.configured);
      const capabilities = result.capabilities || {};
      companionEndpoint.value = result.endpoint || '';
      document.getElementById('companion-disconnect').disabled = !configured;
      for (const capability of ['email', 'inbound', 'state']) {
        const buttons = capability === 'state' ? ['push', 'pull'] : [capability];
        for (const action of buttons) document.getElementById(`companion-${action}`).disabled = !configured || !capabilities[capability];
      }
      document.getElementById('companion-summary-help').hidden = !configured || !capabilities.summarize?.length;
      document.getElementById('companion-schedule-help').hidden = !configured || !capabilities.scheduler;
      const labels = { summarize: 'summaries', email: 'email', inbound: 'email feedback', state: 'state transfer', scheduler: 'scheduled brews' };
      const names = Object.keys(labels).filter(key => Array.isArray(capabilities[key]) ? capabilities[key].length : capabilities[key]).map(key => labels[key]);
      companionStatus.textContent = configured ? `Connected to ${result.endpoint}. Available: ${names.join(', ') || 'no optional services'}.` : 'No companion configured.';
    } catch (error) { companionStatus.textContent = error.message; }
  }
  let computeApplying = false;
  async function refreshCompute({ keepSelection = false } = {}) {
    try {
      const result = await companionCall({ type: 'compute-status' });
      const select = document.getElementById('compute-mode');
      if (!keepSelection) select.value = result.mode;
      select.querySelector('[value="gpu"]').disabled = !result.capabilities.gpu;
      const canEnableIsolation = 'serviceWorker' in navigator && isSecureContext && navigator.hardwareConcurrency > 1;
      select.querySelector('[value="multicore"]').disabled = !result.capabilities.multicore && !canEnableIsolation;
      document.getElementById('compute-support').textContent =
        `${result.capabilities.gpu ? 'WebGPU is exposed by this browser; hardware availability is checked when used.' : 'GPU acceleration is unavailable in this browser.'} ${result.capabilities.multicore ? `Up to ${result.capabilities.threads} CPU cores are available.` : canEnableIsolation ? 'Selecting multiple cores will try enabling browser isolation and reload once. If unavailable, computation stays on one core.' : 'Multiple CPU cores are unavailable on this site or browser.'}`;
      document.getElementById('compute-status').textContent = computeStatusText(result);
      document.getElementById('compute-apply').disabled = result.busy || computeApplying;
    } catch (error) { document.getElementById('compute-status').textContent = error.message; }
  }
  document.getElementById('compute-menu').addEventListener('toggle', event => {
    if (event.target.open && ready) refreshCompute({ keepSelection: true });
  });
  document.getElementById('compute-form').onsubmit = async event => {
    event.preventDefault();
    if (computeApplying) return;
    computeApplying = true;
    document.getElementById('compute-apply').disabled = true;
    document.getElementById('compute-status').textContent = 'Saving your compute choice before reloading…';
    try {
      const mode = document.getElementById('compute-mode').value;
      const needsIsolation = mode === 'multicore' && !crossOriginIsolated;
      await companionCall({ type: 'configure-compute', mode, allowIsolationReload: needsIsolation });
      if ('serviceWorker' in navigator && (needsIsolation || isolationRequested && mode !== 'multicore')) {
        const enable = mode === 'multicore';
        try { localStorage.setItem(isolationKey, String(enable)); } catch { /* no persistent opt-in */ }
        document.getElementById('compute-status').textContent = 'Preparing browser isolation before reloading…';
        const registration = await registerWorker(enable);
        const candidate = registration.installing || registration.waiting;
        if (candidate) await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error('Browser isolation could not start. Reload to continue on CPU.')), 15000);
          const changed = () => {
            if (candidate.state === 'activated') { clearTimeout(timeout); resolve(); }
            else if (candidate.state === 'redundant') { clearTimeout(timeout); reject(new Error('Browser isolation is unavailable; use one CPU core.')); }
          };
          candidate.addEventListener('statechange', changed); changed();
        });
      }
      location.reload();
    } catch (error) {
      computeApplying = false;
      await refreshCompute({ keepSelection: true });
      document.getElementById('compute-status').textContent = error.message;
    }
  };
  const download = (body, type, filename) => {
    const url = URL.createObjectURL(new Blob([body], { type }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  async function navigate(path, options = {}) {
    const version = ++navigation;
    for (const stream of streams.values()) stream.cancel();
    streams.clear();
    setStatus('Opening DailyDigest…');
    try {
      path = localPath(path);
      let response = await client.call({ type: 'request', path, method: 'GET', ...options });
      for (let redirects = 0; response.status >= 300 && response.status < 400; redirects++) {
        if (redirects >= 8) throw new Error('DailyDigest returned too many redirects.');
        const headers = new Headers(response.headers);
        if (!headers.has('location')) throw new Error('DailyDigest redirect is missing its destination.');
        path = localPath(headers.get('location'), path);
        response = await client.call({ type: 'request', path, method: 'GET' });
      }
      if (version !== navigation) return;
      if (response.status >= 400) throw new Error(`DailyDigest HTTP ${response.status}: ${response.body}`);
      const headers = new Headers(response.headers);
      if ((headers.get('content-type') || '').includes('text/calendar')) {
        download(response.body, 'text/calendar', 'dailydigest.ics');
        history.replaceState(null, '', '#' + activePath);
        if (frame.hidden) await navigate(activePath);
        else setStatus('Calendar exported.');
        return;
      }
      if (!(headers.get('content-type') || '').includes('text/html')) throw new Error('Expected a DailyDigest page.');
      activePath = path;
      clearInterval(computeTimer); latestCompute = null;
      history.replaceState(null, '', '#' + path);
      token = crypto.randomUUID();
      frame.srcdoc = renderedDocument(response.body, token, path, location.href);
      frame.hidden = false; startup.hidden = true;
      startup.dataset.state = 'ready';
      setStatus('Your reading desk · saved in this browser');
    } catch (error) { if (version === navigation) report(error); }
  }
  worker.addEventListener('message', ({ data }) => {
    if (data.type === 'ready') {
      ready = true;
      document.getElementById('export').disabled = false;
      document.getElementById('import').disabled = false;
      document.getElementById('companion-connect').disabled = false;
      refreshCompanion();
      refreshCompute();
      refreshStorageProtection();
      companionCall({ type: 'recovery-status' }).then(result => {
        const key = result.quarantined?.at(-1);
        const button = document.getElementById('retained-recovery');
        button.hidden = !key;
        document.getElementById('retained-recovery-help').hidden = !key;
        if (key) button.dataset.key = key;
      }).catch(() => {});
      navigate(routeFromHash(location.hash));
    } else if (data.type === 'compute-state') {
      if (data.busy) document.getElementById('compute-apply').disabled = true;
      refreshCompute({ keepSelection: true });
    } else if (data.type === 'progress') {
      if (data.compute) {
        clearInterval(computeTimer); latestCompute = data.compute; computeReceivedAt = performance.now();
        displayCompute();
        if (data.compute.phase === 'encoding') computeTimer = setInterval(displayCompute, 1000);
        if (['encoding', 'complete', 'fallback'].includes(data.compute.phase) && (!data.compute.completed || data.compute.phase !== 'encoding')) refreshCompute({ keepSelection: true });
      } else setStatus(data.message || friendlyLoadingStatus(data.stage, data.detail) || 'Putting the kettle on…');
    }
    else if (data.type === 'storage-warning') {
      storageWarning = data.message || '';
      const notice = document.getElementById('storage-warning');
      notice.textContent = storageWarning; notice.hidden = !storageWarning;
      document.getElementById('storage-status').textContent = storageWarning || 'Your latest changes are saved in this browser.';
      if (storageWarning) setStatus('Storage needs attention. Open Browser data to back up your changes.');
    }
    else if (data.type === 'error' && !data.id) report(new Error(data.message || data.error || 'Python startup failed'));
  });
  worker.addEventListener('error', () => report(new Error('Python worker stopped. Reload to reconnect.')));
  window.addEventListener('hashchange', () => { if (ready) navigate(routeFromHash(location.hash)); });
  window.addEventListener('message', async ({ source, data }) => {
    if (source !== frame.contentWindow || data?.token !== token) return;
    if (data.type === 'compute-stop') { clearInterval(computeTimer); latestCompute = null; setStatus('Brew needs attention'); return; }
    const reply = (payload) => source.postMessage({ ...payload, token: data.token, id: data.id }, '*');
    try {
      if (data.type === 'navigate') {
        const path = localPath(data.path, activePath);
        if (/^\/calendar\/[^/]+\.ics$/.test(path) || location.hash === '#' + path) navigate(path);
        else location.hash = path;
      }
      else if (data.type === 'replace-route') history.replaceState(null, '', '#' + localPath(data.path));
      else if (data.type === 'submit') await navigate(data.path, { method: data.method, body: data.body, headers: data.headers });
      else if (data.type === 'cancel') { streams.get(data.id)?.cancel(); streams.delete(data.id); }
      else if (data.type === 'request') {
        const request = client.call({ type: 'request', path: localPath(data.path), method: data.method, headers: data.headers, body: data.body }, body => reply({ type: 'stream', body }));
        streams.set(data.id, request);
        try { reply(await request); } finally { streams.delete(data.id); }
      }
    } catch (error) {
      reply({ error: error.message });
      if (error.message !== 'Request disconnected') report(error);
    }
  });
  document.getElementById('retry').onclick = () => location.reload();
  document.getElementById('recovery-export').onclick = async () => {
    try {
      const response = await client.call({ type: 'recovery-export' });
      if (response.status >= 400) throw new Error(response.body);
      download(response.body, 'application/json', 'dailydigest-raw-recovery.json');
      setStatus('Original browser data downloaded. Recovery files can contain credentials; keep this file private.');
    } catch (error) { report(error); }
  };
  document.getElementById('retained-recovery').onclick = async event => {
    try {
      const response = await client.call({ type: 'recovery-export', key: event.currentTarget.dataset.key });
      if (response.status >= 400) throw new Error(response.body);
      download(response.body, 'application/json', 'dailydigest-original-recovery.json');
    } catch (error) { report(error); }
  };
  document.getElementById('recovery-import').onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      if (!confirm('Restore this backup? Your current stored state will be retained for recovery.')) return;
      const body = await file.text();
      await companionCall({ type: 'recovery-import', body, overwrite: true });
      rememberImportNotice({ credentialsOmitted: JSON.parse(body).credentialsIncluded === false });
      location.reload();
    } catch (error) { report(new Error(`Recovery import failed: ${error.message}`)); }
    finally { event.target.value = ''; }
  };
  document.getElementById('recovery-reset').onclick = async () => {
    if (!confirm('Start a fresh reading desk? Your original stored state will be retained for recovery. Download it first if you need a separate copy.')) return;
    try { await companionCall({ type: 'recovery-reset', overwrite: true }); location.reload(); }
    catch (error) { report(error); }
  };
  const storagePromptKey = `dailydigest:storage-prompt:${location.pathname}`;
  const storageNotice = document.getElementById('storage-consent');
  function dismissStoragePrompt() {
    storageNotice.hidden = true;
    try { localStorage.setItem(storagePromptKey, 'dismissed'); } catch { /* optional reminder */ }
  }
  async function refreshStorageProtection() {
    try {
      const granted = Boolean(await navigator.storage?.persisted?.());
      document.getElementById('storage-status').textContent = granted ? 'Persistent storage is enabled.' : 'Stored on this device. Export a backup before clearing browser data.';
      document.getElementById('protect-storage').disabled = granted || !navigator.storage?.persist;
      let dismissed = false;
      try { dismissed = localStorage.getItem(storagePromptKey) === 'dismissed'; } catch { /* optional reminder */ }
      storageNotice.hidden = granted || dismissed || !navigator.storage?.persist;
    } catch { storageNotice.hidden = true; }
  }
  async function protectStorage() {
    document.getElementById('protect-storage').disabled = true;
    document.getElementById('consent-protect-storage').disabled = true;
    try {
      const result = await requestStorageProtection(navigator.storage);
      document.getElementById('storage-status').textContent = result.message;
      setStatus(result.message);
      document.getElementById('protect-storage').disabled = result.granted || !navigator.storage?.persist;
      dismissStoragePrompt();
    } catch (error) {
      document.getElementById('storage-status').textContent = `Storage protection unavailable: ${error.message}`;
      document.getElementById('protect-storage').disabled = false;
    } finally { document.getElementById('consent-protect-storage').disabled = false; }
  }
  document.getElementById('protect-storage').onclick = protectStorage;
  document.getElementById('consent-protect-storage').onclick = protectStorage;
  document.getElementById('dismiss-storage-consent').onclick = dismissStoragePrompt;
  document.getElementById('export').onclick = async () => {
    try {
      const includeCredentials = document.getElementById('include-credentials').checked;
      if (includeCredentials && !confirm('Include saved API keys and credentials in this backup? Keep the downloaded file private.')) return;
      const response = await client.call({ type: 'export', includeCredentials });
      if (response.status >= 400) throw new Error(response.body);
      download(response.body, 'application/json', 'dailydigest-browser-backup.json');
      const exported = new Date().toISOString();
      try { localStorage.setItem(backupKey, exported); } catch { /* export remains usable */ }
      showLastBackup(exported);
      setStatus('Browser backup exported.');
    } catch (error) { report(error); }
  };
  document.getElementById('import').onchange = async (event) => {
    const file = event.target.files[0]; if (!file) return;
    try {
      const body = await file.text(); JSON.parse(body);
      if (!confirm('Replace this browser’s DailyDigest profile, votes and saved reading with this backup? Export your current data first if you need to keep it.')) return;
      const response = await client.call({ type: 'import', body, overwrite: true });
      if (response.status >= 400) throw new Error(response.body);
      rememberImportNotice(JSON.parse(response.body));
      location.reload();
    } catch (error) { report(new Error(`Backup import failed: ${error.message}`)); }
    finally { event.target.value = ''; }
  };
  document.getElementById('companion-form').onsubmit = async (event) => {
    event.preventDefault();
    companionStatus.textContent = 'Checking companion capabilities…';
    try {
      await companionCall({ type: 'configure-companion', endpoint: companionEndpoint.value.trim(), token: companionToken.value });
      companionToken.value = '';
      await refreshCompanion();
    } catch (error) { companionStatus.textContent = error.message; }
  };
  document.getElementById('companion-disconnect').onclick = async () => {
    try { await companionCall({ type: 'configure-companion', endpoint: '', token: '' }); companionToken.value = ''; await refreshCompanion(); }
    catch (error) { companionStatus.textContent = error.message; }
  };
  for (const action of ['email', 'inbound', 'push', 'pull']) {
    const button = document.getElementById(`companion-${action}`);
    button.onclick = async () => {
      if (action === 'email' && !confirm('Send your latest saved digest to the recipient configured on your companion?')) return;
      const includeCredentials = action === 'push' && document.getElementById('sync-credentials').checked;
      if (action === 'push' && !confirm(`Send this browser’s profile, feedback and reading history to your companion? ${includeCredentials ? 'This also sends saved API keys and credentials.' : 'Saved API keys and credentials are excluded.'}`)) return;
      if (action === 'pull' && !confirm('Replace this browser’s DailyDigest profile, feedback and reading history with companion state? Export a browser backup first if you need to keep it.')) return;
      button.disabled = true;
      companionStatus.textContent = 'Running requested companion action…';
      try {
        const result = await companionCall({ type: `companion-${action}`, ...(action === 'pull' ? { overwrite: true } : {}), ...(action === 'push' ? { includeCredentials } : {}) });
        if (action === 'pull' && result.reload) { rememberImportNotice(result); location.reload(); return; }
        if (action === 'inbound') {
          const up = Number(result.votes_up || 0);
          const down = Number(result.votes_down || 0);
          const unresolved = Number(result.unresolved_messages || 0);
          companionStatus.textContent = `Email feedback processed: ${up} upvotes, ${down} downvotes.${unresolved ? ` ${unresolved} messages need review.` : ''}${result.errors?.length ? ' ' + result.errors.join(' ') : ''}`;
          if (up + down > 0) await navigate(activePath);
        } else companionStatus.textContent = action === 'email' ? 'Latest digest sent.' : action === 'push' ? 'Browser state sent.' : 'Companion state restored. Reload to open it.';
      } catch (error) { companionStatus.textContent = error.message; }
      finally { button.disabled = false; }
    };
  }
  worker.postMessage({ type: 'init' });
}

if (typeof document !== 'undefined') boot().catch(error => {
  document.getElementById('startup').dataset.state = 'error';
  document.getElementById('startup-title').textContent = 'Couldn’t open your reading desk';
  document.getElementById('startup-status').textContent = error.message;
  document.getElementById('status').textContent = error.message;
  document.getElementById('retry').hidden = Boolean(browserSupportError());
  document.getElementById('retry').onclick = () => location.reload();
});
