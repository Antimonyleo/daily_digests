const LOCAL_ORIGIN = 'http://127.0.0.1:8765';
import { formatComputeProgress } from './compute.mjs';
import { DeferredWorker, extractReadingView, readReadingView, readingViewDocument, readingViewRoute, hasSavedDesk, SHARED_READING_STYLES } from './reading-view.mjs';
import { createTabHandoff } from './tab-handoff.mjs';
import { initializeQuickGuide } from './guide.mjs';
import { createLinkedBackup, linkedBackupStore, createBackupScheduler } from './linked-backup.mjs';
import { prepareDefaultCPU, readComputeChoice } from './cpu-startup.mjs';

export function backupReminderDue({lastExport, firstOpen, now = Date.now()}) {
  const exported = lastExport ? new Date(lastExport).getTime() : NaN;
  const first = firstOpen ? Number(firstOpen) : NaN;
  const last = Number.isFinite(exported) ? exported : first;
  return Number.isFinite(last) && now - last >= 7 * 86400000;
}

export async function backupFileText(file) {
  if (file.size > 128 * 1024 * 1024) throw new Error('Backup exceeds the 128 MB restore limit.');
  return file.text();
}

export function httpErrorMessage(response) {
  const prefix = `DailyDigest HTTP ${response.status}: `;
  if ((new Headers(response.headers).get('content-type') || '').toLowerCase().includes('json')) {
    try {
      const detail = JSON.parse(response.body)?.detail;
      if (typeof detail === 'string' && detail.trim() && !/[<>]/.test(detail) && !/^[{\[]/.test(detail.trim())) {
        return prefix + detail.trim().slice(0, 500);
      }
    } catch { /* Malformed or structured errors must not dump submitted values. */ }
  }
  return prefix + 'The request could not be completed. Please try again.';
}

export function computeStatusText(result) {
  const completed = result.stats?.completedInputs || 0;
  const backend = result.device === 'webgpu' ? 'GPU' : result.threads > 1 ? `CPU · up to ${result.threads} cores` : 'CPU · one core';
  const actual = result.mode === 'gpu' && result.device !== 'webgpu'
    ? 'GPU requested; using CPU fallback.'
    : completed ? `Used ${backend} for ${completed} new text comparisons this session.`
      : `${backend} selected; no new text comparisons completed this session.`;
  const fallback = result.stats?.fallbackReason === 'device-lost' ? ' The GPU device was lost.' : result.stats?.fallbackReason === 'model-failed' ? ' The GPU could not run this model.' : '';
  const reason = fallback || (result.mode === 'gpu' && result.device !== 'webgpu' && result.gpu?.message ? ` ${result.gpu.message}` : '');
  const reused = result.publicCache?.hits ? ` ${result.publicCache.hits} public text comparisons reused.` : '';
  return actual + reason + reused + (result.busy ? ' A brew is running. Stop it or wait for it to finish before changing Compute.' : ' Changes apply to the next brew.');
}


export function publicSourceFreshness(summary, now = Date.now()) {
  const fetched = Date.parse(summary?.oldest_fetched_at);
  if (!Number.isFinite(fetched) || fetched > now + 300000) return null;
  const hours = Math.max(0, (now - fetched) / 3600000);
  if (hours > 48) return { level: 'expired', hours,
    message: `Some public sources are ${Math.floor(hours)} hours old and have expired. A new brew may have reduced coverage; your saved digest is kept if research coverage fails.` };
  if (hours > 24 || summary.carried_forward_sources > 0) return { level: 'warning', hours,
    message: `Some public sources were last collected ${Math.floor(hours)} hours ago${summary.carried_forward_sources > 0 ? '; earlier successful records are being reused' : ''}. Check source coverage after brewing.` };
  return { level: 'fresh', hours, message: '' };
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

export function skipToReadingDesk(event, frame, startup) {
  event.preventDefault(); // A focus target must not replace the hash-router URL.
  const target = frame.hidden ? startup : frame.contentDocument?.querySelector('main') || frame;
  if (target === startup) target.tabIndex = -1;
  target.focus();
}

/** Mount only build-owned Pip markup in the trusted shell, never article HTML. */
export function mountCachedReadingPip(frame, view) {
  const document = frame.ownerDocument, window = document.defaultView;
  const controller = new window.AbortController();
  const host = document.createElement('div'); host.id = 'cached-pip';
  const stylesheet = document.createElement('link'); stylesheet.rel = 'stylesheet';
  stylesheet.href = new URL('./pip.css', import.meta.url).href;
  const styled = new Promise((resolve, reject) => {
    stylesheet.addEventListener('load', resolve, {once:true});
    stylesheet.addEventListener('error', () => reject(new Error('Pip styles unavailable')), {once:true});
    controller.signal.addEventListener('abort', resolve, {once:true});
  });
  document.head.append(stylesheet); frame.after(host);
  const placeDock = () => host.style.setProperty('--cached-pip-bottom', `${Math.max(18, window.innerHeight - frame.getBoundingClientRect().bottom + 18)}px`);
  const placement = new window.ResizeObserver(placeDock);
  placement.observe(frame);
  window.addEventListener('resize', placeDock, {signal:controller.signal});
  placeDock();
  const toggle = document.createElement('button'); toggle.type = 'button'; toggle.id = 'cached-pip-toggle';
  document.querySelector('#reading-view-notice > div').append(toggle);
  let disposed = false, unmount = () => {}, compactOpen = false, focusRequested = false;
  const focusPip = () => (host.querySelector('#tea-another:not(:disabled)') || host.querySelector('#tea-pet'))?.focus();
  const dock = window.matchMedia('(min-width: 1440px)');
  toggle.setAttribute('aria-controls', host.id);
  const visibility = () => {
    const hidden = document.documentElement.dataset.pip === 'hidden';
    host.hidden = hidden || (!dock.matches && !compactOpen);
    toggle.textContent = hidden ? 'Show Pip' : host.hidden ? 'Visit Pip' : 'Hide Pip';
    toggle.setAttribute('aria-expanded', String(!host.hidden));
  };
  try { document.documentElement.dataset.pip = window.localStorage.getItem('dailydigest-pip') === 'hidden' ? 'hidden' : 'visible'; }
  catch { document.documentElement.dataset.pip = 'visible'; }
  visibility();
  dock.addEventListener('change', visibility, {signal:controller.signal});
  toggle.addEventListener('click', () => {
    const value = host.hidden ? 'visible' : 'hidden';
    compactOpen = value === 'visible';
    focusRequested = compactOpen;
    document.documentElement.dataset.pip = value;
    try { window.localStorage.setItem('dailydigest-pip', value); } catch { /* Optional preference. */ }
    visibility(); document.dispatchEvent(new window.Event('pipvisibilitychange'));
    if (focusRequested) focusPip();
  }, {signal:controller.signal});
  const dispose = () => {
    if (disposed) return;
    disposed = true; controller.abort(); placement.disconnect(); unmount(); host.remove(); toggle.remove(); stylesheet.remove();
  };
  window.addEventListener('pagehide', event => { if (!event.persisted) dispose(); }, {signal:controller.signal});
  const ready = Promise.all([
    fetch(new URL('./pip.html', import.meta.url), {signal:controller.signal}).then(response => {
      if (!response.ok) throw new Error('Pip markup unavailable');
      return response.text();
    }),
    import(new URL('./pip.mjs', import.meta.url).href),
    styled,
  ]).then(([html, module]) => {
    if (disposed) return;
    // This is the static, same-build component. Cached article data enters only
    // the controller's textContent-based notes API, never this HTML assignment.
    host.innerHTML = html;
    unmount = module.mountPip(host, {digestId:view.date, notes:view.notes});
    host.dataset.state = 'ready';
    if (focusRequested) focusPip();
  }).catch(error => { dispose(); if (error.name !== 'AbortError') throw error; });
  return {ready, dispose};
}

export async function settleRenderedEdits(frame, {timeoutMs = 5000} = {}) {
  if (!frame.contentDocument) return;
  const pending = frame.contentWindow?.ddFlushRatings?.();
  if (!pending) return;
  let timer;
  try {
    await Promise.race([pending, new Promise((_resolve, reject) => {
      timer = setTimeout(() => reject(new Error('The reading desk is still finishing an action. Wait for your ratings, then try again.')), timeoutMs);
    })]);
  } finally { clearTimeout(timer); }
}

export function trackSetupDraft(form, changed, {initiallyDirty = false} = {}) {
  if (!form) return;
  const snapshot = () => JSON.stringify([...new FormData(form)]);
  const baseline = snapshot();
  const update = () => changed(initiallyDirty || snapshot() !== baseline);
  for (const name of ['input', 'change', 'click']) form.addEventListener(name, update);
}

export class WorkerClient {
  constructor(worker) {
    this.worker = worker;
    this.pending = new Map();
    this.inFlight = new Map();
    this.paused = false;
    this.sequence = 0;
    worker.addEventListener('message', ({ data }) => {
      if (data.type === 'response' || data.type === 'error') {
        this.inFlight.get(data.id)?.finish(); this.inFlight.delete(data.id);
      }
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
    if (this.failure || this.paused && !['prepare-reload', 'cancel-reload', 'configure-compute'].includes(payload.type)) {
      const rejected = Promise.reject(this.failure || new Error('DailyDigest is switching tabs. Try again after the switch.'));
      rejected.cancel = () => {};
      return rejected;
    }
    const id = ++this.sequence;
    let finish;
    const settled = new Promise(resolve => {finish = resolve;});
    this.inFlight.set(id, {promise:settled, finish, mutating:payload.type === 'request' && payload.method && payload.method.toUpperCase() !== 'GET'});
    const promise = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, chunk });
      try { this.worker.postMessage({ ...payload, id }); }
      catch (error) { this.pending.delete(id); this.inFlight.delete(id); finish(); reject(error); }
    });
    promise.cancel = () => {
      if (!this.failure && this.inFlight.has(id)) this.worker.postMessage({ id, type: 'cancel-request' });
      const pending = this.pending.get(id);
      this.pending.delete(id);
      pending?.reject(new Error('Request disconnected'));
    };
    return promise;
  }
  fail(error) {
    this.failure = error;
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    for (const request of this.inFlight.values()) request.finish();
    this.inFlight.clear();
  }
  async quiesce({timeoutMs = 5000} = {}) {
    this.paused = true;
    let timer;
    try {
      await Promise.race([
        Promise.all([...this.inFlight.values()].map(request => request.promise)),
        new Promise((_resolve, reject) => {timer = setTimeout(() => reject(new Error('The other tab is still finishing an action. Try switching again in a moment.')), timeoutMs);}),
      ]);
    } catch (error) { this.paused = false; throw error; }
    finally { clearTimeout(timer); }
  }
  resume() { this.paused = false; }
}

// This function is serialized into each trusted, Python-rendered document.
function frameBridge(token, path, origin, publicBase) {
  let sequence = 0;
  const pending = new Map();
  const parentOrigin = publicBase ? new URL(publicBase).origin : window.location.origin;
  const notify = (payload) => parent.postMessage({ ...payload, token }, parentOrigin);
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
  window.addEventListener('message', ({ source, origin: senderOrigin, data }) => {
    if (source !== parent || senderOrigin !== parentOrigin || data?.token !== token) return;
    if (data.type === 'storage-committed') { window.dispatchEvent(new Event('dd-storage-committed')); return; }
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
    const storageNote = document.getElementById('browser-storage-note');
    if (storageNote) storageNote.hidden = false;
    for (const button of document.querySelectorAll('[data-browser-action]')) {
      button.addEventListener('click', () => notify({type:button.dataset.browserAction}));
    }
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
  // Route focus announces the new page without outlining its non-interactive
  // heading/container. Interactive controls retain their keyboard focus rings.
  const routeFocusStyle = '<style>:is(h1, main, [role="main"])[tabindex="-1"]:focus { outline: none; }</style>';
  const nonce = crypto.randomUUID().replaceAll('-', '');
  const imageOrigins = new Set();
  for (const match of html.matchAll(/<img\b[^>]*\bdata-src=["']([^"']+)["']/gi)) {
    try { const url = new URL(match[1]); if (url.protocol === 'https:') imageOrigins.add(url.origin); } catch { /* invalid figures remain blocked */ }
  }
  html = html.replace(/<script\b/gi, `<script nonce="${nonce}"`);
  const bridge = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'nonce-${nonce}'; style-src 'unsafe-inline'; font-src data:; img-src 'self' data: blob: ${[...imageOrigins].join(' ')}; connect-src 'none'; form-action 'none'; base-uri 'none'">${routeFocusStyle}<script nonce="${nonce}">(${frameBridge.toString()})(${args});</script>`;
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
  document.querySelector('.skip-link').onclick = event => skipToReadingDesk(event, frame, startup);
  const status = document.getElementById('status');
  const startupStatus = document.getElementById('startup-status');
  let ready = false, companionBackends = [], themeObserver, restoringBackup = false;
  const dataSettings = document.getElementById('data-settings');
  const startupDataIntentKey = `dailydigest:startup-data-intent:${location.pathname}`;
  let dataControlsReady = false, pendingDataSection = null;
  try {
    const section = sessionStorage.getItem(startupDataIntentKey);
    if (['backup', 'data', 'guide'].includes(section)) pendingDataSection = section;
  } catch { /* An ordinary startup still works without tab storage. */ }
  function openDataSection(section) {
    if (!dataControlsReady) {
      pendingDataSection = section;
      status.textContent = section === 'backup' ? 'Opening backup…' : section === 'guide' ? 'Opening guide…' : 'Opening device settings…';
      document.getElementById('announcements').textContent = status.textContent;
      return;
    }
    pendingDataSection = null;
    try { sessionStorage.removeItem(startupDataIntentKey); } catch { /* Optional startup intent. */ }
    if (section === 'guide') { guide.open(); return; }
    if (!dataSettings.open) dataSettings.showModal();
    if (section === 'data') return;
    const menu = document.getElementById(section === 'backup' ? 'browser-data-menu' : `${section}-menu`);
    menu.open = true;
    menu.scrollIntoView({block:'nearest'});
  }
  document.getElementById('open-data-settings').onclick = () => openDataSection('data');
  document.getElementById('close-data-settings').onclick = () => dataSettings.close();
  document.getElementById('open-backup').onclick = () => openDataSection('backup');
  document.getElementById('open-guide').onclick = () => openDataSection('guide');
  document.getElementById('backup-reminder-open').onclick = () => openDataSection('backup');
  for (const id of ['open-backup', 'open-guide', 'open-data-settings']) {
    document.getElementById(id).disabled = false;
  }
  const topicDraftForm = document.getElementById('startup-topics-form');
  const topicDraft = document.getElementById('startup-topics');
  const topicDraftKey = `dailydigest:startup-topics:${location.pathname}`;
  let draftSubmitted = false;
  try {
    topicDraft.value = sessionStorage.getItem(topicDraftKey) || '';
    draftSubmitted = sessionStorage.getItem(topicDraftKey + ':submitted') === 'true';
  } catch { /* optional draft retention */ }
  topicDraft.addEventListener('input', () => {
    try { sessionStorage.setItem(topicDraftKey, topicDraft.value); } catch { /* keep the on-screen draft */ }
  });
  topicDraftForm.addEventListener('submit', event => {
    event.preventDefault();
    const topics = topicDraft.value.split('\n').map(value => value.trim()).filter(Boolean);
    if (topics.length > 10 || topics.some(value => value.length > 160 || value.includes('|'))) {
      document.getElementById('startup-topics-message').textContent = 'Use up to ten topics, each at most 160 characters. Adjust percentages on the next page.';
      return;
    }
    draftSubmitted = true;
    try { sessionStorage.setItem(topicDraftKey + ':submitted', 'true'); } catch { /* current page still keeps it */ }
    document.getElementById('startup-topics-message').textContent = 'Keeping these topics while the reading desk opens…';
    if (ready) navigate('/setup');
  });

  const updateStartupStages = stage => {
    const dataStage = ['Restoring your data', 'Preparing reading desk', 'Ready'].includes(stage);
    for (const node of document.querySelectorAll('[data-startup-stage]')) {
      const name = node.dataset.startupStage;
      node.dataset.state = ['Waiting for previous tab', 'Waiting for another tab'].includes(stage) ? 'waiting' : stage === 'Ready' ? 'done' : name === 'python' ? dataStage ? 'done' : 'active' : name === 'data' ? dataStage ? 'active' : 'waiting' : 'waiting';
    }
  };
  const announcement = document.getElementById('announcements');
  const importNoticeKey = `dailydigest:import-notice:${location.pathname}`;
  const backupKey = `dailydigest:last-backup:${location.pathname}`;
  const backupSnoozeKey = `dailydigest:backup-snooze:${location.pathname}`;
  let firstBrewBackupDue = false;
  const remindBackup = value => {
    try {
      const firstKey = `dailydigest:first-open:${location.pathname}`;
      let first = localStorage.getItem(firstKey);
      if (!first) { first = String(Date.now()); localStorage.setItem(firstKey, first); }
      const snoozed = Number(localStorage.getItem(backupSnoozeKey)) > Date.now();
      const stale = !snoozed && (backupReminderDue({lastExport:value, firstOpen:first}) || firstBrewBackupDue && !value);
      document.getElementById('backup-reminder').hidden = !stale;
      document.getElementById('backup-reminder-text').textContent = value
        ? 'Your last backup export started over a week ago. Keep a recent copy outside this browser.'
        : 'Keep a copy of your reading desk before switching browsers or profiles. Download a backup or link a backup file.';
    } catch { /* Optional reminder cannot block startup. */ }
  };
  const showLastBackup = value => {
    remindBackup(value);
    const timestamp = new Date(value);
    if (Number.isFinite(timestamp.getTime())) document.getElementById('backup-status').textContent = `Last file copy saved or backup download started: ${timestamp.toLocaleString()}. Keep the file private.`;
  };
  document.getElementById('backup-reminder-dismiss').onclick = () => {
    document.getElementById('backup-reminder').hidden = true;
    try { localStorage.setItem(backupSnoozeKey, String(Date.now() + 86400000)); } catch { /* optional reminder */ }
  };
  try { const value = localStorage.getItem(backupKey); remindBackup(value); if (value) showLastBackup(value); } catch { /* optional history */ }
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
    if (storageWarning && (message.includes('saved in this browser') || message.startsWith('Saved in this profile'))) message = 'Your reading desk · changes need a backup';
    status.textContent = message; status.title = message;
    if (!startup.hidden) startupStatus.textContent = message;
    status.setAttribute('aria-label', message);
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
  // Keep the topic draft usable during the brief, one-time CPU preparation.
  hasSavedDesk(indexedDB, `dailydigest:${new URL('.', import.meta.url).pathname}`)
    .then(present => { if (!present && !startupFailedForDraft()) topicDraftForm.hidden = false; }).catch(() => {});
  let cpuStartup = {isolate: isolationRequested, reload: false};
  try {
    cpuStartup = await prepareDefaultCPU({navigator, storage: localStorage, session: sessionStorage,
      key: isolationKey, isolated: crossOriginIsolated, secure: isSecureContext,
      readChoice: () => readComputeChoice(indexedDB, `dailydigest:${new URL('.', import.meta.url).pathname}`),
      register: registerWorker});
  } catch { /* Storage-restricted browsers retain their existing CPU path. */ }
  isolationRequested = cpuStartup.isolate;
  if (cpuStartup.reload) {
    try { if (pendingDataSection) sessionStorage.setItem(startupDataIntentKey, pendingDataSection); } catch { /* Tab storage is optional. */ }
    location.reload(); return;
  }
  if ('serviceWorker' in navigator) registerWorker(isolationRequested)
    .catch(error => { setStatus(`App shortcut unavailable: ${error.message}`); });
  let token = crypto.randomUUID();
  const worker = new DeferredWorker(() => new Worker(new URL('./worker.mjs?compute=v1', import.meta.url), { type: 'module' }));
  const client = new WorkerClient(worker);
  const streams = new Map();
  let runtimeStarted = false, frameGeneration = null, cacheTimer, cachedPip;
  const scrollPositions = new Map();
  const aboutDisclosureStates = new Map();
  let tabMoved = false, dirtySetup = false, hasUnflushedEdits = false, handoff;
  const tabActions = document.getElementById('tab-wait-actions');
  const useThisTab = document.getElementById('use-this-tab');
  const fullDeskKey = `dailydigest:full-desk:${new URL('.', import.meta.url).pathname}`;
  let forceFullDesk = false;
  try { forceFullDesk = sessionStorage.getItem(fullDeskKey) === 'true'; sessionStorage.removeItem(fullDeskKey); } catch { /* optional intent */ }
  const previewNotice = document.getElementById('reading-view-notice');
  const interactiveSandbox = frame.getAttribute('sandbox');
  const startRuntime = (path) => {
    if (path) history.replaceState(null, '', '#' + localPath(path));
    if (runtimeStarted) return;
    runtimeStarted = true;
    cachedPip?.dispose(); cachedPip = null;
    for (const button of previewNotice.querySelectorAll('button')) button.disabled = true;
    document.getElementById('reading-view-status').textContent = 'Opening controls… You can keep reading below.';
    worker.postMessage({ type: 'init' });
  };
  const cacheReadingView = async () => {
    if (!ready || activePath.split('?')[0] !== '/' || frameGeneration === null || frame.hidden) return;
    try {
      const document = frame.contentDocument;
      if (!document || new URL(frame.contentWindow.ddLocation).pathname !== '/') return;
      const view = extractReadingView(document);
      if (view) await client.call({type:'cache-reading-view', generation:frameGeneration, view});
    } catch { /* A disposable reading copy must never block the authoritative desk. */ }
  };
  let computeBusy = false, dismissedRelease = '';
  const releaseNotice = document.getElementById('release-notice');
  const releaseReload = document.getElementById('release-reload');
  const setReleaseState = () => {
    releaseReload.disabled = computeBusy || (runtimeStarted && !ready);
    document.getElementById('release-message').textContent = computeBusy
      ? 'A new version is available. Finish or stop this brew before reloading.' : 'A new version is available.';
  };
  const checkRelease = async () => {
    if (document.hidden) return;
    try {
      const current = document.querySelector('meta[name="dailydigest-build"]')?.content;
      if (!current) return;
      const manifestURL = new URL('./assets/build-manifest.json', import.meta.url);
      manifestURL.search = ''; // The update check must not stay tied to this release's cache token.
      const response = await fetch(manifestURL, { cache: 'no-store' });
      if (!response.ok) return;
      const manifest = await response.json();
      const freshness = publicSourceFreshness(manifest.public_sources);
      const sourceNotice = document.getElementById('source-freshness');
      sourceNotice.textContent = freshness?.message || ''; sourceNotice.hidden = !freshness?.message;
      const next = manifest.build_id;
      if (typeof next !== 'string' || !next || next === current || next === dismissedRelease) return;
      releaseNotice.dataset.build = next; releaseNotice.hidden = false; setReleaseState();
    } catch { /* An unavailable update check must not interrupt reading or brewing. */ }
  };
  document.getElementById('release-dismiss').onclick = () => { dismissedRelease = releaseNotice.dataset.build; releaseNotice.hidden = true; };
  releaseReload.onclick = async () => {
    releaseReload.disabled = true;
    if (dirtySetup) {
      document.getElementById('release-message').textContent = 'Save your unsaved settings, or leave Settings to discard them, before reloading.';
      setReleaseState();
      return;
    }
    if (!runtimeStarted) { location.reload(); return; }
    try { await settleRenderedEdits(frame); await client.quiesce(); await companionCall({ type: 'prepare-reload' }); location.reload(); }
    catch (error) { client.resume(); setReleaseState(); document.getElementById('release-message').textContent = error.message; }
  };
  document.addEventListener('visibilitychange', () => { if (!document.hidden) checkRelease(); });
  setInterval(checkRelease, 300000);
  document.getElementById('interruption-dismiss').onclick = () => { document.getElementById('interruption-notice').hidden = true; };

  const sendCompanionCapabilities = () => frame.contentWindow.postMessage({ type: 'companion-capabilities', token, backends: companionBackends }, location.origin);
  let navigation = 0;
  let activePath = '/';
  let computeTimer, latestCompute, computeReceivedAt, lastAnnouncement;
  const computeNotice = 'Brew runs locally on this device. Keep this tab open until it finishes. Change computation mode in Data & compute.';
  frame.addEventListener('load', () => {
    frame.contentWindow.postMessage({ type: 'compute-config', token, message: computeNotice }, location.origin);
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
    const about = frame.contentDocument?.querySelector('.about-brew');
    if (about && aboutDisclosureStates.has(activePath)) about.open = aboutDisclosureStates.get(activePath);
    if (scrollPositions.has(activePath)) frame.contentWindow.scrollTo(0, scrollPositions.get(activePath));
    if (frame.contentDocument?.title) document.title = frame.contentDocument.title + ' · DailyDigest';
    frame.contentDocument?.addEventListener('click', () => {
      for (const menu of document.querySelectorAll('.browser-bar > details[open]')) menu.open = false;
    });
    trackSetupDraft(frame.contentDocument?.querySelector('form[action="/setup"]'),
      value => { dirtySetup = value; }, {initiallyDirty:dirtySetup});
    const topics = frame.contentDocument?.getElementById('topics');
    if (draftSubmitted && topics && !topics.value.trim() && topicDraft.value.trim()) {
      topics.value = topicDraft.value;
      topics.dispatchEvent(new frame.contentWindow.Event('input', {bubbles:true}));
      dirtySetup = true;
      try {
        sessionStorage.removeItem(topicDraftKey);
        sessionStorage.removeItem(topicDraftKey + ':submitted');
      } catch { /* harmless retained draft */ }
      topicDraft.value = '';
      draftSubmitted = false;
    }
    void cacheReadingView();
  });
  const displayCompute = () => {
    if (!latestCompute) return;
    const info = { ...latestCompute };
    if (info.phase === 'encoding') info.elapsed_ms += performance.now() - computeReceivedAt;
    info.message = formatComputeProgress(info);
    const milestone = announcementKey(info);
    setStatus(info.message, milestone !== lastAnnouncement);
    lastAnnouncement = milestone;
    frame.contentWindow.postMessage({ type: 'compute-progress', token, compute: info }, location.origin);
  };
  const errorPanel = document.getElementById('app-error');
  let errorFocus, errorSettingsOpen = false;
  const dismissError = () => { errorPanel.hidden = true; if (errorSettingsOpen) dataSettings.showModal(); errorSettingsOpen = false; errorFocus?.focus(); };
  document.getElementById('reload-app').onclick = () => location.reload();
  document.getElementById('dismiss-error').onclick = dismissError;
  errorPanel.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); dismissError(); }
  });
  const report = (error) => {
    if (tabMoved) return;
    const message = error.message || String(error);
    if (!ready && /open in another tab/.test(message)) { showTabWait(message); return; }
    if (!ready || frame.hidden) {
      startup.hidden = false; frame.hidden = true; previewNotice.hidden = true;
      document.getElementById('backup-runtime-notice').hidden = true;
      startup.dataset.state = 'error'; document.getElementById('startup-title').textContent = 'Couldn’t open your reading desk';
      setStatus(message); document.getElementById('retry').hidden = false;
      document.getElementById('recovery-tools').hidden = false;
      document.getElementById('recovery-reset').hidden = ready;
      document.querySelector('.recovery-import-label').hidden = ready;
      return;
    }
    setStatus('Action needs attention');
    if (errorPanel.hidden) { errorFocus = document.activeElement; errorSettingsOpen = dataSettings.open; }
    if (dataSettings.open) dataSettings.close();
    document.getElementById('app-error-message').textContent = message;
    errorPanel.hidden = false;
    document.getElementById('dismiss-error').focus();
  };
  const showTabWait = message => {
    startup.hidden = false; frame.hidden = true; previewNotice.hidden = true;
    startup.dataset.state = tabMoved ? 'moved' : 'waiting';
    document.getElementById('startup-title').textContent = tabMoved ? 'Reading desk moved to another tab' : 'DailyDigest is open in another tab';
    document.getElementById('recovery-tools').hidden = true;
    document.getElementById('retry').hidden = tabMoved;
    tabActions.hidden = false; useThisTab.hidden = !handoff;
    setStatus(!handoff ? 'Switch to your other DailyDigest tab, or close it and Retry startup here.'
      : message || 'Switch to your other DailyDigest tab, or use this tab instead.');
  };
  const companionStatus = document.getElementById('companion-status');
  document.querySelector('.companion-command').textContent = `dd companion --allowed-origin ${location.origin}`;
  const companionEndpoint = document.getElementById('companion-endpoint');
  const companionToken = document.getElementById('companion-token');
  const guideKey = `dailydigest:guide:${location.pathname}`;
  const guide = initializeQuickGuide({document,
    hasSeen: () => { try { return localStorage.getItem(guideKey) === 'seen'; } catch { return false; } },
    remember: () => { try { localStorage.setItem(guideKey, 'seen'); } catch { /* optional guide history */ } },
    openSection:openDataSection, restore:() => document.getElementById('import').click()});
  const linkedBackup = createLinkedBackup(linkedBackupStore(indexedDB, `dailydigest:backup-file:${new URL('.', import.meta.url).pathname}`));
  const linkedStatus = document.getElementById('linked-backup-status');
  const automaticBackup = document.getElementById('automatic-backup');
  let linkedAuto = false;
  async function refreshLinkedBackup() {
    const info = await linkedBackup.inspect();
    const granted = info.permission === 'granted';
    linkedAuto = Boolean(info.auto);
    automaticBackup.checked = linkedAuto;
    document.getElementById('automatic-backup-label').hidden = !info.linked || !info.writable;
    automaticBackup.disabled = !granted || info.missing;
    document.getElementById('reconnect-backup').hidden = !info.linked || granted;
    document.getElementById('write-linked-backup').hidden = !info.writable || !granted || info.missing;
    document.getElementById('write-linked-backup').disabled = !ready || computeBusy;
    document.getElementById('restore-linked-backup').hidden = !info.exists || !granted;
    document.getElementById('forget-linked-backup').hidden = !info.linked;
    linkedStatus.textContent = !info.linked ? 'No backup file linked.'
      : !granted ? `${info.name}: allow file access to check your backup.`
      : info.missing ? `${info.name} could not be found. Choose the file again or restore another backup.`
      : !info.writable && info.exists ? `${info.name}: backup found, last modified ${new Date(info.modified).toLocaleString()}. Restore it to use this reading desk, or choose a new file for a separate copy.`
      : info.updatedAt ? `${info.name}: last copy saved ${new Date(info.updatedAt).toLocaleString()}. ${info.auto ? 'Automatic copies are on while this tab is open.' : 'Automatic copies are off.'}`
      : `${info.name} is linked. Save a copy now, or turn on automatic copies.`;
    return info;
  }
  async function saveLinkedBackup() {
    await settleRenderedEdits(frame);
    const response = await client.call({type:'export',includeCredentials:false});
    if (response.status >= 400) throw new Error(response.body);
    // A backup that cannot be restored should not replace the last usable file.
    if (response.body.length > 128 * 1024 * 1024) throw new Error('This reading desk exceeds the 128 MB restore limit. Download a backup and keep it; automatic copies are paused.');
    const result = await linkedBackup.write(response.body);
    try { localStorage.setItem(backupKey,result.updatedAt); } catch { /* copy is already on disk */ }
    showLastBackup(result.updatedAt);
    await refreshLinkedBackup();
  }
  const backupScheduler = createBackupScheduler({save:saveLinkedBackup,
    canSave:() => ready && linkedAuto && !computeBusy && !tabMoved && !client.paused && !restoringBackup,
    onError:error => {
      linkedAuto = false; automaticBackup.checked = false;
      // Preserve the reader's opt-in across a temporary permission failure.
      // The scheduler stays paused until they reconnect or explicitly retry.
      linkedStatus.textContent = `Automatic copies paused: ${error.message}`;
      document.getElementById('backup-reminder-text').textContent = 'Your folder backup needs attention. Your working reading desk is still in this browser profile.';
      document.getElementById('backup-reminder').hidden = false;
    }});
  const scheduleLinkedBackup = () => { if (linkedAuto && !tabMoved) backupScheduler.changed(); };
  const linkedSupported = typeof window.showSaveFilePicker === 'function';
  document.getElementById('link-backup').hidden = !linkedSupported;
  if (!linkedSupported) {
    linkedStatus.textContent = 'This browser cannot keep a link to a backup file. Use Download backup and Restore backup instead.';
    document.getElementById('linked-backup-help').textContent = 'You can save a downloaded backup in any folder and restore it in another browser.';
  } else {
    void linkedBackup.open().then(refreshLinkedBackup).catch(() => {
      linkedStatus.textContent = 'Could not reopen the linked backup file. Choose it again, or download a backup.';
    });
  }
  document.getElementById('link-backup').onclick = async () => {
    try {
      const handle = await window.showSaveFilePicker({id:'dailydigest-backup',startIn:'documents',
        suggestedName:'dailydigest-browser-backup.json',types:[{description:'DailyDigest backup',accept:{'application/json':['.json']}}]});
      backupScheduler.cancel();
      await linkedBackup.connect(handle);
      const info = await refreshLinkedBackup();
      if (!info.exists && ready && !computeBusy) await saveLinkedBackup();
    } catch (error) { if (error.name !== 'AbortError') linkedStatus.textContent = error.message; }
  };
  document.getElementById('reconnect-backup').onclick = async () => {
    try { await linkedBackup.requestAccess(); await refreshLinkedBackup(); scheduleLinkedBackup(); }
    catch (error) { linkedStatus.textContent = error.message; }
  };
  document.getElementById('write-linked-backup').onclick = async event => {
    event.currentTarget.disabled = true;
    try { await saveLinkedBackup(); }
    catch (error) { linkedStatus.textContent = error.message; }
    finally { document.getElementById('write-linked-backup').disabled = !ready || computeBusy; }
  };
  automaticBackup.onchange = async () => {
    try {
      await linkedBackup.setAutomatic(automaticBackup.checked);
      linkedAuto = automaticBackup.checked;
      if (linkedAuto) scheduleLinkedBackup(); else backupScheduler.cancel();
      await refreshLinkedBackup();
    } catch (error) { linkedStatus.textContent = error.message; }
  };
  document.getElementById('forget-linked-backup').onclick = async () => {
    try { backupScheduler.cancel(); await linkedBackup.disconnect(); await refreshLinkedBackup(); }
    catch (error) { linkedStatus.textContent = error.message; }
  };
  const browserMenus = [...document.querySelectorAll('.device-sections > details')];
  document.addEventListener('click', event => {
    if (event.target === dataSettings) dataSettings.close();
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
        for (const action of buttons) {
          const button = document.getElementById(`companion-${action}`);
          button.disabled = !configured || !capabilities[capability];
          button.hidden = !configured || !capabilities[capability];
        }
      }
      document.getElementById('sync-credentials').closest('label').hidden = !configured || !capabilities.state;
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
      select.disabled = false;
      if (!keepSelection) select.value = result.mode;
      select.querySelector('[value="gpu"]').disabled = !result.capabilities.gpu;
      const canEnableIsolation = 'serviceWorker' in navigator && isSecureContext && navigator.hardwareConcurrency > 1;
      select.querySelector('[value="multicore"]').disabled = !result.capabilities.multicore && !canEnableIsolation;
      document.getElementById('compute-support').textContent =
        `${result.gpu?.message || (result.capabilities.gpu ? 'WebGPU is exposed by this browser; hardware availability is checked when used.' : 'GPU acceleration is unavailable in this browser.')} ${result.capabilities.multicore ? `Up to ${result.capabilities.threads} CPU cores are available.` : canEnableIsolation ? 'Selecting multiple cores will try enabling browser isolation and reload once. If unavailable, computation stays on one core.' : 'Multiple CPU cores are unavailable on this site or browser.'}`;
      document.getElementById('compute-status').textContent = computeStatusText(result);
      computeBusy = result.busy; setReleaseState();
      document.getElementById('compute-apply').disabled = result.busy || computeApplying;
    } catch (error) { document.getElementById('compute-status').textContent = error.message; }
  }
  document.getElementById('compute-menu').addEventListener('toggle', event => {
    if (event.target.open && ready) refreshCompute({ keepSelection: true });
  });
  document.getElementById('test-gpu').onclick = async () => {
    const button = document.getElementById('test-gpu'), output = document.getElementById('gpu-test-result');
    button.disabled = true; output.textContent = 'Checking GPU adapter; no model is downloaded…';
    try {
      const result = await companionCall({ type: 'probe-gpu' });
      output.textContent = result.message + (result.info ? ` Vendor: ${result.info.vendor}; architecture: ${result.info.architecture}.` : '');
    } catch (error) { output.textContent = error.message; }
    finally { button.disabled = false; }
  };
  document.getElementById('compute-form').onsubmit = async event => {
    event.preventDefault();
    if (computeApplying) return;
    if (dirtySetup) {
      document.getElementById('compute-status').textContent = 'Save your unsaved settings, or leave Settings to discard them, before changing Compute.';
      return;
    }
    computeApplying = true;
    document.getElementById('compute-apply').disabled = true;
    document.getElementById('compute-status').textContent = 'Saving your compute choice before reloading…';
    let reloadPrepared = false;
    try {
      await settleRenderedEdits(frame);
      await client.quiesce();
      const mode = document.getElementById('compute-mode').value;
      const needsIsolation = mode === 'multicore' && !crossOriginIsolated;
      await companionCall({ type: 'configure-compute', mode, allowIsolationReload: needsIsolation });
      reloadPrepared = true;
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
      let recoveryError;
      if (reloadPrepared) {
        try { await companionCall({ type: 'cancel-reload' }); }
        catch (failed) { recoveryError = failed; }
      }
      computeApplying = false;
      client.resume();
      if (!recoveryError) await refreshCompute({ keepSelection: true });
      document.getElementById('compute-status').textContent = error.message
        + (recoveryError ? ' Reload preparation could not be cancelled. Reload the page to reconnect.' : '');
    }
  };
  const download = (body, type, filename) => {
    const url = URL.createObjectURL(new Blob([body], { type }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  async function navigate(path, options = {}) {
    if (tabMoved) return;
    if (dirtySetup && activePath.split('?')[0] === '/setup' && localPath(path).split('?')[0] !== '/setup'
        && (!options.method || options.method.toUpperCase() === 'GET')
        && !confirm('Discard unsaved settings changes?')) {
      history.replaceState(null, '', '#' + activePath);
      return;
    }
    if (!frame.hidden && frame.contentDocument) {
      scrollPositions.set(activePath, frame.contentWindow.scrollY);
      const about = frame.contentDocument.querySelector('.about-brew');
      if (about) aboutDisclosureStates.set(activePath, about.open);
    }
    if (typeof path === 'string' && path.startsWith('/done')) {
      scrollPositions.delete('/'); aboutDisclosureStates.delete('/');
    }
    const version = ++navigation;
    setStatus('Opening DailyDigest…');
    frame.inert = true; frame.setAttribute('aria-busy', 'true');
    try {
      await settleRenderedEdits(frame);
      if (version !== navigation) return;
      for (const stream of streams.values()) stream.cancel();
      streams.clear();
      path = localPath(path);
      let requestMethod = options.method?.toUpperCase() || 'GET';
      let response = await client.call({ type: 'request', path, method: 'GET', ...options });
      for (let redirects = 0; response.status >= 300 && response.status < 400; redirects++) {
        if (redirects >= 8) throw new Error('DailyDigest returned too many redirects.');
        const headers = new Headers(response.headers);
        if (!headers.has('location')) throw new Error('DailyDigest redirect is missing its destination.');
        path = localPath(headers.get('location'), path);
        requestMethod = 'GET';
        response = await client.call({ type: 'request', path, method: 'GET' });
      }
      if (version !== navigation) return;
      const headers = new Headers(response.headers);
      const setupValidation = response.status === 400 && requestMethod === 'POST'
        && path.split('?')[0] === '/setup' && (headers.get('content-type') || '').toLowerCase().split(';')[0].trim() === 'text/html'
        && /<form\b[^>]*\baction=["']\/setup["'][^>]*>/i.test(response.body);
      if (response.status >= 400 && !setupValidation) throw new Error(httpErrorMessage(response));
      if ((headers.get('content-type') || '').includes('text/calendar')) {
        download(response.body, 'text/calendar', 'dailydigest.ics');
        history.replaceState(null, '', '#' + activePath);
        if (frame.hidden) await navigate(activePath);
        else setStatus('Calendar exported.');
        return;
      }
      if (!(headers.get('content-type') || '').includes('text/html')) throw new Error('Expected a DailyDigest page.');
      activePath = path;
      if (path.split('?')[0] === '/done') {
        firstBrewBackupDue = true;
        try { remindBackup(localStorage.getItem(backupKey)); } catch { /* optional reminder */ }
      }
      dirtySetup = path.split('?')[0] === '/setup' && options.method?.toUpperCase() === 'POST';
      const generation = headers.get('x-dailydigest-generation');
      frameGeneration = generation === null ? null : Number(generation);
      clearInterval(computeTimer); latestCompute = null;
      history.replaceState(null, '', '#' + path);
      token = crypto.randomUUID();
      frame.setAttribute('sandbox', interactiveSandbox);
      frame.srcdoc = renderedDocument(response.body, token, path, location.href);
      frame.hidden = false; startup.hidden = true;
      previewNotice.hidden = true;
      document.getElementById('backup-runtime-notice').hidden = true;
      startup.dataset.state = 'ready';
      setStatus(setupValidation ? 'Settings need correction · changes have not been saved.'
        : headers.get('x-dailydigest-storage') === 'pending' ? 'Saving recent changes…' : 'Saved in this profile · Backup');
      if (response.status < 400 && options.method && options.method.toUpperCase() !== 'GET') scheduleLinkedBackup();
    } catch (error) { if (version === navigation) report(error); }
    finally { if (version === navigation) { frame.inert = false; frame.removeAttribute('aria-busy'); } }
  }
  worker.addEventListener('message', ({ data }) => {
    if (data.type === 'ready') {
      ready = true;
      tabActions.hidden = true;
      document.getElementById('retry').hidden = true;
      document.getElementById('test-gpu').disabled = false;
      checkRelease();
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
      if (!topicDraftForm.hidden && topicDraft.value.trim() && !draftSubmitted) {
        document.getElementById('startup-topics-message').textContent = 'Ready. Continue to review your topics before saving.';
        document.getElementById('startup-title').textContent = 'Your reading desk is ready';
      } else if (!restoringBackup) navigate(draftSubmitted ? '/setup' : routeFromHash(location.hash));
      if (linkedSupported) void refreshLinkedBackup().catch(() => {});
    } else if (data.type === 'storage-waiting') {
      showTabWait();
    } else if (data.type === 'brew-interrupted') {
      document.getElementById('interruption-message').textContent = data.message;
      document.getElementById('interruption-notice').hidden = false;
    } else if (data.type === 'startup-timings') {
      // Public timing measurements only: no profile, content or saved-state details.
      document.getElementById('startup').dataset.durationMs = String(Math.round(data.totalMs));
    } else if (data.type === 'compute-state') {
      computeBusy = data.busy; setReleaseState();
      if (data.busy) document.getElementById('compute-apply').disabled = true;
      else scheduleLinkedBackup();
      document.getElementById('write-linked-backup').disabled = !ready || computeBusy;
      refreshCompute({ keepSelection: true });
    } else if (data.type === 'progress') {
      if (!ready) updateStartupStages(data.stage);
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
      if (storageWarning) setStatus('Storage needs attention. Open Data & compute, then Backup to save your changes.');
    }
    else if (data.type === 'error' && !data.id) report(new Error(data.message || data.error || 'Python startup failed'));
  });
  worker.addEventListener('error', () => report(new Error('Python worker stopped. Reload to reconnect.')));
  const flushWhenHidden = () => { if (ready && !tabMoved && !client.paused) void client.call({type:'flush-state'}).catch(() => {}); };
  document.addEventListener('visibilitychange', () => { if (document.hidden) flushWhenHidden(); });
  window.addEventListener('pagehide', flushWhenHidden);
  window.addEventListener('beforeunload', event => {
    if (!tabMoved && (hasUnflushedEdits || dirtySetup || computeBusy || (frame.contentDocument && frame.contentWindow?.ddHasPendingRatings?.()) || [...client.inFlight.values()].some(request => request.mutating))) {
      event.preventDefault(); event.returnValue = ''; flushWhenHidden();
    }
  });
  worker.addEventListener('message', ({data}) => { if (data.type === 'storage-dirty') hasUnflushedEdits = true; });
  worker.addEventListener('message', ({data}) => { if (data.type === 'storage-checkpoint') {
    hasUnflushedEdits = false;
    void cacheReadingView();
    if (data.generation === frameGeneration) {
      setStatus('Saved in this profile · Backup');
      frame.contentWindow?.postMessage({type:'storage-committed', token}, location.origin);
    }
  } });
  window.addEventListener('hashchange', () => {
    if (ready) navigate(routeFromHash(location.hash));
    else startRuntime();
  });
  window.addEventListener('message', async ({ source, origin: senderOrigin, data }) => {
    if (source !== frame.contentWindow || senderOrigin !== location.origin || data?.token !== token) return;
    if (data.type === 'compute-stop') { clearInterval(computeTimer); latestCompute = null; setStatus('Brew needs attention'); return; }
    const reply = (payload) => source.postMessage({ ...payload, token: data.token, id: data.id }, location.origin);
    try {
      if (data.type === 'navigate') {
        const path = localPath(data.path, activePath);
        if (/^\/calendar\/[^/]+\.ics$/.test(path) || location.hash === '#' + path) navigate(path);
        else location.hash = path;
      }
      else if (data.type === 'replace-route') history.replaceState(null, '', '#' + localPath(data.path));
      else if (data.type === 'open-guide') guide.open();
      else if (data.type === 'open-backup') openDataSection('backup');
      else if (data.type === 'submit') await navigate(data.path, { method: data.method, body: data.body, headers: data.headers });
      else if (data.type === 'cancel') { streams.get(data.id)?.cancel(); streams.delete(data.id); }
      else if (data.type === 'request') {
        const request = client.call({ type: 'request', path: localPath(data.path), method: data.method, headers: data.headers, body: data.body }, body => reply({ type: 'stream', body }));
        streams.set(data.id, request);
        try {
          const response = await request;
          const generation = new Headers(response.headers).get('x-dailydigest-generation');
          if (generation !== null) frameGeneration = Number(generation);
          reply(response);
          if (response.status < 400 && data.method && data.method.toUpperCase() !== 'GET') scheduleLinkedBackup();
          clearTimeout(cacheTimer); cacheTimer = setTimeout(() => { void cacheReadingView(); }, 100);
        } finally { streams.delete(data.id); }
      }
    } catch (error) {
      reply({ error: error.message });
      if (error.message !== 'Request disconnected') report(error);
    }
  });
  document.getElementById('retry').onclick = () => location.reload();
  async function confirmRestore(message, {fresh = false} = {}) {
    const dialog = document.getElementById('restore-confirmation');
    if (dialog.open) return false;
    document.getElementById('restore-title').textContent = fresh ? 'Start fresh?' : 'Restore saved data?';
    document.getElementById('restore-proceed').textContent = fresh ? 'Start fresh' : 'Restore';
    document.getElementById('restore-description').textContent = message;
    return new Promise(resolve => {
      let accepted = false;
      document.getElementById('restore-proceed').onclick = () => { accepted = true; dialog.close(); };
      document.getElementById('restore-cancel').onclick = () => dialog.close();
      dialog.addEventListener('close', () => resolve(accepted), {once:true});
      dialog.showModal();
      document.getElementById('restore-cancel').focus();
    });
  }
  function restoreProgress(message) {
    for (const id of ['restore-progress','backup-restore-progress']) {
      const progress = document.getElementById(id);
      progress.textContent = message; progress.hidden = !message;
    }
  }
  async function pauseLinkedCopies() {
    const info = await linkedBackup.inspect();
    if (info.linked && info.auto) await linkedBackup.setAutomatic(false);
    linkedAuto = false; automaticBackup.checked = false; backupScheduler.cancel();
  }
  async function replaceWithCopiesPaused(action) {
    const previous = await linkedBackup.inspect();
    await pauseLinkedCopies();
    try { return await action(); }
    catch (error) {
      if (previous.auto) {
        try { await linkedBackup.setAutomatic(true); await refreshLinkedBackup(); scheduleLinkedBackup(); }
        catch { linkedStatus.textContent = 'Restore failed. Check your automatic-copy preference before continuing.'; }
      }
      throw error;
    }
  }
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
      if (file.size > 128 * 1024 * 1024) throw new Error('Backup exceeds the 128 MB restore limit.');
      if (!await confirmRestore('Restore this backup? It replaces this browser’s current data.')) return;
      restoreProgress('Checking and restoring your backup… Keep this tab open.');
      const body = await backupFileText(file);
      await replaceWithCopiesPaused(() => companionCall({ type: 'recovery-import', body, overwrite: true }));
      rememberImportNotice({ credentialsOmitted: JSON.parse(body).credentialsIncluded === false });
      location.reload();
    } catch (error) { restoreProgress(''); report(new Error(`Recovery import failed: ${error.message}`)); }
    finally { event.target.value = ''; }
  };
  document.getElementById('recovery-reset').onclick = async () => {
    if (!await confirmRestore('Start with empty browser data? Download your original data first if you need a separate copy.', {fresh:true})) return;
    try { await replaceWithCopiesPaused(() => companionCall({ type: 'recovery-reset', overwrite: true })); location.reload(); }
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
      await settleRenderedEdits(frame);
      const response = await client.call({ type: 'export', includeCredentials });
      if (response.status >= 400) throw new Error(response.body);
      download(response.body, 'application/json', 'dailydigest-browser-backup.json');
      const exported = new Date().toISOString();
      try { localStorage.setItem(backupKey, exported); } catch { /* export remains usable */ }
      showLastBackup(exported);
      setStatus('Browser backup exported.');
    } catch (error) { report(error); }
  };
  async function restoreBackup(body, revision) {
    if (restoringBackup) throw new Error('Wait for the current restore to finish.');
    JSON.parse(body);
    if (!await confirmRestore('Replace this browser’s interests, ratings, saved papers and reading history with this backup? Changes from different browsers are not merged.')) return;
    restoringBackup = true; backupScheduler.cancel();
    try {
      restoreProgress('Checking and restoring your backup… Keep this tab open.');
      const recovery = startup.dataset.state === 'error' && !ready;
      if (!recovery) startRuntime();
      const apply = async () => {
        const response = await client.call({type:recovery ? 'recovery-import' : 'import',body,overwrite:true});
        if (response.status >= 400) throw new Error(response.body);
        return response;
      };
      const response = await (revision ? apply() : replaceWithCopiesPaused(apply));
      if (revision) await linkedBackup.accept(revision).catch(() => {});
      rememberImportNotice(recovery ? {credentialsOmitted:JSON.parse(body).credentialsIncluded === false} : JSON.parse(response.body));
      location.reload();
    } catch (error) { restoringBackup = false; restoreProgress(''); throw error; }
  }
  document.getElementById('restore-linked-backup').onclick = async () => {
    try { const copy = await linkedBackup.read(); await restoreBackup(copy.body,copy.revision); }
    catch (error) { linkedStatus.textContent = `Could not restore this copy: ${error.message}`; }
  };
  document.getElementById('import').onchange = async (event) => {
    const file = event.target.files[0]; if (!file) return;
    try {
      if (file.size > 128 * 1024 * 1024) throw new Error('Backup exceeds the 128 MB restore limit.');
      await restoreBackup(await backupFileText(file));
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
  document.getElementById('copy-companion-command').onclick = async () => {
    try { await navigator.clipboard.writeText(document.querySelector('.companion-command').textContent); companionStatus.textContent = 'Command copied. Run it in a terminal on your computer.'; }
    catch { companionStatus.textContent = 'Select and copy the command above, then run it in a terminal.'; }
  };
  document.querySelector('.companion-command').textContent = `dd companion --allowed-origin ${location.origin}`;
  document.getElementById('shared-origin-warning').hidden = !location.hostname.endsWith('.github.io');
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
      if (action === 'pull' && !await confirmRestore('Replace this browser’s data with Companion state? Changes are not merged.')) return;
      button.disabled = true;
      companionStatus.textContent = 'Running requested companion action…';
      try {
        if (action === 'push') await settleRenderedEdits(frame);
        if (action === 'pull') restoreProgress('Restoring Companion data… Keep this tab open.');
        const apply = () => companionCall({ type: `companion-${action}`, ...(action === 'pull' ? { overwrite: true } : {}), ...(action === 'push' ? { includeCredentials } : {}) });
        const result = await (action === 'pull' ? replaceWithCopiesPaused(apply) : apply());
        if (action === 'pull' && result.reload) { rememberImportNotice(result); location.reload(); return; }
        if (action === 'inbound') {
          const up = Number(result.votes_up || 0);
          const down = Number(result.votes_down || 0);
          const unresolved = Number(result.unresolved_messages || 0);
          companionStatus.textContent = `Email feedback processed: ${up} upvotes, ${down} downvotes.${unresolved ? ` ${unresolved} messages need review.` : ''}${result.errors?.length ? ' ' + result.errors.join(' ') : ''}`;
          if (up + down > 0) await navigate(activePath);
        } else companionStatus.textContent = action === 'email' ? 'Latest digest sent.' : action === 'push' ? 'Browser state sent.' : 'Companion state restored. Reload to open it.';
      } catch (error) { restoreProgress(''); companionStatus.textContent = error.message; }
      finally { button.disabled = false; }
    };
  }
  let tabChannel;
  try {
    if (typeof BroadcastChannel !== 'undefined') tabChannel = new BroadcastChannel(`dailydigest:tabs:${new URL('.', import.meta.url).pathname}`);
  } catch { /* Browser privacy restrictions may disable cross-tab messaging. */ }
  if (tabChannel) {
    const channel = tabChannel;
    handoff = createTabHandoff({channel, id:crypto.randomUUID(), owns:() => ready && !tabMoved,
      release:async () => {
        if (dirtySetup) throw new Error('Save the unsaved settings in the other tab before switching, or leave its Settings page.');
        if (computeBusy) throw new Error('Finish or stop the brew in the other tab, then try switching again.');
        frame.inert = true; toolbar.inert = true; dataSettings.inert = true;
        try {
          setStatus('Saving changes before switching tabs…');
          await settleRenderedEdits(frame);
          await client.quiesce();
          await companionCall({type:'prepare-reload'});
        } catch (error) {
          client.resume(); frame.inert = false; toolbar.inert = false; dataSettings.inert = false;
          setStatus('This reading desk stays open. ' + error.message);
          throw error;
        }
        tabMoved = true; ready = false; clearTimeout(cacheTimer); backupScheduler.cancel();
        worker.terminate();
        client.fail(new Error('Reading desk moved to another tab.'));
        for (const menu of browserMenus) menu.open = false;
        dataSettings.close();
        errorPanel.hidden = true; releaseNotice.hidden = true;
        document.getElementById('backup-runtime-notice').hidden = true;
        showTabWait('Your latest changes were saved before moving. Use this tab instead to switch back.');
        useThisTab.textContent = 'Use this tab instead';
      }});
  }
  useThisTab.onclick = async () => {
    if (!handoff || ready) return;
    useThisTab.disabled = true;
    // A waiting worker owns no editable desk. Remove its queued lock request;
    // the replacement page must restore the previous owner's final checkpoint.
    if (!tabMoved) { worker.terminate(); client.fail(new Error('Switching tabs.')); }
    setStatus('Asking the other tab to save and switch…');
    try {
      const result = await handoff.request();
      const locks = await navigator.locks.query();
      const name = `dailydigest:${new URL('.', import.meta.url).pathname}`;
      if (result.ok || !locks.held.some(lock => lock.name === name)) {
        try { sessionStorage.setItem(fullDeskKey, 'true'); } catch { /* deep links still start normally */ }
        location.reload();
      } else setStatus(result.message);
    } catch (error) { setStatus(error.message); }
    finally { useThisTab.disabled = false; }
  };
  document.getElementById('reading-view-open').onclick = () => startRuntime(readingViewRoute(preview));
  document.getElementById('reading-view-brew').onclick = () => startRuntime('/run');
  dataControlsReady = true;
  if (pendingDataSection) openDataSection(pendingDataSection);
  // Read one small derived record, never the full database or vector cache.
  // Unavailable browser storage falls back to the existing startup/recovery path.
  let preview;
  if (routeFromHash(location.hash) === '/' && !forceFullDesk) {
    preview = await Promise.race([
      readReadingView(indexedDB, `dailydigest:${new URL('.', import.meta.url).pathname}`).catch(() => null),
      new Promise(resolve => setTimeout(() => resolve(null), 800)),
    ]);
  }
  if (preview && !runtimeStarted) {
    guide.showWelcome(true);
    frame.setAttribute('sandbox', 'allow-popups allow-popups-to-escape-sandbox');
    document.title = `Saved DailyDigest ${preview.date}`;
    frame.srcdoc = readingViewDocument(preview, document.documentElement.dataset.theme, import.meta.url, new Date(), SHARED_READING_STYLES);
    frame.hidden = false; startup.hidden = true;
    previewNotice.hidden = false;
    cachedPip = mountCachedReadingPip(frame, preview);
    cachedPip.ready.catch(() => {
      if (!runtimeStarted) document.getElementById('reading-view-status').textContent += ' Pip could not load; open controls to retry.';
    });
    startup.dataset.state = 'reading';
    document.getElementById('reading-view-status').textContent = `Saved digest · ${preview.date}`;
    setStatus('Ready to read');
    document.getElementById('compute-support').textContent = 'Open controls to choose acceleration.';
    document.getElementById('compute-status').textContent = 'Computation is paused while you read the saved digest.';
    companionStatus.textContent = 'Open controls to connect a companion.';
    document.getElementById('backup-runtime-notice').hidden = false;
    checkRelease();
  } else {
    // Check only key presence, never deserialize the full private snapshot.
    hasSavedDesk(indexedDB, `dailydigest:${new URL('.', import.meta.url).pathname}`).then(present => {
      guide.showWelcome(present);
      if (!present && !ready && !startupFailedForDraft()) topicDraftForm.hidden = false;
    }).catch(() => {});
    startRuntime();
  }
  function startupFailedForDraft() { return ['error', 'waiting', 'moved'].includes(startup.dataset.state); }
}

if (typeof document !== 'undefined') boot().catch(error => {
  document.getElementById('startup').dataset.state = 'error';
  document.getElementById('startup-title').textContent = 'Couldn’t open your reading desk';
  document.getElementById('startup-status').textContent = error.message;
  document.getElementById('status').textContent = error.message;
  document.getElementById('retry').hidden = Boolean(browserSupportError());
  document.getElementById('retry').onclick = () => location.reload();
});
