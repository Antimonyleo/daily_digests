const LOCAL_ORIGIN = 'http://127.0.0.1:8765';

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

export function browserSupportError(environment = globalThis) {
  if (!environment.WebAssembly?.Suspending || !environment.WebAssembly?.promising || !environment.navigator?.locks) {
    return 'This browser is missing features DailyDigest needs. Try a current desktop Chrome browser.';
  }
  return null;
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
function frameBridge(token, path, origin) {
  let sequence = 0;
  const pending = new Map();
  const notify = (payload) => parent.postMessage({ ...payload, token }, '*');
  const route = (value) => {
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
  document.addEventListener('click', (event) => {
    const anchor = event.target.closest?.('a[href]');
    if (!anchor || event.defaultPrevented || event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    const href = anchor.getAttribute('href');
    if (href.startsWith('#')) return;
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

export function renderedDocument(html, token, path) {
  // Templates use location for a small number of navigation operations. Route
  // these through the outer hash router; the Pages subpath never becomes an API.
  html = html.replace(/window\.location\.href\s*=\s*([^;]+);/g, 'window.ddNavigate($1);')
    .replace(/window\.location\.reload\(\)/g, 'window.ddReload()')
    .replace(/window\.location\.href/g, 'window.ddLocation')
    .replace(/window\.history\.replaceState/g, 'window.ddReplaceState')
    .replace(/navigator\.serviceWorker\.register\("\/sw\.js"\)/g, 'Promise.reject(new Error("Browser worker owns runtime"))')
    .replace(/<link[^>]+href="\/(?:manifest\.webmanifest|app-icon\.svg)"[^>]*>/g, '');
  const args = [token, path, LOCAL_ORIGIN].map(value => JSON.stringify(value).replace(/</g, '\\u003c')).join(',');
  const bridge = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; form-action 'none'; base-uri 'none'"><script>(${frameBridge.toString()})(${args});</script>`;
  return html.replace(/<head[^>]*>/i, match => match + bridge);
}

async function boot() {
  const supportError = browserSupportError();
  if (supportError) throw new Error(supportError);
  const frame = document.getElementById('app');
  const startup = document.getElementById('startup');
  const status = document.getElementById('status');
  const startupStatus = document.getElementById('startup-status');
  const setStatus = (message) => { status.textContent = message; startupStatus.textContent = message; };
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
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register(new URL('./sw.js', import.meta.url), { scope: new URL('.', import.meta.url).pathname })
      .catch(error => { status.textContent = `App shortcut unavailable: ${error.message}`; });
  }
  let token = crypto.randomUUID();
  const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
  const client = new WorkerClient(worker);
  const streams = new Map();
  let ready = false;
  let navigation = 0;
  let activePath = '/';
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
    if (!ready) { setStatus(message); document.getElementById('retry').hidden = false; return; }
    setStatus('Action needs attention');
    if (errorPanel.hidden) errorFocus = document.activeElement;
    document.getElementById('app-error-message').textContent = message;
    errorPanel.hidden = false;
    document.getElementById('dismiss-error').focus();
  };
  const companionStatus = document.getElementById('companion-status');
  const companionEndpoint = document.getElementById('companion-endpoint');
  const companionToken = document.getElementById('companion-token');
  const browserMenus = [...document.querySelectorAll('.browser-bar > details')];
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
    try { result = JSON.parse(response.body); } catch { throw new Error('Companion returned an unreadable response.'); }
    if (response.status >= 400) throw new Error(result.detail || result.message || `Companion HTTP ${response.status}`);
    return result;
  }
  async function refreshCompanion() {
    try {
      const result = await companionCall({ type: 'companion-status' });
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
      history.replaceState(null, '', '#' + path);
      token = crypto.randomUUID();
      frame.srcdoc = renderedDocument(response.body, token, path);
      frame.hidden = false; startup.hidden = true;
      setStatus('Python ready · data stays in this browser');
    } catch (error) { if (version === navigation) report(error); }
  }
  worker.addEventListener('message', ({ data }) => {
    if (data.type === 'ready') {
      ready = true;
      document.getElementById('export').disabled = false;
      document.getElementById('import').disabled = false;
      document.getElementById('companion-connect').disabled = false;
      refreshCompanion();
      navigate(routeFromHash(location.hash));
    } else if (data.type === 'progress') setStatus(data.message || [data.stage, data.detail].filter(Boolean).join(' · ') || 'Loading Python…');
    else if (data.type === 'error' && !data.id) report(new Error(data.message || data.error || 'Python startup failed'));
  });
  worker.addEventListener('error', () => report(new Error('Python worker stopped. Reload to reconnect.')));
  window.addEventListener('hashchange', () => { if (ready) navigate(routeFromHash(location.hash)); });
  window.addEventListener('message', async ({ source, data }) => {
    if (source !== frame.contentWindow || data?.token !== token) return;
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
  document.getElementById('export').onclick = async () => {
    try {
      const response = await client.call({ type: 'export' });
      if (response.status >= 400) throw new Error(response.body);
      download(response.body, 'application/json', 'dailydigest-browser-backup.json');
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
      if (action === 'push' && !confirm('Send this browser’s profile, feedback and reading history to your companion?')) return;
      if (action === 'pull' && !confirm('Replace this browser’s DailyDigest profile, feedback and reading history with companion state? Export a browser backup first if you need to keep it.')) return;
      button.disabled = true;
      companionStatus.textContent = 'Running requested companion action…';
      try {
        const result = await companionCall({ type: `companion-${action}`, ...(action === 'pull' ? { overwrite: true } : {}) });
        if (action === 'pull' && result.reload) { location.reload(); return; }
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
  document.getElementById('startup-status').textContent = error.message;
  document.getElementById('status').textContent = error.message;
  document.getElementById('retry').hidden = false;
  document.getElementById('retry').onclick = () => location.reload();
});
