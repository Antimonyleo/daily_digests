/** A small, disposable reading copy; SQLite snapshots remain authoritative. */
export class DeferredWorker extends EventTarget {
  constructor(create) { super(); this.create = create; this.instance = null; }
  postMessage(data) {
    if (this.terminated) throw new Error('This reading desk moved to another tab. Reload to reconnect.');
    if (!this.instance) {
      this.instance = this.create();
      this.instance.addEventListener('message', event => this.dispatchEvent(new MessageEvent('message', {data:event.data})));
      this.instance.addEventListener('error', () => this.dispatchEvent(new Event('error')));
    }
    this.instance.postMessage(data);
  }
  terminate() { this.terminated = true; this.instance?.terminate(); }
}

export function validateReadingView(value) {
  if (!value || value.version !== 1 || typeof value.date !== 'string' ||
      !/^\d{4}-\d{2}-\d{2}$/.test(value.date) || typeof value.savedAt !== 'string' ||
      !Number.isFinite(Date.parse(value.savedAt)) || !Array.isArray(value.items) ||
      !value.items.length || value.items.length > 500 ||
      Object.keys(value).some(key => !['version','date','savedAt','items'].includes(key))) return null;
  const limits = {section:128, title:2000, url:8000, meta:2000, summary:30000, reason:20000, opportunity:20000};
  for (const item of value.items) {
    if (!item || Object.keys(item).length !== Object.keys(limits).length ||
        Object.entries(limits).some(([key, limit]) => typeof item[key] !== 'string' || item[key].length > limit) ||
        !item.title.trim()) return null;
    try {
      const url = new URL(item.url);
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return null;
    } catch { return null; }
  }
  return JSON.stringify(value).length <= 1000000 ? value : null;
}

export function extractReadingView(document, now = new Date()) {
  const text = (node, selector) => node.querySelector(selector)?.textContent.trim() || '';
  const value = {version:1, date:text(document, '.digest-date'), savedAt:now.toISOString(),
    items:Array.from(document.querySelectorAll('.item[data-section]'), node => ({
      section:text(node.closest('.section-block') || node, '.section-heading') || node.dataset.section,
      title:text(node, '.title-link'), url:node.querySelector('.title-link')?.getAttribute('href') || '',
      meta:text(node, '.meta'), summary:text(node, '.summary-primary .summary-text, .summary'),
      reason:text(node, '.match-topic, .reason-line'), opportunity:text(node, '.opportunity-facts'),
    }))};
  return validateReadingView(value);
}

export async function readReadingView(indexedDB, databaseName) {
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open(databaseName, 1);
    request.onupgradeneeded = () => request.result.createObjectStore('state');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction('state');
      const request = transaction.objectStore('state').get('reading-view');
      request.onsuccess = () => resolve(validateReadingView(request.result));
      request.onerror = () => reject(request.error);
    });
  } finally { db.close(); }
}

export function readingViewDocument(value, theme = 'light') {
  if (!validateReadingView(value)) throw new Error('Invalid saved reading view');
  const escape = text => text.replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  let section = '';
  const items = value.items.map(item => {
    const heading = section !== item.section ? `<h2>${escape(item.section)}</h2>` : '';
    section = item.section;
    return heading + `<article><h3><a href="${escape(item.url)}" target="_blank" rel="noopener noreferrer">${escape(item.title)}</a></h3><p class="meta">${escape(item.meta)}</p>`
      + (item.opportunity ? '<p class="meta">Saved opportunity details. Verify current deadlines and eligibility at the official source.</p>' : '')
      + [item.opportunity, item.reason, item.summary].filter(Boolean).map(text => `<p>${escape(text)}</p>`).join('') + '</article>';
  }).join('');
  const savedAt = new Intl.DateTimeFormat(undefined, {dateStyle:'medium',timeStyle:'short'}).format(new Date(value.savedAt));
  return `<!doctype html><html lang="en" data-theme="${theme === 'dark' ? 'dark' : 'light'}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; form-action 'none'; base-uri 'none'"><title>Saved DailyDigest ${escape(value.date)}</title><style>
  *{box-sizing:border-box}body{margin:0;background:#f6f7f4;color:#15221b;font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:960px;margin:auto;padding:28px 20px 48px}h1{font-size:30px;line-height:1.2;margin:4px 0}h2{font-size:18px;margin-top:28px}h3{font-size:18px;line-height:1.4;margin:0}article{background:#fff;border:1px solid #dde3dc;border-radius:12px;padding:20px;margin:12px 0}a{color:#256f52;text-decoration:none}a:hover{text-decoration:underline}a:focus-visible{outline:2px solid;outline-offset:3px}p{margin:9px 0 0;white-space:pre-line;overflow-wrap:anywhere}.meta{color:#657169;font-size:13px}.eyebrow{color:#256f52;font-size:12px;letter-spacing:.09em;text-transform:uppercase}[data-theme=dark] body{background:#0e1511;color:#edf4ef}[data-theme=dark] article{background:#151f19;border-color:#314038}[data-theme=dark] :is(a,.eyebrow){color:#92d2ac}[data-theme=dark] .meta{color:#afc1b3}
  </style></head><body><main><p class="eyebrow">DailyDigest · saved reading view</p><h1>Your reading desk</h1><p class="meta">Digest ${escape(value.date)} · saved ${escape(savedAt)}</p>${items}</main></body></html>`;
}
