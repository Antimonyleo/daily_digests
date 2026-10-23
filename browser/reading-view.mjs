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

function safeArticleURL(value) {
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password;
  } catch { return false; }
}

export function validateReadingView(value) {
  // Earlier copies omitted evidence, personalization or publication companions.
  // Regenerate this disposable copy from the authoritative saved desk.
  if (!value || value.version !== 5 || typeof value.date !== 'string' ||
      !/^\d{4}-\d{2}-\d{2}$/.test(value.date) || typeof value.savedAt !== 'string' ||
      !Number.isFinite(Date.parse(value.savedAt)) || !Array.isArray(value.items) ||
      value.items.length > 500 ||
      typeof value.profileName !== 'string' || value.profileName.length > 200 ||
      !Array.isArray(value.notes) || value.notes.length > 32 ||
      value.notes.some(note => typeof note !== 'string' || note.length > 2000) ||
      Object.keys(value).some(key => !['version','date','savedAt','items','related','profileName','notes'].includes(key))) return null;
  const limits = {section:128, title:2000, url:8000, meta:2000, summary:30000, reason:20000, opportunity:20000, uncertainty:20000, details:60000, why:30000, caveat:30000};
  const figureFields = {figureURL:8000, figureLabel:40};
  for (const item of value.items) {
    if (!item || Object.keys(item).some(key=>!Object.hasOwn(limits,key) && !Object.hasOwn(figureFields,key) && key !== 'companions') ||
        Object.entries(figureFields).some(([key, limit])=>key in item && (typeof item[key] !== 'string' || item[key].length > limit)) ||
        Object.entries(limits).some(([key, limit]) => typeof item[key] !== 'string' || item[key].length > limit) ||
        !item.title.trim()) return null;
    if (item.companions !== undefined && (!Array.isArray(item.companions) || item.companions.length > 50 ||
        item.companions.some(link => !link || Object.keys(link).some(key => !['title','url','publicationLabel'].includes(key)) ||
          typeof link.title !== 'string' || !link.title.trim() || link.title.length > 2000 ||
          typeof link.publicationLabel !== 'string' || !link.publicationLabel.trim() || link.publicationLabel.length > 80 ||
          typeof link.url !== 'string' || link.url.length > 8000 || !safeArticleURL(link.url)))) return null;
    try {
      const url = new URL(item.url);
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return null;
      if (item.figureURL) {
        const figure = new URL(item.figureURL);
        if (figure.protocol !== 'https:' || figure.username || figure.password ||
            !['Graphical abstract','Article preview'].includes(item.figureLabel)) return null;
      }
    } catch { return null; }
  }
  if (value.related !== undefined) {
    const related = value.related;
    if (!related || Object.keys(related).some(key => !['heading','context','links','total'].includes(key)) ||
        typeof related.heading !== 'string' || !related.heading.trim() || related.heading.length > 500 ||
        typeof related.context !== 'string' || related.context.length > 10000 ||
        !Array.isArray(related.links) || !related.links.length || related.links.length > 50 ||
        ('total' in related && (!Number.isSafeInteger(related.total) || related.total < related.links.length || related.total > 1000000)) ||
        related.links.some(link => !link || Object.keys(link).some(key => !['title','url'].includes(key)) ||
          typeof link.title !== 'string' || !link.title.trim() || link.title.length > 2000 ||
          typeof link.url !== 'string' || link.url.length > 8000 || !safeArticleURL(link.url))) return null;
  }
  if (!value.items.length && !value.related) return null;
  return JSON.stringify(value).length <= 1000000 ? value : null;
}

/** Open controls for the edition the reader is already viewing. */
export function readingViewRoute(value) {
  const view = validateReadingView(value);
  return view ? '/?digest_id=' + encodeURIComponent(view.date) : '/';
}

export function extractReadingView(document, now = new Date()) {
  const text = (node, selector) => node.querySelector(selector)?.textContent.trim() || '';
  const summaryField = (node, label) => {
    const field = Array.from(node.querySelectorAll?.('.summary-field') || []).find(field =>
      text(field, '.summary-label').replace(/:$/, '').toLowerCase() === label.toLowerCase());
    return field ? text(field, '.summary-text') : '';
  };
  const detailsText = node => {
    const body = node.querySelector('.item-details-body');
    if (!body) return '';
    if (body.cloneNode) {
      // textContent retains collapsed evidence; structural separators keep
      // nested labels/paragraphs readable without copying executable markup.
      const copy = body.cloneNode(true);
      // Promoted context is rendered once beside the finding. Preserve every
      // other evidence field and its attribution in the complete disclosure.
      for (const field of copy.querySelectorAll('.summary-field')) {
        const label = text(field, '.summary-label').replace(/:$/, '');
        if (['Why read', 'Caveat'].includes(label) && summaryField(node, label)) field.remove();
      }
      for (const block of copy.querySelectorAll('div, p, blockquote, summary, li, br')) {
        block.before('\n\n'); block.after('\n\n');
      }
      return copy.textContent.split('\n').map(line => line.trim()).join('\n').trim().replace(/\n{3,}/g, '\n\n');
    }
    return body.children?.length
      ? Array.from(body.children, child => child.textContent.trim()).filter(Boolean).join('\n\n')
      : body.textContent.trim();
  };
  let personalization;
  try { personalization = JSON.parse(text(document, '#reading-personalization')); }
  catch { return null; }
  const value = {version:5, date:text(document, '.digest-date'), savedAt:now.toISOString(),
    profileName:personalization?.name, notes:personalization?.notes,
    items:Array.from(document.querySelectorAll('.item[data-section]'), node => ({
      section:text(node.closest('.section-block') || node, '.section-heading') || node.dataset.section,
      title:text(node, '.title-link'), url:node.querySelector('.title-link')?.getAttribute('href') || '',
      meta:text(node, '.meta'), summary:text(node, '.summary-primary .summary-text, .summary'),
      why:summaryField(node, 'Why read'), caveat:summaryField(node, 'Caveat'),
      reason:text(node, '.match-topic, .reason-line'), opportunity:text(node, '.opportunity-facts'),
      uncertainty:text(node, '.topic-support, [data-reading-uncertainty]'),
      details:detailsText(node),
      companions:Array.from(node.querySelectorAll?.('.related-commentaries li') || [], child => ({
        title:text(child, 'a[href]'), url:child.querySelector('a[href]')?.getAttribute('href') || '',
        publicationLabel:text(child, '.publication-label'),
      })),
      ...(node.querySelector('.figure-link') ? {
        figureURL:node.querySelector('.figure-link').getAttribute('href'),
        figureLabel:text(node, '.article-figure summary').split(' · ')[0].trim(),
      } : {}),
    }))};
  const related = document.querySelector('#related-reading');
  if (related) {
    const links = Array.from(related.querySelectorAll('a[href]'), link => ({
      title:link.textContent.trim(), url:link.getAttribute('href'),
    })).filter(link => link.title && safeArticleURL(link.url));
    if (links.length) value.related = {heading:text(related, 'h2, h3') || 'Related reading',
      context:text(related, '[data-reading-context], :scope > p'), links:links.slice(0,50),
      ...(links.length > 50 ? {total:links.length} : {})};
  }
  return validateReadingView(value);
}

export async function hasSavedDesk(indexedDB, databaseName) {
  const request = indexedDB.open(databaseName, 1);
  request.onupgradeneeded = () => request.result.createObjectStore('state');
  const db = await new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  try {
    return await new Promise((resolve, reject) => {
      const key = db.transaction('state').objectStore('state').getKey('snapshot');
      key.onsuccess = () => resolve(key.result !== undefined);
      key.onerror = () => reject(key.error);
    });
  } finally { db.close(); }
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

// Build replaces this sentinel with exact shared token/surface CSS. Keeping it
// in the existing module avoids opaque-frame COEP requests and extra startup I/O.
export const SHARED_READING_STYLES = null;

export function readingGreeting(name, now = new Date()) {
  const hour = now.getHours();
  const greeting = hour < 5 ? 'You’re up early' : hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';
  return greeting + (name ? `, ${name}` : '');
}

export function readingViewDocument(value, theme = 'light', controlsBaseURL = import.meta.url, now = new Date(), sharedCSS = null) {
  if (!validateReadingView(value)) throw new Error('Invalid saved reading view');
  const escape = text => text.replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const sections = [...new Set(value.items.map(item => item.section))];
  const contents = sections.map((name, index) => `<a href="about:srcdoc#cached-section-${index}">${escape(name)} <span>${value.items.filter(item => item.section === name).length}</span></a>`).join('');
  const items = sections.map((section, index) => `<section class="section-block" id="cached-section-${index}" tabindex="-1"><h2 class="section-heading">${escape(section)}</h2>`
    + value.items.filter(item => item.section === section).map(item =>
      `<article><h3><a href="${escape(item.url)}" target="_blank" rel="noopener noreferrer">${escape(item.title)}</a></h3><p class="meta">${escape(item.meta)}</p>`
      + (item.opportunity ? '<p class="meta">Saved opportunity details. Verify current deadlines and eligibility at the official source.</p>' : '')
      + [...new Set([item.opportunity, item.reason, item.uncertainty].filter(Boolean))].map(text => `<p class="reading-context">${escape(text)}</p>`).join('')
      + (item.summary ? `<p class="paper-finding">${escape(item.summary)}</p>` : '')
      + [['Why read', item.why], ['Caveat', item.caveat]].filter(([, text]) => text).map(([label, text]) => `<div class="summary-field"><span class="summary-label">${label}</span><p class="summary-text">${escape(text)}</p></div>`).join('')
      + (item.figureURL ? `<p class="meta"><a href="${escape(item.figureURL)}" target="_blank" rel="noopener noreferrer" referrerpolicy="no-referrer">${escape(item.figureLabel)} · open publisher image</a></p>` : '')
      + (item.companions?.length ? `<ul class="related-commentaries" aria-label="Commentary on ${escape(item.title)}">`
        + item.companions.map(link => `<li><span class="publication-label">${escape(link.publicationLabel)}</span> · <a href="${escape(link.url)}" target="_blank" rel="noopener noreferrer">${escape(link.title)}</a></li>`).join('') + '</ul>' : '')
      + (item.details ? `<details><summary>Details</summary><p>${escape(item.details)}</p></details>` : '') + '</article>'
    ).join('') + '</section>').join('');
  let controlsLink = 'Use Open controls above to see all.';
  if (sharedCSS !== null && (typeof sharedCSS !== 'string' || sharedCSS.length > 400001)) throw new Error('Invalid shared reading styles');
  let sharedStyles = sharedCSS === null ? '' : '<style>' + sharedCSS.replace(/</g, '\\3c ') + '</style>', styleOrigin = '';
  try {
    const url = new URL('./', controlsBaseURL);
    if (safeArticleURL(url.href)) {
      if (sharedCSS === null) {
        styleOrigin = ' ' + url.origin;
        sharedStyles = ['./design-tokens.css','./reading-surface.css'].map(file => `<link rel="stylesheet" href="${escape(new URL(file,url).href)}">`).join('');
      }
      url.hash = readingViewRoute(value);
      controlsLink = `<a href="${escape(url.href)}" target="_blank" rel="noopener noreferrer">Open controls</a> in a new tab to see all.`;
    }
  } catch { /* Non-browser consumers can use the surrounding controls. */ }
  const related = value.related ? `<section aria-labelledby="related-reading-title"><h2 id="related-reading-title">${escape(value.related.heading)}</h2><p>${escape(value.related.context)}</p><ul>`
    + value.related.links.map(link => `<li><a href="${escape(link.url)}" target="_blank" rel="noopener noreferrer">${escape(link.title)}</a></li>`).join('') + '</ul>'
    + (value.related.total > value.related.links.length
      ? `<p class="meta">Showing ${value.related.links.length} of ${value.related.total} related papers. ${controlsLink}</p>` : '') + '</section>' : '';
  const savedAt = new Intl.DateTimeFormat(undefined, {dateStyle:'medium',timeStyle:'short'}).format(new Date(value.savedAt));
  return `<!doctype html><html lang="en" data-theme="${theme === 'dark' ? 'dark' : 'light'}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'none'; style-src 'unsafe-inline'${styleOrigin}; font-src data:; form-action 'none'; base-uri 'none'"><title>Saved DailyDigest ${escape(value.date)}</title>${sharedStyles}<style>
  *{box-sizing:border-box}body{line-height:1.5}main{position:relative;z-index:1;max-width:1100px;margin:auto;padding:24px 20px 48px}h1{font-size:30px;font-weight:650;line-height:1.2;margin:4px 0 20px;letter-spacing:-.02em}h2{font-size:17px;margin:0;padding:14px 20px;border-bottom:1px solid var(--border)}h3{font-size:22px;font-weight:650;line-height:1.3;margin:0;letter-spacing:-.015em}article{padding:24px;border-bottom:1px solid var(--border)}article:last-child{border-bottom:0}a{color:var(--accent);text-decoration:none}h3 a{color:var(--text)}a:hover{text-decoration:underline;text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}p{margin:8px 0 0;white-space:pre-line;overflow-wrap:anywhere}.meta,.reading-context{color:var(--muted);font-size:14px;line-height:1.5}.digest-card{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}.contents-nav{display:flex;align-items:center;flex-wrap:wrap;gap:8px 20px;padding:10px 20px;border-bottom:1px solid var(--border)}.contents-nav a{padding:8px 0;font-size:14px}.contents-nav time{margin-left:auto;font-size:14px;color:var(--muted);font-variant-numeric:tabular-nums}.contents-nav span{margin-left:4px}.paper-finding{font-size:17px;line-height:1.55;color:var(--text);max-width:70ch;margin-top:16px}.summary-field{display:grid;grid-template-columns:82px minmax(0,1fr);gap:10px;align-items:baseline;margin-top:8px;max-width:74ch}.summary-label{font-size:13px;font-weight:600;color:var(--muted)}.summary-text{font-size:15px;line-height:1.5;margin:0;color:var(--copy-secondary)}summary{cursor:pointer;min-height:40px;padding:8px 0;font-size:14px;color:var(--accent-strong)}details>p{font-size:14px;line-height:1.5;max-width:74ch}h1,a{overflow-wrap:anywhere}.section-block{scroll-margin-top:12px}.cache-updated{font-size:13px;color:var(--muted);margin:16px 0}
  @media(max-width:680px){main{padding:18px 12px 40px}h1{font-size:27px}article{padding:20px 16px}h3{font-size:20px}.paper-finding{font-size:16px}.contents-nav{padding:8px 14px;gap:4px 14px}.contents-nav time{font-size:13px}.summary-field{grid-template-columns:1fr;gap:2px}}
  .related-commentaries{margin:10px 0;padding:0 0 0 12px;border-left:2px solid var(--border);list-style:none;font-size:14px;line-height:1.45}.related-commentaries li+li{margin-top:6px}.related-commentaries .publication-label{color:var(--muted)}.related-commentaries a{overflow-wrap:anywhere;text-underline-offset:3px}
  </style></head><body><main><h1 id="page-salutation">${escape(readingGreeting(value.profileName,now))}</h1><div class="digest-card"><nav class="contents-nav" aria-label="In this digest">${contents}<time class="digest-date" datetime="${escape(value.date)}">${escape(value.date)}</time></nav>${items}</div>${related}<p class="cache-updated">Saved ${escape(savedAt)}</p></main></body></html>`;
}
