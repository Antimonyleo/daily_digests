// Real-browser acceptance against a built static site. All source requests are
// intercepted fixtures; application routes, ranking and persistence run normally.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { chromium } from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const base = process.env.DAILYDIGEST_BROWSER_URL || 'http://127.0.0.1:8767/daily_digests/';
const output = resolve(process.env.DAILYDIGEST_BROWSER_REPORT || '.full-review/browser-launch-2026-10-13');
const fixture = process.env.DAILYDIGEST_BROWSER_FIXTURE || `${output}/fixture-backup.json`;
const feedPath = process.env.DAILYDIGEST_BROWSER_FEED || `${output}/research-feed.xml`;
await mkdir(output, { recursive: true });
const report = { url: base, started: new Date().toISOString(), checks: [], console: [], errors: [], requests: [], fixtureRequests: [], workerEvents: [] };
const scopeCorpus = JSON.parse(await readFile(new URL('../tests/fixtures/research_scope_benchmark.json', import.meta.url), 'utf8')).examples;
const fixtureLabels = new Map(scopeCorpus.map(row => [row.title, row.label]));
const companionRequests = [];
let companionSnapshot = '';
let companionBackends = ['claude_cli'];
let companionProviderIdentity = 'a'.repeat(64);
let companionSummaryMarker = 'Mock API A summary';
let providerSnapshotSequence = 0;
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--enable-experimental-webassembly-jspi'] });
const context = await browser.newContext({ viewport: { width: 1360, height: 900 }, acceptDownloads: true });
let sourceMode = 'success';
let feed = '';
try { feed = await readFile(feedPath, 'utf8'); } catch { /* Startup checks can run before fixtures are ready. */ }
await context.route('**/*', async route => {
  const url = route.request().url();
  if (url.startsWith(base)) return route.continue();
  if (url.startsWith('http://127.0.0.1:8768/')) {
    const request = route.request();
    const path = new URL(url).pathname;
    const cors = { 'access-control-allow-origin': new URL(base).origin, 'access-control-allow-methods': 'GET,POST,PUT,OPTIONS', 'access-control-allow-headers': 'Content-Type,X-DailyDigest-Token' };
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: cors });
    assert.equal(request.headers()['x-dailydigest-token'], 'fictional-acceptance-token', 'Companion requests require the configured authentication token.');
    companionRequests.push({ path, method: request.method(), body: request.postData() });
    let result = { ok: true };
    if (path === '/health') result = { ok: true, capabilities: { summarize: companionBackends, email: true, inbound: true, state: true, scheduler: false }, provider_identities: Object.fromEntries(companionBackends.map(backend => [backend, companionProviderIdentity])) };
    if (path === '/summarize') {
      const payload = JSON.parse(request.postData());
      result = { summaries: Object.fromEntries(payload.items.map(row => [row.id, `Key finding: ${companionSummaryMarker} ${row.id}.\nWhy read: Fictional acceptance metadata.\nCaveat: Mocked provider response.`])), provenance: Object.fromEntries(payload.items.map(row => [row.id, payload.backend])), diagnostics: { provider_calls: 1, fallback_items: 0 } };
    }
    if (path === '/text') result = { text: 'Mocked acceptance overview.' };
    if (path === '/email') result = { ok: true, sent: true };
    if (path === '/inbound') {
      const email = companionRequests.find(row => row.path === '/email');
      result = { ok: true, votes: [{ mail_id: 'fictional-mail-1', line: '+R1', digest_id: JSON.parse(email.body).digest_id }], errors: [] };
    }
    if (path === '/state' && request.method() === 'PUT') companionSnapshot = JSON.parse(request.postData()).snapshot;
    if (path === '/state' && request.method() === 'GET') result = { snapshot: companionSnapshot };
    return route.fulfill({ status: 200, contentType: 'application/json', headers: cors, body: JSON.stringify(result) });
  }
  if (url.startsWith('https://acceptance.invalid/')) {
    report.fixtureRequests.push({ url, mode: sourceMode });
    if (sourceMode === 'failure') return route.fulfill({ status: 503, body: 'Acceptance fixture unavailable', headers: { 'access-control-allow-origin': '*' } });
    return route.fulfill({ status: 200, contentType: 'application/rss+xml', body: feed, headers: { 'access-control-allow-origin': '*' } });
  }
  report.requests.push({ url, action: 'blocked-external' });
  return route.abort('blockedbyclient');
});
const page = await context.newPage();
page.setDefaultTimeout(30000);
page.on('console', event => { if (['warning', 'error'].includes(event.type())) report.console.push({ type: event.type(), text: event.text() }); });
page.on('pageerror', error => report.errors.push(error.message));
await page.addInitScript(() => {
  const NativeWorker = window.Worker;
  window.__workerEvents = [];
  window.Worker = class AcceptanceWorker extends NativeWorker {
    constructor(...args) {
      super(...args);
      window.__appWorker = this;
      this.addEventListener('message', event => {
        const data = event.data;
        window.__workerEvents.push({ type: data.type, stage: data.stage, detail: data.detail, message: data.message, id: data.id, status: data.status, body: data.type === 'stream' || data.status >= 400 ? data.body : undefined });
        if (data.type === 'ready') window.__workerReady = true;
      });
    }
  };
  let id = 1000000;
  window.__rpc = payload => new Promise((resolve, reject) => {
    const requestId = ++id;
    const timeout = setTimeout(() => { window.__appWorker.removeEventListener('message', listener); reject(new Error('Acceptance RPC timed out')); }, 300000);
    function listener(event) {
      if (event.data.id !== requestId || event.data.type !== 'response') return;
      clearTimeout(timeout); window.__appWorker.removeEventListener('message', listener); resolve(event.data);
    }
    window.__appWorker.addEventListener('message', listener);
    window.__appWorker.postMessage({ ...payload, id: requestId });
  });
});
const frame = () => page.frameLocator('#app');
async function check(name, work) {
  const started = performance.now();
  try {
    const evidence = await work();
    report.checks.push({ name, passed: true, elapsedMs: Math.round(performance.now() - started), evidence });
    console.log(JSON.stringify({ check: name, passed: true, evidence }));
    return evidence;
  } catch (error) {
    const pageState = await page.evaluate(() => ({ status: document.getElementById('status')?.textContent, hash: location.hash, text: document.querySelector('#app')?.contentDocument?.body?.innerText })).catch(() => null);
    report.checks.push({ name, passed: false, elapsedMs: Math.round(performance.now() - started), error: error.stack, pageState });
    console.log(JSON.stringify({ check: name, passed: false, error: error.message }));
    await page.screenshot({ path: `${output}/failure-${name.replace(/[^a-z0-9]/gi, '-')}.png`, fullPage: true }).catch(() => {});
    return null;
  } finally {
    report.workerEvents = await page.evaluate(() => window.__workerEvents).catch(() => []);
    await writeFile(`${output}/browser-acceptance.json`, JSON.stringify(report, null, 2));
  }
}
async function startup() {
  await page.waitForFunction(() => window.__workerReady || document.getElementById('retry')?.hidden === false, undefined, { timeout: 300000 });
  assert.equal(await page.evaluate(() => Boolean(window.__workerReady)), true, await page.locator('#startup-status').textContent());
  await page.locator('#app').waitFor({ state: 'visible' });
  return page.evaluate(() => ({ jspi: typeof WebAssembly.Suspending === 'function', title: document.querySelector('#app').contentDocument.title }));
}
async function route(path, selector) {
  await page.waitForFunction(() => typeof document.querySelector('#app')?.contentWindow?.ddNavigate === 'function');
  await page.evaluate(path => {
    const app = document.querySelector('#app');
    window.__priorAppDocument = app.contentDocument;
    app.contentWindow.ddNavigate(path);
  }, path);
  await page.waitForFunction(path => {
    const app = document.querySelector('#app');
    return app?.contentDocument !== window.__priorAppDocument && app?.contentWindow?.ddLocation?.endsWith(path);
  }, path, { timeout: 60000 });
  if (selector) await frame().locator(selector).first().waitFor({ state: 'visible' });
}
async function snapshot() {
  const response = await page.evaluate(() => window.__rpc({ type: 'export' }));
  assert.equal(response.status, 200, response.body);
  return JSON.parse(response.body);
}
async function openSection(section) {
  const details = frame().locator(`details.section-block[data-section="${section}"]`);
  if (await details.getAttribute('open') === null) await details.locator('summary.section-toggle').click();
}
async function openCompanionMenu() {
  if (await page.locator('#companion-menu').getAttribute('open') === null) await page.locator('#companion-menu > summary').click();
}
async function digestIdentity() {
  await route('/', '.digest-date');
  return (await frame().locator('.digest-date').textContent()).trim();
}
async function selectedStoredSummaries() {
  const exported = await snapshot();
  const directory = `${output}/provider-state-${report.started.replace(/\D/g, '')}-${++providerSnapshotSequence}`;
  await mkdir(directory, { recursive: true });
  for (const suffix of ['', '-wal', '-shm']) {
    const encoded = exported.files[`data/digest.db${suffix}`];
    if (encoded) await writeFile(`${directory}/digest.db${suffix}`, Buffer.from(encoded, 'base64'));
  }
  const digest = await digestIdentity();
  const result = spawnSync(new URL('../.venv/bin/python', import.meta.url).pathname, ['-c', `import sqlite3,json,sys
connection=sqlite3.connect(sys.argv[1])
rows=connection.execute("SELECT i.id,i.summary_backend,i.summary_signature,i.summary FROM items i JOIN digest_items d ON i.id=d.item_id WHERE d.digest_id=? AND i.section='research' ORDER BY d.item_label",(sys.argv[2],)).fetchall()
print(json.dumps([dict(zip(['id','backend','signature','summary'],row)) for row in rows]))`, `${directory}/digest.db`, digest], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}
async function successfulBrew(mode = 'full') {
  await route(`/run?reading_mode=${mode}`, '#brew-button');
  await frame().locator('#brew-button').click();
  await page.waitForFunction(() => location.hash.startsWith('#/done') || document.querySelector('#app')?.contentDocument?.querySelector('#err:not([hidden])'), undefined, { timeout: 300000 });
  assert.ok((await page.evaluate(() => location.hash)).startsWith('#/done'), await frame().locator('#err').textContent().catch(() => 'Brew did not finish'));
  await frame().locator('a.cta').click();
  await frame().locator('.item[data-section="research"]').first().waitFor();
}
let initialReady = false;
let itemId;
try {
  await page.goto(base + '#/setup');
  initialReady = Boolean(await check('startup-and-deep-setup', async () => {
    const evidence = await startup();
    await frame().locator('#topics').waitFor();
    return evidence;
  }));
  if (!initialReady) throw new Error('Startup failed; dependent acceptance checks cannot run.');
  await check('setup-native-form', async () => {
    await frame().locator('#topics').fill('RNA delivery\nProtein design\nHydrogels');
    await frame().locator('details:has(#name) > summary').click();
    await frame().locator('#name').fill('Browser Acceptance');
    await frame().locator('#bio').fill('Fictional browser acceptance reader researching RNA delivery, protein design and biomaterials.');
    await frame().locator('form[action="/setup"] button[type="submit"]').click();
    await page.waitForFunction(() => {
      const path = document.querySelector('#app')?.contentWindow?.ddLocation;
      return path && !path.endsWith('/setup');
    }, undefined, { timeout: 60000 });
    const exported = await snapshot();
    assert.ok(Object.hasOwn(exported.files, '.env'));
    return { privateConfigPersisted: true, route: await page.evaluate(() => location.hash) };
  });
  const backupText = await readFile(fixture, 'utf8');
  await check('import-fixture-backup', async () => {
    const response = await page.evaluate(body => window.__rpc({ type: 'import', body, overwrite: true }), backupText);
    assert.equal(response.status, 200, response.body);
    await page.reload(); await startup();
    await route('/', '.item[data-item-id]');
    return { digest: await digestIdentity(), items: await frame().locator('.item[data-item-id]').count() };
  });
  await check('history-calendar-display-and-pip', async () => {
    await route('/', '.item[data-item-id]');
    const history = frame().locator('.time-machine-link');
    assert.ok(await history.count() >= 2, 'Fixture must include at least two historical digests.');
    await history.first().click();
    await frame().locator('.digest-date').waitFor();
    await frame().locator('#display-menu-toggle').click();
    await frame().locator('#theme-toggle').click();
    const root = frame().locator('html');
    assert.equal(await root.getAttribute('data-theme'), 'dark');
    await frame().locator('#display-toggle').click();
    assert.equal(await root.getAttribute('data-display'), 'compact');
    const pip = frame().locator('#pip-toggle');
    if (await pip.isVisible()) { await pip.click(); assert.equal(await root.getAttribute('data-pip'), 'hidden'); }
    await route('/', '.item[data-item-id]');
    await openSection('events');
    const calendar = frame().locator('a[href^="/calendar/"]').first();
    await calendar.waitFor();
    const readingHash = await page.evaluate(() => location.hash);
    const downloadEvent = page.waitForEvent('download');
    await calendar.click();
    const download = await downloadEvent;
    await download.saveAs(`${output}/calendar.ics`);
    const contents = await readFile(`${output}/calendar.ics`, 'utf8');
    assert.match(contents, /BEGIN:VCALENDAR/);
    assert.equal(await page.evaluate(() => location.hash), readingHash, 'Calendar export must preserve the reading route for reload.');
    return { historyEntries: await history.count(), calendar: true, display: 'compact', theme: 'dark' };
  });
  await check('feedback-vote-reason-bookmark-known', async () => {
    await route('/', '.item[data-item-id]');
    await openSection('events');
    const eventCard = frame().locator('.item[data-section="events"]').first();
    await eventCard.locator('.known-btn').click();
    await eventCard.locator('.known-btn[aria-pressed="true"]').waitFor({ state: 'attached' });
    const card = frame().locator('.item[data-section="research"]').first();
    itemId = await card.getAttribute('data-item-id');
    await card.locator('.vote-btn[data-grade="10"]').click();
    await card.locator('.vote-btn[data-grade="10"][aria-pressed="true"]').waitFor();
    await card.locator('.reason-chip[data-reason="off_topic"]').click();
    await card.locator('.reason-chip[data-reason="off_topic"][aria-pressed="true"]').waitFor();
    const bookmark = card.locator('.bookmark-btn:not(.known-btn)');
    if (await bookmark.getAttribute('aria-pressed') !== 'true') await bookmark.click();
    await card.locator('.bookmark-btn:not(.known-btn)[aria-pressed="true"]').waitFor();
    return { itemId, grade: 10, reason: 'off_topic', bookmarked: true, known: true };
  });
  await check('reload-persists-feedback-and-saved-reading', async () => {
    await page.reload(); await startup();
    await route('/', '.time-machine-link');
    const archived = frame().locator('.time-machine-link').last();
    const archivedPath = await archived.getAttribute('href');
    await archived.click();
    await page.waitForFunction(path => document.querySelector('#app')?.contentWindow?.ddLocation?.endsWith(path), archivedPath);
    await openSection('events');
    await frame().locator('.item[data-section="events"] .known-btn[aria-pressed="true"]').waitFor();
    await route('/saved', '.saved-card');
    const card = frame().locator(`.saved-card[data-item-id="${itemId}"]`);
    await card.locator('.vote-btn[data-grade="10"][aria-pressed="true"]').waitFor();
    await card.locator('.reason-chip[data-reason="off_topic"][aria-pressed="true"]').waitFor();
    return { itemId, savedAcrossReload: true };
  });
  await check('saved-search-and-clear-vote', async () => {
    const card = frame().locator(`.saved-card[data-item-id="${itemId}"]`);
    await card.locator('.clear-rating').click();
    await card.locator('.clear-rating').waitFor({ state: 'hidden' });
    assert.equal(await card.locator('.vote-btn[aria-pressed="true"]').count(), 0);
    const search = frame().locator('form[role="search"] input[name="q"]');
    await search.fill('impossibleAcceptanceSearchToken');
    await search.press('Enter');
    await page.waitForFunction(() => document.querySelector('#app')?.contentWindow?.ddLocation?.includes('impossibleAcceptanceSearchToken'));
    assert.equal(await frame().locator('.saved-card').count(), 0);
    return { cleared: true, searchFilters: true };
  });
  await check('mobile-layout-and-keyboard', async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    await route('/setup', '#topics');
    await frame().locator('details:has(#name) > summary').click();
    const overflow = await page.evaluate(() => {
      const doc = document.querySelector('#app').contentDocument;
      return { width: doc.documentElement.clientWidth, scrollWidth: doc.documentElement.scrollWidth, outerWidth: document.documentElement.clientWidth, outerScrollWidth: document.documentElement.scrollWidth };
    });
    assert.ok(overflow.scrollWidth <= overflow.width + 1, JSON.stringify(overflow));
    assert.ok(overflow.outerScrollWidth <= overflow.outerWidth + 1, JSON.stringify(overflow));
    await openCompanionMenu();
    const companionBox = await page.locator('#companion-menu .data-tools').boundingBox();
    assert.ok(companionBox.x >= 0 && companionBox.x + companionBox.width <= 390, JSON.stringify(companionBox));
    await page.locator('#companion-menu > summary').press('Escape');
    await frame().locator('#topics').focus(); await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => document.querySelector('#app').contentDocument.activeElement.tagName), 'SUMMARY');
    await page.screenshot({ path: `${output}/mobile-setup.png`, fullPage: true });
    await page.setViewportSize({ width: 1360, height: 900 });
    return overflow;
  });
  await check('brew-cancellation-preserves-digest', async () => {
    const before = await digestIdentity();
    await route('/run?reading_mode=usual', '#brew-button');
    await frame().locator('#brew-button').click();
    await frame().locator('#cancel-button').click();
    await frame().locator('#err').waitFor({ state: 'visible', timeout: 300000 });
    assert.match(await frame().locator('#err').textContent(), /stop|cancel/i);
    const after = await digestIdentity(); assert.equal(after, before);
    return { before, after };
  });
  await check('brew-status-reconnect-without-starting-a-new-brew', async () => {
    const sourcesBefore = report.fixtureRequests.length;
    await route('/run?resume=1&run_id=deadbeef1234', '#reconnect-button');
    assert.match(await frame().locator('#err').textContent(), /lost|reconnect/i);
    await frame().locator('#reconnect-button').click();
    await frame().locator('#err').waitFor({ state: 'visible' });
    assert.equal(report.fixtureRequests.length, sourcesBefore);
    return { recoveredStatusError: true, noNewBrew: true };
  });
  await check('real-browser-brew-and-local-model-inference', async () => {
    assert.ok(feed, 'Missing research-feed.xml fixture');
    await route('/run?reading_mode=full', '#brew-button');
    await frame().locator('#mode-full').check();
    await frame().locator('#brew-button').click();
    await page.waitForFunction(() => location.hash.startsWith('#/done') || document.querySelector('#app')?.contentDocument?.querySelector('#err:not([hidden])'), undefined, { timeout: 300000 });
    assert.ok((await page.evaluate(() => location.hash)).startsWith('#/done'), await frame().locator('#err').textContent().catch(() => 'Brew did not finish'));
    await frame().locator('a.cta').click();
    await frame().locator('.item[data-item-id]').first().waitFor();
    const events = await page.evaluate(() => window.__workerEvents);
    assert.ok(events.some(event => event.stage === 'Computing embeddings'), 'Real embedding inference was not observed.');
    assert.ok(events.some(event => event.type === 'stream' && event.body.includes('"stage": "done"')), 'Native terminal brew event missing.');
    const titles = await frame().locator('.item[data-section="research"] .title-link').allTextContents();
    const slate = titles.map(title => ({ title, fixtureLabel: fixtureLabels.get(title.trim()) || 'seed' }));
    assert.ok(slate.length > 0);
    assert.equal(slate.filter(row => row.fixtureLabel === 'off_interest').length, 0, JSON.stringify(slate));
    await page.screenshot({ path: `${output}/brewed-digest.png`, fullPage: true });
    return { digest: await digestIdentity(), selected: await frame().locator('.item[data-item-id]').count(), slate, fixtureRequests: report.fixtureRequests.length };
  });
  await check('degraded-research-preserves-last-digest', async () => {
    const before = await digestIdentity(); sourceMode = 'failure';
    await route('/run?reading_mode=minimal', '#brew-button');
    await frame().locator('#mode-minimal').check(); await frame().locator('#brew-button').click();
    await frame().locator('#err').waitFor({ state: 'visible', timeout: 300000 });
    const message = await frame().locator('#err').textContent();
    assert.match(message, /research|source|coverage|fetch|ingest/i);
    const after = await digestIdentity(); assert.equal(after, before);
    sourceMode = 'success'; return { before, after, message };
  });
  await check('backup-download-and-import-confirmation', async () => {
    const exported = await snapshot();
    await writeFile(`${output}/exported-backup.json`, JSON.stringify(exported));
    await page.locator('.browser-bar > details:not(#companion-menu) > summary').click();
    const downloadEvent = page.waitForEvent('download'); await page.locator('#export').click();
    const download = await downloadEvent; await download.saveAs(`${output}/downloaded-backup.json`);
    const downloaded = JSON.parse(await readFile(`${output}/downloaded-backup.json`, 'utf8'));
    assert.equal(downloaded.format, 'dailydigest-browser-v1');
    const dismissal = page.waitForEvent('dialog');
    await page.locator('#import').setInputFiles(`${output}/downloaded-backup.json`);
    await (await dismissal).dismiss();
    await page.waitForFunction(() => document.getElementById('import').value === '');
    assert.equal(await page.evaluate(() => Boolean(window.__workerReady)), true);
    const reload = page.waitForEvent('framenavigated', { predicate: changedFrame => changedFrame === page.mainFrame() });
    const confirmation = page.waitForEvent('dialog');
    await page.locator('#import').setInputFiles(`${output}/downloaded-backup.json`);
    await (await confirmation).accept();
    await reload; await startup();
    return { files: Object.keys(exported.files), confirmationDismissed: true, confirmedImportReloaded: true };
  });
  await check('optional-companion-explicit-actions', async () => {
    assert.equal(companionRequests.length, 0, 'No companion contact should occur before explicit setup.');
    await openCompanionMenu();
    await page.locator('#companion-endpoint').fill('http://127.0.0.1:8768');
    await page.locator('#companion-token').fill('fictional-acceptance-token');
    await page.locator('#companion-connect').click();
    await page.waitForFunction(() => document.getElementById('companion-status').textContent.includes('Connected'));
    assert.deepEqual(companionRequests.map(row => row.path), ['/health']);
    assert.equal(await page.locator('#companion-token').inputValue(), '');
    const status = await page.evaluate(() => window.__rpc({ type: 'companion-status' }));
    assert.doesNotMatch(status.body, /fictional-acceptance-token/);
    assert.doesNotMatch(JSON.stringify(await snapshot()), /fictional-acceptance-token/);
    for (const action of ['email', 'push']) {
      const confirmation = page.waitForEvent('dialog').then(dialog => dialog.accept());
      await page.locator(`#companion-${action}`).click();
      await confirmation;
      await page.waitForFunction(() => !document.getElementById('companion-status').textContent.includes('Running requested'));
    }
    const email = companionRequests.find(row => row.path === '/email');
    assert.ok(JSON.parse(email.body).html.includes('<html'), 'Email must use actual rendered digest HTML.');
    assert.equal(JSON.parse(companionSnapshot).format, 'dailydigest-browser-v1');
    await page.locator('#companion-inbound').click();
    await page.waitForFunction(() => document.getElementById('companion-status').textContent.includes('Email feedback processed'));
    assert.match(await page.locator('#companion-status').textContent(), /1 upvotes, 0 downvotes/);
    await frame().locator('.item[data-section="research"]').first().locator('.vote-btn[data-grade="70"][aria-pressed="true"]').waitFor();
    assert.ok(companionRequests.some(row => row.path === '/inbound/ack'), 'Mailbox ACK must follow committed native browser feedback.');
    const confirmation = page.waitForEvent('dialog').then(dialog => dialog.accept());
    const reload = page.waitForEvent('framenavigated', { predicate: changedFrame => changedFrame === page.mainFrame() });
    await page.locator('#companion-pull').click(); await confirmation;
    await reload; await startup();
    await openCompanionMenu();
    await page.locator('#companion-disconnect').click();
    await page.waitForFunction(() => document.getElementById('companion-status').textContent.includes('No companion'));
    assert.equal(await page.locator('#companion-email').isDisabled(), true);
    return { mockedCompanion: true, requests: companionRequests.map(({ path, method }) => ({ path, method })), nativeEmailFeedbackApplied: true, nativeEmailFeedbackGrade: 70, mailboxAcknowledged: true, stateRestored: true, disconnected: true };
  });
  await check('pwa-project-subpath-and-install-control', async () => {
    const evidence = await page.evaluate(async () => {
      const manifest = new URL(document.querySelector('link[rel="manifest"]').getAttribute('href'), location.href).href;
      const registration = await navigator.serviceWorker.ready;
      const offered = new Event('beforeinstallprompt', { cancelable: true });
      offered.prompt = async () => { window.__installPromptCalled = true; };
      offered.userChoice = Promise.resolve({ outcome: 'accepted' });
      window.dispatchEvent(offered);
      return { manifest, scope: registration.scope, promptPrevented: offered.defaultPrevented, offeredButtonVisible: !document.getElementById('install').hidden };
    });
    assert.equal(evidence.manifest, base + 'manifest.webmanifest');
    assert.equal(evidence.scope, base);
    assert.equal(evidence.promptPrevented, true);
    assert.equal(evidence.offeredButtonVisible, true);
    await page.locator('#install').click();
    await page.waitForFunction(() => document.getElementById('install').hidden);
    assert.equal(await page.evaluate(() => window.__installPromptCalled), true);
    return { ...evidence, mockedInstallPrompt: true, promptInvokedByClick: true };
  });
  await check('companion-api-dispatch-provenance-and-provider-cache', async () => {
    sourceMode = 'success'; companionBackends = ['api'];
    await openCompanionMenu();
    await page.locator('#companion-endpoint').fill('http://127.0.0.1:8768');
    await page.locator('#companion-token').fill('fictional-acceptance-token');
    await page.locator('#companion-connect').click();
    await page.waitForFunction(() => document.getElementById('companion-status').textContent.includes('Connected'));
    await page.locator('#companion-menu > summary').press('Escape');
    await route('/setup', '#topics');
    await frame().locator('details:has(input[name="llm_backend"]) > summary').click();
    await frame().locator('input[name="llm_backend"][value="api"]').check();
    await frame().locator('#llm_api_key').fill('');
    await frame().locator('#llm_model').fill('fictional-companion-model');
    await frame().locator('form[action="/setup"] button[type="submit"]').click();
    await page.waitForFunction(() => document.querySelector('#app')?.contentWindow?.ddLocation?.startsWith('http://127.0.0.1:8765/run'));
    const configured = await snapshot();
    assert.match(Buffer.from(configured.files['.env'], 'base64').toString(), /LLM_BACKEND=api/);
    const initialRequests = companionRequests.filter(row => row.path === '/summarize').length;
    await successfulBrew();
    const first = await selectedStoredSummaries();
    assert.ok(first.length > 0);
    for (const row of first) { assert.equal(row.backend, 'api'); assert.match(row.summary, /Mock API A summary/); }
    const dispatched = companionRequests.filter(row => row.path === '/summarize').slice(initialRequests);
    assert.ok(dispatched.length > 0, 'Native browser API selection must dispatch through its advertised companion without a browser key.');
    for (const request of dispatched) {
      const payload = JSON.parse(request.body);
      assert.deepEqual(Object.keys(payload).sort(), ['backend', 'items', 'profile', 'timeout']);
      assert.equal(payload.backend, 'api');
      assert.ok(payload.timeout > 0 && payload.timeout <= 60);
      assert.deepEqual(Object.keys(payload.profile).sort(), ['bio', 'keywords']);
      assert.ok(payload.profile.bio.length <= 400 && payload.profile.keywords.length <= 20);
      assert.ok(payload.items.every(row => row.abstract.length <= 1500 && first.some(selected => selected.id === row.id)));
      assert.doesNotMatch(request.body, /fictional-acceptance-token|llm_api_key|digest\.db/);
    }
    companionProviderIdentity = 'b'.repeat(64); companionSummaryMarker = 'Mock API B summary';
    const firstCalls = companionRequests.filter(row => row.path === '/summarize').length;
    await successfulBrew();
    const second = await selectedStoredSummaries();
    assert.ok(companionRequests.filter(row => row.path === '/summarize').length > firstCalls, 'Changed server provider identity must invalidate cached browser summaries.');
    assert.deepEqual(second.map(row => row.id), first.map(row => row.id));
    for (const row of second) { assert.equal(row.backend, 'api'); assert.match(row.summary, /Mock API B summary/); assert.notEqual(row.signature, first.find(prior => prior.id === row.id).signature); }
    await route('/setup', '#topics');
    await frame().locator('details:has(input[name="llm_backend"]) > summary').click();
    await frame().locator('input[name="llm_backend"][value="extractive"]').check();
    await frame().locator('form[action="/setup"] button[type="submit"]').click();
    await page.waitForFunction(() => document.querySelector('#app')?.contentWindow?.ddLocation?.startsWith('http://127.0.0.1:8765/run'));
    const restored = await snapshot();
    assert.match(Buffer.from(restored.files['.env'], 'base64').toString(), /LLM_BACKEND=extractive/);
    await openCompanionMenu();
    await page.locator('#companion-disconnect').click();
    return { nativeApiWithNoBrowserKey: true, providerSummaryBackend: 'api', selectedItemsOnly: first.length, serverIdentityInvalidatesCache: true, extractiveRestored: true };
  });
  await check('public-snapshot-through-real-browser-httpx', async () => {
    // A separate ephemeral worker reads public metadata only. The other cases
    // use fixture feeds, which cannot expose Pages' real compression headers.
    const evidence = await page.evaluate(async base => {
      const source = `
        import { loadPyodide } from ${JSON.stringify(base + 'node_modules/pyodide/pyodide.mjs')};
        try {
          const py = await loadPyodide({ indexURL: ${JSON.stringify(base + 'node_modules/pyodide/')} });
          await py.loadPackage(['httpx']);
          const bundle = await fetch(${JSON.stringify(base + 'assets/app.zip')});
          if (!bundle.ok) throw new Error('Application archive unavailable');
          py.unpackArchive(await bundle.arrayBuffer(), 'zip', { extractDir: '/app' });
          py.runPython("import sys\\nsys.path.insert(0, '/app/browser')\\nimport network\\nnetwork.install_httpx_transport()");
          py.globals.set('snapshot_url', ${JSON.stringify(base + 'assets/public-sources.json')});
          const result = py.runPython("import httpx, json\\nwith httpx.Client(timeout=20, follow_redirects=False) as client:\\n    response = client.get(snapshot_url)\\n    response.raise_for_status()\\n    snapshot = response.json()\\njson.dumps({'public_only': snapshot['public_only'], 'sources': len(snapshot['sources']), 'decoded_bytes': len(response.content), 'content_encoding': response.headers.get('content-encoding'), 'content_length': int(response.headers['content-length'])})");
          postMessage({ result: JSON.parse(result) });
        } catch (error) { postMessage({ error: String(error) }); }
      `;
      const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
      try {
        return await new Promise((resolve, reject) => {
          const worker = new Worker(url, { type: 'module' });
          const finish = callback => value => { clearTimeout(timer); worker.terminate(); callback(value); };
          const timer = setTimeout(finish(reject), 90000, new Error('Public snapshot transport check timed out'));
          worker.onmessage = finish(event => event.data.error ? reject(new Error(event.data.error)) : resolve(event.data.result));
          worker.onerror = finish(event => reject(new Error(event.message)));
        });
      } finally { URL.revokeObjectURL(url); }
    }, base);
    assert.equal(evidence.public_only, true);
    assert.ok(evidence.sources > 0);
    assert.equal(evidence.content_encoding, null);
    assert.equal(evidence.content_length, evidence.decoded_bytes);
    return evidence;
  });
  await check('visible-backup-error-and-keyboard-recovery', async () => {
    const before = await digestIdentity();
    const dataMenu = page.locator('.browser-bar > details:has(#export)');
    if (await dataMenu.getAttribute('open') === null) await dataMenu.locator('summary').click();
    await page.locator('#export').focus();
    await page.locator('#import').setInputFiles({ name: 'invalid-backup.json', mimeType: 'application/json', buffer: Buffer.from('{invalid') });
    await page.locator('#app-error').waitFor({ state: 'visible' });
    assert.match(await page.locator('#app-error-message').textContent(), /Backup import failed/);
    await page.locator('#dismiss-error').press('Escape');
    await page.locator('#app-error').waitFor({ state: 'hidden' });
    assert.equal(await page.evaluate(() => document.activeElement.id), 'export');
    assert.equal(await digestIdentity(), before);
    return { visibleError: true, keyboardDismissed: true, focusRestored: true, savedDigestPreserved: true };
  });
  await check('unsupported-browser-explained-before-runtime-download', async () => {
    const unsupported = await browser.newContext();
    const testPage = await unsupported.newPage();
    const workerRequests = [];
    testPage.on('request', request => { if (request.url().includes('/worker.mjs')) workerRequests.push(request.url()); });
    await testPage.addInitScript(() => Object.defineProperty(WebAssembly, 'Suspending', { value: undefined, configurable: true }));
    try {
      await testPage.goto(base, { waitUntil: 'domcontentloaded' });
      await testPage.locator('#retry').waitFor({ state: 'visible' });
      assert.match(await testPage.locator('#startup-status').textContent(), /current desktop Chrome/);
      assert.equal(workerRequests.length, 0);
      return { simulatedMissingCapability: true, clearMessage: true, runtimeNotDownloaded: true };
    } finally { await unsupported.close(); }
  });
} catch (error) { report.fatal = error.stack; console.error(error.message); }
finally {
  report.workerEvents = await page.evaluate(() => window.__workerEvents).catch(() => []);
  report.finished = new Date().toISOString();
  report.passed = !report.fatal && report.checks.every(check => check.passed) && report.errors.length === 0;
  await writeFile(`${output}/browser-acceptance.json`, JSON.stringify(report, null, 2));
  await browser.close();
}
process.exitCode = report.passed ? 0 : 1;
