// Real-browser acceptance against a built static site. All source requests are
// intercepted fixtures; application routes, ranking and persistence run normally.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { chromium } from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const base = process.env.DAILYDIGEST_BROWSER_URL || 'http://127.0.0.1:8767/daily_digests/';
const output = resolve(process.env.DAILYDIGEST_BROWSER_REPORT || '.full-review/browser-launch-2026-10-13');
const fixture = process.env.DAILYDIGEST_BROWSER_FIXTURE || `${output}/fixture-backup.json`;
const feedPath = process.env.DAILYDIGEST_BROWSER_FEED || `${output}/research-feed.xml`;
function publicCacheProbeTexts(snapshot, hashes) {
  const covered = new Set(hashes);
  return [...new Set(Object.values(snapshot.sources || {}).filter(row => row.status === 'ok')
    .flatMap(row => row.items || []).filter(item => item.section === 'research' && item.title?.trim())
    .map(item => item.abstract?.trim() ? `${item.title.trim()}. ${item.abstract.trim()}` : item.title.trim()))]
    .filter(text => covered.has(createHash('sha256').update(text).digest('hex'))).slice(0, 12);
}
await mkdir(output, { recursive: true });
const report = { url: base, started: new Date().toISOString(), checks: [], console: [], errors: [], requests: [], fixtureRequests: [], workerEvents: [] };
report.verifierSha256 = createHash('sha256').update(await readFile(new URL(import.meta.url))).digest('hex');
const scopeCorpus = JSON.parse(await readFile(new URL('../tests/fixtures/research_scope_benchmark.json', import.meta.url), 'utf8')).examples;
const fixtureLabels = new Map(scopeCorpus.map(row => [row.title, row.label]));
const companionRequests = [];
const spellingRequests = [];
const apiTestRequests = [];
let apiTestMode = 'success', releaseApiTest, apiTestStarted;
let spellingMode = 'success', releaseSpelling, spellingStarted;
const checkpointEvidence = [];
let companionSnapshot = '';
let companionBackends = ['claude_cli'];
let companionProviderIdentity = 'a'.repeat(64);
let companionSummaryMarker = 'Mock API A summary';
let providerSnapshotSequence = 0;
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--enable-experimental-webassembly-jspi'] });
report.browserVersion = browser.version();
console.log(JSON.stringify({browserVersion:report.browserVersion}));
const context = await browser.newContext({ viewport: { width: 1360, height: 900 }, acceptDownloads: true });
let sourceMode = 'success';
let pendingSlowSource;
let feed = '';
try { feed = await readFile(feedPath, 'utf8'); } catch { /* Startup checks can run before fixtures are ready. */ }
await context.route('**/*', async route => {
  const url = route.request().url();
  if (url.startsWith(base) && new URL(url).pathname.endsWith('/worker.mjs')) {
    const response = await route.fetch();
    let source = await response.text();
    // Test-only instrumentation; never included in the published worker.
    source = source.replace("self.onmessage = async ({ data }) => {", `self.onmessage = async ({ data }) => {
      if (data.type === 'acceptance-calibration-obstruction') {
        try {
          await initialize();
          state.py.runPython('from dailydigest.rank.calibrate import _calibrator_path; _acceptance_calibration_tmp = _calibrator_path().with_suffix(".json.tmp")');
          state.py.runPython(data.block ? '_acceptance_calibration_tmp.mkdir()' : '_acceptance_calibration_tmp.rmdir()');
          self.postMessage(response(data.id,JSON.stringify({blocked:Boolean(data.block)})));
        } catch(error) {self.postMessage(response(data.id,String(error),500));}
        return;
      }
      if (data.type === 'acceptance-embedding-cache') {
        try {
          await initialize();
          const rows=state.py.runPython(\`import json
from sqlalchemy import select
from dailydigest.store import ItemEmbeddingRow, session_scope
with session_scope() as _acceptance_session:
    _acceptance_rows = _acceptance_session.execute(select(ItemEmbeddingRow).order_by(ItemEmbeddingRow.item_id, ItemEmbeddingRow.model)).scalars().all()
    _acceptance_cache = [[int(row.item_id), row.model, row.text_hash, int(row.dim), row.vector.hex()] for row in _acceptance_rows]
json.dumps(_acceptance_cache)\`);
          self.postMessage(response(data.id,rows));
        } catch(error) {self.postMessage(response(data.id,String(error),500));}
        return;
      }
      if (data.type === 'acceptance-committed-snapshot') {
        try {await initialize();await persist({force:true});self.postMessage(response(data.id,(await readSnapshot()) || '{}'));}
        catch(error) {self.postMessage(response(data.id,String(error),500));}
        return;
      }
      if (data.type === 'acceptance-public-cache') {
        try {
          await initialize();
          const before = state.engine.stats.completedInputs, hitsBefore = state.publicEmbeddings.stats.hits;
          state.py.globals.set('_acceptance_public_texts', JSON.stringify(data.texts));
          state.py.runPython(\`def _acceptance_public_vectors():
    import json, numpy as np
    from dailydigest.rank.embed import embed_texts
    vectors = embed_texts(json.loads(_acceptance_public_texts), is_query=False)
    return json.dumps({'vectors':int(vectors.shape[0]), 'valid':bool(vectors.shape[1] == 384 and np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5))})\`);
          const vectors = JSON.parse(await state.py.globals.get('_acceptance_public_vectors').callPromising());
          self.postMessage(response(data.id, JSON.stringify({before, after:state.engine.stats.completedInputs,
            hits:state.publicEmbeddings.stats.hits-hitsBefore, ...vectors})));
        } catch(error) {self.postMessage(response(data.id,String(error),500));}
        return;
      }
      if (data.type === 'acceptance-state') {
        try {
          if (data.action === 'begin') state.py.runPython(\`from dailydigest.store import _engine
from sqlalchemy import text
_acceptance_connection = _engine().connect()
_acceptance_transaction = _acceptance_connection.begin()
_acceptance_connection.execute(text("CREATE TABLE IF NOT EXISTS acceptance_checkpoint (id INTEGER)"))
_acceptance_connection.execute(text("INSERT INTO acceptance_checkpoint VALUES (1)"))\`);
          if (data.action === 'reset-view') state.py.runPython(\`from dailydigest import store
from sqlalchemy import text
with store.session_scope() as session:
    digest_id = session.execute(text("SELECT id FROM digests ORDER BY id DESC LIMIT 1")).scalar_one()
    item_id = session.execute(text("SELECT item_id FROM digest_items WHERE digest_id=:digest ORDER BY item_label LIMIT 1"), {"digest":digest_id}).scalar_one()
store.write_impressions(digest_id, [('research', item_id, 0, 0.9)], model_version='acceptance-checkpoint')\`);
          if (data.action === 'commit') state.py.runPython('_acceptance_transaction.commit(); _acceptance_connection.close()');
          const ready = Boolean(state.py.globals.get('snapshot_ready')());
          const saved = await persist({force: true});
          const full = ready ? state.py.globals.get('export_snapshot')() : null;
          if (full) { state.py.globals.set('_acceptance_snapshot', full); state.py.runPython('validate_snapshot_sqlite(decode_snapshot(_acceptance_snapshot))'); }
          self.postMessage(response(data.id, JSON.stringify({ready, saved, generation: Number(state.py.globals.get('state_generation')()), equal: full === await readSnapshot()})));
        } catch (error) { self.postMessage(response(data.id, String(error), 500)); }
        return;
      }`);
    source = source.replace("    const result = await state.py.globals.get('handle_request')", "    const acceptanceBefore = Number(state.py.globals.get('state_generation')());\n    const result = await state.py.globals.get('handle_request')");
    source = source.replace('    const delivered = JSON.parse(result);', `    const saved = await persist({force:true});
    if (saved && !jobRunning) {
      const persisted = await readSnapshot();
      const generation = Number(state.py.globals.get('state_generation')());
      const equal = persisted === state.py.globals.get('export_snapshot')();
      self.postMessage({type:'acceptance-checkpoint', method:data.method || 'GET', path:data.path,
        before:acceptanceBefore, after:generation, equal});
    }
    const delivered = JSON.parse(result);`);
    return route.fulfill({response, body: source});
  }
  if (url.startsWith(base)) return route.continue();
  if (url.startsWith('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi')) {
    const topic = new URL(url).searchParams.get('term');
    spellingRequests.push(topic);
    if (spellingMode === 'slow') {
      await new Promise(resolve => { releaseSpelling = resolve; spellingStarted?.(); });
    }
    if (spellingMode === 'failure') return route.fulfill({status:503, body:'Fixture unavailable', headers:{'access-control-allow-origin':'*'}});
    const corrected = {'induced provimity':'induced proximity', 'assymetric catalysis':'asymmetric catalysis', 'KRAS G12D inhibtor':'KRAS G12C inhibitor'}[topic] || '';
    return route.fulfill({contentType:'application/xml', headers:{'access-control-allow-origin':'*'},
      body:`<eSpellResult><CorrectedQuery>${corrected}</CorrectedQuery><ERROR/></eSpellResult>`});
  }
  if (url.startsWith('https://api-fixture.invalid/') || url.startsWith('https://api.anthropic.com/v1/messages')) {
    const request = route.request();
    const cors = {'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'access-control-allow-methods': 'POST,OPTIONS'};
    if (request.method() === 'OPTIONS') return route.fulfill({status:204, headers:cors});
    apiTestRequests.push({url, method:request.method(), headers:request.headers(), body:JSON.parse(request.postData())});
    if (apiTestMode === 'slow') await new Promise(resolve => { releaseApiTest = resolve; apiTestStarted?.(); });
    if (apiTestMode === 'failure') return route.fulfill({status:401, headers:cors, body:'Fictional authentication failure: fixture key must not be reflected'});
    const anthropic = new URL(url).pathname.endsWith('/messages');
    return route.fulfill({status:200, contentType:'application/json', headers:cors,
      body:JSON.stringify(anthropic ? {content:[{type:'text', text:'OK'}]} : {choices:[{message:{content:'OK'}}]})});
  }
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
      const email = JSON.parse(companionRequests.findLast(row => row.path === '/email').body);
      const run = email.subject.match(/\[run:([0-9a-f]{32})\]/i);
      assert.ok(run, 'Mock replies must identify the actual sent email edition.');
      result = { ok: true, votes: [{ mail_id: 'fictional-mail-1', line: '+R1', digest_id: email.digest_id, run_id: run[1].toLowerCase() }], errors: [] };
    }
    if (path === '/state' && request.method() === 'PUT') companionSnapshot = JSON.parse(request.postData()).snapshot;
    if (path === '/state' && request.method() === 'GET') result = { snapshot: companionSnapshot };
    return route.fulfill({ status: 200, contentType: 'application/json', headers: cors, body: JSON.stringify(result) });
  }
  if (url.startsWith('https://acceptance.invalid/')) {
    report.fixtureRequests.push({ url, mode: sourceMode });
    if (sourceMode === 'slow') {
      pendingSlowSource?.();
      await new Promise(resolve => setTimeout(resolve, 8000));
      return route.fulfill({ status: 200, contentType: 'application/rss+xml', body: feed, headers: { 'access-control-allow-origin': '*' } }).catch(() => {});
    }
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
let acceptDiscardedDraft = true;
const discardedDraftPrompts = [];
page.on('dialog', dialog => {
  report.dialogs ||= [];
  report.dialogs.push({type:dialog.type(),message:dialog.message(),at:new Date().toISOString()});
  void writeFile(`${output}/dialog-events.json`,JSON.stringify(report.dialogs,null,2));
  if (dialog.type() === 'beforeunload') { void dialog.accept(); return; }
  if (dialog.message() === 'Discard unsaved settings changes?') {
    discardedDraftPrompts.push(acceptDiscardedDraft);
    void (acceptDiscardedDraft ? dialog.accept() : dialog.dismiss());
  }
});
await page.addInitScript(() => {
  const NativeWorker = window.Worker;
  window.__workerEvents = [];
  window.Worker = class AcceptanceWorker extends NativeWorker {
    constructor(...args) {
      super(...args);
      window.__appWorker = this;
      this.addEventListener('message', event => {
        const data = event.data;
        window.__workerEvents.push({ type: data.type, stage: data.stage, detail: data.detail, message: data.message, timings: data.type === 'startup-timings' ? {totalMs:data.totalMs, stages:data.stages} : undefined, id: data.id, status: data.status, body: data.type === 'stream' || data.status >= 400 ? data.body : undefined });
        if (data.type === 'acceptance-checkpoint') { window.__checkpoints ||= []; window.__checkpoints.push(data); window.__recordCheckpoint?.(data); }
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
await page.exposeFunction('__recordCheckpoint', data => checkpointEvidence.push(data));
const frame = () => page.frameLocator('#app');
async function check(name, work) {
  const started = performance.now();
  let sampling = false;
  const progress = setInterval(async () => {
    if (sampling) return;
    sampling = true;
    try {
      const state = await page.evaluate(() => ({hash:location.hash,
        status:document.getElementById('status')?.textContent,
        text:document.querySelector('#app')?.contentDocument?.body?.innerText?.slice(-6000),
        workerEvents:window.__workerEvents?.slice(-20)}));
      await writeFile(`${output}/active-check.json`, JSON.stringify({name,elapsedMs:Math.round(performance.now()-started),sampledAt:new Date().toISOString(),...state},null,2));
    } catch { /* Diagnostic sampling must not alter acceptance behavior. */ }
    finally { sampling = false; }
  }, 20000);
  try {
    const evidence = await work();
    report.checks.push({ name, passed: true, elapsedMs: Math.round(performance.now() - started), evidence });
    console.log(JSON.stringify({ check: name, passed: true, evidence }));
    return evidence;
  } catch (error) {
    const pageState = await page.evaluate(() => {
      const iframe=document.querySelector('#app'),doc=iframe?.contentDocument;
      const describe=node=>node?{tag:node.tagName,id:node.id,type:node.type,role:node.getAttribute('role'),label:node.getAttribute('aria-label')}:null;
      return {status:document.getElementById('status')?.textContent,hash:location.hash,text:doc?.body?.innerText,
        focus:{outer:describe(document.activeElement),reading:describe(doc?.activeElement),frameInert:iframe?.inert,
          frameHidden:iframe?.hidden,dataSettingsOpen:document.querySelector('#data-settings')?.open,
          openDialogs:[...document.querySelectorAll('dialog[open]')].map(node=>node.id),
          openReadingGroups:[...doc?.querySelectorAll('details[open]')||[]].map(node=>node.id||node.className)}};
    }).catch(() => null);
    report.checks.push({ name, passed: false, elapsedMs: Math.round(performance.now() - started), error: error.stack, pageState });
    console.log(JSON.stringify({ check: name, passed: false, error: error.message, stack:error.stack, focus:pageState?.focus }));
    await page.screenshot({ path: `${output}/failure-${name.replace(/[^a-z0-9]/gi, '-')}.png`, fullPage: true }).catch(() => {});
    return null;
  } finally {
    clearInterval(progress);
    report.workerEvents = await page.evaluate(() => window.__workerEvents).catch(() => []);
    await writeFile(`${output}/browser-acceptance.json`, JSON.stringify(report, null, 2));
  }
}
async function startup() {
  await page.waitForFunction(() => window.__workerReady || document.getElementById('reading-view-notice')?.hidden === false || document.getElementById('retry')?.hidden === false, undefined, { timeout: 300000 });
  // These checks exercise editable routes. The independent reading-view check
  // proves that refresh remains usable without starting the runtime at all.
  if (await page.locator('#reading-view-notice').isVisible()) await page.locator('#reading-view-open').click();
  await page.waitForFunction(() => window.__workerReady || document.getElementById('retry')?.hidden === false, undefined, { timeout: 300000 });
  assert.equal(await page.evaluate(() => Boolean(window.__workerReady)), true, await page.locator('#startup-status').textContent());
  await page.locator('#app').waitFor({ state: 'visible' });
  await page.waitForFunction(() => {
    const document = window.document.querySelector('#app')?.contentDocument;
    return document && typeof document.defaultView?.ddNavigate === 'function';
  });
  return page.evaluate(() => ({ startupDurationMs: Number(document.querySelector('#startup').dataset.durationMs), timings: window.__workerEvents.find(row=>row.timings)?.timings, jspi: typeof WebAssembly.Suspending === 'function', title: document.querySelector('#app').contentDocument.title }));
}
async function route(path, selector) {
  await page.evaluate(() => document.getElementById('data-settings')?.close());
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
async function openSettingsGroup(id) {
  const group = frame().locator('#' + id);
  if (await group.getAttribute('open') === null) await group.locator(':scope > summary').click();
}
async function editInterest(row, matching = false) {
  if (await row.locator('[data-interest-name]').isHidden()) await row.locator('[data-interest-edit]').click();
  if (matching && await row.locator('details').getAttribute('open') === null) await row.locator('details > summary').click();
}
async function assertPercentagesReady() {
  const inputs = frame().locator('[data-interest-weight]');
  const values = await inputs.evaluateAll(nodes => nodes.map(node => Number(node.value)));
  assert.equal(values.reduce((sum, value) => sum + value, 0), 100);
  assert.ok(await inputs.evaluateAll(nodes => nodes.every(node => node.checkValidity())));
}
async function openAboutDigest() {
  const about=frame().locator('.about-brew');
  if (await about.getAttribute('open') === null) await about.locator(':scope>summary').click();
}
async function chooseDigest(value) {
  await page.evaluate(()=>{window.__priorEditionDocument=document.querySelector('#app').contentDocument;});
  await frame().locator('#digest-date').selectOption(value);
  await page.waitForFunction(()=>document.querySelector('#app')?.contentDocument!==window.__priorEditionDocument);
  await frame().locator('#digest-date').waitFor();
}
async function setTopics(value) {
  await frame().locator('#topics').evaluate((node, text) => { node.value = text; node.dispatchEvent(new Event('input', { bubbles: true })); }, value);
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
async function toggleMenu(id) {
  const open = await page.locator('#'+id).getAttribute('open') !== null;
  if (!await page.locator('#data-settings').evaluate(node => node.open)) await page.locator('#open-data-settings').click();
  await page.locator('#'+id+' > summary').click();
  if (open) await page.locator('#close-data-settings').click();
}
async function openCompanionMenu() {
  if (!await page.locator('#data-settings').evaluate(node => node.open)) await page.locator('#open-data-settings').click();
  if (await page.locator('#companion-menu').getAttribute('open') === null) await page.locator('#companion-menu > summary').click();
}
async function digestIdentity(digestId = '') {
  await route(digestId ? '/?digest_id=' + encodeURIComponent(digestId) : '/', '#digest-date');
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
  const checkpoint = await page.evaluate(() => window.__rpc({type:'acceptance-state'}));
  assert.equal(checkpoint.status, 200, checkpoint.body);
  assert.equal(JSON.parse(checkpoint.body).equal, true, 'Mocked brew completion must persist the full committed snapshot');
  report.brewCheckpoints ||= []; report.brewCheckpoints.push(JSON.parse(checkpoint.body));
}
function readerControlsFixture(backupText) {
  // Prepare a private fictional restore, not special app handlers or a forced
  // button state. The real vote/feature thresholds decide whether training runs.
  const result = spawnSync(new URL('../.venv/bin/python', import.meta.url).pathname, ['-c', `
import base64,json,os,sys,tempfile
from pathlib import Path, PurePosixPath
from datetime import datetime,timezone
import yaml
backup=json.load(sys.stdin)
state=Path(tempfile.mkdtemp(prefix='dailydigest-reader-controls-',dir=os.environ.get('TMPDIR')))
for name,encoded in backup['files'].items():
    relative=PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    assert name=='.env' or relative.parts[0]=='data'
    target=state/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(base64.b64decode(encoded))
os.chdir(state)
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
os.environ.update(DB_PATH=str(state/'data/digest.db'),PROFILE_PATH=str(state/'data/profile.yaml'),LLM_BACKEND='extractive',LLM_API_KEY='',RESEND_API_KEY='')
from dailydigest import store
from dailydigest.rank.source_quality import RANKER_VERSION
profile_path=state/'data/profile.yaml'
profile=yaml.safe_load(profile_path.read_text());profile.pop('name',None)
profile_path.write_text(yaml.safe_dump(profile))
store.init_db()
with store.session_scope() as session:
    original_saved=session.query(store.BookmarkRow).count()
    session.add(store.DigestRow(id='1900-01-01',item_count=0));session.flush()
    for number in range(1,56):
        item=store.ItemRow(source='Fictional controls fixture',section='research',external_id=f'reader-controls-{number}',title=f'Acceptance saved pagination {number:02}',url=f'https://acceptance.invalid/controls/{number}',abstract='Fictional archive and calibration control input.',published_at=datetime(1900,1,1,tzinfo=timezone.utc))
        session.add(item);session.flush()
        session.add(store.BookmarkRow(item_id=item.id))
        if number<=30:
            positive=number%2==0
            session.add(store.VoteRow(item_id=item.id,value=1 if positive else -1,grade=100 if positive else 10))
            session.add(store.DigestItemFeatureRow(digest_id='1900-01-01',item_id=item.id,item_label=f'R{number}',final_score=.85 if positive else .25,features_json=json.dumps({'ranker_version':RANKER_VERSION})))
store._ENGINE.dispose()
backup['files']={str(path.relative_to(state)):base64.b64encode(path.read_bytes()).decode() for path in (state/'data').rglob('*') if path.is_file()}
backup['files']['.env']=base64.b64encode((state/'.env').read_bytes()).decode()
print(json.dumps({'backup':backup,'originalSaved':original_saved,'addedSaved':55,'addedRatings':30,'policy':RANKER_VERSION}))
`, new URL('../', import.meta.url).pathname], {input:backupText,encoding:'utf8',maxBuffer:20*1024*1024});
  assert.equal(result.status,0,result.stderr);
  return JSON.parse(result.stdout);
}
async function importFixture(backupText) {
  const response = await page.evaluate(body => window.__rpc({type:'import',body,overwrite:true}),backupText);
  assert.equal(response.status,200,response.body);
  await page.reload(); await startup();
}
async function completedCalibration() {
  const completed=await page.waitForFunction(async () => {
    const response=await window.__rpc({type:'request',path:'/ranking/status'});
    if(response.status!==200)return false;
    const status=JSON.parse(response.body);
    return !status.training_job?.running && status.training_job?.last_result ? status : false;
  },undefined,{timeout:120000,polling:250});
  const status=await completed.jsonValue();
  await completed.dispose();
  await page.waitForFunction(() => !document.querySelector('#app')?.contentDocument?.querySelector('#train-ranker-btn')?.disabled);
  return status;
}
async function clickCalibration() {
  // Observe the real fetch acknowledgement before polling, so a preceding
  // job's result cannot satisfy the next job's completion check.
  const previous=await frame().locator('#train-ranker-btn').evaluate(button=>{
    const win=button.ownerDocument.defaultView;
    if(!win.__acceptanceTrainingResponses){
      win.__acceptanceTrainingResponses=[];
      const nativeFetch=win.fetch;
      win.fetch=async(...args)=>{
        const response=await nativeFetch(...args);
        if(args[0]==='/ranking/train')win.__acceptanceTrainingResponses.push({status:response.status,body:await response.clone().json()});
        return response;
      };
    }
    return win.__acceptanceTrainingResponses.length;
  });
  await frame().locator('#train-ranker-btn').click();
  await page.waitForFunction(previous=>document.querySelector('#app')?.contentWindow?.__acceptanceTrainingResponses?.length>previous,previous,{timeout:120000});
  const acknowledged=await frame().locator('#train-ranker-btn').evaluate(button=>button.ownerDocument.defaultView.__acceptanceTrainingResponses.at(-1));
  assert.equal(acknowledged.status,200);
  assert.equal(acknowledged.body.started,true);
}
let initialReady = false;
let itemId;
try {
  await page.goto(base + '#/setup');
  initialReady = Boolean(await check('startup-and-deep-setup', async () => {
    // CPU preparation may reload this document. The actual worker starts after
    // that boundary, while Python is still loading and the warming UI is visible.
    await page.waitForFunction(() => Boolean(window.__appWorker));
    await page.locator('#startup .tea-cup').waitFor({ state: 'visible' });
    await page.screenshot({ path: `${output}/warming-kettle.png`, fullPage: true });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    assert.equal(await page.locator('#startup .steam').first().evaluate(node => getComputedStyle(node).animationName), 'none');
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    const evidence = await startup();
    report.buildId = (await (await page.request.get(base + 'assets/build-manifest.json')).json()).build_id;
    await frame().locator('#interest-editor').waitFor();
    await frame().locator('#browser-storage-note').waitFor({ state: 'visible' });
    assert.match(await page.locator('#data-settings > .storage-location').textContent(), /Saved in this browser/);
    assert.ok((await page.locator('.browser-bar').boundingBox()).height <= 60);
    return evidence;
  }));
  if (!initialReady) throw new Error('Startup failed; dependent acceptance checks cannot run.');
  await check('first-run-has-visible-interests-equal-topics-and-reachable-save', async () => {
    const interest = await frame().locator('[data-interest-name]').first().evaluate(node => {
      const box = node.getBoundingClientRect();
      return {top:box.top,bottom:box.bottom,viewport:innerHeight};
    });
    assert.ok(interest.top >= 0 && interest.bottom <= interest.viewport, JSON.stringify(interest));
    assert.equal(await frame().locator('[data-interest-weight]').first().isVisible(), true);
    assert.equal(await frame().getByText('Back to today’s digest', {exact:true}).count(), 0);
    assert.equal(await frame().locator('input[id^="include_"]:checked').count(), 0);
    await frame().locator('[data-interest-name]').first().fill('DNA nanotechnology');
    await frame().locator('#add-interest').click();
    assert.deepEqual(await frame().locator('[data-interest-weight]').evaluateAll(nodes=>nodes.map(node=>node.value)), ['50','50']);
    assert.equal(await frame().locator('#normalize-interests').isHidden(),true);
    const names=await frame().locator('[data-interest-name],[data-interest-weight],[data-interest-aliases],[data-interest-description]').evaluateAll(nodes=>nodes.map(node=>node.getAttribute('aria-label')));
    assert.equal(new Set(names).size,names.length);
    assert.equal(await frame().locator('.settings-actions').evaluate(node=>getComputedStyle(node).position), 'static');
    await frame().locator('.settings-actions button[type="submit"]').scrollIntoViewIfNeeded();
    const save=await frame().locator('.settings-actions button[type="submit"]').evaluate(node=>{const box=node.getBoundingClientRect();return {top:box.top,bottom:box.bottom,viewport:innerHeight};});
    assert.ok(save.top>=0 && save.bottom<=save.viewport+1,JSON.stringify(save));
    assert.equal(await frame().locator('main').count(),1);
    assert.match(await page.title(), /Set up DailyDigest/);
    const beforeSkip=page.url();
    await page.locator('body > .skip-link').focus();await page.locator('body > .skip-link').press('Enter');
    assert.equal(page.url(),beforeSkip);
    assert.equal(await frame().locator('main').evaluate(node=>node===document.activeElement),true);
    await setTopics('');
    return {firstInterestVisible:interest,equalShares:true,researchOnly:true,inlinePercentages:true,uniqueFieldNames:true,saveReachable:save};
  });
  await check('fresh-profile-reuses-covered-public-vectors-or-encodes-locally', async () => {
    const response = await page.request.get(base+'assets/public-sources.json');
    assert.equal(response.ok(),true);
    const snapshot=await response.json();
    const build = await (await page.request.get(base+'assets/build-manifest.json')).json();
    const asset = await page.request.get(base+'assets/public-embeddings.json');
    if (build.public_embeddings.published > 0) assert.equal(asset.ok(), true, 'Declared public cache must be present');
    const cache = asset.ok() ? await asset.json() : {hashes:[]};
    const covered = publicCacheProbeTexts(snapshot, cache.hashes);
    if (build.public_embeddings.complete) assert.equal(covered.length, 12, 'Complete cache must cover the public research probe');
    const texts = covered.length ? covered : [...new Set(Object.values(snapshot.sources)
      .filter(row => row.status === 'ok').flatMap(row => row.items || [])
      .filter(item => item.section === 'research' && item.title?.trim())
      .map(item => item.abstract?.trim() ? `${item.title.trim()}. ${item.abstract.trim()}` : item.title.trim()))].slice(0,3);
    assert.ok(texts.length > 0, 'Research probe requires real public source text');
    const result=await page.evaluate(texts=>window.__rpc({type:'acceptance-public-cache',texts}),texts);
    assert.equal(result.status,200,result.body);
    const evidence=JSON.parse(result.body);
    assert.equal(evidence.before,0); assert.equal(evidence.valid,true);
    if (covered.length) {
      assert.equal(evidence.after,0); assert.equal(evidence.hits,texts.length);
    } else {
      assert.equal(evidence.after,texts.length); assert.equal(evidence.hits,0);
    }
    return {...evidence, published:build.public_embeddings.published,
      missing:build.public_embeddings.missing, localFallbackVerified:!covered.length};
  });
  await check('dynamic-internal-links-support-modified-clicks', async () => {
    await page.evaluate(() => {
      const doc = document.querySelector('#app').contentDocument;
      const link = doc.createElement('a'); link.id = 'dynamic-navigation-check';
      link.textContent = 'Open saved reading in another tab'; link.target = '_blank';
      doc.body.prepend(link); link.href = '/saved';
    });
    await page.waitForFunction(() => document.querySelector('#app').contentDocument.querySelector('#dynamic-navigation-check').href.includes('#/saved'));
    const target = frame().locator('#dynamic-navigation-check');
    assert.equal(await target.getAttribute('href'), base + '#/saved');
    const opened = await context.newPage();
    const response = await opened.goto(await target.getAttribute('href'), { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200);
    assert.equal(opened.url(), base + '#/saved');
    await opened.close(); await target.evaluate(node => node.remove());
    return { dynamicRoutePreserved: true, newTabTargetPublicSubpath: true };
  });
  await check('storage-protection-is-optional-and-local', async () => {
    await page.evaluate(() => {
      window.__storagePermissionCalls = 0;
      Object.defineProperty(navigator.storage, 'persist', { configurable: true, value: async () => { window.__storagePermissionCalls++; return false; } });
    });
    assert.equal(await page.evaluate(() => window.__storagePermissionCalls), 0);
    await toggleMenu('browser-data-menu');
    assert.match(await page.locator('#browser-data-menu').textContent(), /browser manages the local storage directory/);
    assert.equal(await page.locator('#protect-storage').isHidden(),true,'Advanced storage controls start collapsed');
    await page.locator('#browser-data-menu .data-tools > details > summary').click();
    assert.equal(await page.locator('#protect-storage').isVisible(),true);
    await page.locator('#protect-storage').click();
    await page.waitForFunction(() => document.querySelector('#storage-status').textContent.includes('not granted'));
    assert.equal(await page.evaluate(() => window.__storagePermissionCalls), 1);
    assert.equal(await page.locator('#storage-consent').isHidden(), true);
    await toggleMenu('browser-data-menu');
    return { explicitOnly: true, denialReported: true, noFolderMigration: true };
  });
  await check('explicit-compute-controls-and-device-warning', async () => {
    await route('/setup', '#interest-editor');
    await editInterest(frame().locator('.interest-row').first());
    await frame().locator('[data-interest-name]').first().fill('Unsaved compute test topic');
    await toggleMenu('compute-menu');
    await page.locator('#compute-apply').click();
    await page.locator('#compute-status').filter({hasText:'Save your unsaved settings'}).waitFor();
    assert.equal(await frame().locator('[data-interest-name]').first().inputValue(),'Unsaved compute test topic');
    assert.equal(await page.locator('#compute-apply').isEnabled(),true);
    await toggleMenu('compute-menu');
    // A fresh Settings navigation explicitly discards this fictional draft.
    await route('/setup', '#interest-editor');
    await frame().locator('[data-browser-action="open-guide"]').click();
    await page.locator('#quick-guide').waitFor();
    await page.locator('#close-guide').click();
    const current = await page.evaluate(async () => JSON.parse((await window.__rpc({ type: 'compute-status' })).body));
    await toggleMenu('compute-menu');
    const expectedDefault = await page.evaluate(() => crossOriginIsolated
      && navigator.hardwareConcurrency >= 4 && (navigator.deviceMemory ?? 4) >= 4 ? 'multicore' : 'cpu');
    assert.equal(current.mode, expectedDefault);
    assert.equal(await page.locator('#compute-mode').inputValue(), expectedDefault);
    assert.equal(await page.locator('#compute-mode option[value="multicore"]').isDisabled(), false);
    assert.equal(await page.locator('#compute-mode option[value="gpu"]').isDisabled(), !current.capabilities.gpu);
    assert.match(await page.locator('#compute-menu').textContent(), /heat, fan noise and battery use/);
    if (current.capabilities.gpu) {
      await page.locator('#compute-mode').selectOption('gpu');
      await Promise.all([page.waitForEvent('load'), page.locator('#compute-apply').click()]);
      await startup();
      await page.waitForFunction(() => document.querySelector('#compute-mode').value === 'gpu');
      await toggleMenu('compute-menu');
      await page.locator('#compute-mode').selectOption('cpu');
      await Promise.all([page.waitForEvent('load'), page.locator('#compute-apply').click()]);
      await startup();
    } else await toggleMenu('compute-menu');
    await frame().locator('#interest-editor').waitFor();
    return { unsavedSettingsPreserved:true, helpDoesNotDirtySettings:true,
      optInPersistsAcrossReload: current.capabilities.gpu, unsupportedChoicesDisabled: true, deviceWarning: true, cpuDefault: expectedDefault };
  });
  await check('muted-research-examples-require-explicit-use-and-save', async () => {
    const topics = frame().locator('#topics');
    assert.equal(await topics.inputValue(), '');
    assert.match(await topics.getAttribute('placeholder'), /RNA nanotechnology/);
    assert.equal(await frame().locator('#bio').inputValue(), '');
    await page.screenshot({ path: `${output}/research-placeholders.png`, fullPage: true });
    const disciplines = [];
    for (const [discipline, count, topic] of [['biomolecular',10,'RNA nanotechnology'], ['engineering',4,'Battery materials'], ['chemistry',4,'Asymmetric catalysis'], ['ecology',4,'Biodiversity monitoring'], ['astronomy',4,'Exoplanet atmospheres']]) {
      if (disciplines.length) await route('/setup', '#interest-editor');
      await frame().locator('#example-discipline').selectOption(discipline);
      assert.equal(await topics.inputValue(), '', 'Choosing a discipline must not load or save it.');
      await frame().locator('#use-example-profile').click();
      assert.equal((await topics.inputValue()).split('\n').length, count);
      assert.match(await topics.inputValue(), new RegExp(topic));
      assert.equal(await frame().locator('#bio').inputValue(), '', 'Examples must not write an implicit bio.');
      assert.equal(await frame().locator('#use-example-profile').isDisabled(), true);
      await frame().locator('#example-discipline').selectOption(discipline === 'ecology' ? 'engineering' : 'ecology');
      assert.match(await topics.inputValue(), new RegExp(topic), 'Changing an example choice must preserve current edits.');
      disciplines.push(discipline);
    }
    const beforeSave = await snapshot();
    assert.equal(Object.hasOwn(beforeSave.files, 'data/profile.yaml'), false);
    await page.screenshot({ path: `${output}/research-example.png`, fullPage: true });
    await route('/setup', '#interest-editor');
    return { examplesOnlyUntilChosen:true, disciplines, noImplicitBio:true, noProfileWrittenUntilSave:true };
  });
  await check('individual-interest-percentages-must-total-100', async () => {
    await setTopics('RNA delivery | 3\nProtein engineering | 1');
    await openSettingsGroup('priority-options');
    const weights = frame().locator('[data-interest-weight]');
    assert.deepEqual(await weights.evaluateAll(inputs => inputs.map(input => input.value)), ['75', '25']);
    assert.equal(await frame().locator('#topics').inputValue(), 'RNA delivery | 3\nProtein engineering | 1', 'Viewing rounded percentages must preserve untouched relative inputs.');
    await weights.first().fill('0');
    assert.match(await frame().locator('#interest-total').textContent(), /25%.*75% left/);
    assert.equal(await weights.first().evaluate(input => input.checkValidity()), false);
    await frame().locator('form[action="/setup"] button[type="submit"]').click();
    assert.equal(await page.evaluate(() => location.hash), '#/setup');
    await frame().locator('#normalize-interests').click();
    await assertPercentagesReady();
    await frame().locator('#add-interest').click();
    assert.equal(await weights.count(), 3);
    await assertPercentagesReady();
    await frame().locator('.interest-row').last().getByRole('button', { name: /Remove research topic/ }).click();
    await assertPercentagesReady();
    await weights.first().fill('90');
    await weights.first().press('Tab');
    assert.deepEqual(await weights.evaluateAll(inputs => inputs.map(input => input.value)), ['90', '10']);
    await weights.first().fill('101');
    assert.match(await frame().locator('#interest-total').textContent(), /111%.*11% over/);
    assert.equal(await weights.first().evaluate(input => input.checkValidity()), false);
    await frame().locator('#even-interests').click();
    assert.deepEqual(await weights.evaluateAll(inputs => inputs.map(input => input.value)), ['50', '50']);
    await route('/setup', '#interest-editor');
    return {separateFields:true, untouchedRawWeightsPreserved:true, underAndOverBlocked:true, addRemoveNormalizes:true, validEditBalancesOtherTopics:true};
  });
  await check('server-settings-validation-remains-editable-and-private', async () => {
    const before = await snapshot();
    await frame().locator('form[action="/setup"]').evaluate(form => {
      // A listener-free clone exercises server validation independently of the
      // editor's client checks, through the actual document submission bridge.
      const submitted = form.cloneNode(true);
      submitted.hidden = true;
      submitted.querySelector('#name').value = 'Fictional unsaved validation name';
      submitted.querySelector('#topics').value = 'RNA delivery | -1';
      submitted.querySelector('#llm_api_key').value = 'fixture-key-must-not-return';
      form.after(submitted);
      submitted.dispatchEvent(new SubmitEvent('submit', {bubbles:true, cancelable:true}));
    });
    await frame().locator('.errors').waitFor();
    assert.equal(await frame().locator('#name').inputValue(), 'Fictional unsaved validation name');
    assert.equal(await frame().locator('#llm_api_key').inputValue(), '');
    assert.equal((await frame().locator('body').innerHTML()).includes('fixture-key-must-not-return'), false);
    assert.match(await page.locator('#status').textContent(), /changes have not been saved/);
    assert.deepEqual((await snapshot()).files, before.files);
    await toggleMenu('compute-menu');
    await page.locator('#compute-apply').click();
    await page.waitForFunction(() => document.querySelector('#compute-status').textContent.includes('Save your unsaved settings'));
    await toggleMenu('compute-menu');
    await route('/setup', '#interest-editor');
    return {editableForm:true, draftPreserved:true, keyNotReflected:true, stateUnchanged:true, controlledReloadBlocked:true};
  });
  await check('local-wording-variants-are-explicit-and-do-not-autosave', async () => {
    await setTopics('quantum error correction (QEC) | 100');
    const row = frame().locator('.interest-row').first();
    const before=await snapshot(), spellingBefore=spellingRequests.length, externalBefore=report.requests.length;
    await editInterest(row, true);
    assert.equal(await row.locator('[data-interest-aliases]').inputValue(),'');
    await row.getByRole('button',{name:'Suggest wording variants',exact:true}).click();
    assert.equal(await row.locator('[data-interest-aliases]').inputValue(),'');
    await row.getByRole('button',{name:'Add “QEC”',exact:true}).click();
    assert.equal(await row.locator('[data-interest-aliases]').inputValue(),'QEC');
    assert.equal(await row.locator('[data-interest-name]').inputValue(),'quantum error correction (QEC)');
    await assertPercentagesReady();
    assert.deepEqual((await snapshot()).files,before.files,'Local suggestions change only the unsaved form');
    assert.equal(spellingRequests.length,spellingBefore);
    assert.equal(report.requests.length,externalBefore);
    await row.locator('[data-interest-name]').fill('Drug development');
    await row.locator('[data-interest-aliases]').fill('');
    await row.getByRole('button',{name:'Suggest wording variants',exact:true}).click();
    await row.locator('.hint').filter({hasText:'No additional wording variants found locally'}).waitFor();
    assert.equal(await row.locator('[data-interest-aliases]').inputValue(),'');
    assert.equal(spellingRequests.length,spellingBefore);
    assert.deepEqual((await snapshot()).files,before.files);
    await route('/setup','#interest-editor');
    return {noExternalRequest:true,noAutosave:true,explicitAcronym:true,topicAndPercentagePreserved:true,honestMissingVariants:true};
  });
  await check('interest-spelling-is-explicit-and-preserves-reader-control', async () => {
    await setTopics('induced provimity | 75\nassymetric catalysis | 25');
    const first = frame().locator('.interest-row').nth(0), second = frame().locator('.interest-row').nth(1);
    const firstStatus = first.locator('.interest-spelling .hint');
    assert.equal(spellingRequests.length, 0, 'Typing must not send an interest to the spelling provider.');
    assert.equal(await first.getByRole('button', {name:'Check spelling in PubMed', exact:true}).isHidden(), true);
    await editInterest(first, true);
    await editInterest(second, true);
    await openSettingsGroup('author-settings');
    await frame().locator('#authors_of_interest').fill('Fictional author');
    await openSettingsGroup('advanced-interests');
    await frame().locator('#seed_works').fill('Fictional representative paper');
    await frame().locator('#context_keywords').fill('Fictional background topic');
    const before = await snapshot();
    await editInterest(first, true);
    await first.getByRole('button', {name:'Check spelling in PubMed', exact:true}).click();
    await first.getByRole('button', {name:'Use suggestion', exact:true}).waitFor();
    assert.equal(await first.locator('[data-interest-name]').inputValue(), 'induced provimity');
    assert.deepEqual((await snapshot()).files, before.files, 'Checking wording must not save private state.');
    await first.getByRole('button', {name:'Use suggestion', exact:true}).click();
    assert.equal(await first.locator('[data-interest-name]').inputValue(), 'induced proximity');
    assert.equal(await first.locator('[data-interest-weight]').inputValue(), '75');
    await editInterest(second, true);
    await second.getByRole('button', {name:'Check spelling in PubMed', exact:true}).click();
    await second.getByRole('button', {name:'Keep wording', exact:true}).waitFor();
    await second.getByRole('button', {name:'Keep wording', exact:true}).click();
    assert.equal(await second.locator('[data-interest-name]').inputValue(), 'assymetric catalysis');
    await editInterest(first, true);
    await first.locator('[data-interest-name]').fill('RNA delivery');
    await editInterest(first, true);
    await first.getByRole('button', {name:'Check spelling in PubMed', exact:true}).click();
    await firstStatus.filter({hasText:'No spelling suggestion found'}).waitFor();
    assert.equal(await first.getByRole('button', {name:'Use suggestion', exact:true}).isHidden(), true);
    spellingMode = 'failure';
    await editInterest(first, true);
    await first.getByRole('button', {name:'Check spelling in PubMed', exact:true}).click();
    await firstStatus.filter({hasText:'Couldn’t check wording'}).waitFor();
    assert.equal(await first.locator('[data-interest-name]').inputValue(), 'RNA delivery');
    spellingMode = 'slow';
    await editInterest(first, true);
    await first.locator('[data-interest-name]').fill('induced provimity');
    const started = new Promise(resolve => { spellingStarted = resolve; });
    await editInterest(first, true);
    try {
      await first.getByRole('button', {name:'Check spelling in PubMed', exact:true}).click(); await started;
      assert.equal(await second.locator('[data-check-wording]').isDisabled(), true);
      await editInterest(first, true);
      await first.locator('[data-interest-name]').fill('New specialist topic');
      releaseSpelling(); spellingStarted = null;
      await first.getByRole('button', {name:'Check spelling in PubMed', exact:true}).waitFor({state:'visible'});
      await page.waitForFunction(() => !document.querySelector('#app').contentDocument.querySelector('[data-check-wording]').disabled);
      assert.equal(await first.getByRole('button', {name:'Use suggestion', exact:true}).isHidden(), true);
      assert.equal(await first.locator('[data-interest-name]').inputValue(), 'New specialist topic');
    } finally {
      releaseSpelling?.(); spellingStarted = null; spellingMode = 'success';
    }
    spellingMode = 'success';
    await editInterest(first, true);
    await first.locator('[data-interest-name]').fill('KRAS G12D inhibtor');
    await editInterest(first, true);
    await first.getByRole('button', {name:'Check spelling in PubMed', exact:true}).click();
    await firstStatus.filter({hasText:'No spelling suggestion found'}).waitFor();
    assert.equal(await first.locator('[data-interest-name]').inputValue(), 'KRAS G12D inhibtor');
    await editInterest(first, true);
    await first.locator('[data-interest-name]').fill('induced provimity');
    await editInterest(second, true);
    await second.locator('[data-interest-name]').fill('induced proximity');
    await editInterest(first, true);
    await first.getByRole('button', {name:'Check spelling in PubMed', exact:true}).click();
    await first.getByRole('button', {name:'Use suggestion', exact:true}).click();
    await firstStatus.filter({hasText:'Another interest already uses'}).waitFor();
    assert.equal(await first.locator('[data-interest-name]').inputValue(), 'induced provimity');
    await setTopics('induced provimity | 75\nassymetric catalysis | 25');
    for (const row of [first, second]) {
      await editInterest(row, true);
      await row.getByRole('button', {name:'Check spelling in PubMed', exact:true}).click();
      await row.getByRole('button', {name:'Use suggestion', exact:true}).click();
    }
    await editInterest(first, true);
    await first.locator('[data-interest-aliases]').fill('molecular glue');
    await first.locator('[data-interest-description]').fill('Validated proximity-inducing design methods.');
    await frame().locator('form[action="/setup"] button[type="submit"]').click();
    await page.waitForFunction(() => location.hash.startsWith('#/run'));
    const saved = await snapshot();
    const parsed = spawnSync(new URL('../.venv/bin/python', import.meta.url).pathname,
      ['-c', 'import yaml,json,sys; print(json.dumps(yaml.safe_load(sys.stdin.read())))'],
      {input:Buffer.from(saved.files['data/profile.yaml'],'base64').toString(),encoding:'utf8'});
    assert.equal(parsed.status,0,parsed.stderr);
    const profile = JSON.parse(parsed.stdout);
    assert.deepEqual(profile.keywords, ['induced proximity','asymmetric catalysis']);
    assert.deepEqual(profile.authors_of_interest, ['Fictional author']);
    assert.deepEqual(profile.seed_works, ['Fictional representative paper']);
    assert.deepEqual(profile.context_keywords, ['Fictional background topic']);
    assert.equal(profile.canonical_facets['induced proximity'].priority,75);
    assert.deepEqual(profile.canonical_facets['induced proximity'].aliases,['molecular glue']);
    assert.equal(profile.canonical_facets['induced proximity'].description,'Validated proximity-inducing design methods.');
    assert.equal(Object.hasOwn(profile.canonical_facets['induced proximity'],'previous_name'),false);
    await route('/setup','#interest-editor');
    assert.equal(await frame().locator('#authors_of_interest').inputValue(), 'Fictional author');
    assert.equal(await frame().locator('#seed_works').inputValue(), 'Fictional representative paper');
    assert.equal(await frame().locator('#context_keywords').inputValue(), 'Fictional background topic');
    return {advancedInterestsPreserved:true, explicitOnly:true,acceptAndDismiss:true,unavailable:true,staleReplyIgnored:true,identifiersPreserved:true,
      duplicatePrevented:true,correctedNamesSaved:true,weightsAndDetailsPreserved:true,providerRequests:spellingRequests.length};
  });
  await check('setup-native-form', async () => {
    await setTopics('DNA nanotechnology | 5\nRNA nanotechnology | 5\nnucleic acid delivery | 15\nmRNA therapeutics | 10\ngene therapy | 10\nnucleic acid therapeutics | 15\nTarget protein degradation | 15\nInduce proximity | 15\nDrug development | 10');
    await assertPercentagesReady();
    const degradation = frame().locator('.interest-row').nth(6);
    await editInterest(degradation, true);
    await degradation.locator('[data-interest-aliases]').fill('targeted protein degradation; PROTAC');
    await degradation.locator('[data-interest-description]').fill('New degrader designs and transferable methods.');
    const proximity = frame().locator('.interest-row').nth(7);
    await editInterest(proximity, true);
    await proximity.locator('[data-interest-aliases]').fill('induced proximity; molecular glue');
    await openSettingsGroup('profile-options');
    await frame().locator('#name').fill('Browser Acceptance');
    await frame().locator('#bio').fill('Fictional browser acceptance reader researching RNA delivery, protein design and biomaterials.');
    await frame().locator('form[action="/setup"] button[type="submit"]').click();
    await page.waitForFunction(() => {
      const path = document.querySelector('#app')?.contentWindow?.ddLocation;
      return path && !path.endsWith('/setup');
    }, undefined, { timeout: 60000 });
    const exported = await snapshot();
    assert.ok(Object.hasOwn(exported.files, '.env'));
    const parsed = spawnSync(new URL('../.venv/bin/python', import.meta.url).pathname,
      ['-c', 'import yaml,json,sys; print(json.dumps(yaml.safe_load(sys.stdin.read())))'],
      { input: Buffer.from(exported.files['data/profile.yaml'], 'base64').toString(), encoding: 'utf8' });
    assert.equal(parsed.status, 0, parsed.stderr);
    const profile = JSON.parse(parsed.stdout);
    assert.equal(profile.keywords.length, 9);
    assert.equal(Object.keys(profile.canonical_facets).length, 9);
    assert.equal(profile.canonical_facets['Target protein degradation'].priority, 15);
    assert.deepEqual(profile.canonical_facets['Induce proximity'].aliases, ['induced proximity', 'molecular glue']);
    assert.equal(exported.credentialsIncluded, false);
    return { privateConfigPersisted: true, allNineInterests: true, relativeWeightsPreserved: true, aliasesPersisted: true, credentialsExcluded: true, route: await page.evaluate(() => location.hash) };
  });
  await check('api-connection-tests-are-explicit-private-and-unsaved', async () => {
    await route('/setup', '#interest-editor');
    await openSettingsGroup('summary-options');
    const before = await snapshot();
    assert.equal(apiTestRequests.length, 0, 'Rendering settings must not contact an API.');
    for (const backend of ['api', 'anthropic']) {
      const initialCount = apiTestRequests.length;
      await frame().locator('#summary-backend').selectOption(backend);
      await frame().locator('#llm_base_url').fill(backend === 'anthropic' ? 'https://api.anthropic.com/v1' : 'https://api-fixture.invalid/v1');
      await frame().locator('#llm_api_key').fill('fictional-api-test-key');
      await frame().locator('#llm_model').fill('fictional-test-model');
      assert.equal(apiTestRequests.length, initialCount, 'Selecting and typing provider settings must not test them.');
      const inputState = await Promise.all(['llm_base_url','llm_api_key','llm_model'].map(id => frame().locator(`#${id}`).inputValue()));
      await frame().locator('#test-api-connection').click();
      await frame().locator('#connection-check-status').filter({hasText:'API connection verified'}).waitFor({timeout:60000});
      assert.equal(apiTestRequests.length, initialCount + 1);
      const request = apiTestRequests.at(-1);
      assert.equal(request.method, 'POST');
      assert.equal(new URL(request.url).pathname, backend === 'api' ? '/v1/chat/completions' : '/v1/messages');
      assert.deepEqual(request.body, {model:'fictional-test-model',max_tokens:8,messages:[{role:'user',content:'Reply OK.'}]});
      assert.doesNotMatch(JSON.stringify(request.body), /Browser Acceptance|RNA delivery|Fictional author|profile|keywords|digest\.db/);
      if (backend === 'anthropic') {
        assert.equal(request.headers['anthropic-dangerous-direct-browser-access'], 'true');
        assert.equal(request.headers['x-api-key'], 'fictional-api-test-key');
      } else {
        assert.equal(request.headers.authorization, 'Bearer fictional-api-test-key');
        assert.equal(request.headers['anthropic-dangerous-direct-browser-access'], undefined);
      }
      assert.deepEqual((await snapshot()).files, before.files, 'A connection test must not save profile, key or model settings.');
      apiTestMode = 'failure';
      await frame().locator('#test-api-connection').click();
      await frame().locator('#connection-check-status').filter({hasText:'Authentication was rejected'}).waitFor({timeout:60000});
      assert.deepEqual(await Promise.all(['llm_base_url','llm_api_key','llm_model'].map(id => frame().locator(`#${id}`).inputValue())), inputState);
      assert.doesNotMatch(await frame().locator('#connection-check-status').textContent(), /fictional-api-test-key|fixture key must not/);
      assert.deepEqual((await snapshot()).files, before.files);
      apiTestMode = 'success';
    }
    apiTestMode = 'slow';
    const started = new Promise(resolve => { apiTestStarted = resolve; });
    await frame().locator('#test-api-connection').click(); await started;
    assert.equal(await frame().locator('#test-api-connection').isDisabled(), true);
    await frame().locator('#llm_model').fill('new-unsaved-model');
    releaseApiTest(); apiTestStarted = null;
    await frame().locator('#test-api-connection:not([disabled])').waitFor();
    assert.equal(await frame().locator('#connection-check-status').textContent(), '');
    assert.equal(await frame().locator('#llm_model').inputValue(), 'new-unsaved-model');
    assert.deepEqual((await snapshot()).files, before.files);
    apiTestMode = 'success';
    await route('/setup', '#interest-editor');
    return {apiAndAnthropic:true, genericPromptOnly:true, explicitClickOnly:true, browserAnthropicHeaderScoped:true, noStateSaved:true, failurePreservesFields:true, staleReplyIgnored:true};
  });
  await check('legacy-priorities-show-percentages-without-changing-untouched-raw-weights', async () => {
    const backup = await snapshot();
    backup.files['data/profile.yaml'] = Buffer.from('bio: General reader.\nkeywords: [Astrophysics, History]\ncanonical_facets:\n  Astrophysics: {priority: 0.1}\n  History: {priority: 100}\n').toString('base64');
    const imported = await page.evaluate(body => window.__rpc({ type: 'import', body, overwrite: true }), JSON.stringify(backup));
    assert.equal(imported.status, 200, imported.body);
    await page.reload(); await startup();
    await route('/setup', '#interest-editor');
    assert.deepEqual(await frame().locator('[data-interest-weight]').evaluateAll(inputs => inputs.map(input => input.value)), ['1', '99']);
    const beforeSave = Buffer.from((await snapshot()).files['data/profile.yaml'], 'base64').toString();
    assert.match(beforeSave, /priority: 0\.1\b/);
    await openSettingsGroup('profile-options');
    await frame().locator('#name').fill('Renamed reader');
    await frame().locator('form[action="/setup"] button[type="submit"]').click();
    await page.waitForFunction(() => location.hash.startsWith('#/run'));
    const stored = await snapshot();
    const yaml = Buffer.from(stored.files['data/profile.yaml'], 'base64').toString();
    assert.match(yaml, /priority: 0\.1\b/);
    assert.match(yaml, /priority: 100(?:\.0)?\b/);
    return { noChangeUntilSave:true, untouchedExactWeightsPreserved:true, displayedPercentagesTotal100:true };
  });
  const backupText = await readFile(fixture, 'utf8');
  await check('import-fixture-backup', async () => {
    const response = await page.evaluate(body => window.__rpc({ type: 'import', body, overwrite: true }), backupText);
    assert.equal(response.status, 200, response.body);
    await page.reload(); await startup();
    await route('/', '.item[data-item-id]');
    return { digest: await digestIdentity(), items: await frame().locator('.item[data-item-id]').count() };
  });
  await check('calibration-disabled-until-real-rating-threshold',async()=>{
    await route('/','.item[data-item-id]');
    await openAboutDigest();
    assert.equal(await frame().locator('#train-ranker-btn').isDisabled(),true);
    return {disabledWithoutRequiredRatings:true};
  });
  const controlsFixture=readerControlsFixture(backupText);
  await writeFile(`${output}/reader-controls-fixture.json`,JSON.stringify(controlsFixture.backup));
  try {
    await importFixture(JSON.stringify(controlsFixture.backup));
    await check('inline-name-save-persists-in-heading',async()=>{
      await route('/','.name-prompt');
      await frame().locator('.name-prompt input[name="name"]').fill('Mira Control');
      await frame().locator('.name-prompt button[type="submit"]').click();
      await page.waitForFunction(()=>document.querySelector('#app')?.contentDocument?.querySelector('#page-salutation')?.dataset.profileName==='Mira Control');
      assert.equal(await frame().locator('.name-prompt').count(),0);
      await page.reload();await startup();await route('/','#page-salutation');
      assert.equal(await frame().locator('#page-salutation').getAttribute('data-profile-name'),'Mira Control');
      assert.match(Buffer.from((await snapshot()).files['data/profile.yaml'],'base64').toString(),/name: Mira Control/);
      return {savedThroughForm:true,retainedAfterReload:true};
    });
    await check('saved-search-preserves-query-across-next-and-previous-pages',async()=>{
      await route('/saved','.saved-card');
      await frame().locator('input[name="q"]').fill('Acceptance saved pagination');
      await frame().locator('form[role="search"] button').click();
      await frame().locator('#archive-count').filter({hasText:'55 saved items'}).waitFor();
      assert.equal(await frame().locator('.saved-card').count(),50);
      const first=await frame().locator('.saved-card').evaluateAll(nodes=>nodes.map(node=>node.dataset.itemId));
      await frame().getByRole('link',{name:'Next page',exact:true}).click();
      await frame().getByText('Page 2 of 2',{exact:true}).waitFor();
      assert.equal(await frame().locator('input[name="q"]').inputValue(),'Acceptance saved pagination');
      const second=await frame().locator('.saved-card').evaluateAll(nodes=>nodes.map(node=>node.dataset.itemId));
      assert.equal(second.length,5);assert.equal(new Set([...first,...second]).size,55);
      assert.equal(await frame().getByRole('link',{name:'Next page',exact:true}).count(),0);
      await frame().getByRole('link',{name:'Previous page',exact:true}).click();
      await frame().getByText('Page 1 of 2',{exact:true}).waitFor();
      assert.deepEqual(await frame().locator('.saved-card').evaluateAll(nodes=>nodes.map(node=>node.dataset.itemId)),first);
      await frame().locator('input[name="q"]').fill('Acceptance saved pagination 55');
      await frame().locator('form[role="search"] button').click();
      await frame().locator('#archive-count').filter({hasText:'1 saved item matching'}).waitFor();
      assert.equal(await frame().locator('.saved-card').count(),1);
      assert.equal(await frame().getByRole('navigation',{name:'Saved reading pages'}).count(),0);
      await frame().locator('input[name="q"]').fill('');
      await frame().locator('form[role="search"] button').click();
      await frame().locator('#archive-count').filter({hasText:`${55+controlsFixture.originalSaved} saved items`}).waitFor();
      assert.equal(await frame().locator('input[name="q"]').inputValue(),'');
      return {matched:55,firstPage:50,secondPage:5,queryRetained:true,clearRestoresArchive:true};
    });
    await check('calibration-button-runs-and-persists-real-calibrator',async()=>{
      await route('/','.item[data-item-id]');
      await openAboutDigest();
      assert.equal(await frame().locator('#train-ranker-btn').isEnabled(),true);
      await clickCalibration();
      const status=await completedCalibration();
      assert.equal(status.training_job.last_result.ok,true);
      assert.equal(status.training_job.last_result.trained,true);
      const stored=await snapshot();
      const calibration=JSON.parse(Buffer.from(stored.files['data/calibrator.json'],'base64').toString());
      assert.equal(calibration.n,30);assert.equal(calibration.policy,controlsFixture.policy);
      return {signedRatings:status.status.vote_counts.signed,calibratedVotes:calibration.n,policy:calibration.policy};
    });
    await check('calibration-failure-is-visible-and-can-retry',async()=>{
      const before=(await snapshot()).files['data/calibrator.json'];
      const blocked=await page.evaluate(()=>window.__rpc({type:'acceptance-calibration-obstruction',block:true}));
      assert.equal(blocked.status,200,blocked.body);
      try {
        await clickCalibration();
        const status=await completedCalibration();
        assert.equal(status.training_job.last_result.ok,false);
        assert.equal(status.training_job.last_result.reason,'training_error');
        report.calibrationFailure = status.training_job.last_result;
        await frame().locator('#toast').filter({hasText:/fail|could not/i}).waitFor();
        await frame().locator('#ranking-message').filter({hasText:/fail|could not/i}).waitFor();
        assert.equal((await snapshot()).files['data/calibrator.json'],before,'A failed refresh must retain the prior calibration');
      } finally {
        const cleared=await page.evaluate(()=>window.__rpc({type:'acceptance-calibration-obstruction',block:false}));
        assert.equal(cleared.status,200,cleared.body);
      }
      await clickCalibration();
      const retried=await completedCalibration();
      assert.equal(retried.training_job.last_result.ok,true);
      return {failedJobVisible:true,priorCalibrationRetained:true,retrySucceeded:true};
    });
  } finally {
    await importFixture(backupText);
    await route('/','.item[data-item-id]');
  }
  // Saved reading requires a configured profile; run after fixture import.
  await check('navigation-focus-has-no-black-box-and-keyboard-controls-stay-visible', async () => {
    const evidence = [];
    for (const path of ['/saved', '/setup']) {
      await route(path, 'h1');
      await page.waitForFunction(() => {
        const doc = document.querySelector('#app').contentDocument;
        return doc.activeElement === doc.querySelector('h1, main, [role="main"]');
      });
      assert.equal(await frame().locator('h1').evaluate(node => getComputedStyle(node).outlineStyle), 'none');
      await page.keyboard.press('Tab');
      const control = await page.evaluate(() => {
        const doc = document.querySelector('#app').contentDocument, node = doc.activeElement;
        return { tag: node.tagName, visible: node.matches(':focus-visible'), outline: doc.defaultView.getComputedStyle(node).outlineStyle };
      });
      assert.equal(control.visible, true);
      assert.notEqual(control.outline, 'none');
      evidence.push({ path, headingFocused: true, headingOutline: 'none', keyboardControl: control });
    }
    return evidence;
  });
  await check('route-title-landmark-and-reading-position-return', async () => {
    await route('/', '#digest-date');
    await openAboutDigest();
    await frame().locator('#source-details').scrollIntoViewIfNeeded();
    const previous=await page.evaluate(()=>document.querySelector('#app').contentWindow.scrollY);
    assert.ok(previous>100);
    await route('/setup', '#interest-editor');assert.match(await page.title(), /Set up DailyDigest/);
    assert.equal(await frame().locator('main').count(),1);
    await route('/', '#digest-date');
    const returned=await page.evaluate(()=>document.querySelector('#app').contentWindow.scrollY);
    assert.ok(Math.abs(returned-previous)<5,JSON.stringify({previous,returned}));
    assert.match(await page.title(), /DailyDigest/);
    return {previous,returned,routeTitles:true,oneMain:true};
  });
  await check('history-calendar-display-and-pip', async () => {
    await route('/', '.item[data-item-id]');
    const history = frame().locator('#digest-date option');
    const historyCount = await history.count();
    assert.ok(historyCount >= 2, 'Fixture must include at least two historical digests.');
    const selected = await history.first().getAttribute('value');
    await chooseDigest(selected);
    await frame().locator('#display-menu-toggle').click();
    await frame().locator('#theme-toggle').click();
    const root = frame().locator('html');
    assert.equal(await root.getAttribute('data-theme'), 'dark');
    await page.waitForFunction(() => document.documentElement.dataset.theme === 'dark');
    await frame().locator('#display-toggle').click();
    assert.equal(await root.getAttribute('data-display'), 'compact');
    const pip = frame().locator('#pip-toggle');
    if (await pip.isVisible()) { await pip.click(); assert.equal(await root.getAttribute('data-pip'), 'hidden'); }
    await route('/', '.item[data-item-id]');
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
    await openSection('events');
    const calendar = frame().locator('a[href*="#/calendar/"]').first();
    await calendar.waitFor();
    const readingHash = await page.evaluate(() => location.hash);
    const downloadEvent = page.waitForEvent('download');
    await calendar.click();
    const download = await downloadEvent;
    await download.saveAs(`${output}/calendar.ics`);
    const contents = await readFile(`${output}/calendar.ics`, 'utf8');
    assert.match(contents, /BEGIN:VCALENDAR/);
    assert.equal(await page.evaluate(() => location.hash), readingHash, 'Calendar export must preserve the reading route for reload.');
    return { historyEntries: historyCount, calendar: true, display: 'compact', theme: 'dark' };
  });
  await check('real-browser-mutation-and-impression-checkpoints', async () => {
    await route('/', '.item[data-section="research"]');
    const id = await frame().locator('.item[data-section="research"]').first().getAttribute('data-item-id');
    const token = await frame().locator('body').evaluate(() => {
      const text = [...document.scripts].map(node => node.textContent).join('\n');
      return JSON.parse(text.match(/const CSRF_TOKEN\s*=\s*("[^"\n]+")/)[1]);
    });
    const evidence = [];
    for (const [method, path] of [['POST', `/vote/${id}/70`], ['POST', `/bookmark/${id}`],
      ['DELETE', `/bookmark/${id}`], ['POST', `/known/${id}`], ['DELETE', `/known/${id}`],
      ['POST', `/vote/${id}/10`], ['POST', `/vote/${id}/reason/off_topic`],
      ['DELETE', `/vote/${id}/reason/off_topic`], ['DELETE', `/vote/${id}`]]) {
      const before = JSON.parse((await page.evaluate(() => window.__rpc({type:'acceptance-state'}))).body).generation;
      const result = await page.evaluate(({method,path,token}) => window.__rpc({type:'request', method, path, headers:{'X-CSRF-Token':token}}), {method,path,token});
      assert.equal(result.status, 200, result.body);
      const after = JSON.parse((await page.evaluate(() => window.__rpc({type:'acceptance-state'}))).body);
      assert.ok(after.generation > before, `Untracked ${method} ${path}`);
      assert.equal(after.saved && after.equal, true);
      evidence.push({method,path,generation:after.generation});
    }
    const reset = await page.evaluate(() => window.__rpc({type:'acceptance-state', action:'reset-view'}));
    assert.equal(reset.status, 200, reset.body);
    const before = JSON.parse(reset.body).generation;
    const digest = await page.evaluate(() => window.__rpc({type:'request',method:'GET',path:'/'}));
    assert.equal(digest.status, 200, digest.body.slice(0,200));
    const after = JSON.parse((await page.evaluate(() => window.__rpc({type:'acceptance-state'}))).body);
    assert.ok(after.generation > before, 'Digest GET must track its impression writes');
    assert.equal(after.saved && after.equal, true);
    return {mutations:evidence, impressionGet:true};
  });
  await check('sql-checkpoint-defers-mid-transaction-and-validates-after-commit', async () => {
    const probe = action => page.evaluate(action => window.__rpc({type:'acceptance-state', action}), action);
    const during = await probe('begin');
    assert.equal(during.status, 200, during.body);
    assert.deepEqual([JSON.parse(during.body).ready, JSON.parse(during.body).saved], [false, false]);
    const after = await probe('commit');
    assert.equal(after.status, 200, after.body);
    const evidence = JSON.parse(after.body);
    assert.equal(evidence.ready && evidence.saved && evidence.equal, true);
    return evidence;
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
    await route('/', '#digest-date');
    const archivedPath = await frame().locator('#digest-date option').last().getAttribute('value');
    await chooseDigest(archivedPath);
    const archivedId=new URL(archivedPath,base).searchParams.get('digest_id');
    assert.equal((await frame().locator('.digest-date').textContent()).trim(),archivedId);
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
    const gradeBefore=await card.getAttribute('data-current-grade');
    await card.evaluate(node=>{node.__acceptanceOriginalCard=true;});
    await card.locator('.remove-saved').click();
    await frame().locator(`.saved-card[data-item-id="${itemId}"][data-saved-state="removed"]`).waitFor();
    assert.equal(await card.locator('.saved-content').isHidden(),true);
    assert.equal(await card.locator('.saved-removal').isVisible(),true);
    const removed=await page.evaluate(()=>window.__rpc({type:'request',path:'/saved'}));
    assert.equal(removed.status,200,removed.body);
    assert.ok(!removed.body.includes(`data-item-id="${itemId}"`),'Removed bookmark must be absent from the actual Saved route');
    const removedCheckpoint=await page.evaluate(()=>window.__rpc({type:'acceptance-state'}));
    assert.equal(removedCheckpoint.status,200,removedCheckpoint.body);
    assert.equal(JSON.parse(removedCheckpoint.body).equal,true,'Removal must be durably checkpointed');
    await card.locator('.undo-saved').click();
    await frame().locator(`.saved-card[data-item-id="${itemId}"][data-saved-state="saved"]`).waitFor();
    assert.equal(await card.locator('.saved-content').isVisible(),true);
    assert.equal(await card.locator('.saved-removal').isHidden(),true);
    assert.equal(await card.evaluate(node=>node.__acceptanceOriginalCard),true,'Undo restores the original node');
    assert.equal(await card.getAttribute('data-current-grade'),gradeBefore);
    await card.locator('.vote-btn[data-grade="10"][aria-pressed="true"]').waitFor();
    await card.locator('.reason-chip[data-reason="off_topic"][aria-pressed="true"]').waitFor();
    const restored=await page.evaluate(()=>window.__rpc({type:'request',path:'/saved'}));
    assert.equal(restored.status,200,restored.body);
    assert.ok(restored.body.includes(`data-item-id="${itemId}"`),'Undo must restore the actual bookmark');
    const restoredCheckpoint=await page.evaluate(()=>window.__rpc({type:'acceptance-state'}));
    assert.equal(restoredCheckpoint.status,200,restoredCheckpoint.body);
    assert.equal(JSON.parse(restoredCheckpoint.body).equal,true,'Undo must be durably checkpointed');
    await card.locator('.clear-rating').click();
    await card.locator('.clear-rating').waitFor({ state: 'hidden' });
    assert.equal(await card.locator('.vote-btn[aria-pressed="true"]').count(), 0);
    const search = frame().locator('form[role="search"] input[name="q"]');
    await search.fill('impossibleAcceptanceSearchToken');
    await search.press('Enter');
    await page.waitForFunction(() => document.querySelector('#app')?.contentWindow?.ddLocation?.includes('impossibleAcceptanceSearchToken'));
    assert.equal(await frame().locator('.saved-card').count(), 0);
    return {cleared:true,searchFilters:true,durableTombstone:true,undoRestoresOriginalNodeAndRating:true};
  });
  await check('failed-compute-isolation-apply-leaves-feedback-usable', async () => {
    await page.evaluate(()=>{
      window.__originalSWRegister=navigator.serviceWorker.register;
      navigator.serviceWorker.register=async()=>{throw Error('Fixture registration failed');};
    });
    try {
      await toggleMenu('compute-menu');
      await page.locator('#compute-mode').selectOption('multicore');
      await page.locator('#compute-apply').click();
      await page.waitForFunction(()=>document.querySelector('#compute-status').textContent.includes('Fixture registration failed'));
      assert.equal(await page.locator('#compute-apply').isDisabled(),false);
      const csrf=await page.evaluate(()=>{
        const scripts=[...document.querySelector('#app').contentDocument.scripts].map(node=>node.textContent).join('\n');
        return JSON.parse(scripts.match(/const CSRF_TOKEN\s*=\s*("[^"\n]+")/)[1]);
      });
      const vote=await page.evaluate(({id,csrf})=>window.__rpc({type:'request',method:'POST',path:`/vote/${id}/70`,headers:{'x-csrf-token':csrf}}),{id:itemId,csrf});
      assert.equal(vote.status,200,vote.body);
      assert.equal(JSON.parse(vote.body).new_grade,70);
      const clear=await page.evaluate(({id,csrf})=>window.__rpc({type:'request',method:'DELETE',path:`/vote/${id}`,headers:{'x-csrf-token':csrf}}),{id:itemId,csrf});
      assert.equal(clear.status,200,clear.body);
    } finally {
      await page.evaluate(()=>{
        navigator.serviceWorker.register=window.__originalSWRegister;
        localStorage.removeItem(`dailydigest:isolation:${location.pathname}`);
      });
      const reset=await page.evaluate(()=>window.__rpc({type:'configure-compute',mode:'cpu'}));
      assert.equal(reset.status,200,reset.body);
      await page.reload();await startup();
    }
    return {registrationRejected:true,applyUsableAgain:true,realFeedbackPostSucceeded:true,readingStatePreserved:true};
  });
  await check('mobile-layout-and-keyboard', async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    await route('/setup', '#interest-editor');
    const draftSnapshot = await snapshot();
    const draftHash = await page.evaluate(() => location.hash);
    const draftRow = frame().locator('.interest-row').first();
    await editInterest(draftRow);
    await draftRow.locator('[data-interest-name]').fill('Unsaved draft navigation sample');
    await page.evaluate(() => { window.__retainedDraftDocument = document.querySelector('#app').contentDocument; });
    const promptsBefore = discardedDraftPrompts.length;
    try {
      acceptDiscardedDraft = false;
      const rejectedNavigation = page.waitForEvent('dialog', {predicate: dialog => dialog.message() === 'Discard unsaved settings changes?'});
      await page.evaluate(() => document.querySelector('#app').contentWindow.ddNavigate('/saved'));
      await rejectedNavigation;
      assert.deepEqual(discardedDraftPrompts.slice(promptsBefore), [false]);
      assert.equal(await page.evaluate(() => location.hash), draftHash);
      assert.equal(await page.evaluate(() => document.querySelector('#app').contentDocument === window.__retainedDraftDocument), true,'Rejecting draft navigation retains the same Settings document');
      assert.equal(await draftRow.locator('[data-interest-name]').inputValue(), 'Unsaved draft navigation sample');
      assert.deepEqual((await snapshot()).files, draftSnapshot.files);
      acceptDiscardedDraft = true;
      await route('/saved', 'h1');
      assert.deepEqual(discardedDraftPrompts.slice(promptsBefore), [false, true]);
      assert.equal(await page.evaluate(() => location.hash), '#/saved');
      assert.deepEqual((await snapshot()).files, draftSnapshot.files);
      await route('/setup', '#interest-editor');
      assert.notEqual(await frame().locator('[data-interest-name]').first().inputValue(), 'Unsaved draft navigation sample');
    } finally { acceptDiscardedDraft = true; }
    await openSettingsGroup('profile-options');
    const overflow = await page.evaluate(() => {
      const doc = document.querySelector('#app').contentDocument;
      return { width: doc.documentElement.clientWidth, scrollWidth: doc.documentElement.scrollWidth, outerWidth: document.documentElement.clientWidth, outerScrollWidth: document.documentElement.scrollWidth };
    });
    assert.ok(overflow.scrollWidth <= overflow.width + 1, JSON.stringify(overflow));
    assert.ok(overflow.outerScrollWidth <= overflow.outerWidth + 1, JSON.stringify(overflow));
    const statusBox = await page.locator('#status').boundingBox();
    assert.ok(statusBox.width > 100, JSON.stringify(statusBox));
    await openCompanionMenu();
    const companionBox = await page.locator('#companion-menu .data-tools').boundingBox();
    assert.ok(companionBox.x >= 0 && companionBox.x + companionBox.width <= 390, JSON.stringify(companionBox));
    await page.locator('#companion-menu > summary').press('Escape');
    assert.equal(await frame().locator('[data-interest-weight]').first().isVisible(), true,'390px: inline topic priority remains visible after Companion Escape');
    const firstRow=frame().locator('.interest-row').first();
    await editInterest(firstRow);
    await firstRow.locator('[data-interest-name]').focus(); await page.keyboard.press('Tab');
    assert.equal(await firstRow.locator('[data-interest-weight]').evaluate(node => node === document.activeElement), true,'390px: Tab from topic name focuses its inline percentage');
    await page.keyboard.press('Tab');
    assert.equal(await firstRow.locator('[data-interest-edit]').evaluate(node => node === document.activeElement), true,'390px: Tab from inline percentage focuses Edit');
    await firstRow.locator('[data-interest-done]').click();
    await firstRow.locator('[data-interest-edit]').press('Enter');
    assert.equal(await firstRow.locator('[data-interest-name]').isVisible(),true,'390px: Enter on Edit reopens the topic-name editor');
    await frame().locator('#priority-options > summary').focus(); await page.keyboard.press('Enter');
    // Native details dispatches toggle after Enter; that handler reveals these controls.
    await frame().locator('#normalize-interests').waitFor({state:'visible'});
    assert.equal(await frame().locator('#normalize-interests').isVisible(),true,'390px: Enter on Balance priorities reveals Normalize');
    await page.screenshot({ path: `${output}/mobile-setup.png`, fullPage: true });
    await page.setViewportSize({ width: 320, height: 720 });
    const narrow = await page.evaluate(() => {
      const doc = document.querySelector('#app').contentDocument;
      return {inner:doc.documentElement.scrollWidth <= doc.documentElement.clientWidth + 1, outer:document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1};
    });
    assert.deepEqual(narrow, {inner:true, outer:true});
    await frame().locator('#author-settings > summary').focus(); await page.keyboard.press('Enter');
    assert.notEqual(await frame().locator('#author-settings').getAttribute('open'), null);
    await frame().locator('#authors_of_interest').focus(); await page.keyboard.press('Tab');
    assert.equal(await frame().locator('#author-settings .settings-help>summary').evaluate(node=>node===document.activeElement),true,'320px: Tab from followed researchers focuses its Help disclosure');
    await frame().locator('#advanced-interests > summary').focus(); await page.keyboard.press('Enter');
    assert.notEqual(await frame().locator('#advanced-interests').getAttribute('open'), null);
    await frame().locator('#seed_works').focus(); await page.keyboard.press('Tab');
    assert.equal(await frame().locator('#context_keywords').evaluate(node => node === document.activeElement), true,'320px: Tab from paper examples focuses background interests');
    await frame().locator('.settings-actions button[type="submit"]').focus();
    assert.equal(await frame().locator('.settings-actions button[type="submit"]').evaluate(node => node === document.activeElement), true,'320px: Save settings accepts keyboard focus');
    await page.screenshot({path:`${output}/mobile-320-settings.png`,fullPage:true});
    await page.setViewportSize({ width: 1280, height: 900 });
    const zoomed = await page.evaluate(() => {
      const doc = document.querySelector('#app').contentDocument;
      document.documentElement.style.zoom = '2';
      return {inner:doc.documentElement.scrollWidth <= doc.documentElement.clientWidth + 1, outer:document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1};
    });
    assert.deepEqual(zoomed, {inner:true, outer:true});
    await page.screenshot({path:`${output}/settings-200-percent.png`,fullPage:true});
    await page.evaluate(() => { document.documentElement.style.zoom = ''; document.querySelector('#app').contentDocument.documentElement.style.zoom = ''; });
    await page.setViewportSize({ width: 800, height: 900 });
    await toggleMenu('compute-menu');
    const computeBox = await page.locator('#compute-menu .data-tools').boundingBox();
    assert.ok(computeBox.x >= 0 && computeBox.x + computeBox.width <= 800, JSON.stringify(computeBox));
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#compute-menu').getAttribute('open'), null);
    await page.setViewportSize({ width: 1360, height: 900 });
    return {...overflow, rejectedDraftNavigationRetainsDocumentAndDraft:true, acceptedDraftNavigationDiscardsOnlyUnsavedChanges:true};
  });
  await check('gpu-probe-does-not-load-model-and-explains-result', async () => {
    const modelRequests = [];
    const listener = request => {if (/\.onnx(?:\?|$)/.test(request.url())) modelRequests.push(request.url());};
    page.on('request', listener);
    try {
      await toggleMenu('compute-menu');
      await page.locator('#test-gpu').click();
      await page.waitForFunction(() => !document.querySelector('#test-gpu').disabled);
      const message = await page.locator('#gpu-test-result').textContent();
      assert.match(message, /GPU|WebGPU/);
      assert.equal(modelRequests.length, 0);
      await toggleMenu('compute-menu');
      return {message, modelRequests:0};
    } finally {page.off('request', listener);}
  });
  await check('release-and-stale-source-notices-are-nondisruptive', async () => {
    const hash = await page.evaluate(() => location.hash);
    let stale = true;
    const handler = async request => {
      const response = await request.fetch();
      const manifest = await response.json();
      if (stale) {manifest.build_id='acceptance-next-release';manifest.public_sources={oldest_fetched_at:new Date(Date.now()-49*3600000).toISOString()};}
      else manifest.public_sources={oldest_fetched_at:new Date().toISOString()};
      await request.fulfill({response,body:JSON.stringify(manifest)});
    };
    await page.route('**/assets/build-manifest.json*', handler);
    try {
      await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
      await page.locator('#release-notice').waitFor({state:'visible'});
      await page.locator('#source-freshness').waitFor({state:'visible'});
      assert.match(await page.locator('#source-freshness').textContent(), /49 hours.*expired/);
      assert.equal(await page.locator('#release-reload').isDisabled(),false);
      assert.equal(await page.evaluate(()=>location.hash),hash);
      await page.locator('#release-dismiss').click();
      await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
      await page.waitForTimeout(50);
      assert.equal(await page.locator('#release-notice').isHidden(),true);
      stale=false;
      await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
      await page.locator('#source-freshness').waitFor({state:'hidden'});
      return {noAutomaticReload:true, originalTimestampAge:true, expiryVisible:true, dismissRememberedInTab:true};
    } finally {await page.unroute('**/assets/build-manifest.json*',handler);}
  });
  await check('brew-cancellation-preserves-digest', async () => {
    const before = await digestIdentity();
    await route('/run?reading_mode=usual', '#brew-button');
    await frame().locator('#browser-compute-notice').waitFor({ state: 'visible' });
    assert.match(await frame().locator('#browser-compute-notice').textContent(), /runs locally.*Keep this tab open/);
    await frame().locator('#brew-button').click();
    await frame().locator('#cancel-button').click();
    await frame().locator('#err').waitFor({ state: 'visible', timeout: 300000 });
    assert.match(await frame().locator('#err').textContent(), /stop|cancel/i);
    assert.equal(await frame().locator('#progress-panel').getAttribute('data-state'), 'paused');
    assert.equal(await frame().locator('.cup .steam').first().evaluate(node => getComputedStyle(node).animationName), 'none');
    assert.match(await frame().locator('#brew-title').textContent(), /needs attention/);
    const after = await digestIdentity(before); assert.equal(after, before);
    return { before, after, waitingAnimationStopped: true };
  });
  await check('stop-aborts-a-pending-source-fetch', async () => {
    const before = await digestIdentity();
    const csrf = await page.evaluate(() => {
      const scripts = [...document.querySelector('#app').contentDocument.scripts].map(node => node.textContent).join('\n');
      const value = scripts.match(/const CSRF_TOKEN\s*=\s*("[^"\n]+")/);
      if (!value) throw new Error('Fictional reading page did not expose its CSRF token');
      return JSON.parse(value[1]);
    });
    let timeout;
    const started = new Promise((resolve, reject) => {
      pendingSlowSource = resolve;
      timeout = setTimeout(() => reject(new Error('Slow fixture was not requested')), 30000);
    });
    sourceMode = 'slow';
    try {
      await route('/run?reading_mode=usual', '#brew-button');
      await frame().locator('#brew-button').click();
      await started; clearTimeout(timeout);
      await toggleMenu('compute-menu');
      await page.waitForFunction(() => document.querySelector('#compute-apply').disabled && document.querySelector('#compute-status').textContent.includes('brew is running'));
      const refused = await page.evaluate(() => window.__rpc({ type: 'configure-compute', mode: 'gpu' }));
      assert.equal(refused.status, 409);
      await toggleMenu('compute-menu');
      const vote = await page.evaluate(({ id, csrf }) => window.__rpc({ type: 'request',
        method: 'POST', path: `/vote/${id}/70`, headers: { 'x-csrf-token': csrf } }), { id: itemId, csrf });
      assert.equal(vote.status, 200, vote.body);
      assert.equal(JSON.parse(vote.body).new_grade, 70);
      const beginning = performance.now();
      await frame().locator('#cancel-button').click();
      await frame().locator('#err').waitFor({ state: 'visible', timeout: 5000 });
      const elapsedMs = performance.now() - beginning;
      assert.match(await frame().locator('#err').textContent(), /stop|cancel/i);
      await page.waitForFunction(() => !document.querySelector('#compute-apply').disabled);
      assert.equal(await digestIdentity(), before);
      await page.reload(); await startup();
      await route('/saved', '.saved-card');
      const card = frame().locator(`.saved-card[data-item-id="${itemId}"]`);
      await card.locator('.vote-btn[data-grade="70"][aria-pressed="true"]').waitFor();
      await card.locator('.clear-rating').click();
      await card.locator('.clear-rating').waitFor({ state: 'hidden' });
      return { cancelledDuringFetch: true, elapsedMs: Math.round(elapsedMs), savedDigestPreserved: true, midBrewVoteDurableAfterReload: true, computeSwitchBlockedDuringBrew: true, computeSwitchEnabledAfterStop: true };
    } finally { sourceMode = 'success'; pendingSlowSource = null; clearTimeout(timeout); }
  });
  await check('reload-during-brew-keeps-last-checkpoint-and-can-brew-again', async () => {
    await successfulBrew('usual'); // Populate the actual SQLite item-embedding cache first.
    const cachedBefore=await page.evaluate(()=>window.__rpc({type:'acceptance-embedding-cache'}));
    assert.equal(cachedBefore.status,200,cachedBefore.body);
    const embeddingRows=JSON.parse(cachedBefore.body);
    assert.ok(embeddingRows.length>0,'Require real cached vectors before testing interrupted restoration');
    const before = await digestIdentity();
    let timeout;
    const started = new Promise((resolve, reject) => {
      pendingSlowSource = resolve;
      timeout = setTimeout(() => reject(new Error('Slow reload fixture was not requested')), 30000);
    });
    sourceMode = 'slow';
    try {
      await route('/run?reading_mode=usual', '#brew-button');
      await frame().locator('#brew-button').click();
      await started; clearTimeout(timeout);
      const checkpoint = await page.evaluate(()=>window.__rpc({type:'acceptance-committed-snapshot'}));
      assert.equal(checkpoint.status,200,checkpoint.body);
      const during = JSON.parse(checkpoint.body);
      await page.route('**/assets/build-manifest.json*', async request => {
        const response = await request.fetch(); const manifest = await response.json();
        manifest.build_id = 'acceptance-busy-next-release';
        await request.fulfill({response, body:JSON.stringify(manifest)});
      }, {times:1});
      await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
      await page.locator('#release-notice').waitFor({state:'visible'});
      assert.equal(await page.locator('#release-reload').isDisabled(), true, 'Release update must wait for the active brew');
      assert.match(await page.locator('#release-message').textContent(), /Finish or stop this brew/);
      const requestsBefore = report.fixtureRequests.length;
      await page.reload(); await startup();
      await page.locator('#interruption-notice').waitFor({state:'visible'});
      assert.match(await page.locator('#interruption-message').textContent(), /interrupted.*last saved digest and feedback/);
      assert.ok((await page.evaluate(()=>location.hash)).startsWith('#/run'), 'Run route survives reload');
      assert.equal(report.fixtureRequests.length, requestsBefore, 'Reload must not silently restart a brew');
      const after = await snapshot();
      assert.equal(after.files['data/digest.db'], during.files['data/digest.db'], 'Interrupted brew restores the exact committed SQLite checkpoint');
      const cachedAfter=await page.evaluate(()=>window.__rpc({type:'acceptance-embedding-cache'}));
      assert.equal(cachedAfter.status,200,cachedAfter.body);
      assert.deepEqual(JSON.parse(cachedAfter.body),embeddingRows,'Committed vector bytes, model identity and text hashes must survive reload unchanged');
      assert.equal(await digestIdentity(), before);
      sourceMode = 'success'; pendingSlowSource = null;
      await page.locator('#interruption-dismiss').click();
      await successfulBrew('usual');
      return {interruptedNotice:true, checkpointExact:true, committedEmbeddingRows:embeddingRows.length, runRouteRestored:true, noSilentRestart:true, releaseWaitsForBrew:true, nextBrewSucceeded:true};
    } finally {
      sourceMode = 'success'; pendingSlowSource = null; clearTimeout(timeout);
      // A failed assertion must not leave a running fixture job for later cases.
      const status=await page.evaluate(async()=>JSON.parse((await window.__rpc({type:'compute-status'})).body));
      if(status.busy && await frame().locator('#cancel-button').count()) {
        await frame().locator('#cancel-button').click();
        await page.waitForFunction(()=>!document.querySelector('#compute-apply').disabled);
      }
    }
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
    assert.equal(await frame().locator('#mode-minimal + label .mode-title').textContent(), 'Short');
    assert.equal(await frame().locator('#mode-usual + label .mode-title').textContent(), 'Standard');
    assert.equal(await frame().locator('#mode-full + label .mode-title').textContent(), 'Full');
    await page.evaluate(()=>{
      window.__qualifiedPreview=[];
      const doc=document.querySelector('#app').contentDocument, preview=doc.querySelector('#brew-preview');
      new doc.defaultView.MutationObserver(()=>{
        if(!preview.hidden)window.__qualifiedPreview=[...preview.querySelectorAll('a')].map(node=>node.textContent);
      }).observe(preview,{attributes:true,attributeFilter:['hidden']});
    });
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
    const preview=await page.evaluate(()=>window.__qualifiedPreview);
    assert.ok(preview.length>0 && preview.length<=8,JSON.stringify(preview));
    assert.ok(preview.every(title=>titles.includes(title)), 'Provisional picks must belong to the final qualified slate.');
    assert.match(await frame().locator('.item').first().textContent(), /Fixture Author.*Nature/s);
    assert.equal(await frame().locator('.item').first().locator('.publication-label').textContent(), 'Publication');
    assert.match(await frame().locator('.item').first().locator('.match-topic').textContent(), /Text contains|Topic terms found|Related by meaning|Topic fit is uncertain|Title only/);
    await page.screenshot({ path: `${output}/brewed-digest.png`, fullPage: true });
    return { digest: await digestIdentity(), selected: await frame().locator('.item[data-item-id]').count(), slate, fixtureRequests: report.fixtureRequests.length };
  });
  await check('source-coverage-reports-the-readers-latest-check', async () => {
    const coverage = frame().locator('.source-coverage[aria-label="Source coverage and freshness"]');
    assert.match(await coverage.textContent(), /1 of 1 sources responded/);
    assert.match(await coverage.textContent(), /Checked.*UTC/s);
    await openAboutDigest();
    await coverage.locator('summary').click();
    assert.match(await coverage.locator('li').first().textContent(), /Nature.*Responded.*Checked/s);
    await page.screenshot({ path: `${output}/source-coverage.png`, fullPage: true });
    return { selectedSourceCountOnly: true, checkTimestampVisible: true, sourceDetailsAvailable: true };
  });
  await check('gpu-opt-in-runs-locally-or-recovers-on-cpu', async () => {
    const status = await page.evaluate(async () => JSON.parse((await window.__rpc({ type: 'compute-status' })).body));
    if (!status.capabilities.gpu) return { unavailableChoiceDisabled: true };
    const configured = await page.evaluate(() => window.__rpc({ type: 'configure-compute', mode: 'gpu' }));
    assert.equal(configured.status, 200, configured.body);
    await page.reload(); await startup();
    await successfulBrew();
    const active = await page.evaluate(async () => JSON.parse((await window.__rpc({ type: 'compute-status' })).body));
    const fallback = await page.evaluate(() => window.__workerEvents.some(event => event.type === 'progress' && event.stage === 'Computing embeddings' && /CPU/.test(event.message || '')));
    assert.ok(active.device === 'webgpu' || (active.device === 'wasm' && fallback));
    await toggleMenu('compute-menu');
    await page.waitForFunction(device => document.querySelector('#compute-status').textContent.includes(device === 'webgpu' ? 'Used GPU for' : 'GPU requested; using CPU fallback'), active.device);
    await toggleMenu('compute-menu');
    const restored = await page.evaluate(() => window.__rpc({ type: 'configure-compute', mode: 'cpu' }));
    assert.equal(restored.status, 200, restored.body);
    await page.reload(); await startup();
    await route('/', '.item[data-item-id]');
    return { selectedMode: 'gpu', actualDevice: active.device, cpuFallbackVisible: active.device === 'wasm' && fallback, cpuRestored: true };
  });
  await check('degraded-research-preserves-last-digest', async () => {
    const before = await digestIdentity(); sourceMode = 'failure';
    await route('/run?reading_mode=minimal', '#brew-button');
    await frame().locator('#mode-minimal').check(); await frame().locator('#brew-button').click();
    await frame().locator('#err').waitFor({ state: 'visible', timeout: 300000 });
    const message = await frame().locator('#err').textContent();
    assert.match(message, /research|source|coverage|fetch|ingest/i);
    const after = await digestIdentity(before); assert.equal(after, before);
    assert.match(await frame().locator('.source-coverage[aria-label="Source coverage and freshness"]').textContent(), /0 of 1 sources responded.*1 unavailable/s);
    sourceMode = 'success'; return { before, after, message, unavailableCoverageVisible: true };
  });
  await check('backup-download-and-import-confirmation', async () => {
    const exported = await snapshot();
    await writeFile(`${output}/exported-backup.json`, JSON.stringify(exported));
    await toggleMenu('browser-data-menu');
    const downloadEvent = page.waitForEvent('download'); await page.locator('#export').click();
    const download = await downloadEvent; await download.saveAs(`${output}/downloaded-backup.json`);
    const downloaded = JSON.parse(await readFile(`${output}/downloaded-backup.json`, 'utf8'));
    assert.equal(downloaded.format, 'dailydigest-browser-v1');
    await page.locator('#import').setInputFiles(`${output}/downloaded-backup.json`);
    await page.locator('#restore-cancel').click();
    await page.waitForFunction(() => document.getElementById('import').value === '');
    assert.equal(await page.evaluate(() => Boolean(window.__workerReady)), true);
    const reload = page.waitForEvent('framenavigated', { predicate: changedFrame => changedFrame === page.mainFrame() });
    await page.locator('#import').setInputFiles(`${output}/downloaded-backup.json`);
    await page.locator('#restore-proceed').click();
    await reload; await startup();
    await page.locator('#transfer-notice').waitFor({ state: 'visible' });
    assert.match(await page.locator('#transfer-message').textContent(), /excluded credentials/);
    await page.locator('#dismiss-transfer').click();
    return { files: Object.keys(exported.files), confirmationDismissed: true, confirmedImportReloaded: true, credentialOmissionExplained: true };
  });
  await check('optional-companion-explicit-actions', async () => {
    assert.equal(companionRequests.length, 0, 'No companion contact should occur before explicit setup.');
    await route('/setup', '#interest-editor');
    await openSettingsGroup('summary-options');
    assert.equal(await frame().locator('input[name="llm_backend"][value="claude_cli"]').isDisabled(), true);
    assert.equal(await frame().locator('input[name="llm_backend"][value="codex_cli"]').isDisabled(), true);
    assert.match(await frame().locator('[data-companion-note]').first().textContent(), /connect a browser companion/);
    await openCompanionMenu();
    await page.locator('#companion-endpoint').fill('http://127.0.0.1:8768');
    await page.locator('#companion-token').fill('fictional-acceptance-token');
    await page.locator('#companion-connect').click();
    await page.waitForFunction(() => document.getElementById('companion-status').textContent.includes('Connected'));
    assert.deepEqual(companionRequests.map(row => row.path), ['/health']);
    await frame().locator('input[name="llm_backend"][value="claude_cli"]:not([disabled])').waitFor({state:'attached'});
    assert.equal(await frame().locator('input[name="llm_backend"][value="codex_cli"]').isDisabled(), true);
    assert.match(await frame().locator('[data-companion-note]').first().textContent(), /Connected companion supports/);
    await route('/', '.item[data-section="research"]');
    assert.equal(await page.locator('#companion-token').inputValue(), '');
    const status = await page.evaluate(() => window.__rpc({ type: 'companion-status' }));
    assert.doesNotMatch(status.body, /fictional-acceptance-token/);
    assert.doesNotMatch(JSON.stringify(await snapshot()), /fictional-acceptance-token/);
    await openCompanionMenu();
    for (const action of ['email', 'push']) {
      const confirmation = page.waitForEvent('dialog').then(dialog => dialog.accept());
      await page.locator(`#companion-${action}`).click();
      await confirmation;
      await page.waitForFunction(() => !document.getElementById('companion-status').textContent.includes('Running requested'));
    }
    const email = companionRequests.find(row => row.path === '/email');
    assert.ok(JSON.parse(email.body).html.includes('<html'), 'Email must use actual rendered digest HTML.');
    assert.match(JSON.parse(email.body).subject, /\[run:[0-9a-f]{32}\]/i, 'Sent emails must retain their vote-label edition.');
    assert.equal(JSON.parse(companionSnapshot).format, 'dailydigest-browser-v1');
    await page.locator('#companion-inbound').click();
    await page.waitForFunction(() => document.getElementById('companion-status').textContent.includes('Email feedback processed'));
    assert.match(await page.locator('#companion-status').textContent(), /1 upvotes, 0 downvotes/);
    await frame().locator('.item[data-section="research"]').first().locator('.vote-btn[data-grade="70"][aria-pressed="true"]').waitFor();
    assert.ok(companionRequests.some(row => row.path === '/inbound/ack'), 'Mailbox ACK must follow committed native browser feedback.');
    const reload = page.waitForEvent('framenavigated', { predicate: changedFrame => changedFrame === page.mainFrame() });
    await page.locator('#companion-pull').click(); await page.locator('#restore-proceed').click();
    await reload; await startup();
    await openCompanionMenu();
    await page.locator('#companion-disconnect').click();
    await page.waitForFunction(() => document.getElementById('companion-status').textContent.includes('No companion'));
    assert.equal(await page.locator('#companion-email').isDisabled(), true);
    await route('/setup', '#interest-editor');
    assert.equal(await frame().locator('input[name="llm_backend"][value="claude_cli"]').isDisabled(), true);
    await route('/', '.item[data-section="research"]');
    return { mockedCompanion: true, requests: companionRequests.map(({ path, method }) => ({ path, method })), nativeEmailFeedbackApplied: true, nativeEmailFeedbackGrade: 70, mailboxAcknowledged: true, stateRestored: true, disconnected: true };
  });
  await check('pwa-project-subpath-and-install-control', async () => {
    await page.evaluate(() => document.getElementById('data-settings')?.close());
    const evidence = await page.evaluate(async () => {
      const manifest = new URL(document.querySelector('link[rel="manifest"]').getAttribute('href'), location.href).href;
      const registration = await navigator.serviceWorker.ready;
      const offered = new Event('beforeinstallprompt', { cancelable: true });
      offered.prompt = async () => { window.__installPromptCalled = true; };
      offered.userChoice = Promise.resolve({ outcome: 'accepted' });
      window.dispatchEvent(offered);
      return { manifest, scope: registration.scope, promptPrevented: offered.defaultPrevented, offeredButtonVisible: !document.getElementById('install').hidden };
    });
    assert.equal(new URL(evidence.manifest).pathname, new URL(base + 'manifest.webmanifest').pathname);
    assert.ok(new URL(evidence.manifest).searchParams.has('build'));
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
    await route('/setup', '#interest-editor');
    await openSettingsGroup('summary-options');
    await frame().locator('#summary-backend').selectOption('api');
    await frame().locator('#llm_api_key').fill('');
    await frame().locator('#llm_model').fill('fictional-companion-model');
    await frame().locator('form[action="/setup"] button[type="submit"]').click();
    await page.waitForFunction(() => document.querySelector('#app')?.contentWindow?.ddLocation?.startsWith('http://127.0.0.1:8765/run'));
    const configured = await snapshot();
    assert.match(Buffer.from(configured.files['.env'], 'base64').toString(), /LLM_BACKEND="?api"?/);
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
    await route('/setup', '#interest-editor');
    await openSettingsGroup('summary-options');
    await frame().locator('#summary-backend').selectOption('extractive');
    await frame().locator('form[action="/setup"] button[type="submit"]').click();
    await page.waitForFunction(() => document.querySelector('#app')?.contentWindow?.ddLocation?.startsWith('http://127.0.0.1:8765/run'));
    const restored = await snapshot();
    assert.match(Buffer.from(restored.files['.env'], 'base64').toString(), /LLM_BACKEND="?extractive"?/);
    await openCompanionMenu();
    await page.locator('#companion-disconnect').click();
    return { nativeApiWithNoBrowserKey: true, providerSummaryBackend: 'api', selectedItemsOnly: first.length, serverIdentityInvalidatesCache: true, extractiveRestored: true };
  });
  await check('multicore-opt-in-on-a-host-without-isolation-headers', async () => {
    const initial = await page.evaluate(() => ({ isolated: crossOriginIsolated, cores: navigator.hardwareConcurrency }));
    if (initial.cores <= 1) return { singleCoreDevice: true };
    await toggleMenu('compute-menu');
    await page.locator('#compute-mode').selectOption('multicore');
    await Promise.all([page.waitForEvent('load'), page.locator('#compute-apply').click()]);
    await startup();
    const status = await page.evaluate(async () => ({ isolated: crossOriginIsolated,
      ...JSON.parse((await window.__rpc({ type: 'compute-status' })).body) }));
    assert.equal(status.isolated, true);
    assert.ok(status.threads > 1 && status.threads <= 4, JSON.stringify(status));
    await successfulBrew('usual');
    await toggleMenu('compute-menu');
    await page.locator('#compute-mode').selectOption('cpu');
    await Promise.all([page.waitForEvent('load'), page.locator('#compute-apply').click()]);
    await startup();
    assert.equal(await page.evaluate(() => crossOriginIsolated), false);
    return { initiallyIsolated: initial.isolated, isolatedAfterOptIn: status.isolated, threads: status.threads, realBrewPassed: true, optOutRestored: true };
  });
  await check('public-snapshot-through-real-browser-httpx', async () => {
    // A separate ephemeral worker reads public metadata only. The other cases
    // use fixture feeds, which cannot expose Pages' real compression headers.
    const evidence = await page.evaluate(async ({base, atCollection}) => {
      const manifest = await (await fetch(base + 'assets/build-manifest.json')).json();
      const probe = [
        'def probe():',
        '    import httpx, json',
        '    from datetime import datetime, timezone, timedelta',
        '    with httpx.Client(timeout=20, follow_redirects=False) as client:',
        '        response = client.get(snapshot_url)',
        '        response.raise_for_status()',
        '        snapshot = response.json()',
        '    records = list(snapshot["sources"].values())',
        atCollection ? '    instant = datetime.fromisoformat(snapshot["generated_at"])' : '    instant = datetime.now(timezone.utc)',
        '    healthy = [r for r in records if r.get("status") == "ok" and timedelta(minutes=-5) <= instant - datetime.fromisoformat(r["fetched_at"]) <= timedelta(hours=48)]',
        '    fresh = [r for r in healthy if not r.get("carried_forward")]',
        '    fallback = snapshot.get("publication", {}).get("mode") == "push-carry-forward"',
        '    eligible = healthy if fallback else fresh',
        '    attempts = [r.get("latest_attempt", {}) for r in records]',
        '    if fallback and not any(a.get("status") == "ok" and not a.get("carried_forward") for a in attempts): raise ValueError("All current collection attempts failed")',

        '    return json.dumps({"public_only": snapshot["public_only"], "sources": len(records), "healthy": len(healthy), "fresh": len(fresh), "eligible": len(eligible), "fallback": fallback, "kinds": sorted({r["kind"] for r in records}), "healthy_kinds": sorted({r["kind"] for r in eligible}), "sections": sorted({r["section"] for r in records}), "healthy_sections": sorted({r["section"] for r in eligible}), "decoded_bytes": len(response.content), "content_encoding": response.headers.get("content-encoding"), "content_length": int(response.headers["content-length"])})',
        'probe',
      ].join('\n');
      const source = `
        import { loadPyodide } from ${JSON.stringify(base + manifest.runtime_path + 'pyodide/pyodide.mjs')};
        import { createBrowserHttpBridge } from ${JSON.stringify(base + 'http.mjs')};
        try {
          const bridge = createBrowserHttpBridge();
          globalThis.browserHttpRequest = bridge.request;
          globalThis.browserHttpWait = bridge.wait;
          const py = await loadPyodide({ indexURL: ${JSON.stringify(base + manifest.runtime_path + 'pyodide/')} });
          await py.loadPackage(['httpx']);
          const bundle = await fetch(${JSON.stringify(base + 'assets/app.zip')});
          if (!bundle.ok) throw new Error('Application archive unavailable');
          py.unpackArchive(await bundle.arrayBuffer(), 'zip', { extractDir: '/app' });
          py.runPython("import sys\\nsys.path.insert(0, '/app/browser')\\nimport network\\nnetwork.install_httpx_transport()");
          py.globals.set('snapshot_url', ${JSON.stringify(base + 'assets/public-sources.json')});
          const result = await py.runPython(${JSON.stringify(probe)}).callPromising();
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
    }, {base,atCollection:process.env.DAILYDIGEST_BROWSER_SNAPSHOT_AT_COLLECTION === '1'});
    assert.equal(evidence.public_only, true);
    assert.ok(evidence.healthy > 0);
    assert.ok(evidence.eligible / evidence.sources >= 0.5);
    assert.deepEqual(evidence.healthy_kinds, evidence.kinds);
    assert.deepEqual(evidence.healthy_sections, evidence.sections);
    assert.equal(evidence.content_encoding, null);
    assert.equal(evidence.content_length, evidence.decoded_bytes);
    return {...evidence,validationClock:process.env.DAILYDIGEST_BROWSER_SNAPSHOT_AT_COLLECTION === '1' ? 'offline collection timestamp; not release freshness' : 'current time'};
  });
  await check('visible-backup-error-and-keyboard-recovery', async () => {
    const before = await digestIdentity();
    const dataMenu = page.locator('.device-sections > details:has(#export)');
    if (!await page.locator('#data-settings').evaluate(node => node.open)) await page.locator('#open-data-settings').click();
    if (await dataMenu.getAttribute('open') === null) await dataMenu.locator(':scope > summary').click();
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
      await testPage.locator('#startup[data-state="error"]').waitFor();
      assert.equal(await testPage.locator('#retry').isVisible(), false);
      assert.match(await testPage.locator('#startup-status').textContent(), /current desktop Chrome/);
      assert.equal(workerRequests.length, 0);
      return { simulatedMissingCapability: true, clearMessage: true, runtimeNotDownloaded: true };
    } finally { await unsupported.close(); }
  });
  await check('mixed-html-release-and-cpu-preparation-have-bounded-reloads', async () => {
    const mixed = await browser.newContext();
    await mixed.addInitScript(() => {
      const Original=Worker;
      window.Worker=class extends Original {
        constructor(...args) {super(...args);sessionStorage.setItem('acceptance-worker-count',String(Number(sessionStorage.getItem('acceptance-worker-count')||0)+1));}
      };
    });
    let documents = 0;
    await mixed.route('**/*', async request => {
      if (!request.request().url().startsWith(base)) return request.abort('blockedbyclient');
      if (new URL(request.request().url()).pathname === new URL(base).pathname) {
        const response = await request.fetch();
        let html = await response.text();
        if (++documents === 1) html = html.replace(/(<meta name="dailydigest-build" content=")[^"]+/, '$1old-cached-release');
        return request.fulfill({response, body:html});
      }
      return request.continue();
    });
    try {
      const testPage = await mixed.newPage();
      await testPage.goto(base + '#/setup');
      await testPage.locator('#app').waitFor({state:'visible', timeout:300000});
      const isolated=await testPage.evaluate(()=>crossOriginIsolated);
      assert.equal(documents, isolated ? 3 : 2, 'One release recovery and at most one CPU preparation reload');
      assert.equal(await testPage.evaluate(()=>sessionStorage.getItem('acceptance-worker-count')), '1');
      assert.equal(new URL(testPage.url()).hash, '#/setup');
      assert.match(new URL(testPage.url()).searchParams.get('__dd_build'), /^[a-f0-9]{16}$/);
      await testPage.frameLocator('#app').locator('#interest-editor').waitFor();
      return {documents, isolated, pythonWorkers:1, routePreserved:true, startupRecovered:true};
    } finally { await mixed.close(); }
  });
  await check('corrupt-startup-retains-raw-data-and-can-recover', async () => {
    const recovery = await browser.newContext({ acceptDownloads: true });
    const testPage = await recovery.newPage();
    const corrupt = '{"format":"dailydigest-browser-v1","files":{"data/digest.db":"broken-base64"}}';
    try {
      await testPage.goto(base + 'app-icon.svg');
      await testPage.evaluate(async ({ value, scope }) => {
        const database = await new Promise((resolve, reject) => {
          const request = indexedDB.open(`dailydigest:${scope}`, 1);
          request.onupgradeneeded = () => request.result.createObjectStore('state');
          request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
        });
        await new Promise((resolve, reject) => {
          const transaction = database.transaction('state', 'readwrite');
          transaction.objectStore('state').put(value, 'snapshot');
          transaction.oncomplete = resolve; transaction.onerror = () => reject(transaction.error);
        });
        database.close();
      }, { value: corrupt, scope: new URL(base).pathname });
      await testPage.goto(base);
      await testPage.locator('#recovery-tools').waitFor({ timeout: 300000 });
      assert.equal(await testPage.locator('#startup .steam').first().evaluate(node => getComputedStyle(node).animationName), 'none');
      const downloadEvent = testPage.waitForEvent('download');
      await testPage.locator('#recovery-export').click();
      const downloaded = await downloadEvent;
      await downloaded.saveAs(`${output}/corrupt-raw-recovery.json`);
      assert.equal(await readFile(`${output}/corrupt-raw-recovery.json`, 'utf8'), corrupt);
      await testPage.locator('#recovery-reset').click();
      await testPage.locator('#restore-proceed').click();
      await testPage.locator('#app').waitFor({ state: 'visible', timeout: 300000 });
      const retained = await testPage.evaluate(async scope => {
        const database = await new Promise(resolve => { const request = indexedDB.open(`dailydigest:${scope}`, 1); request.onsuccess = () => resolve(request.result); });
        const rows = await new Promise(resolve => { const request = database.transaction('state').objectStore('state').getAllKeys(); request.onsuccess = () => resolve(request.result); });
        database.close(); return rows;
      }, new URL(base).pathname);
      assert.ok(retained.some(key => String(key).startsWith('recovery:')));
      return { rawDataUnchanged: true, explicitReset: true, originalRetained: true, startupRecovered: true };
    } finally { await recovery.close(); }
  });
  await check('real-browser-route-checkpoints-match-full-export', async () => {
    assert.ok(checkpointEvidence.length > 20, 'Expected real app requests across setup, feedback, reading and mocked brew.');
    assert.deepEqual(checkpointEvidence.filter(row => !row.equal), []);
    const changed = checkpointEvidence.filter(row => row.after > row.before);
    for (const path of ['/setup', '/vote/', '/bookmark/', '/known/']) assert.ok(changed.some(row => row.path.includes(path)), `Missing mutation evidence: ${path}`);
    assert.ok(checkpointEvidence.some(row => row.method === 'DELETE' && row.path.startsWith('/bookmark/')), 'Unsave must be checkpointed');
    assert.ok(checkpointEvidence.some(row => row.path.includes('/reason/')), 'Rating reason must be checkpointed');
    assert.ok(report.brewCheckpoints?.length > 0, 'Mocked brew completion must be checkpointed');
    assert.ok(checkpointEvidence.some(row => row.method === 'GET' && row.before === row.after && row.path.startsWith('/setup')), 'Read-only setup must not dirty state.');
    return {requests: checkpointEvidence.length, changed: changed.length, checkpoints: checkpointEvidence};
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
