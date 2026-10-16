// Real Chromium refresh regression. Only synthetic private files are persisted;
// all non-local requests are blocked, and no brew/provider request is issued.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { chromium } from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const site = resolve(process.env.DAILYDIGEST_STORAGE_SITE || 'browser/dist');
const output = resolve(process.env.DAILYDIGEST_STORAGE_REPORT || '.full-review/storage-startup-2026-10-15');
const validator = process.env.DAILYDIGEST_STORAGE_VALIDATOR;
const expectedFailure = process.env.DAILYDIGEST_STORAGE_EXPECT_FAILURE === '1';
await mkdir(output, { recursive: true });
const generated = spawnSync(resolve('.venv/bin/python'), ['-c', `
import base64, io, json, pathlib, sqlite3, sys
import numpy as np
root = pathlib.Path(sys.argv[1])
with sqlite3.connect(':memory:') as connection:
    connection.execute('CREATE TABLE synthetic_embeddings (id INTEGER PRIMARY KEY, payload BLOB)')
    connection.execute('INSERT INTO synthetic_embeddings VALUES (1, ?)', (bytes(range(256))*32768,))
    assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    database_bytes = connection.serialize()
array = io.BytesIO()
np.save(array, np.arange(1024*1024, dtype=np.float32), allow_pickle=False)
files = {'data/synthetic-embedding-cache.sqlite': database_bytes,
         'data/embeddings/synthetic-vectors.npy': array.getvalue(),
         'data/profile.yaml': b'bio: Synthetic storage regression reader.\\nkeywords: [astronomy]\\n',
         '.env': b'LLM_BACKEND=extractive\\n'}
(root/'synthetic-snapshot.json').write_text(json.dumps({'format':'dailydigest-browser-v1','files':{name:base64.b64encode(value).decode() for name,value in files.items()}}))
`, output], { encoding: 'utf8' });
assert.equal(generated.status, 0, generated.stderr);
const payload = await readFile(resolve(output, 'synthetic-snapshot.json'), 'utf8');
const fixture = JSON.parse(payload);
const digest = value => createHash('sha256').update(value).digest('hex');
const report = { started: new Date().toISOString(), site, validator: validator || 'built artifact', expectedFailure,
  validatorSha256: digest(await readFile(resolve(validator || site + '/storage.mjs'))), snapshotSha256:digest(payload),
  snapshotCharacters: payload.length, files: Object.fromEntries(Object.entries(fixture.files).map(([name, value]) => {
    const bytes = Buffer.from(value, 'base64'); return [name, { bytes: bytes.length, sha256: digest(bytes) }];
  })), checks: [], stacks: [], errors: [], blocked: [] };
const mime = { '.html':'text/html', '.mjs':'text/javascript', '.js':'text/javascript', '.json':'application/json', '.wasm':'application/wasm', '.css':'text/css', '.svg':'image/svg+xml', '.whl':'application/octet-stream' };
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (pathname === '/__seed__') { response.writeHead(200, {'content-type':'text/html'}); response.end('<title>Synthetic storage seed</title>'); return; }
    let file = resolve(site, '.' + pathname);
    assert.ok(file === site || file.startsWith(site + sep));
    if ((await stat(file)).isDirectory()) file = resolve(file, 'index.html');
    response.writeHead(200, { 'content-type':mime[extname(file)] || 'application/octet-stream', 'cache-control':'no-store' });
    response.end(await readFile(file));
  } catch { response.writeHead(404); response.end('Not found'); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch({ executablePath:process.env.CHROME_PATH || '/usr/bin/google-chrome', headless:true,
  args:['--no-sandbox', '--enable-experimental-webassembly-jspi'] });
try {
  const context = await browser.newContext({ acceptDownloads:true });
  await context.route('**/*', async route => {
    const url = route.request().url();
    if (!url.startsWith(base)) { report.blocked.push(url); return route.abort('blockedbyclient'); }
    if (new URL(url).pathname.endsWith('/storage.mjs') && validator) return route.fulfill({ contentType:'text/javascript', body:await readFile(resolve(validator), 'utf8') });
    if (new URL(url).pathname.endsWith('/worker.mjs')) {
      const upstream = await route.fetch();
      const source = (await upstream.text()).replace('validateStoredSnapshot(snapshot).warning',
        `(() => { try { return validateStoredSnapshot(snapshot).warning; } catch(error) { self.postMessage({type:'acceptance-stack', stack:error.stack}); throw error; } })()`);
      return route.fulfill({ response:upstream, body:source });
    }
    return route.continue();
  });
  const page = await context.newPage();
  page.setDefaultTimeout(60000);
  page.on('pageerror', error => report.errors.push(error.stack));
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    window.__startupEvents = [];
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args); const pending = new Map(); let id = 0;
        window.__storageRPC = data => new Promise(resolve => { const key = `storage-${++id}`; pending.set(key, resolve); this.postMessage({...data, id:key}); });
        this.addEventListener('message', ({data}) => {
          window.__startupEvents.push(data.type === 'acceptance-stack' ? data : {type:data.type, message:data.message, stage:data.stage});
          if (data.type === 'ready') window.__storageReady = true;
          if (pending.has(data.id)) { pending.get(data.id)(data); pending.delete(data.id); }
        });
      }
    };
  });
  // A same-origin inert document seeds IndexedDB before the application boots.
  await page.goto(base + '__seed__');
  await page.evaluate(async value => {
    const database = await new Promise((resolve, reject) => {
      const request = indexedDB.open('dailydigest:/', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('state');
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    await new Promise((resolve, reject) => {
      const transaction = database.transaction('state', 'readwrite');
      transaction.objectStore('state').put(value, 'snapshot');
      transaction.oncomplete = resolve; transaction.onerror = () => reject(transaction.error);
    }); database.close();
  }, payload);
  const started = Date.now();
  await page.goto(base, { waitUntil:'domcontentloaded' });
  await page.waitForFunction(() => window.__storageReady || document.querySelector('#startup')?.dataset.state === 'error', undefined, {timeout:300000});
  report.startupMilliseconds = Date.now() - started;
  report.events = await page.evaluate(() => window.__startupEvents);
  report.stacks = report.events.filter(event => event.type === 'acceptance-stack').map(event => event.stack);
  const ready = await page.evaluate(() => Boolean(window.__storageReady));
  assert.equal(ready, !expectedFailure, JSON.stringify(report.events));
  const raw = await page.evaluate(async () => {
    const database = await new Promise(resolve => { const request=indexedDB.open('dailydigest:/',1); request.onsuccess=()=>resolve(request.result); });
    const result = await new Promise(resolve => { const request=database.transaction('state').objectStore('state').get('snapshot'); request.onsuccess=()=>resolve(request.result); });
    database.close(); return result;
  });
  const persisted = JSON.parse(raw);
  for (const [name, value] of Object.entries(fixture.files)) assert.equal(persisted.files[name], value, `Startup must preserve persisted ${name}`);
  report.checks.push('all seeded IndexedDB file bytes remain identical');
  if (expectedFailure) {
    assert.equal(raw, payload, 'Failed startup must preserve the entire original checkpoint');
    assert.ok(report.stacks.some(stack => /Maximum call stack size exceeded/.test(stack)), 'Capture actual browser stack');
    const recovered = await page.evaluate(() => window.__storageRPC({type:'recovery-export'}));
    assert.equal(recovered.body, payload);
    report.checks.push('failed-startup recovery exports exact original checkpoint');
  } else {
    await page.locator('#app').waitFor({state:'visible', timeout:30000});
    await page.frameLocator('#app').locator('body').waitFor();
    report.checks.push('reading desk iframe loads');
    const exported = await page.evaluate(() => window.__storageRPC({type:'export', includeCredentials:true}));
    assert.equal(exported.status, 200, exported.body);
    const restored = JSON.parse(exported.body);
    for (const [name, value] of Object.entries(fixture.files)) assert.equal(restored.files[name], value, `Restore/export must preserve ${name} exactly`);
    report.checks.push('all seeded SQLite, NumPy, profile and environment bytes survive restore/export');
    await page.locator('#open-data-settings').click();
    await page.locator('#browser-data-menu > summary').click();
    const [backup] = await Promise.all([page.waitForEvent('download'), page.locator('#export').click()]);
    const downloaded = JSON.parse(await readFile(await backup.path(), 'utf8'));
    for (const [name, value] of Object.entries(fixture.files).filter(([name]) => name !== '.env')) assert.equal(downloaded.files[name], value);
    report.checks.push('Browser data export download preserves all seeded data files');
    const refreshStarted = Date.now();
    await page.reload({waitUntil:'domcontentloaded'});
    await page.waitForFunction(() => window.__storageReady, undefined, {timeout:300000});
    await page.locator('#app').waitFor({state:'visible', timeout:30000});
    const refreshed = await page.evaluate(() => window.__storageRPC({type:'export', includeCredentials:true}));
    assert.equal(refreshed.status, 200, refreshed.body);
    const again = JSON.parse(refreshed.body);
    for (const [name, value] of Object.entries(fixture.files)) assert.equal(again.files[name], value);
    report.refreshMilliseconds = Date.now() - refreshStarted;
    report.checks.push('second page refresh opens the desk and preserves all seeded bytes');
  }
  assert.deepEqual(report.blocked, [], 'No external data or provider requests should be attempted');
  report.success = true;
} catch (error) { report.success = false; report.failure = error.stack; throw error; }
finally {
  await writeFile(resolve(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  await browser.close(); await new Promise(resolve => server.close(resolve));
}
console.log(JSON.stringify({success:report.success, snapshotCharacters:report.snapshotCharacters, startupMilliseconds:report.startupMilliseconds, checks:report.checks, stacks:report.stacks}, null, 2));
