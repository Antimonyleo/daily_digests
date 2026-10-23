// Exercise the real service-worker isolation/reload path with an empty desk.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {resolve, sep, extname} from 'node:path';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const site = resolve(process.env.DAILYDIGEST_CPU_SITE || 'browser/dist');
const output = resolve(process.env.DAILYDIGEST_CPU_REPORT || '.full-review/cpu-startup');
await mkdir(output, {recursive: true});
const report = {checks: [], errors: [], external: []};
const types = {'.mjs': 'text/javascript', '.js': 'text/javascript', '.html': 'text/html',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm'};
let workerRequests = 0;
const server = createServer(async (request, response) => {
  try {
    let path = new URL(request.url, 'http://localhost').pathname;
    assert.ok(path.startsWith('/daily_digests/'));
    if (path.endsWith('__seed__')) { response.end('<title>Isolated fixture</title>'); return; }
    if (path.endsWith('/')) path += 'index.html';
    const file = resolve(site, '.' + path.slice('/daily_digests'.length));
    assert.ok(file.startsWith(site + sep));
    // Leave time to submit the topic draft before the one-time reload.
    if (path.endsWith('/sw.js') && workerRequests++ === 0) await new Promise(done => setTimeout(done, 350));
    response.setHeader('Content-Type', types[extname(file)] || 'application/octet-stream');
    response.end(await readFile(file));
  } catch { response.writeHead(404).end(); }
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const base = `http://127.0.0.1:${server.address().port}/daily_digests/`;
const browser = await chromium.launch({executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true, args: ['--no-sandbox', '--enable-experimental-webassembly-jspi']});
try {
  for (const explicitCPU of [false, true]) {
    const context = await browser.newContext();
    await context.route('**/*', route => {
      if (route.request().url().startsWith(base)) return route.continue();
      report.external.push(route.request().url()); return route.abort();
    });
    const page = await context.newPage();
    let workers = 0, documents = 0;
    page.on('pageerror', error => report.errors.push(error.message));
    page.on('request', request => {
      if (request.resourceType() === 'document' && request.frame() === page.mainFrame()) documents++;
    });
    await page.exposeFunction('__workerStarted', () => { workers++; });
    await page.addInitScript(() => {
      window.__ready = false;
      const Native = Worker;
      window.Worker = class extends Native {
        constructor(...args) {
          super(...args); window.__worker = this; void window.__workerStarted();
          this.addEventListener('message', ({data}) => { if (data.type === 'ready') window.__ready = true; });
        }
      };
    });
    if (explicitCPU) {
      await page.goto(base + '__seed__');
      await page.evaluate(async () => {
        const request = indexedDB.open('dailydigest:/daily_digests/', 1);
        request.onupgradeneeded = () => request.result.createObjectStore('state');
        const db = await new Promise(done => { request.onsuccess = () => done(request.result); });
        await new Promise(done => {
          const transaction = db.transaction('state', 'readwrite');
          transaction.objectStore('state').put('cpu', 'compute'); transaction.oncomplete = done;
        }); db.close();
      });
      documents = 0;
    }
    const started = performance.now();
    await page.goto(base + '#/setup');
    if (!explicitCPU) {
      await page.locator('#startup-topics-form').waitFor();
      await page.locator('#startup-topics').fill('DNA nanotechnology\nProtein design');
      await page.locator('#startup-topics-continue').click();
    }
    await page.waitForFunction(() => window.__ready, undefined, {timeout: 120000});
    const frame = page.frameLocator('#app');
    await frame.locator('#interest-editor').waitFor();
    const status = await page.evaluate(() => new Promise(done => {
      const listener = ({data}) => {
        if (data.id !== 'cpu-verification') return;
        window.__worker.removeEventListener('message', listener); done(JSON.parse(data.body));
      };
      window.__worker.addEventListener('message', listener);
      window.__worker.postMessage({id: 'cpu-verification', type: 'compute-status'});
    }));
    assert.equal(status.mode, explicitCPU ? 'cpu' : 'multicore');
    assert.equal(status.threads, explicitCPU ? 1 : 4);
    assert.equal(await page.evaluate(() => crossOriginIsolated), !explicitCPU);
    assert.equal(workers, 1, 'CPU preparation must not initialize Python twice');
    assert.equal(documents, explicitCPU ? 1 : 2, 'At most one preparation reload');
    if (!explicitCPU) assert.deepEqual(await frame.locator('[data-interest-name]').evaluateAll(nodes => nodes.map(node => node.value)),
      ['DNA nanotechnology', 'Protein design']);
    report.checks.push({explicitCPU, mode: status.mode, threads: status.threads, workers, documents,
      readyMs: performance.now() - started});
    await context.close();
  }
  assert.deepEqual(report.errors, []); assert.deepEqual(report.external, []);
  report.success = true;
} catch (error) { report.failure = error.stack; throw error; }
finally {
  await writeFile(resolve(output, 'report.json'), JSON.stringify(report, null, 2));
  await browser.close(); await new Promise(done => server.close(done));
}
console.log(JSON.stringify(report));
