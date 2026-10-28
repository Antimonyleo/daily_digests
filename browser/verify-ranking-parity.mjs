// Offline full-Brew replay across captured research fields. Never uses a reader's
// database or live services. Compare reports from two builds/backends by URL.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {resolve, extname, sep} from 'node:path';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const site = resolve(process.env.DAILYDIGEST_PARITY_SITE || 'browser/dist');
const out = resolve(process.env.DAILYDIGEST_PARITY_REPORT || '.full-review/ranking-parity');
const mode = process.env.DAILYDIGEST_PARITY_MODE || 'cpu';
const rounds = Number(process.env.DAILYDIGEST_PARITY_ROUNDS || 1);
assert.ok(['cpu', 'multicore', 'gpu'].includes(mode));
assert.ok(Number.isInteger(rounds) && rounds >= 1 && rounds <= 5);
const fixturePath = process.env.DAILYDIGEST_PARITY_FIXTURE || new URL('../tests/fixtures/cross_field_alignment.json', import.meta.url);
const corpus = JSON.parse(await readFile(fixturePath, 'utf8'));
const mixed = ['asymmetric_organocatalysis', 'induced_proximity'].map(id=>corpus.profiles.find(profile=>profile.id===id));
const mixedTopics = mixed.map(profile=>profile.profile.keywords[0]);
for (const shares of [[70,30],[30,70]]) corpus.profiles.push({
  id: 'mixed_priorities_' + shares.join('_'), pool: mixed.flatMap(profile=>profile.pool),
  profile: {bio:'', keywords:mixedTopics, canonical_facets:Object.fromEntries(mixedTopics.map((topic,index)=>[topic,{priority:shares[index]}]))},
});
const requested = process.env.DAILYDIGEST_PARITY_PROFILES?.split(',');
if (requested?.includes('watchlist_boundaries')) corpus.profiles.push({
  id:'watchlist_boundaries',
  profile:{...mixed[0].profile,authors_of_interest:['John Smith','Иван Петров']},
  pool:mixed[0].pool.slice(0,3).map((row,index)=>({...row,
    authors:['John Doe and Jane Smith','John Smith','Иван Петров'][index]})),
});
const profiles = corpus.profiles.filter(profile => !requested || requested.includes(profile.id));
assert.ok(profiles.length);
const fixtureDate = '2026-10-17T12:00:00Z';
const report = {scope: 'Captured public text/metadata and fictional profiles; full local extractive Brew. Publication dates are rebased to the preceding day to exercise the entire pool. Fixed clock, no live sources/providers. Labels are provisional, not independent reader judgments.',
  mode, rounds, fixtureDate, headless:process.env.DAILYDIGEST_PARITY_HEADED !== '1',
  gpuFlags: process.env.DAILYDIGEST_PARITY_GPU_FLAGS === '1', runs: [], errors: [], blocked: []};
await mkdir(out, {recursive: true});
const types = {'.html':'text/html', '.mjs':'text/javascript', '.js':'text/javascript', '.json':'application/json', '.wasm':'application/wasm', '.css':'text/css', '.svg':'image/svg+xml'};
const server = createServer(async (req, res) => {
  try {
    if (mode === 'multicore') {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    }
    let path = new URL(req.url, 'http://local').pathname;
    if (path === '/daily_digests/seed') {res.setHeader('Content-Type','text/html');res.end('<title>Isolated test desk</title>');return;}
    assert.ok(path.startsWith('/daily_digests/'));
    if (path.endsWith('/')) path += 'index.html';
    const file = resolve(site, '.' + path.slice('/daily_digests'.length));
    assert.ok(file.startsWith(site + sep));
    res.setHeader('Content-Type', types[extname(file)] || 'application/octet-stream');
    res.end(await readFile(file));
  } catch {res.writeHead(404).end();}
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const base = `http://127.0.0.1:${server.address().port}/daily_digests/`;
const args = ['--no-sandbox', ...(report.headless ? ['--enable-experimental-webassembly-jspi'] : [])];
if (report.gpuFlags) args.push('--enable-gpu', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan');
const browser = await chromium.launch({executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: report.headless, args});
const patch = `self.onmessage = async ({ data }) => {
  if(data.type === 'parity-fixture') {
    try {
      await initialize(); await prepareComputePackages();
      state.py.globals.set('_parity_items', JSON.stringify(data.items));
      state.py.runPython(\`import json
from types import SimpleNamespace
from dailydigest import pipeline
from dailydigest.models import Item
_parity_pool = [Item.model_validate(row) for row in json.loads(_parity_items)]
pipeline.dispatch_source = lambda spec: SimpleNamespace(fetch=lambda spec, days=2: [row.model_copy(deep=True) for row in _parity_pool if row.source == spec.name])
\`);
      self.postMessage(response(data.id, '{}'));
    } catch(error) {self.postMessage(response(data.id, String(error.stack || error), 500));}
    return;
  }
  if(data.type === 'parity-evidence') {
    try {
      const snapshot = state.py.runPython(\`import json
from sqlalchemy import text
from dailydigest.store import session_scope, load_ranking_snapshot, load_digest_audit
with session_scope() as _parity_session:
    _parity_run = _parity_session.execute(text('SELECT run_id FROM ranking_snapshots ORDER BY rowid DESC LIMIT 1')).scalar()
_parity_snapshot = load_ranking_snapshot(_parity_run) if _parity_run else None
if _parity_snapshot:
    _parity_snapshot['related_reading'] = load_digest_audit(_parity_snapshot['digest_id'], 'related_reading')
json.dumps(_parity_snapshot)
\`);
      self.postMessage(response(data.id, JSON.stringify({snapshot: JSON.parse(snapshot), device:state.engine.device,
        stats:state.engine.stats, backend:state.engine.cacheBackend, threads:state.engine.threads,
        publicCache:state.publicEmbeddings.stats, gpu:gpuProbe && {available:gpuProbe.available,reason:gpuProbe.reason,info:gpuProbe.info}})));
    } catch(error) {self.postMessage(response(data.id, String(error.stack || error), 500));}
    return;
  }
`;
try {
  for (const profile of profiles) {
    const context = await browser.newContext({timezoneId: 'UTC', serviceWorkers: 'block'});
    try {
      await context.route('**/*', async route => {
        const url = route.request().url();
        if (!url.startsWith(base)) {report.blocked.push(url);return route.abort();}
        if (new URL(url).pathname.endsWith('/worker.mjs')) {
          const response = await route.fetch();
          const original = await response.text();
          assert.ok(original.includes('self.onmessage = async ({ data }) => {'));
          const clock = `const OriginalDate=Date;globalThis.Date=class extends OriginalDate{constructor(...args){super(...(args.length?args:[${JSON.stringify(fixtureDate)}]));}static now(){return OriginalDate.parse(${JSON.stringify(fixtureDate)});}};\n`;
          return route.fulfill({response, body: clock + original.replace('self.onmessage = async ({ data }) => {', patch)});
        }
        return route.continue();
      });
      const page = await context.newPage();
      page.on('pageerror', error => report.errors.push(error.message));
      await page.addInitScript(() => {
        window.__events = [];
        const Original = Worker;
        window.Worker = class extends Original {
          constructor(...args) {super(...args);window.__worker=this;this.addEventListener('message',({data})=>{window.__events.push(data);if(data.type==='ready')window.__ready=true;});}
        };
        let sequence = 0;
        window.__rpc = data => new Promise((resolve, reject) => {
          const id = 'parity-' + ++sequence;
          const timer = setTimeout(() => {window.__worker.removeEventListener('message', listener);reject(Error('Evidence RPC timed out'));}, 120000);
          const listener = ({data}) => {if(data.id!==id)return;clearTimeout(timer);window.__worker.removeEventListener('message',listener);data.status===200?resolve(data.body):reject(Error(data.body));};
          window.__worker.addEventListener('message',listener);window.__worker.postMessage({...data,id});
        });
      });
      await page.goto(base + 'seed');
      const files = {
        '.env': 'DB_PATH=data/digest.db\nPROFILE_PATH=data/profile.yaml\nSOURCES_PATH=data/acceptance-sources.yaml\nLLM_BACKEND=extractive\nUSER_TZ=UTC\nINCLUDE_EVENTS=false\nINCLUDE_OPPORTUNITIES=false\nINCLUDE_INDUSTRY=false\nINCLUDE_AI=false\nINCLUDE_WORLD=false\nINCLUDE_REGULATORY=false\nTOP_RESEARCH=10\n',
        'data/profile.yaml': JSON.stringify({...profile.profile, name:'Fictional test reader'}),
        'data/acceptance-sources.yaml': JSON.stringify({research:[...new Set(profile.pool.map(row=>row.source))]
          .map(name=>({kind:'rss',name,url:'https://acceptance.invalid/'+encodeURIComponent(name)+'.rss'}))}),
      };
      const snapshot = JSON.stringify({format:'dailydigest-browser-v1',files:Object.fromEntries(Object.entries(files).map(([name,text])=>[name,Buffer.from(text).toString('base64')]))});
      await page.evaluate(async ({snapshot,mode}) => {
        const request=indexedDB.open('dailydigest:/daily_digests/',1);
        request.onupgradeneeded=()=>request.result.createObjectStore('state');
        const db=await new Promise((resolve,reject)=>{request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});
        await new Promise((resolve,reject)=>{const tx=db.transaction('state','readwrite');tx.objectStore('state').put(snapshot,'snapshot');tx.objectStore('state').put(mode,'compute');tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);});db.close();
      }, {snapshot,mode});
      await page.goto(base + '#/run');
      await page.waitForFunction(()=>window.__ready, undefined, {timeout:120000});
      await page.evaluate(items=>window.__rpc({type:'parity-fixture',items}),
        profile.pool.map(row=>({...row,published_at:'2026-10-16T12:00:00Z'})));
      const frame=()=>page.frameLocator('#app');
      for(let trial=0;trial<rounds;trial++) {
        if(trial) await page.evaluate(()=>location.hash='#/run');
        await frame().locator('#brew-button').waitFor();
        await frame().locator('#mode-full').check();
        await page.evaluate(()=>window.__events=[]);
        const started=performance.now();
        await frame().locator('#brew-button').click();
        await page.waitForFunction(()=>(location.hash.startsWith('#/done')||location.hash.startsWith('#/?digest_id='))||document.querySelector('#app')?.contentDocument?.querySelector('#err:not([hidden])'),undefined,{timeout:600000});
        const elapsedMs=performance.now()-started;
        assert.ok(/^#\/(?:done|\?digest_id=)/.test(await page.evaluate(()=>location.hash)),await frame().locator('body').innerText());
        await page.waitForFunction(()=>window.__events.some(event=>event.type==='compute-state'&&event.busy===false),undefined,{timeout:60000});
        const evidence=JSON.parse(await page.evaluate(()=>window.__rpc({type:'parity-evidence'})));
        assert.ok(evidence.snapshot?.complete_candidate_pool);
        assert.ok(evidence.snapshot.candidates.length > 0, 'A parity replay must exercise candidate ranking');
        if(profile.id==='watchlist_boundaries') {
          // Only these test bylines are fictional; exercise the actual pipeline
          // boost and its persisted evidence, not just the matcher in isolation.
          for(const [index,item] of profile.pool.entries()) {
            const row=evidence.snapshot.candidates.find(row=>row.metadata.url===item.url);
            assert.ok(row, item.url);
            assert.equal(row.features.author_match || 0,index===0?0:1);
            assert.equal(row.features.author_boost || 0,index===0?0:0.12);
          }
        }
        const events=await page.evaluate(()=>window.__events);
        const run={profile:profile.id,trial,inputCount:profile.pool.length,elapsedMs,...evidence,events};
        report.runs.push(run);
        await writeFile(resolve(out,'report.json'),JSON.stringify(report,null,2)+'\n');
        console.log(JSON.stringify({profile:profile.id,trial,elapsedMs,candidates:evidence.snapshot.candidates.length,
          selected:evidence.snapshot.candidates.filter(row=>row.selected).length,device:evidence.device,threads:evidence.threads,gpu:evidence.gpu}));
      }
    } finally {await context.close();}
  }
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.blocked,[]);report.success=true;
} catch(error) {report.failure=error.stack;throw error;}
finally {await writeFile(resolve(out,'report.json'),JSON.stringify(report,null,2)+'\n');await browser.close();await new Promise(done=>server.close(done));}
