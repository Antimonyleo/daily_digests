// Focused acceptance for the shipped CPU cross-encoder and Python JSPI bridge.
// Fresh browser storage and synthetic article records; external requests blocked.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { chromium } from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const base = process.env.DAILYDIGEST_BROWSER_URL || 'http://127.0.0.1:8767/';
const output = resolve(process.env.DAILYDIGEST_SEMANTIC_REPORT || '.full-review/semantic-bridge-2026-10-15/browser-acceptance');
await mkdir(output, { recursive: true });
const report = { url: base, started: new Date().toISOString(), checks: [], modelRequests: [], blocked: [], errors: [] };
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: true,
  args: ['--no-sandbox', '--enable-experimental-webassembly-jspi'] });
const context = await browser.newContext();
const python = `
import json
from types import SimpleNamespace
from dailydigest.rank import semantic_alignment as _sa

def _acceptance_semantic(payload, action):
    records = json.loads(payload)
    backend = _sa._build_backend(None)
    if action == 'score':
        return json.dumps(backend.score_records(records))
    original_cache = _sa._PAIR_CACHE
    _sa._PAIR_CACHE = type(original_cache)()
    original_score = backend.score_records
    calls = []
    def counted(batch, **kwargs):
        calls.append(len(batch))
        return original_score(batch, **kwargs)
    backend.score_records = counted
    try:
        profile = SimpleNamespace(keywords=[records[0]['query']], canonical_facets={})
        rows = [SimpleNamespace(id=i+1, title=row['title'], abstract=row['abstract'], section='research') for i,row in enumerate(records)]
        for attempt in range(2):
            features = {row.id: {'topic_score': .7} for row in rows}
            ranked = _sa.apply_topic_alignment([(row,.6) for row in rows], features, profile)
        return json.dumps({'calls': calls, 'same_backend': _sa._build_backend(None) is backend,
                          'raw_topic_preserved': all(feature['topic_score'] == .7 for feature in features.values()),
                          'alignments': [features[row.id]['topic_alignment'] for row in rows]})
    finally:
        backend.score_records = original_score
        _sa._PAIR_CACHE = original_cache
`;
await context.route('**/*', async route => {
  const url = route.request().url();
  if (!url.startsWith(base)) {
    report.blocked.push(url);
    return route.abort('blockedbyclient');
  }
  if (url.includes('/assets/semantic-models/')) report.modelRequests.push(url);
  if (new URL(url).pathname.endsWith('/worker.mjs')) {
    const upstream = await route.fetch();
    const source = (await upstream.text()).replace('self.onmessage = async ({ data }) => {', `self.onmessage = async ({ data }) => {
      if (data.type === 'acceptance-semantic') {
        try {
          await initialize();
          state.py.runPython(${JSON.stringify(python)});
          const body = await state.py.globals.get('_acceptance_semantic').callPromising(JSON.stringify(data.records), data.action);
          self.postMessage(response(data.id, body));
        } catch (error) { self.postMessage(response(data.id, error.message || String(error), 500)); }
        return;
      }`);
    return route.fulfill({ response: upstream, body: source });
  }
  return route.continue();
});
const page = await context.newPage();
page.on('pageerror', error => report.errors.push(error.message));
await page.addInitScript(() => {
  const NativeWorker = window.Worker;
  window.__semanticEvents = [];
  window.Worker = class extends NativeWorker {
    constructor(...args) {
      super(...args);
      const pending = new Map(); let sequence = 0;
      window.__semanticRPC = data => new Promise(resolve => {
        const id = `semantic-${++sequence}`; pending.set(id, resolve); this.postMessage({ ...data, id });
      });
      this.addEventListener('message', ({ data }) => {
        window.__semanticEvents.push({ type: data.type, stage: data.stage, compute: data.compute });
        if (data.type === 'ready') window.__semanticReady = true;
        if (pending.has(data.id)) { pending.get(data.id)(data); pending.delete(data.id); }
      });
    }
  };
});
const records = [
  { query: 'comparative genomics of plant genome evolution', title: 'Plant evolution and genomic synteny',
    abstract: 'We compare conserved synteny and genome duplication across flowering plants.' },
  { query: 'plant resistance genes', title: 'Wheat immunity', abstract: 'Plant genomic synteny and evolutionary relationships. '.repeat(150) },
  { query: 'Café α-β resistance', title: '植物 immunity — Müller\u0000 genes', abstract: 'évolution αβγ 中文 🤖 immune receptors\nwith\ttabs' },
  { query: 'plant evolution', title: 'Genomic synteny '.repeat(90), abstract: '' },
];
try {
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__semanticReady || document.getElementById('retry')?.hidden === false, undefined, { timeout: 300000 });
  assert.equal(await page.evaluate(() => Boolean(window.__semanticReady)), true, 'Python worker failed to start');
  assert.equal(report.modelRequests.length, 0, 'Semantic assets must not load at boot');
  report.checks.push('semantic assets load lazily');
  const score = await page.evaluate(records => window.__semanticRPC({ type: 'acceptance-semantic', action: 'score', records }), records);
  assert.equal(score.status, 200, score.body);
  const actual = JSON.parse(score.body), expected = [3.847635507583618, -5.078920364379883, -9.381305694580078, -10.60063648223877];
  assert.equal(actual.length, 4);
  for (let i = 0; i < actual.length; i++) assert.ok(Math.abs(actual[i].logit - expected[i]) < .0001, `Native/browser logit mismatch at ${i}`);
  assert.deepEqual(actual.map(row => [row.windows, row.complete]), [[1,true],[3,false],[1,true],[1,false]]);
  report.scores = actual;
  report.checks.push('real Pyodide bridge and WASM scores match native across Unicode and windowed records');
  assert.ok(report.modelRequests.some(url => url.endsWith('/onnx/model.onnx')));
  const alignmentRecords = [{ ...records[0], abstract: records[0].abstract + ' Additional conserved gene-order evidence.' }, records[1]];
  const alignment = await page.evaluate(records => window.__semanticRPC({ type: 'acceptance-semantic', action: 'alignment', records }), alignmentRecords);
  assert.equal(alignment.status, 200, alignment.body);
  report.alignment = JSON.parse(alignment.body);
  assert.deepEqual(report.alignment.calls, [2]);
  assert.equal(report.alignment.same_backend, true);
  assert.equal(report.alignment.raw_topic_preserved, true);
  report.checks.push('real alignment uses stable backend identity and reuses cached pairs');
  const bounded = await page.evaluate(records => window.__semanticRPC({ type: 'acceptance-semantic', action: 'score', records }), [{ ...records[0], query: 'genome '.repeat(193), topic: 'plant evolution' }]);
  assert.equal(bounded.status, 200, bounded.body);
  const partial = JSON.parse(bounded.body)[0];
  assert.equal(partial.query_truncated, true);
  assert.equal(partial.complete, false);
  assert.ok(Number.isFinite(partial.logit));
  report.checks.push('oversized query retains its topic and reports partial coverage');
  const events = await page.evaluate(() => window.__semanticEvents);
  assert.ok(events.some(event => event.compute?.backend === 'CPU · local topic matching' && event.compute.phase === 'encoding'));
  report.checks.push('topic matching reaches existing CPU progress display');
  assert.deepEqual(report.blocked, []);
  assert.deepEqual(report.errors, []);
  report.status = 'passed';
} catch (error) {
  report.status = 'failed'; report.failure = error.stack || String(error);
  process.exitCode = 1;
} finally {
  report.finished = new Date().toISOString();
  await writeFile(resolve(output, 'semantic-report.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify({ status: report.status, checks: report.checks, failure: report.failure, report: resolve(output, 'semantic-report.json') }));
