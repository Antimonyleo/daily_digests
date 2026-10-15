import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
const source = readFileSync(new URL('./worker.mjs', import.meta.url), 'utf8');
function production(name, globals) {
  const match = source.match(new RegExp(`async function ${name}\\([^]*?\\n\\}`));
  assert.ok(match, `production ${name} exists`);
  const context = vm.createContext({ setTimeout, clearTimeout, AbortController, URL, Error, ...globals });
  vm.runInContext(match[0], context);
  return context[name];
}

test('storage ownership waits for a departing worker rather than refusing immediately', async () => {
  const events = [];
  const own = production('ownStorage', { ownership: null, progress: (...value) => events.push(value),
    self: { location: { href: 'https://example.test/digests/worker.mjs' } },
    navigator: { locks: { request: async (_name, options, callback) => {
      assert.equal(options.ifAvailable, undefined);
      await new Promise(resolve => setTimeout(resolve, 10));
      return callback({ name: 'fixture' });
    } } } });
  await own({ timeoutMs: 200 });
});

test('storage lock wait is bounded and a timeout can be retried', async () => {
  let requests = 0;
  const own = production('ownStorage', { ownership: null, progress: () => {},
    self: { location: { href: 'https://example.test/digests/worker.mjs' } },
    navigator: { locks: { request: async (_name, options, callback) => {
      if (++requests > 1) return callback({});
      await new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(new Error('aborted'))));
    } } } });
  await assert.rejects(own({ timeoutMs: 10 }), /another tab|previous tab/);
  await own({ timeoutMs: 100 });
  assert.equal(requests, 2);
});

for (const [name, gpu, expected] of [
  ['missing API', undefined, 'no-api'],
  ['null adapter', { requestAdapter: async () => null }, 'no-adapter'],
  ['software adapter', { requestAdapter: async () => ({ info: { isFallbackAdapter: true, vendor: 'software' } }) }, 'software-adapter'],
  ['hardware adapter', { requestAdapter: async () => ({ info: { vendor: 'fixture', architecture: 'fixture-arch' } }) }, 'available'],
  ['probe error', { requestAdapter: async () => { throw Error('driver unavailable'); } }, 'probe-failed'],
]) test(`GPU probe explains ${name} without loading a model or requesting a device`, async () => {
  const probe = production('probeGPU', { navigator: { gpu } });
  const result = await probe();
  assert.equal(result.reason, expected);
  assert.equal(result.available, expected === 'available');
  if (result.available) assert.equal(result.info.vendor, 'fixture');
});

test('a suspended real job records recovery state until its final checkpoint commits', async () => {
  let release;
  const suspended = new Promise(resolve => { release = resolve; });
  let jobs = true;
  const events = [], writes = [];
  const run = production('pumpJobs', { jobRunning:false,
    saveSnapshot:async(value,key)=>writes.push({value,key}),
    self:{postMessage:value=>events.push(value)}, persist:async()=>{events.push({checkpoint:true});return true;},
    state:{py:{globals:{get:key=>key==='has_jobs'?()=>jobs:{callPromising:async()=>{await suspended;jobs=false;}}}}} });
  const pending=run();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(writes[0].key,'brew-in-progress');
  assert.ok(writes[0].value.startedAt);
  assert.equal(writes.length,1,'Running job retains its recovery marker');
  release();await pending;
  assert.equal(writes.at(-1).value,null);
  assert.equal(events.at(-1).busy,false);
  assert.ok(events.some(value=>value.checkpoint));
});

test('startup skips micropip entirely when the built archive already contains every pure dependency', async () => {
  const calls=[];
  const load=production('loadPythonPackages',{self:{location:{href:'https://example.test/digests/worker.mjs'}}});
  await load({loadPackage:async packages=>calls.push(packages),globals:{set:()=>assert.fail('no wheel install is needed')},
    runPythonAsync:()=>assert.fail('no empty micropip install')},{pyodide:['numpy','micropip'],micropip:[]});
  assert.deepEqual(Array.from(calls[0]),['numpy']);
});

test('startup retains explicitly listed wheel installation for compatible manifests', async () => {
  const calls=[];
  const load=production('loadPythonPackages',{self:{location:{href:'https://example.test/digests/worker.mjs'}}});
  await load({loadPackage:async packages=>calls.push(['packages',...packages]),globals:{set:(key,value)=>calls.push([key,value])},
    runPythonAsync:async code=>calls.push(['python',code])},{pyodide:['numpy'],micropip:['./assets/wheels/example.whl']});
  assert.ok(calls[0].includes('micropip'));
  assert.match(calls[1][1],/https:\/\/example.test\/digests\/assets\/wheels\/example.whl/);
  assert.match(calls[2][1],/await micropip.install/);
});
