import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { validateRecoverySnapshot } from './storage.mjs';
import { validateReadingView } from './reading-view.mjs';

// Exercise the production dispatcher and idle guard without Pyodide or a DB.
const source = readFileSync(new URL('./worker.mjs', import.meta.url), 'utf8');
const idle = source.match(/function idleForCompanion\(\) \{[\s\S]*?\n\}/)[0];
const replace = source.match(/async function replaceState\(payload\) \{[\s\S]*?\n\}/)[0];
const handler = source.slice(source.indexOf('self.onmessage = async ({ data }) => {'));
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
const turn = () => new Promise(resolve => setImmediate(resolve));
function harness(overrides = {}) {
  const replies = [], writes = [], requests = [];
  const context = vm.createContext({ self: { postMessage: row => replies.push(row) },
    initialize: async () => {}, persist: async () => true, stateReplacing: false,
    validateRecoverySnapshot, validateReadingView, configurationChanging: false, reloadPending: false, emailSending: false, jobRunning: false, companion: null, gpuProbe: null,
    state: { py: { globals: { get: key => key === 'handle_request' ? { callPromising: async payload => {
      requests.push(JSON.parse(payload)); return '{}';
    } } : () => false } } },
    companionEndpoint: value => value, companionCall: async () => JSON.stringify({ ok: true }),
    saveSnapshot: async (value, key) => writes.push({ value, key }),
    replaceSnapshot: async value => writes.push({value, recovery:true}),
    computeCapabilities: () => ({}), computeSelection: () => ({}), save: () => {},
    response: (id, body, status = 200) => ({ id, body, status }), emit: () => {},
    globalThis: { browserCompanionSupports: () => false }, JSON, setTimeout: () => {}, ...overrides });
  vm.runInContext(idle + '\n' + replace + '\n' + handler, context);
  return { context, replies, writes, requests,
    send: data => context.self.onmessage({ data }),
    configure: { id: 1, type: 'configure-companion', endpoint: 'http://127.0.0.1:9999', token: 'fixture' } };
}

function requestHarness({status = 200, method = 'POST', path = '/bookmark/5', busy = false} = {}) {
  const checkpoint = deferred(), delivered = [];
  let saved = false, flushes = 0;
  const h = harness({
    jobRunning: busy,
    prepareComputePackages: async () => {},
    storageWriter: {status: () => ({dirty: !saved, durable: saved}), schedule: () => {}},
    persist: async () => { flushes++; saved = await checkpoint.promise; return saved; },
    emit: value => delivered.push(JSON.parse(value)),
    state: {py: {globals: {get: key => key === 'handle_request'
      ? {callPromising: async () => JSON.stringify({id: 91, type: 'response', status, headers: {}, body: '{}'})}
      : () => key === 'state_generation' ? 3 : false}}},
  });
  return {...h, checkpoint, delivered, flushes: () => flushes,
    request: {id: 91, type: 'request', method, path}};
}

test('successful idle reader edits acknowledge only after the browser checkpoint', async () => {
  for (const path of ['/vote/5/100', '/bookmark/5', '/known/5', '/setup']) {
    const h = requestHarness({path});
    const pending = h.send(h.request);
    await turn();
    assert.equal(h.delivered.length, 0, path + ' acknowledged before durable storage');
    assert.equal(h.flushes(), 1);
    h.checkpoint.resolve(true); await pending;
    assert.equal(h.delivered[0].headers['x-dailydigest-storage'], 'saved');
  }
});

test('a failed reader checkpoint remains pending and does not claim durability', async () => {
  const h = requestHarness();
  const pending = h.send(h.request);
  await turn(); h.checkpoint.resolve(false); await pending;
  assert.equal(h.flushes(), 1);
  assert.equal(h.delivered[0].headers['x-dailydigest-storage'], 'pending');
  assert.ok(h.replies.some(row => row.type === 'storage-dirty'));
});

test('reads, rejected changes and running jobs retain nonblocking checkpoint scheduling', async () => {
  for (const options of [{method: 'GET'}, {status: 403}, {busy: true},
    {path: '/run/start'}, {path: '/refresh'}, {path: '/ranking/train'}]) {
    const h = requestHarness(options);
    await h.send(h.request);
    assert.equal(h.flushes(), 0, JSON.stringify(options));
    assert.equal(h.delivered.length, 1);
  }
});

test('reading copies require the current committed generation and cannot be stored during brewing', async () => {
  const view={version:5,profileName:'Ada',notes:[],date:'2026-10-15',savedAt:'2026-10-15T12:00:00Z',items:[{
    section:'Research',title:'Fictional paper',url:'https://example.org/paper',meta:'Nature',summary:'Finding',reason:'Topic fit',opportunity:'',uncertainty:'',details:'',why:'',caveat:''}]};
  const h=harness({storageWriter:{status:()=>({durable:true})},state:{py:{globals:{get:()=>()=>7}}}});
  await h.send({id:1,type:'cache-reading-view',generation:6,view});
  assert.equal(JSON.parse(h.replies.at(-1).body).cached,false);
  await h.send({id:2,type:'cache-reading-view',generation:7,view});
  assert.equal(JSON.parse(h.replies.at(-1).body).cached,true);
  assert.equal(h.writes.length,1);
  h.context.jobRunning=true;
  await h.send({id:3,type:'cache-reading-view',generation:7,view});
  assert.equal(JSON.parse(h.replies.at(-1).body).cached,false);
  assert.equal(h.writes.length,1);
  h.context.jobRunning=false;
  h.context.storageWriter.status=()=>({durable:false});
  await h.send({id:4,type:'cache-reading-view',generation:7,view});
  assert.equal(JSON.parse(h.replies.at(-1).body).cached,false);
  assert.equal(h.writes.length,1);
});

test('compute status distinguishes the requested GPU, actual CPU fallback and active brew', async () => {
  const h = harness({ computeMode: 'gpu', jobRunning: true, state: {
    capabilities: { gpu: false }, engine: { device: 'wasm', threads: 1,
      stats: { calls: 2, completedInputs: 12, cpuFallbacks: 1 } },
    py: { globals: { get: () => () => true } },
  } });
  await h.send({ id: 10, type: 'compute-status' });
  const status = JSON.parse(h.replies[0].body);
  assert.equal(status.mode, 'gpu');
  assert.equal(status.device, 'wasm');
  assert.equal(status.busy, true);
  assert.equal(status.stats.completedInputs, 12);
});

test('a brew starting during companion health prevents later configuration mutation', async () => {
  const health = deferred();
  const h = harness({ companionCall: () => health.promise });
  const pending = h.send(h.configure);
  await turn(); h.context.jobRunning = true;
  health.resolve(JSON.stringify({ ok: true })); await pending;
  assert.equal(h.replies[0].status, 409);
  assert.equal(h.writes.length, 0);
  assert.equal(h.context.companion, null);
});

test('companion writes block a concurrent brew and release the guard after commit', async () => {
  const saved = deferred(), entered = deferred();
  const h = harness({ saveSnapshot: async () => { entered.resolve(); await saved.promise; } });
  const pending = h.send(h.configure); await entered.promise;
  await h.send({ id: 2, type: 'request', method: 'POST', path: '/run/start' });
  assert.equal(h.replies.find(row => row.id === 2).status, 409);
  assert.equal(h.requests.length, 0);
  saved.resolve(); await pending;
  assert.equal(h.replies.find(row => row.id === 1).status, 200);
  assert.equal(h.context.configurationChanging, false);
});

test('compute changes recheck idle after checkpoint and protect their settings write', async () => {
  const checkpoint = deferred();
  const h = harness({ persist: () => checkpoint.promise });
  const pending = h.send({ id: 1, type: 'configure-compute', mode: 'cpu' });
  await turn(); h.context.jobRunning = true; checkpoint.resolve(true); await pending;
  assert.equal(h.replies[0].status, 409);
  assert.equal(h.writes.length, 0);
});

test('compute writes block concurrent requests and failed commits release the guard', async () => {
  const saved = deferred(), entered = deferred();
  const h = harness({ saveSnapshot: async () => { entered.resolve(); await saved.promise; throw Error('quota'); } });
  const pending = h.send({ id: 1, type: 'configure-compute', mode: 'cpu' }); await entered.promise;
  await h.send({ id: 2, type: 'request', method: 'POST', path: '/run/start' });
  assert.equal(h.replies.find(row => row.id === 2).status, 409);
  assert.equal(h.requests.length, 0);
  saved.resolve(); await pending;
  assert.equal(h.replies.find(row => row.id === 1).status, 500);
  assert.equal(h.context.configurationChanging, false);
});

test('stale brew health cannot overwrite a newly configured companion', async () => {
  const health = deferred(), entered = deferred();
  const old = { endpoint: 'http://127.0.0.1:8000', token: 'old' };
  const h = harness({ companion: old,
    globalThis: { browserCompanionSupports: () => true },
    companionCall: (_path, _payload, _method, _timeout, configuration) => {
      if (configuration === old) { entered.resolve(); return health.promise; }
      return Promise.resolve(JSON.stringify({ ok: true, capabilities: { summarize: ['api'] }, provider_identities: { api: 'new' } }));
    } });
  const get = h.context.state.py.globals.get;
  h.context.state.py.globals.get = key => key === 'selected_browser_backend' ? () => 'api' : get(key);
  const brew = h.send({ id: 2, type: 'request', method: 'POST', path: '/run/start' });
  await entered.promise;
  await h.send(h.configure);
  health.resolve(JSON.stringify({ ok: true, capabilities: { summarize: ['api'] }, provider_identities: { api: 'old' } }));
  await brew;
  assert.equal(h.replies.find(row => row.id === 2).status, 409);
  assert.equal(h.writes.length, 1);
  assert.equal(h.context.companion.provider_identities.api, 'new');
  assert.equal(h.requests.length, 0);
});

test('an import finishing during a health check blocks subsequent configuration writes', async () => {
  const health = deferred();
  const h = harness({ companionCall: () => health.promise });
  const pending = h.send(h.configure); await turn();
  h.context.stateReplacing = true;
  health.resolve(JSON.stringify({ ok: true })); await pending;
  assert.equal(h.replies[0].status, 409);
  assert.equal(h.writes.length, 0);
});


test('oversized import is rejected before checkpoint or private state replacement', async () => {
  let checkpoints = 0, imports = 0;
  const h = harness({ persist: async () => { checkpoints++; return true; } });
  const get = h.context.state.py.globals.get;
  h.context.state.py.globals.get = key => key === 'handle_import'
    ? { callPromising: async () => { imports++; } } : get(key);
  const body = JSON.stringify({ format: 'dailydigest-browser-v1',
    files: Object.fromEntries(Array.from({ length: 10001 }, (_, i) => [`data/f${i}`, ''])) });
  await h.send({ id: 1, type: 'import', overwrite: true, body });
  assert.equal(h.replies[0].status, 500);
  assert.match(h.replies[0].body, /recovery limit/);
  assert.equal(checkpoints, 0);
  assert.equal(imports, 0);
  assert.equal(h.context.stateReplacing, false);
});

test('restore cannot replace a desk whose latest changes failed to checkpoint', async () => {
  let imports = 0;
  const h = harness({persist:async()=>false});
  h.context.state.py.globals.get = key => key === 'handle_import'
    ? {callPromising:async()=>{imports++;}} : ()=>false;
  await h.send({id:50,type:'import',overwrite:true,body:JSON.stringify({format:'dailydigest-browser-v1',files:{}})});
  assert.equal(h.replies[0].status,500);
  assert.match(h.replies[0].body,/save.*current/i);
  assert.equal(imports,0);
  assert.equal(h.context.stateReplacing,false);
});


test('release reload never bypasses an active brew or an unsaved checkpoint', async () => {
  const busy = harness({ jobRunning: true });
  await busy.send({ id: 11, type: 'prepare-reload' });
  assert.equal(busy.replies[0].status, 409);
  const unsaved = harness({ persist: async () => false });
  await unsaved.send({ id: 12, type: 'prepare-reload' });
  assert.equal(unsaved.replies[0].status, 500);
  assert.match(unsaved.replies[0].body, /backup before reloading/);
  const saved = harness();
  await saved.send({ id: 13, type: 'prepare-reload' });
  assert.equal(saved.replies[0].status, 200);
});

test('a confirmed release reload fences later mutations before the old worker is terminated', async () => {
  const h = harness();
  await h.send({id:20,type:'prepare-reload'});
  await h.send({id:21,type:'request',method:'POST',path:'/setup'});
  assert.equal(h.replies.find(row=>row.id===20).status,200);
  assert.equal(h.replies.find(row=>row.id===21).status,409);
  assert.equal(h.requests.length,0);
});

test('failed reload preparation can explicitly reopen the reading desk after a committed Compute choice', async () => {
  const h=harness();
  await h.send({id:30,type:'configure-compute',mode:'multicore',allowIsolationReload:true});
  assert.equal(h.replies.find(row=>row.id===30).status,200);
  await h.send({id:31,type:'request',method:'POST',path:'/setup'});
  assert.equal(h.replies.find(row=>row.id===31).status,409);
  assert.equal(h.requests.length,0);
  await h.send({id:32,type:'cancel-reload'});
  assert.equal(h.replies.find(row=>row.id===32).status,200);
  await h.send({id:33,type:'request',method:'POST',path:'/setup'});
  assert.equal(h.requests.length,1,'An isolation setup failure must leave subsequent settings/votes usable');
});

for(const flag of ['stateReplacing','configurationChanging']) test(`cancel-reload cannot interrupt ${flag}`,async()=>{
  const h=harness({reloadPending:true,[flag]:true});
  await h.send({id:40,type:'cancel-reload'});
  assert.equal(h.replies[0].status,409);
  assert.equal(h.context.reloadPending,true);
  assert.equal(h.context[flag],true);
});

function emailHarness(overrides = {}) {
  const sent = [];
  const h = harness(overrides);
  const get = h.context.state.py.globals.get;
  h.context.state.py.globals.get = key => ({
    companion_email_payload: () => JSON.stringify({ html: '<p>Fixture</p>', subject: 'Fixture', digest_id: '2026-10-16' }),
    companion_email_sent: payload => sent.push(payload),
    handle_import: { callPromising: async () => {} },
  }[key] || get(key));
  return { ...h, sent };
}

test('email fences desk replacement, configuration, brewing and reload through delivery checkpoint', async () => {
  const remote = deferred(), checkpoint = deferred(), checkpointStarted = deferred();
  const h = emailHarness({
    companionCall: path => path === '/email' ? remote.promise : Promise.resolve(JSON.stringify({ ok: true })),
    persist: async () => { checkpointStarted.resolve(); return checkpoint.promise; },
  });
  const sending = h.send({ id: 60, type: 'companion-email' });
  await turn();
  const blocked = [
    { type: 'import', overwrite: true, body: JSON.stringify({ format: 'dailydigest-browser-v1', files: {} }) },
    { type: 'companion-pull', overwrite: true }, h.configure,
    { type: 'request', method: 'POST', path: '/run/start' }, { type: 'prepare-reload' },
  ];
  // Rejections must not wait for the in-flight delivery checkpoint.
  h.context.persist = async () => true;
  for (const [index, data] of blocked.entries()) {
    await h.send({ ...data, id: 70 + index });
    assert.equal(h.replies.find(row => row.id === 70 + index).status, 409);
  }
  h.context.persist = async () => { checkpointStarted.resolve(); return checkpoint.promise; };
  remote.resolve(JSON.stringify({ sent: true }));
  await checkpointStarted.promise;
  assert.equal(h.sent.length, 1);
  assert.equal(h.context.emailSending, true);
  h.context.persist = async () => true;
  await h.send({ id: 80, type: 'prepare-reload' });
  assert.equal(h.replies.find(row => row.id === 80).status, 409);
  checkpoint.resolve(true); await sending;
  assert.equal(h.replies.find(row => row.id === 60).status, 200);
  assert.equal(h.context.emailSending, false);
  await h.send({ id: 81, type: 'prepare-reload' });
  assert.equal(h.replies.find(row => row.id === 81).status, 200);
});

for (const failure of ['network', 'checkpoint']) test(`email releases its fence after ${failure} failure`, async () => {
  const remote = deferred();
  const h = emailHarness({ companionCall: () => remote.promise, persist: async () => failure !== 'checkpoint' });
  const sending = h.send({ id: 90, type: 'companion-email' });
  await turn();
  assert.equal(h.context.emailSending, true);
  if (failure === 'network') remote.resolve(Promise.reject(Error('Fixture network failure')));
  else remote.resolve(JSON.stringify({ sent: true }));
  await sending;
  assert.equal(h.replies.find(row => row.id === 90).status, 500);
  assert.equal(h.sent.length, failure === 'network' ? 0 : 1);
  assert.equal(h.context.emailSending, false);
  h.context.persist = async () => true;
  await h.send({ id: 91, type: 'prepare-reload' });
  assert.equal(h.replies.find(row => row.id === 91).status, 200);
});
