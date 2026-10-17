import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { localPath, routeFromHash, renderedDocument, WorkerClient, browserSupportError, friendlyLoadingStatus, publicRoute, announcementKey, computeStatusText, requestStorageProtection, publicSourceFreshness, settleRenderedEdits, skipToReadingDesk, backupFileText } from './app.mjs';

test('an oversized backup is rejected before reading its contents', async () => {
  await assert.rejects(backupFileText({size:128*1024*1024+1,text:()=>assert.fail('must not allocate the file contents')}), /128 MB/);
  assert.equal(await backupFileText({size:4,text:async()=> 'test'}),'test');
});

test('recovery confirmation names the selected action and defaults to cancellation', async () => {
  const source = readFileSync(new URL('./app.mjs', import.meta.url), 'utf8');
  const start = source.indexOf('  async function confirmRestore(');
  const handler = source.slice(start, source.indexOf('  function restoreProgress(', start));
  const nodes = {'restore-title':{}, 'restore-description':{}, 'restore-proceed':{},
    'restore-cancel':{focus(){this.focused = true;}}};
  const dialog = nodes['restore-confirmation'] = {
    open:false, showModal(){this.open = true;},
    addEventListener(_name, callback){this.onclose = callback;},
    close(){this.open = false;this.onclose();},
  };
  const confirm = runInNewContext(handler + '\nconfirmRestore', {document:{getElementById:id => nodes[id]}});
  const reset = confirm('Create empty data?', {fresh:true});
  assert.equal(nodes['restore-title'].textContent, 'Start fresh?');
  assert.equal(nodes['restore-proceed'].textContent, 'Start fresh');
  assert.equal(nodes['restore-cancel'].focused, true);
  nodes['restore-cancel'].onclick();
  assert.equal(await reset, false);
  const restore = confirm('Replace current data?');
  assert.equal(nodes['restore-title'].textContent, 'Restore saved data?');
  assert.equal(nodes['restore-proceed'].textContent, 'Restore');
  nodes['restore-proceed'].onclick();
  assert.equal(await restore, true);
});

test('acceptance probes covered public texts when an optional cache is partial', () => {
  const source = readFileSync(new URL('./verify-browser.mjs', import.meta.url), 'utf8');
  const match = source.match(/function publicCacheProbeTexts\([\s\S]*?\n\}/);
  assert.ok(match, 'Acceptance needs coverage-aware probe selection');
  const choose = runInNewContext(`(${match[0]})`, {createHash});
  const snapshot = {sources: {feed: {status:'ok', items:[
    {section:'research', title:'Uncached first article', abstract:'New text'},
    {section:'research', title:'Covered article', abstract:'Existing text'},
  ]}}};
  const hash = createHash('sha256').update('Covered article. Existing text').digest('hex');
  assert.equal(choose(snapshot, [hash]).join('|'), 'Covered article. Existing text');
  assert.equal(choose(snapshot, []).length, 0);
  assert.equal(choose({sources:{}}, [hash]).length, 0);
});

test('compute feedback distinguishes selected, exercised, fallback and busy states', () => {
  const selected = { mode: 'gpu', device: 'webgpu', stats: { completedInputs: 0 } };
  assert.match(computeStatusText(selected), /GPU selected; no new text comparisons completed/);
  assert.match(computeStatusText({ ...selected, stats: { completedInputs: 16 } }), /Used GPU for 16/);
  assert.match(computeStatusText({ ...selected, device: 'wasm' }), /GPU requested; using CPU fallback/);
  assert.match(computeStatusText({ ...selected, busy: true }), /Stop it or wait/);
  assert.match(computeStatusText({ ...selected, publicCache: {hits:12} }), /12 public text comparisons reused/);
  assert.match(computeStatusText({ mode: 'multicore', device: 'wasm', threads: 4 }), /up to 4 cores/);
});

test('storage permission reports grants, denial, unsupported browsers and errors honestly', async () => {
  let calls = 0;
  assert.equal((await requestStorageProtection({ persist: async () => { calls++; return true; } })).granted, true);
  assert.equal(calls, 1);
  assert.match((await requestStorageProtection({ persist: async () => false })).message, /not granted/);
  assert.match((await requestStorageProtection(undefined)).message, /cannot request/);
  await assert.rejects(requestStorageProtection({ persist: async () => { throw new Error('denied'); } }), /denied/);
});

test('modified internal links retain the public project path and deep route', () => {
  assert.equal(publicRoute('/saved?section=research', 'https://example.test/daily_digests/#/setup'),
    'https://example.test/daily_digests/#/saved?section=research');
  const html = renderedDocument('<html><head></head><body><a href="/saved">Saved</a></body></html>', 'token', '/', 'https://example.test/daily_digests/');
  assert.match(html, /href="https:\/\/example.test\/daily_digests\/#\/saved"/);
  assert.throws(() => publicRoute('//another.test/'), /local DailyDigest/);
});

test('screen reader progress announces milestones, not each clock tick', () => {
  assert.equal(announcementKey({ phase: 'encoding', completed: 4, total: 100, elapsed_ms: 1000 }),
    announcementKey({ phase: 'encoding', completed: 5, total: 100, elapsed_ms: 2000 }));
  assert.notEqual(announcementKey({ phase: 'encoding', completed: 4, total: 100 }),
    announcementKey({ phase: 'encoding', completed: 14, total: 100 }));
});

test('waiting messages describe real stages without a made-up countdown', () => {
  assert.match(friendlyLoadingStatus('Loading Python'), /kettle/);
  assert.match(friendlyLoadingStatus('Computing embeddings', '4 / 20'), /Steeping.*4 \/ 20/);
  assert.match(friendlyLoadingStatus('Loading embedding model'), /First-time/);
  assert.equal(friendlyLoadingStatus('Other stage', 'actual detail'), 'Other stage · actual detail');
});

test('startup identifies missing browser capabilities before downloading the worker', () => {
  const supported = { WebAssembly: { Suspending() {}, promising() {} }, navigator: { locks: {} } };
  assert.equal(browserSupportError(supported), null);
  assert.match(browserSupportError({}), /current desktop Chrome/);
  assert.ok(browserSupportError({ ...supported, navigator: {} }));
  assert.ok(browserSupportError({ ...supported, WebAssembly: { promising() {} } }));
});

class FakeWorker extends EventTarget {
  sent = [];
  postMessage(data) { this.sent.push(data); }
  respond(data) { this.dispatchEvent(new MessageEvent('message', { data })); }
}

function bridgedFrame() {
  const messages = [];
  const listeners = {};
  const parent = { postMessage: (data, origin) => { assert.equal(origin, 'https://reader.example'); messages.push(data); } };
  const events = [];
  const window = { addEventListener: (name, handler) => { listeners[name] = handler; }, dispatchEvent: event => events.push(event) };
  const document = { addEventListener() {}, getElementById() {} };
  const html = renderedDocument('<html><head></head></html>', 'token', '/run', 'https://reader.example/daily_digests/');
  const script = html.match(/<script\b[^>]*>([\s\S]*?)<\/script>/)[1];
  runInNewContext(script, { parent, window, document, URL, URLSearchParams, FormData, Headers, Request, Response, EventTarget, Event, MessageEvent, CustomEvent });
  return { window, messages, events, deliver: (data, origin='https://reader.example') => listeners.message({ source: parent, origin, data: { ...data, token: 'token' } }) };
}

test('frame ignores authenticated-looking messages from a different origin', () => {
  const frame = bridgedFrame();
  frame.deliver({type:'compute-progress',compute:{completed:4}}, 'https://another.example');
  assert.equal(frame.events.length,0);
});

test('rendered scripts need a per-document nonce and figures permit only listed origins', () => {
  const render=()=>renderedDocument('<html><head></head><body><script>window.example=true;</script><img data-src="https://publisher.example/figure.png"></body></html>', 'token', '/', 'https://reader.example/');
  const first=render(), second=render();
  assert.doesNotMatch(first, /script-src 'unsafe-inline'/);
  assert.match(first, /img-src 'self' data: blob: https:\/\/publisher.example;/);
  assert.notEqual(first.match(/nonce="([^"]+)"/)[1],second.match(/nonce="([^"]+)"/)[1]);
  assert.equal([...first.matchAll(/<script nonce=/g)].length,2);
});

test('live compute progress reaches the brew panel independently of stream polling', () => {
  const frame = bridgedFrame();
  frame.deliver({ type: 'compute-progress', compute: { completed: 4, total: 2340, message: 'Comparing research text' } });
  assert.equal(frame.events.length, 1);
  assert.equal(frame.events[0].type, 'dd-compute-progress');
  assert.equal(frame.events[0].detail.completed, 4);
});

test('companion capabilities reach settings through the authenticated frame bridge', () => {
  const frame = bridgedFrame();
  frame.deliver({ type: 'companion-capabilities', backends: ['api', 'codex_cli'] });
  assert.equal(frame.events.length, 1);
  assert.equal(frame.events[0].type, 'dd-companion-capabilities');
  assert.deepEqual(Array.from(frame.events[0].detail.backends), ['api', 'codex_cli']);
});

test('frame fetch uses worker responses and blocks arbitrary external requests', async () => {
  const frame = bridgedFrame();
  const request = frame.window.fetch('/vote/7/2', { method: 'POST', headers: { 'X-CSRF-Token': 'csrf' }, body: '{}' });
  assert.equal(frame.messages[0].path, '/vote/7/2');
  assert.equal(frame.messages[0].headers['x-csrf-token'], 'csrf');
  frame.deliver({ id: 1, status: 200, headers: { 'content-type': 'application/json' }, body: '{"ok":true}' });
  assert.deepEqual(await (await request).json(), { ok: true });
  await assert.rejects(frame.window.fetch('https://example.org/secret'), /explicitly configured/);
  assert.equal(frame.messages.length, 1);
});

test('frame EventSource handles split SSE chunks and disconnects on close', () => {
  const frame = bridgedFrame();
  const stream = new frame.window.EventSource('/run/stream?run_id=123');
  const events = [];
  stream.onmessage = event => events.push(event.data);
  frame.deliver({ id: 1, type: 'stream', body: 'data: {"stage":' });
  frame.deliver({ id: 1, type: 'stream', body: '"ranking"}\n\n: heartbeat\n\ndata: done\n\n' });
  assert.deepEqual(events, ['{"stage":"ranking"}', 'done']);
  assert.equal(stream.readyState, frame.window.EventSource.OPEN);
  stream.close();
  assert.equal(stream.readyState, frame.window.EventSource.CLOSED);
  assert.equal(frame.messages.at(-1).type, 'cancel');
});

test('Pages hashes retain deep app routes without accepting another origin', () => {
  assert.equal(routeFromHash('#/saved?section=research'), '/saved?section=research');
  assert.equal(routeFromHash('#/run?resume=true'), '/run?resume=true');
  assert.equal(routeFromHash('#https://example.org'), '/');
  assert.equal(routeFromHash('#//example.org/path'), '/');
  assert.throws(() => localPath('https://example.org/run'), /local DailyDigest/);
  assert.equal(localPath('?date=2026-10-13', '/saved'), '/saved?date=2026-10-13');
});

test('trusted rendered pages route template navigation and suppress localhost service worker', () => {
  const source = '<html><head><link rel="manifest" href="/manifest.webmanifest"></head><body><script>window.location.href = `/done?digest_id=${id}`; window.location.reload(); window.history.replaceState(null,"",url); const base=window.location.href; navigator.serviceWorker.register("/sw.js").catch(() => {});</script></body></html>';
  const html = renderedDocument(source, 'test-token', '/run');
  assert.match(html, /window\.ddNavigate\(`\/done\?digest_id=\$\{id\}`\)/);
  assert.match(html, /window\.ddReload\(\)/);
  assert.match(html, /window\.ddReplaceState\(null/);
  assert.match(html, /const base=window\.ddLocation/);
  assert.doesNotMatch(html, /register\("\/sw\.js"\)/);
  assert.doesNotMatch(html, /href="\/manifest/);
  assert.match(html, /Content-Security-Policy/);
  assert.match(html, /source !== parent/);
});

test('worker responses correlate concurrent requests and deliver stream progress', async () => {
  const worker = new FakeWorker(); const client = new WorkerClient(worker); const chunks = [];
  const first = client.call({ type: 'request', path: '/run/stream' }, chunk => chunks.push(chunk));
  const second = client.call({ type: 'request', path: '/saved' });
  worker.respond({ id: 1, type: 'response-chunk', body: 'data: started\n\n' });
  worker.respond({ id: 2, type: 'response', status: 200, body: 'saved' });
  assert.equal((await second).body, 'saved');
  assert.deepEqual(chunks, ['data: started\n\n']);
  assert.equal(client.pending.size, 1);
  worker.respond({ id: 1, type: 'response', status: 200, body: '' });
  await first; assert.equal(client.pending.size, 0);
});

test('worker failure rejects outstanding calls and disconnect cancels the matching stream', async () => {
  const worker = new FakeWorker(); const client = new WorkerClient(worker);
  const first = client.call({ type: 'request', path: '/run/stream' });
  first.cancel();
  await assert.rejects(first, /disconnected/);
  assert.deepEqual(worker.sent[1], { id: 1, type: 'cancel-request' });
  const second = client.call({ type: 'request', path: '/saved' });
  worker.dispatchEvent(new Event('error'));
  await assert.rejects(second, /worker stopped/);
  assert.equal(client.pending.size, 0);
  const sent = worker.sent.length;
  second.cancel();
  assert.equal(worker.sent.length, sent, 'Closing a failed stream must not send to a terminated worker');
});

test('worker request errors are visible and release pending calls', async () => {
  const worker = new FakeWorker(); const client = new WorkerClient(worker);
  const request = client.call({ type: 'request', path: '/setup' });
  worker.respond({ id: 1, type: 'error', message: 'Profile write failed' });
  await assert.rejects(request, /Profile write failed/);
  assert.equal(client.pending.size, 0);
});

test('tab handoff drains an in-flight vote and blocks new actions until explicitly resumed', async () => {
  const worker = new FakeWorker(), client = new WorkerClient(worker);
  const vote = client.call({type:'request', method:'POST', path:'/vote/7/70'});
  let drained = false;
  const waiting = client.quiesce({timeoutMs:1000}).then(() => {drained=true;});
  await Promise.resolve();
  assert.equal(drained,false);
  await assert.rejects(client.call({type:'request', method:'POST', path:'/bookmark/7'}), /switching tabs/i);
  assert.equal(worker.sent.length,1);
  worker.respond({id:1,type:'response',status:200,body:'{"ok":true}'});
  await vote;await waiting;
  const checkpoint = client.call({type:'prepare-reload'});
  worker.respond({id:2,type:'response',status:200,body:'{"ok":true}'});
  await checkpoint;
  client.resume();
  const next = client.call({type:'request',path:'/setup'});
  worker.respond({id:3,type:'response',status:200,body:'settings'});
  await next;
});

test('a handoff timeout leaves the original pending action and desk usable', async () => {
  const worker = new FakeWorker(), client = new WorkerClient(worker);
  const vote = client.call({type:'request',path:'/vote/7/70'});
  await assert.rejects(client.quiesce({timeoutMs:10}), /still finishing/i);
  assert.equal(client.pending.size,1);
  worker.respond({id:1,type:'response',status:200,body:'{"ok":true}'});
  await vote;
  assert.equal(client.paused,false);
});

test('cancelling a UI wait cannot hide a still-running mutation from tab handoff', async () => {
  const worker = new FakeWorker(), client = new WorkerClient(worker);
  const request = client.call({type:'request',method:'POST',path:'/vote/7/70'});
  request.cancel();await assert.rejects(request,/disconnected/);
  assert.equal(client.pending.size,0);
  let drained=false;
  const pending=client.quiesce({timeoutMs:1000}).then(()=>{drained=true;});
  await Promise.resolve();assert.equal(drained,false);
  worker.respond({id:1,type:'response',status:200,body:'{"ok":true}'});
  await pending;assert.equal(drained,true);
});


test('public-source warnings use original oldest success time and distinguish 24h warnings from 48h expiry', () => {
  const now = Date.parse('2026-10-15T12:00:00Z');
  const summary = hours => ({oldest_fetched_at:new Date(now - hours * 3600000).toISOString()});
  assert.equal(publicSourceFreshness(summary(12), now).level, 'fresh');
  assert.equal(publicSourceFreshness(summary(25), now).level, 'warning');
  assert.match(publicSourceFreshness(summary(49), now).message, /49 hours.*expired/);
  assert.equal(publicSourceFreshness({...summary(12), carried_forward_sources:1}, now).level, 'warning');
  assert.equal(publicSourceFreshness({oldest_fetched_at:'invalid'}, now), null);
  assert.equal(publicSourceFreshness(summary(-1), now), null);
});


for(const failure of ['registration','activation-timeout','redundant']) test(`Compute ${failure} failure cancels only its completed reload fence`,async()=>{
  const source=readFileSync(new URL('./app.mjs',import.meta.url),'utf8');
  const start=source.indexOf("  document.getElementById('compute-form').onsubmit =");
  const handler=source.slice(start,source.indexOf('  const download =',start));
  const nodes={'compute-form':{},'compute-apply':{disabled:false},'compute-status':{},'compute-mode':{value:'multicore'}};
  const calls=[];let pending=false,refreshes=0,reloads=0;
  runInNewContext(handler,{document:{getElementById:id=>nodes[id]},computeApplying:false,dirtySetup:false,crossOriginIsolated:false,
    frame:{}, settleRenderedEdits:async()=>{}, client:{quiesce:async()=>{},resume(){}},
    navigator:{serviceWorker:{}},isolationRequested:false,isolationKey:'fixture-isolation',localStorage:{setItem(){}},
    companionCall:async data=>{calls.push(data.type);if(data.type==='configure-compute')pending=true;if(data.type==='cancel-reload')pending=false;return{ok:true};},
    registerWorker:async()=>{if(failure==='registration')throw Error('Fixture registration failed');return{installing:{state:failure==='redundant'?'redundant':'installing',addEventListener(){}}};},
    setTimeout:callback=>{if(failure==='activation-timeout')callback();return 1;},clearTimeout(){},
    refreshCompute:async()=>{if(pending)throw Error('HTTP409: reload pending');refreshes++;nodes['compute-apply'].disabled=false;},
    location:{reload:()=>{reloads++;}}});
  await nodes['compute-form'].onsubmit({preventDefault(){}});
  assert.deepEqual(calls,['configure-compute','cancel-reload']);
  assert.equal(pending,false);assert.equal(refreshes,1);assert.equal(reloads,0);
  assert.equal(nodes['compute-apply'].disabled,false);
  assert.match(nodes['compute-status'].textContent,/registration failed|could not start|unavailable/);
});


test('navigation and handoff await the frame queue before pausing its worker', async()=>{
  let finish, settled=false;
  const pending=new Promise(resolve=>{finish=resolve;});
  const frame={contentDocument:{},contentWindow:{ddFlushRatings:()=>pending}};
  const draining=settleRenderedEdits(frame).then(()=>{settled=true;});
  await new Promise(resolve=>setImmediate(resolve));assert.equal(settled,false);
  finish();await draining;assert.equal(settled,true);
  await assert.rejects(settleRenderedEdits({contentDocument:{},contentWindow:{ddFlushRatings:()=>new Promise(()=>{})}},{timeoutMs:5}),/still finishing/);
});


test('the shell skip link focuses the reading desk without changing its route',()=>{
  let prevented=0,focused='';
  const event={preventDefault(){prevented++;}};
  const main={focus(){focused='main';}},startup={focus(){focused='startup';}};
  const frame={hidden:false,contentDocument:{querySelector:()=>main},focus(){focused='frame';}};
  skipToReadingDesk(event,frame,startup);assert.equal(focused,'main');
  frame.contentDocument=null;skipToReadingDesk(event,frame,startup);assert.equal(focused,'frame');
  frame.hidden=true;skipToReadingDesk(event,frame,startup);assert.equal(focused,'startup');
  assert.equal(startup.tabIndex,-1);assert.equal(prevented,3);
});

function snapshotCopyControls(action, frame, snapshotState = () => null) {
  const source=readFileSync(new URL('./app.mjs',import.meta.url),'utf8');
  const nodes=Object.fromEntries(['export','include-credentials','sync-credentials',
    'companion-email','companion-inbound','companion-push','companion-pull','recovery-export'].map(id=>[id,{checked:false}]));
  const copies=[],captured=[],errors=[],companionStatus={};
  const context={document:{getElementById:id=>nodes[id]},frame,settleRenderedEdits,
    client:{call:async payload=>{copies.push(payload);captured.push(snapshotState());return {status:200,body:'snapshot'};}},
    companionCall:async payload=>{copies.push(payload);captured.push(snapshotState());return {};},companionStatus,
    confirm:()=>true,download(){},localStorage:{setItem(){}},backupKey:'fixture',
    showLastBackup(){},setStatus(){},restoreProgress(){},report:error=>errors.push(error.message)};
  const start=action==='download' ? source.indexOf("  document.getElementById('export').onclick =")
    : action==='raw' ? source.indexOf("  document.getElementById('recovery-export').onclick =")
      : source.indexOf("  for (const action of ['email', 'inbound', 'push', 'pull'])");
  const end=action==='download' ? source.indexOf('  async function restoreBackup',start)
    : action==='raw' ? source.indexOf("  document.getElementById('retained-recovery').onclick",start)
      : source.indexOf('  let tabChannel;',start);
  assert.ok(start>=0 && end>start,'Exercise the actual snapshot-copy handler');
  runInNewContext(source.slice(start,end),context);
  return {copies,captured,errors,companionStatus,click:nodes[action==='download'?'export':action==='raw'?'recovery-export':'companion-push'].onclick};
}

for(const action of ['download','push']) {
  test(`${action} copies wait for the latest queued rating before creating a snapshot`,async()=>{
    const window={addEventListener(){}};
    const create=runInNewContext(readFileSync(new URL('../templates/_rating_queue.js.j2',import.meta.url),'utf8')+';createRatingQueue',{window});
    let release,confirmed=null;
    const rate=create({initial:null,select(){},busy(){},success(){},failure:error=>assert.fail(error.message),
      send:async grade=>{if(grade===70)await new Promise(done=>{release=done;});confirmed=grade;return {};}});
    const first=rate(70),latest=rate(100);
    await new Promise(done=>setImmediate(done));
    const controls=snapshotCopyControls(action,{contentDocument:{},contentWindow:window},()=>confirmed);
    const copying=controls.click();
    await new Promise(done=>setImmediate(done));
    const copiedBeforeLatest=controls.copies.length;
    release();await Promise.all([first,latest,copying]);
    assert.equal(copiedBeforeLatest,0,'No snapshot may be created while a visible rating is still queued');
    assert.equal(confirmed,100);assert.equal(controls.copies.length,1);
    assert.deepEqual(controls.captured,[100],'The copy must capture the latest confirmed rating');
    assert.equal(controls.copies[0].includeCredentials,false);
  });

  test(`${action} reports a rating drain failure without creating a stale copy`,async()=>{
    const controls=snapshotCopyControls(action,{contentDocument:{},contentWindow:{
      ddFlushRatings:()=>Promise.reject(Error('Ratings are still finishing; try again.'))}});
    await controls.click();
    assert.equal(controls.copies.length,0);
    const message=action==='download'?controls.errors[0]:controls.companionStatus.textContent;
    assert.match(message,/Ratings are still finishing/);
  });
}

test('snapshot downloads remain available without a rendered frame, including raw recovery',async()=>{
  const normal=snapshotCopyControls('download',{contentDocument:null});await normal.click();
  assert.equal(normal.copies[0].type,'export');assert.deepEqual(normal.errors,[]);
  const recovery=snapshotCopyControls('raw',{contentDocument:{},contentWindow:{
    ddFlushRatings:()=>assert.fail('Raw recovery must work independently of the rendered desk')}});
  await recovery.click();assert.equal(recovery.copies[0].type,'recovery-export');assert.deepEqual(recovery.errors,[]);
});

for(const action of ['release','compute']) test(`${action} reload keeps unsaved settings usable before any reload fence`,async()=>{
  const source=readFileSync(new URL('./app.mjs',import.meta.url),'utf8');
  const nodes={'compute-form':{},'compute-apply':{disabled:false},'compute-status':{},'compute-mode':{value:'cpu'},'release-message':{}};
  const effects=[],releaseReload={disabled:false};
  const context={document:{getElementById:id=>nodes[id]},dirtySetup:true,computeApplying:false,
    releaseReload,runtimeStarted:true,frame:{draft:'unsaved topics'},settleRenderedEdits:async()=>effects.push('drain'),
    client:{quiesce:async()=>effects.push('pause'),resume:()=>effects.push('resume')},
    companionCall:async data=>{effects.push(data.type);return {};},location:{reload:()=>effects.push('reload')},
    setReleaseState:()=>{releaseReload.disabled=false;},crossOriginIsolated:false,isolationRequested:false,navigator:{},
    refreshCompute:async()=>{},localStorage:{setItem:()=>effects.push('write')}};
  const start=source.indexOf(action==='release'?'  releaseReload.onclick =':"  document.getElementById('compute-form').onsubmit =");
  const end=source.indexOf(action==='release'?"  document.addEventListener('visibilitychange'":'  const download =',start);
  runInNewContext(source.slice(start,end),context);
  await (action==='release'?releaseReload.onclick():nodes['compute-form'].onsubmit({preventDefault(){}}));
  assert.deepEqual(effects,[],'Unsaved edits must be checked before pausing, saving a compute choice or reloading');
  assert.equal(context.frame.draft,'unsaved topics');assert.equal(context.computeApplying,false);
  assert.equal(releaseReload.disabled,false);assert.equal(nodes['compute-apply'].disabled,false);
  assert.match(nodes[action==='release'?'release-message':'compute-status'].textContent,/Save.*settings.*discard/);
});

function setupDraftTracking(initiallyDirty=false) {
  const source=readFileSync(new URL('./app.mjs',import.meta.url),'utf8');
  const fields=new Map([['topics','RNA delivery | 100'],['name','Reader']]);
  const formListeners={},documentListeners={};
  const form={fields,addEventListener:(name,handler)=>{(formListeners[name]||=[]).push(handler);}};
  class FormDataDouble {constructor(target){this.entries=[...target.fields];} [Symbol.iterator](){return this.entries[Symbol.iterator]();}}
  const helper=source.match(/export function trackSetupDraft\([\s\S]*?\n\}/);
  const context={dirtySetup:initiallyDirty,activePath:'/setup',
    document:{querySelectorAll:()=>[]},FormData:FormDataDouble,
    frame:{contentDocument:{querySelector:()=>form,addEventListener:(name,handler)=>{(documentListeners[name]||=[]).push(handler);}}}};
  if(helper)context.trackSetupDraft=runInNewContext(helper[0].replace('export ','')+';trackSetupDraft',context);
  const start=source.indexOf("    frame.contentDocument?.addEventListener('click', () => {");
  runInNewContext(source.slice(start,source.indexOf("    const topics =",start)),context);
  return {fields,dirty:()=>context.dirtySetup,
    dispatch(name){const event={target:{closest:selector=>selector==='form'?form:{}}};
      for(const listener of [...formListeners[name]||[],...documentListeners[name]||[]])listener(event);
    }};
}

test('Settings help buttons leave a clean form clean',()=>{
  const draft=setupDraftTracking();draft.dispatch('click');assert.equal(draft.dirty(),false);
});

test('Settings edits become clean when submitted fields return to their original values',()=>{
  const draft=setupDraftTracking();draft.fields.set('topics','RNA vaccines | 100');draft.dispatch('input');assert.equal(draft.dirty(),true);
  draft.fields.set('topics','RNA delivery | 100');draft.dispatch('input');assert.equal(draft.dirty(),false);
});

test('Settings mutator buttons and transferred startup topics still mark changed submitted data dirty',()=>{
  const draft=setupDraftTracking();draft.fields.set('topics','Protein design | 100');draft.dispatch('click');assert.equal(draft.dirty(),true);
});

test('Settings validation rerenders retain their unsaved submission even at the new form baseline',()=>{
  const draft=setupDraftTracking(true);draft.dispatch('change');assert.equal(draft.dirty(),true);
});

function navigationControls(response) {
  const source=readFileSync(new URL('./app.mjs',import.meta.url),'utf8');
  const errors=[],statuses=[],requests=[];
  const frame={hidden:false,contentDocument:{},contentWindow:{scrollY:0},srcdoc:'original draft',setAttribute(){},removeAttribute(){}};
  const context={frame,errors,tabMoved:false,scrollPositions:new Map(),activePath:'/setup',navigation:0,
    setStatus:message=>statuses.push(message),settleRenderedEdits:async()=>{},streams:new Map(),localPath,
    client:{call:async request=>{requests.push(request);return Array.isArray(response)?response.shift():response;}},Headers,
    history:{replaceState(){}},crypto:{randomUUID:()=> 'fixture-token'},renderedDocument,
    location:{href:'https://reader.test/daily_digests/#/setup'},dirtySetup:true,frameGeneration:null,
    computeTimer:null,clearInterval,latestCompute:null,interactiveSandbox:'allow-scripts allow-same-origin',
    startup:{hidden:true,dataset:{}},previewNotice:{hidden:true},
    document:{getElementById:()=>({})},scheduleLinkedBackup:()=>{context.copies++;},copies:0,
    report:error=>errors.push(error.message)};
  const helper=source.match(/export function httpErrorMessage\([\s\S]*?\n\}/);
  if(helper)context.httpErrorMessage=runInNewContext(helper[0].replace('export ','')+';httpErrorMessage',{Headers});
  const start=source.indexOf('  async function navigate(');
  const navigate=runInNewContext(source.slice(start,source.indexOf("  worker.addEventListener('message'",start))+';navigate',context);
  return {context,frame,errors,statuses,requests,navigate};
}

test('a setup POST validation response renders editable server HTML and remains unsaved',async()=>{
  const body='<html><head><title>Settings</title></head><body><main><p role="alert">Priorities must total 100%.</p><form method="post" action="/setup"><input name="topics" value="RNA delivery | 40"></form></main></body></html>';
  const ui=navigationControls({status:400,headers:{'content-type':'text/html'},body});
  await ui.navigate('/setup',{method:'POST',body:'topics=RNA'});
  assert.deepEqual(ui.errors,[]);assert.match(ui.frame.srcdoc,/RNA delivery \| 40/);
  assert.match(ui.frame.srcdoc,/Priorities must total 100%/);assert.equal(ui.context.dirtySetup,true);
  assert.match(ui.statuses.at(-1),/not.*saved/);assert.equal(ui.context.copies,0);
});

for(const [path,method,status] of [['/setup','GET',400],['/setup','POST',400],['/setup','POST',500],['/run','POST',400]]) {
  test(`${method} ${path} HTTP ${status} does not render or echo generic error HTML`,async()=>{
    const ui=navigationControls({status,headers:{'content-type':'text/html'},body:'<html><body>private error document</body></html>'});
    await ui.navigate(path,{method});
    assert.equal(ui.frame.srcdoc,'original draft');assert.equal(ui.errors.length,1);
    assert.doesNotMatch(ui.errors[0],/html|private error document/);assert.match(ui.errors[0],new RegExp(String(status)));
  });
}

test('a JSON navigation error displays its plain detail without dumping the response',async()=>{
  const ui=navigationControls({status:409,headers:{'content-type':'application/json'},body:JSON.stringify({detail:'A brew is running. Try again when it finishes.',internal:'must not display'})});
  await ui.navigate('/setup',{method:'POST'});
  assert.match(ui.errors[0],/A brew is running/);assert.doesNotMatch(ui.errors[0],/[{}]|internal|must not display/);
  assert.equal(ui.frame.srcdoc,'original draft');
});

test('structured JSON validation details never dump submitted values into navigation errors',async()=>{
  const ui=navigationControls({status:422,headers:{'content-type':'application/json'},body:JSON.stringify({detail:[{input:'submitted-secret',msg:'Invalid value'}]})});
  await ui.navigate('/setup',{method:'POST'});assert.doesNotMatch(ui.errors[0],/submitted-secret|\[|\{/);
});

test('a redirected GET error cannot become an editable POST validation response',async()=>{
  const ui=navigationControls([{status:303,headers:{location:'/setup'},body:''},
    {status:400,headers:{'content-type':'text/html'},body:'<html>unexpected GET error</html>'}]);
  await ui.navigate('/setup',{method:'POST'});assert.equal(ui.frame.srcdoc,'original draft');assert.equal(ui.errors.length,1);
});
