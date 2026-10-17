// Real isolated browser, fictional desk only. No brew or provider calls.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, writeFile, mkdir, stat, mkdtemp} from 'node:fs/promises';
import {resolve, extname, sep} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const site=resolve(process.env.DAILYDIGEST_READING_SITE || 'browser/dist');
const output=resolve(process.env.DAILYDIGEST_READING_REPORT || '.full-review/reading-view');
const fixturePath=process.env.DAILYDIGEST_READING_FIXTURE || '.full-review/browser-launch-2026-10-13/fixture-backup.json';
const cacheMiB=Number(process.env.DAILYDIGEST_READING_CACHE_MIB || 40);
await mkdir(output,{recursive:true});
const inflated=resolve(output,'sqlite-fixture.json');
const inflate=spawnSync(new URL('../.venv/bin/python',import.meta.url).pathname,['-c', `
import base64,json,os,sqlite3,sys,tempfile
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text())
root=Path(tempfile.mkdtemp(prefix='dd-reading-sqlite-',dir=os.environ.get('TMPDIR') or None))
for name,encoded in value['files'].items():
 if name in {'data/digest.db','data/digest.db-wal','data/digest.db-shm'}:
  (root/Path(name).name).write_bytes(base64.b64decode(encoded))
source=sqlite3.connect(root/'digest.db')
source.execute('CREATE TABLE reader_benchmark_padding (value BLOB)')
source.execute('INSERT INTO reader_benchmark_padding VALUES (randomblob(?))',(int(sys.argv[3])*1024*1024,))
source.commit()
target=sqlite3.connect(root/'closed.db');source.backup(target);target.close();source.close()
value['files']['data/digest.db']=base64.b64encode((root/'closed.db').read_bytes()).decode()
value['files'].pop('data/digest.db-wal',None);value['files'].pop('data/digest.db-shm',None)
Path(sys.argv[2]).write_text(json.dumps(value))
`,resolve(fixturePath),inflated,String(cacheMiB)],{encoding:'utf8',maxBuffer:1024*1024});
assert.equal(inflate.status,0,inflate.stderr);
const fixturePayload=await readFile(inflated,'utf8');
const fixture=JSON.parse(fixturePayload);
const mime={'.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.css':'text/css','.json':'application/json','.wasm':'application/wasm','.svg':'image/svg+xml'};
const server=createServer(async(request,response)=>{
  try {
    const path=decodeURIComponent(new URL(request.url,'http://localhost').pathname);
    if(path==='/daily_digests/__seed__'){response.writeHead(200,{'content-type':'text/html'});response.end('<title>Isolated seed</title>');return;}
    if(path==='/daily_digests/__fixture__'){response.writeHead(200,{'content-type':'application/json'});response.end(fixturePayload);return;}
    assert.ok(path.startsWith('/daily_digests/'));
    let file=resolve(site,'.'+path.slice('/daily_digests'.length));
    assert.ok(file===site||file.startsWith(site+sep));
    if((await stat(file)).isDirectory())file=resolve(file,'index.html');
    response.writeHead(200,{'content-type':mime[extname(file)]||'application/octet-stream','cache-control':'max-age=600'});
    response.end(await readFile(file));
  }catch{response.writeHead(404);response.end('Not found');}
});
await new Promise(done=>server.listen(0,'127.0.0.1',done));
const base=process.env.DAILYDIGEST_READING_URL || `http://127.0.0.1:${server.address().port}/daily_digests/`;
const scratch=await mkdtemp(resolve(process.env.TMPDIR || tmpdir(), 'dd-reading-view-'));
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',headless:true,
  args:['--no-sandbox','--enable-experimental-webassembly-jspi'],env:{...process.env,TMPDIR:scratch}});
const report={scope:'Fictional desk with incompressible SQLite padding, actual runtime and UI, no brew/providers.',url:base,cacheMiB,checks:[],reloads:[],errors:[],blocked:[]};
try{
  const context=await browser.newContext({viewport:{width:1360,height:900}});
  await context.route('**/*',async route=>{
    if(route.request().url().startsWith(base)){
      if(new URL(route.request().url()).pathname.endsWith('/worker.mjs')){
        const response=await route.fetch();let source=await response.text();
        source=source.replace("function saveSnapshot(snapshot, key = 'snapshot') {", "async function saveSnapshot(snapshot, key = 'snapshot') { if(self.__readerHold && key==='snapshot') await new Promise(done=>{self.__readerRelease=done;});");
        source=source.replace('self.onmessage = async ({ data }) => {', `self.onmessage = async ({ data }) => {
          if(data.type==='reader-save-config'){
            await initialize();
            if(!self.__readerReadHook){const read=state.py.FS.readFile;state.py.FS.readFile=(path,...args)=>{if(String(path).endsWith('/digest.db'))self.__readerLargeReads=(self.__readerLargeReads||0)+1;return read(path,...args);};self.__readerReadHook=true;}
            if(data.sql)state.py.runPython("from dailydigest.store import _engine; _reader_tx = _engine().begin(); _reader_connection = _reader_tx.__enter__(); _reader_connection.exec_driver_sql('CREATE TABLE IF NOT EXISTS reader_benchmark_mutations (value INTEGER)'); _reader_connection.exec_driver_sql('INSERT INTO reader_benchmark_mutations VALUES (1)'); _reader_tx.__exit__(None, None, None)");
            const checkpointStarted=performance.now();
            if(data.hold!==undefined)self.__readerHold=data.hold;
            if(data.mutate)state.py.runPython("from pathlib import Path; Path('data/reader-latency.txt').write_text('a real private file change')");
            if(data.release)self.__readerRelease?.();
            if(data.checkpoint)await persist({force:true});
            self.postMessage(response(data.id,JSON.stringify({checkpointMs:performance.now()-checkpointStarted,largeReads:self.__readerLargeReads||0,status:storageWriter.status(),equal:data.checkpoint ? await readSnapshot()===state.py.globals.get('export_snapshot')() : null})));return;
          }`);
        return route.fulfill({response,body:source});
      }
      return route.continue();
    }
    report.blocked.push(new URL(route.request().url()).hostname);return route.abort();
  });
  if(process.env.DAILYDIGEST_READING_URL){
    await context.route(base+'__seed__',route=>route.fulfill({contentType:'text/html',body:'<title>Isolated seed</title>'}));
    await context.route(base+'__fixture__',route=>route.fulfill({contentType:'application/json',body:fixturePayload}));
  }
  const page=await context.newPage();page.setDefaultTimeout(30000);
  page.on('dialog',async dialog=>{(report.dialogs ||= []).push({type:dialog.type(),message:dialog.message()});await dialog.accept();});
  page.on('pageerror',error=>report.errors.push(error.stack));
  await page.addInitScript(()=>{
    window.__workerStarts=0;window.__ready=false;window.__inflight=new Set();window.__completedPaths=[];
    const requests=new Map();
    const Native=Worker;window.Worker=class extends Native{constructor(...args){super(...args);window.__workerStarts++;
      let rpcSequence=0;const rpcPending=new Map();window.__readingRPC=data=>new Promise((done,fail)=>{const id='reading-rpc-'+(++rpcSequence);const timer=setTimeout(()=>{rpcPending.delete(id);fail(Error('Timed out waiting for '+data.type));},30000);rpcPending.set(id,value=>{clearTimeout(timer);done(value);});this.postMessage({...data,id});});
      this.addEventListener('message',({data})=>{if(data.type==='ready')window.__ready=true;if(data.type==='response'||data.type==='error'){
        if(rpcPending.has(data.id)){rpcPending.get(data.id)(data);rpcPending.delete(data.id);}
        window.__inflight.delete(data.id);if(requests.has(data.id)){window.__completedPaths.push(requests.get(data.id));requests.delete(data.id);}
      }});}
      postMessage(data){if(data.type==='request'){window.__inflight.add(data.id);requests.set(data.id,data.path);}super.postMessage(data);}};
  });
  async function stored(key){return page.evaluate(async key=>{
    const request=indexedDB.open('dailydigest:/daily_digests/',1);
    const db=await new Promise((done,fail)=>{request.onsuccess=()=>done(request.result);request.onerror=()=>fail(request.error);});
    try{return await new Promise((done,fail)=>{const get=db.transaction('state').objectStore('state').get(key);get.onsuccess=()=>done(get.result);get.onerror=()=>fail(get.error);});}finally{db.close();}
  },key);}
  async function canonicalHash(){return page.evaluate(async()=>{
    const request=indexedDB.open('dailydigest:/daily_digests/',1);
    const db=await new Promise(done=>{request.onsuccess=()=>done(request.result);});
    const value=await new Promise(done=>{const get=db.transaction('state').objectStore('state').get('snapshot');get.onsuccess=()=>done(get.result);});db.close();
    const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value));
    return Array.from(new Uint8Array(digest),byte=>byte.toString(16).padStart(2,'0')).join('');
  });}
  await page.goto(base+'__seed__');
  await page.evaluate(async()=>{
    // Avoid sending an entire large backup through the DevTools protocol.
    const value=await (await fetch('./__fixture__')).text();
    const request=indexedDB.open('dailydigest:/daily_digests/',1);
    request.onupgradeneeded=()=>request.result.createObjectStore('state');
    const db=await new Promise(done=>{request.onsuccess=()=>done(request.result);});
    await new Promise((done,fail)=>{const tx=db.transaction('state','readwrite');tx.objectStore('state').put(value,'snapshot');tx.oncomplete=done;tx.onerror=()=>fail(tx.error);});db.close();
  });
  await page.goto(base+'#/');await page.waitForFunction(()=>window.__ready,undefined,{timeout:180000});
  report.buildId=await page.locator('meta[name="dailydigest-build"]').getAttribute('content');
  const frame=()=>page.frameLocator('#app');
  await frame().locator('.item[data-section="research"]').first().waitFor();
  const title=await frame().locator('.title-link').first().textContent();
  // A held IndexedDB save must not delay a route response. The model cache is
  // deliberately large; unchanged binary files must not be read/re-encoded.
  console.log('reading benchmark: runtime ready');
  const baseline=await page.evaluate(()=>window.__readingRPC({type:'flush-state'}));assert.equal(baseline.status,200);
  await page.evaluate(()=>window.__readingRPC({type:'reader-save-config',hold:true,mutate:true}));
  const responseStart=Date.now();
  const fast=await Promise.race([page.evaluate(()=>window.__readingRPC({type:'request',method:'GET',path:'/setup'})),new Promise((_r,reject)=>setTimeout(()=>reject(Error('Route waited for a held checkpoint')),2000))]);
  assert.equal(fast.status,200);report.routeDuringHeldSaveMs=Date.now()-responseStart;
  const heldDeadline=Date.now()+30000;
  let held;
  do {
    held=JSON.parse((await page.evaluate(()=>window.__readingRPC({type:'reader-save-config'}))).body);
    if(held.status.saving)break;
    await page.waitForTimeout(50);
  } while(Date.now()<heldDeadline);
  assert.equal(held.status.saving,true,'The checkpoint writer must reach the held save before release.');
  assert.equal(held.status.durable,false);
  const final=JSON.parse((await page.evaluate(()=>window.__readingRPC({type:'reader-save-config',hold:false,release:true,checkpoint:true}))).body);
  assert.equal(final.equal,true,'Incremental checkpoint must equal a full export byte-for-byte');assert.equal(final.status.durable,true);
  console.log('reading benchmark: held checkpoint released');
  report.sqlCheckpointMs=[];
  for(let trial=0;trial<3;trial++){
    const result=JSON.parse((await page.evaluate(()=>window.__readingRPC({type:'reader-save-config',sql:true,checkpoint:true}))).body);
    assert.equal(result.equal,true);assert.equal(result.status.durable,true);report.sqlCheckpointMs.push(Math.round(result.checkpointMs));
  }
  report.checks.push('route returns before a held checkpoint on a 40 MiB SQLite desk; each SQL checkpoint equals full export');
  console.log('reading benchmark: SQL checkpoints complete');
  // Exercise real feedback before refresh and let the UI's derived copy settle.
  console.log('reading benchmark: rating and saving');
  const card=frame().locator('.item[data-section="research"]').first();
  const itemId=await card.getAttribute('data-item-id');
  await card.locator('.vote-btn[data-grade="70"]').click();
  await card.locator('.vote-btn[data-grade="70"][aria-pressed="true"]').waitFor();
  await card.locator('.bookmark-btn:not(.known-btn)').click();
  await card.locator('.bookmark-btn:not(.known-btn)[aria-pressed="true"]').waitFor();
  await page.waitForFunction(id=>window.__completedPaths.includes('/bookmark/'+id)&&window.__completedPaths.includes('/vote/'+id+'/70'),itemId);
  await page.evaluate(()=>document.querySelector('#app').contentWindow.ddFlushRatings());
  assert.equal((await page.evaluate(()=>window.__readingRPC({type:'flush-state'}))).status,200);
  console.log('reading benchmark: feedback confirmed');
  let projection;
  const deadline=Date.now()+30000;
  while(!(projection=await stored('reading-view')) && Date.now()<deadline) await page.waitForTimeout(50);
  assert.ok(projection, 'The digest must regenerate its reading copy after feedback commits');
  assert.ok(projection.items.some(item=>item.title===title));
  assert.deepEqual(Object.keys(projection).sort(),['date','items','savedAt','version']);
  report.checks.push('real digest and feedback produce a bounded reading copy without credentials');
  console.log('reading benchmark: reading copy available');
  const canonical=await canonicalHash();
  for(let trial=0;trial<3;trial++){
    const start=Date.now();await page.reload({waitUntil:'domcontentloaded'});
    await page.locator('#reading-view-notice').waitFor({state:'visible'});
    await frame().getByRole('link',{name:title,exact:true}).waitFor();
    report.reloads.push({milliseconds:Date.now()-start,workerStarts:await page.evaluate(()=>window.__workerStarts)});
    assert.equal(await page.evaluate(()=>window.__workerStarts),0,'Cached reading must not initialize Python');
    assert.equal(await page.locator('#export').isDisabled(),true);
    assert.equal(await page.locator('#compute-mode').isDisabled(),true);
    assert.match(await page.locator('#compute-status').textContent(), /paused while you read/);
    assert.equal(await page.locator('#app').getAttribute('sandbox'),'allow-popups allow-popups-to-escape-sandbox');
    assert.equal(await canonicalHash(),canonical,'Read-only refresh must leave canonical bytes unchanged');
  }
  report.checks.push('three refreshes show the saved articles without starting any worker or changing saved bytes');
  await page.screenshot({path:output+'/desktop-reading.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.screenshot({path:output+'/mobile-reading.png',fullPage:true});
  console.log('reading benchmark: cached reloads complete');
  await page.locator('#reading-view-open').click();
  await page.waitForFunction(()=>window.__ready,undefined,{timeout:180000});
  await page.locator('#reading-view-notice').waitFor({state:'hidden'});
  await frame().locator('.vote-btn[data-grade="70"][aria-pressed="true"]').first().waitFor();
  await frame().locator('.bookmark-btn:not(.known-btn)[aria-pressed="true"]').first().waitFor();
  assert.equal(await page.evaluate(()=>window.__workerStarts),1);
  assert.match(await page.locator('#app').getAttribute('sandbox'),/allow-scripts/);
  report.checks.push('opening controls starts exactly one runtime and restores real vote/save controls');
  // Fence the old worker before corrupting the disposable cache: an in-flight
  // checkpoint must not regenerate a valid copy between this write and reload.
  assert.equal((await page.evaluate(()=>window.__readingRPC({type:'prepare-reload'}))).status,200);
  // A malformed disposable record must not prevent ordinary startup/recovery.
  await page.evaluate(async()=>{
    const r=indexedDB.open('dailydigest:/daily_digests/',1);const db=await new Promise(done=>{r.onsuccess=()=>done(r.result);});
    await new Promise(done=>{const tx=db.transaction('state','readwrite');tx.objectStore('state').put({version:999},'reading-view');tx.oncomplete=done;});db.close();
  });
  await page.reload();await page.waitForFunction(()=>window.__ready,undefined,{timeout:180000});
  await frame().locator('.item[data-section="research"]').first().waitFor();
  report.checks.push('malformed reading copy falls back to the authoritative runtime');
  assert.deepEqual(report.dialogs || [],[], 'A confirmed clean refresh must not require discarding pending changes');
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.blocked,[]);report.success=true;
}catch(error){
  report.success=false;report.failure=error.stack;
  report.failedPages=[];
  for(const context of browser.contexts())for(const page of context.pages()){
    report.failedPages.push(await page.evaluate(()=>({url:location.href,status:document.querySelector('#status')?.textContent,error:document.querySelector('#app-error')?.textContent,frame:document.querySelector('#app')?.contentDocument?.body?.innerText?.slice(0,5000)})).catch(()=>({closed:true})));
  }
  throw error;
}
finally{await writeFile(output+'/report.json',JSON.stringify(report,null,2)+'\n');await browser.close();await new Promise(done=>server.close(done));}
console.log(JSON.stringify(report,null,2));
