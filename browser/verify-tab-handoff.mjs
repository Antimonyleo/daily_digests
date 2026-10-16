// Real Web Locks and two tabs; fictional state, no live brew or provider calls.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile,writeFile,mkdir,stat,mkdtemp} from 'node:fs/promises';
import {resolve,sep,extname} from 'node:path';
import {tmpdir} from 'node:os';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const site=resolve(process.env.DAILYDIGEST_TAB_SITE||'browser/dist');
const output=resolve(process.env.DAILYDIGEST_TAB_REPORT||'.full-review/tab-handoff');
const fixture=await readFile(process.env.DAILYDIGEST_TAB_FIXTURE||'.full-review/browser-launch-2026-10-13/fixture-backup.json','utf8');
await mkdir(output,{recursive:true});
const mime={'.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.json':'application/json','.css':'text/css','.svg':'image/svg+xml','.wasm':'application/wasm'};
const server=createServer(async(req,res)=>{
  try{
    const path=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    if(path==='/daily_digests/__seed__'){res.writeHead(200,{'content-type':'text/html'});res.end('<title>Fictional test desk</title>');return;}
    assert.ok(path.startsWith('/daily_digests/'));let file=resolve(site,'.'+path.slice('/daily_digests'.length));
    assert.ok(file===site||file.startsWith(site+sep));if((await stat(file)).isDirectory())file=resolve(file,'index.html');
    res.writeHead(200,{'content-type':mime[extname(file)]||'application/octet-stream'});res.end(await readFile(file));
  }catch{res.writeHead(404);res.end('Not found');}
});
await new Promise(done=>server.listen(0,'127.0.0.1',done));
const base=process.env.DAILYDIGEST_TAB_URL||`http://127.0.0.1:${server.address().port}/daily_digests/`;
const scratch=await mkdtemp(resolve(process.env.TMPDIR||tmpdir(),'dd-tab-handoff-'));
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',headless:true,
  args:['--no-sandbox','--enable-experimental-webassembly-jspi'],env:{...process.env,TMPDIR:scratch}});
const report={scope:'Two real tabs, fictional state; pending vote, failed flush and active brew simulated at the actual worker boundary.',checks:[],errors:[],blocked:[]};
try{
  const context=await browser.newContext();
  await context.route('**/*',async route=>{
    const url=route.request().url();
    if(!url.startsWith(base)){report.blocked.push(new URL(url).hostname);return route.abort();}
    if(new URL(url).pathname.endsWith('/__seed__'))return route.fulfill({contentType:'text/html',body:'<title>Fictional test desk</title>'});
    if(new URL(url).pathname.endsWith('/worker.mjs')){
      const response=await route.fetch();let source=await response.text();
      source=source.replace('self.onmessage = async ({ data }) => {',`self.onmessage = async ({ data }) => {
        if(data.type==='tab-test-config'){
          await initialize();
          if(data.hold!==undefined)self.__holdVote=data.hold;
          if(data.quota!==undefined)self.__quota=data.quota;
          if(data.busy!==undefined){jobRunning=data.busy;self.postMessage({type:'compute-state',busy:jobRunning});}
          if(data.release)self.__releaseVote?.();
          if(data.quota===false)await persist({force:true});
          self.postMessage(response(data.id,'{}'));return;
        }`);
      source=source.replace("    const result = await state.py.globals.get('handle_request')",`    if(self.__holdVote && data.method==='POST' && data.path.startsWith('/vote/')){
        self.postMessage({type:'tab-test-held'});await new Promise(done=>{self.__releaseVote=done;});
      }
      const result = await state.py.globals.get('handle_request')`);
      source=source.replace('async function persist(options = {}) {','async function persist(options = {}) { if(self.__quota)return false;');
      return route.fulfill({response,body:source});
    }
    return route.continue();
  });
  await context.addInitScript(()=>{
    window.__ready=false;window.__held=false;window.__workers=0;
    const Native=Worker;window.Worker=class extends Native{constructor(...args){super(...args);window.__workers++;
      let id=0;const pending=new Map();window.__rpc=data=>new Promise(done=>{const key='tab-test-'+(++id);pending.set(key,done);this.postMessage({...data,id:key});});
      this.addEventListener('message',({data})=>{if(data.type==='ready')window.__ready=true;if(data.type==='tab-test-held')window.__held=true;
        if(pending.has(data.id)){pending.get(data.id)(data);pending.delete(data.id);}});
    }};
  });
  async function page(){const p=await context.newPage();p.setDefaultTimeout(30000);p.on('pageerror',e=>report.errors.push(e.stack));return p;}
  async function ready(p){await p.waitForFunction(()=>window.__ready && document.querySelector('#app')?.hidden===false && document.querySelector('#startup')?.dataset.state==='ready',undefined,{timeout:180000});await p.waitForFunction(()=>typeof document.querySelector('#app')?.contentDocument?.defaultView?.ddNavigate==='function');}
  async function route(p,path,selector){await p.evaluate(path=>{window.__old=document.querySelector('#app').contentDocument;document.querySelector('#app').contentWindow.ddNavigate(path);},path);
    await p.waitForFunction(()=>document.querySelector('#app')?.contentDocument!==window.__old);if(selector)await p.frameLocator('#app').locator(selector).first().waitFor();}
  async function config(p,data){const value=await p.evaluate(data=>window.__rpc({type:'tab-test-config',...data}),data);assert.equal(value.status,200,value.body);}
  async function oneOwner(p){assert.equal(await p.evaluate(async()=>{const locks=await navigator.locks.query();return locks.held.filter(lock=>lock.name==='dailydigest:/daily_digests/').length;}),1);}
  const first=await page();await first.goto(base+'__seed__');
  await first.evaluate(async payload=>{const r=indexedDB.open('dailydigest:/daily_digests/',1);r.onupgradeneeded=()=>r.result.createObjectStore('state');
    const db=await new Promise(done=>{r.onsuccess=()=>done(r.result);});await new Promise(done=>{const tx=db.transaction('state','readwrite');tx.objectStore('state').put(payload,'snapshot');tx.oncomplete=done;});db.close();},fixture);
  await first.goto(base+'#/');await ready(first);const card=first.frameLocator('#app').locator('.item[data-section="research"]').first();await card.waitFor();
  await card.locator('.bookmark-btn:not(.known-btn)').click();await card.locator('.bookmark-btn:not(.known-btn)[aria-pressed="true"]').waitFor();
  const second=await page();await second.goto(base+'#/setup');await second.locator('#tab-wait-actions').waitFor({state:'visible'});
  assert.equal(await second.locator('#retry').isVisible(),true);await oneOwner(first);
  report.checks.push('second editable tab has actionable switch/close/retry instructions without acquiring a second lock');
  await config(first,{hold:true});await card.locator('.vote-btn[data-grade="70"]').click();await first.waitForFunction(()=>window.__held);
  await second.locator('#use-this-tab').click();await first.waitForTimeout(300);
  assert.notEqual(await first.locator('#startup').getAttribute('data-state'),'moved');await oneOwner(first);
  await config(first,{hold:false,release:true});
  // A cold model on a real network can legitimately outlast the bounded drain.
  // Refusal must retain the owner and vote; retry only after the vote finishes.
  const switched=await Promise.race([
    first.locator('#startup[data-state="moved"]').waitFor().then(()=>true),
    second.locator('#startup-status').filter({hasText:'still finishing an action'}).waitFor().then(()=>false),
  ]);
  if(!switched){
    await oneOwner(first);
    await first.frameLocator('#app').locator('.vote-btn[data-grade="70"][aria-pressed="true"]').first().waitFor();
    await second.locator('#use-this-tab').click();
    report.slowActionRefusedSafely=true;
  }
  await ready(second);
  await first.locator('#startup[data-state="moved"]').waitFor();await route(second,'/','.item[data-section="research"]');
  await second.frameLocator('#app').locator('.vote-btn[data-grade="70"][aria-pressed="true"]').first().waitFor();
  await second.frameLocator('#app').locator('.bookmark-btn:not(.known-btn)[aria-pressed="true"]').first().waitFor();await oneOwner(second);
  report.checks.push('handoff waits for an in-flight vote, saves before releasing, restores vote/save and fences the old tab');
  await route(second,'/setup','[data-interest-name]');await second.frameLocator('#app').locator('[data-interest-name]').first().fill('Unsaved topic draft');
  await first.locator('#use-this-tab').click();await first.locator('#startup-status').filter({hasText:'unsaved settings'}).waitFor();
  assert.equal(await second.frameLocator('#app').locator('[data-interest-name]').first().inputValue(),'Unsaved topic draft');await oneOwner(second);
  report.checks.push('unsaved setup edits refuse handoff and remain intact');
  await route(second,'/','.item[data-section="research"]');await config(second,{quota:true});
  await second.frameLocator('#app').locator('.vote-btn[data-grade="100"]').first().click();
  await first.locator('#use-this-tab').click();await first.locator('#startup-status').filter({hasText:'could not be saved'}).waitFor();await oneOwner(second);
  await config(second,{quota:false});await first.locator('#use-this-tab').click();await ready(first);
  await first.frameLocator('#app').locator('.vote-btn[data-grade="100"][aria-pressed="true"]').first().waitFor();
  report.checks.push('failed checkpoint keeps the original owner; retry after saving restores the latest vote');
  await config(first,{busy:true});await second.locator('#use-this-tab').click();
  await second.locator('#startup-status').filter({hasText:'Finish or stop the brew'}).waitFor();await oneOwner(first);await config(first,{busy:false});
  report.checks.push('active brew refuses takeover instead of being terminated');
  await second.goto(base+'#/setup');await second.reload();await second.locator('#tab-wait-actions').waitFor({state:'visible'});await first.close();await ready(second);
  await route(second,'/','.item[data-section="research"]');await second.frameLocator('#app').locator('.vote-btn[data-grade="100"][aria-pressed="true"]').first().waitFor();
  await oneOwner(second);report.checks.push('closing the owner releases the waiting tab normally and keeps feedback');
  await second.screenshot({path:output+'/active-desk.png',fullPage:true});
  const restricted=await page();await restricted.addInitScript(()=>{window.BroadcastChannel=class{constructor(){throw new DOMException('Messaging restricted','SecurityError');}};});
  await restricted.goto(base+'#/setup');await restricted.locator('#tab-wait-actions').waitFor({state:'visible'});
  assert.equal(await restricted.locator('#use-this-tab').isHidden(),true);
  assert.equal(await restricted.locator('#retry').isVisible(),true);
  assert.match(await restricted.locator('#startup-status').textContent(), /close it and Retry/);
  await restricted.close();report.checks.push('restricted messaging keeps startup usable with close/retry guidance');
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.blocked,[]);report.success=true;
}catch(error){report.success=false;report.failure=error.stack;throw error;}
finally{await writeFile(output+'/report.json',JSON.stringify(report,null,2)+'\n');await browser.close();await new Promise(done=>server.close(done));}
console.log(JSON.stringify(report,null,2));
