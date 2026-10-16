// Real isolated browser, fictional desk only. No brew or provider calls.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, writeFile, mkdir, stat, mkdtemp} from 'node:fs/promises';
import {resolve, extname, sep} from 'node:path';
import {tmpdir} from 'node:os';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const site=resolve(process.env.DAILYDIGEST_READING_SITE || 'browser/dist');
const output=resolve(process.env.DAILYDIGEST_READING_REPORT || '.full-review/reading-view');
const fixture=JSON.parse(await readFile(process.env.DAILYDIGEST_READING_FIXTURE || '.full-review/browser-launch-2026-10-13/fixture-backup.json','utf8'));
const cacheMiB=Number(process.env.DAILYDIGEST_READING_CACHE_MIB || 40);
fixture.files['data/startup-cache.bin']=Buffer.alloc(cacheMiB*1024*1024).toString('base64');
const fixturePayload=JSON.stringify(fixture);
await mkdir(output,{recursive:true});
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
const report={scope:'Fictional desk, actual runtime and UI, no brew/providers.',url:base,cacheMiB,checks:[],reloads:[],errors:[],blocked:[]};
try{
  const context=await browser.newContext({viewport:{width:1360,height:900}});
  await context.route('**/*',route=>{
    if(route.request().url().startsWith(base))return route.continue();
    report.blocked.push(new URL(route.request().url()).hostname);return route.abort();
  });
  if(process.env.DAILYDIGEST_READING_URL){
    await context.route(base+'__seed__',route=>route.fulfill({contentType:'text/html',body:'<title>Isolated seed</title>'}));
    await context.route(base+'__fixture__',route=>route.fulfill({contentType:'application/json',body:fixturePayload}));
  }
  const page=await context.newPage();page.setDefaultTimeout(30000);
  page.on('pageerror',error=>report.errors.push(error.stack));
  await page.addInitScript(()=>{
    window.__workerStarts=0;window.__ready=false;window.__inflight=new Set();window.__completedPaths=[];
    const requests=new Map();
    const Native=Worker;window.Worker=class extends Native{constructor(...args){super(...args);window.__workerStarts++;
      this.addEventListener('message',({data})=>{if(data.type==='ready')window.__ready=true;if(data.type==='response'||data.type==='error'){
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
  // Exercise real feedback before refresh and let the UI's derived copy settle.
  const card=frame().locator('.item[data-section="research"]').first();
  const itemId=await card.getAttribute('data-item-id');
  await card.locator('.vote-btn[data-grade="70"]').click();
  await card.locator('.vote-btn[data-grade="70"][aria-pressed="true"]').waitFor();
  await card.locator('.bookmark-btn:not(.known-btn)').click();
  await card.locator('.bookmark-btn:not(.known-btn)[aria-pressed="true"]').waitFor();
  await page.waitForFunction(id=>window.__completedPaths.includes('/bookmark/'+id)&&window.__completedPaths.includes('/vote/'+id+'/70'),itemId);
  let projection;
  const deadline=Date.now()+30000;
  while(!(projection=await stored('reading-view')) && Date.now()<deadline) await page.waitForTimeout(50);
  assert.ok(projection, 'The digest must regenerate its reading copy after feedback commits');
  assert.ok(projection.items.some(item=>item.title===title));
  assert.deepEqual(Object.keys(projection).sort(),['date','items','savedAt','version']);
  report.checks.push('real digest and feedback produce a bounded reading copy without credentials');
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
  await page.locator('#reading-view-open').click();
  await page.waitForFunction(()=>window.__ready,undefined,{timeout:180000});
  await page.locator('#reading-view-notice').waitFor({state:'hidden'});
  await frame().locator('.vote-btn[data-grade="70"][aria-pressed="true"]').first().waitFor();
  await frame().locator('.bookmark-btn:not(.known-btn)[aria-pressed="true"]').first().waitFor();
  assert.equal(await page.evaluate(()=>window.__workerStarts),1);
  assert.match(await page.locator('#app').getAttribute('sandbox'),/allow-scripts/);
  report.checks.push('opening controls starts exactly one runtime and restores real vote/save controls');
  // A malformed disposable record must not prevent ordinary startup/recovery.
  await page.evaluate(async()=>{
    const r=indexedDB.open('dailydigest:/daily_digests/',1);const db=await new Promise(done=>{r.onsuccess=()=>done(r.result);});
    await new Promise(done=>{const tx=db.transaction('state','readwrite');tx.objectStore('state').put({version:999},'reading-view');tx.oncomplete=done;});db.close();
  });
  await page.reload();await page.waitForFunction(()=>window.__ready,undefined,{timeout:180000});
  await frame().locator('.item[data-section="research"]').first().waitFor();
  report.checks.push('malformed reading copy falls back to the authoritative runtime');
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.blocked,[]);report.success=true;
}catch(error){report.success=false;report.failure=error.stack;throw error;}
finally{await writeFile(output+'/report.json',JSON.stringify(report,null,2)+'\n');await browser.close();await new Promise(done=>server.close(done));}
console.log(JSON.stringify(report,null,2));
