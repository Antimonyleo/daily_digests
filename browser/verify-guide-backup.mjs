// Offline real app and browser file handles. The native file chooser is replaced
// with an origin-private test file; no reader folders or providers are accessed.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, writeFile, mkdir, mkdtemp} from 'node:fs/promises';
import {resolve, sep, extname} from 'node:path';
import {tmpdir} from 'node:os';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const site=resolve(process.env.DAILYDIGEST_GUIDE_SITE || 'browser/dist');
const output=resolve(process.env.DAILYDIGEST_GUIDE_REPORT || '.full-review/guide-backup');
const fixture=await readFile(process.env.DAILYDIGEST_GUIDE_FIXTURE || '.full-review/browser-launch-2026-10-13/fixture-backup.json','utf8');
await mkdir(output,{recursive:true});
const mime={'.mjs':'text/javascript','.js':'text/javascript','.html':'text/html','.css':'text/css','.json':'application/json','.wasm':'application/wasm','.svg':'image/svg+xml'};
const server=createServer(async(request,response)=>{
  try {
    let path=new URL(request.url,'http://localhost').pathname;
    assert.ok(path.startsWith('/daily_digests/'));
    if(path.endsWith('/'))path+='index.html';
    const file=resolve(site,'.'+path.slice('/daily_digests'.length));
    assert.ok(file.startsWith(site+sep));
    response.setHeader('Content-Type',mime[extname(file)]||'application/octet-stream');
    response.end(await readFile(file));
  }catch{response.writeHead(404).end();}
});
await new Promise(done=>server.listen(0,'127.0.0.1',done));
const base=`http://127.0.0.1:${server.address().port}/daily_digests/`;
const scratch=await mkdtemp(resolve(process.env.TMPDIR||tmpdir(),'dd-guide-backup-'));
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',headless:true,
  args:['--no-sandbox','--enable-experimental-webassembly-jspi'],env:{...process.env,TMPDIR:scratch}});
const report={checks:[],errors:[],blocked:[],nativePickerVerified:false};
async function context() {
  const value=await browser.newContext({viewport:{width:1280,height:900}});
  await value.route('**/*',route=>{
    if(route.request().url().startsWith(base))return route.continue();
    report.blocked.push(route.request().url());return route.abort();
  });
  await value.addInitScript(()=>{
    window.__ready=false;window.__workers=0;
    const Original=Worker;
    window.Worker=class extends Original {
      constructor(...args){super(...args);window.__workers++;window.__worker=this;this.addEventListener('message',({data})=>{if(data.type==='ready')window.__ready=true;});}
    };
    window.__rpc=payload=>new Promise(resolve=>{
      const id='guide-'+crypto.randomUUID();
      const listener=({data})=>{if(data.id===id){window.__worker.removeEventListener('message',listener);resolve(data);}};
      window.__worker.addEventListener('message',listener);window.__worker.postMessage({...payload,id});
    });
    window.showSaveFilePicker=async()=>{
      const root=await navigator.storage.getDirectory();
      return root.getFileHandle('guide-test-backup.json',{create:true});
    };
  });
  return value;
}
async function newPage(owner) {
  const page=await owner.newPage();page.setDefaultTimeout(30000);
  page.on('pageerror',error=>report.errors.push(error.stack));
  page.on('dialog',dialog=>dialog.accept());
  return page;
}
const ready=page=>page.waitForFunction(()=>window.__ready,undefined,{timeout:120000});
const disk=page=>page.evaluate(async()=>await (await (await navigator.storage.getDirectory()).getFileHandle('guide-test-backup.json')).getFile().then(file=>file.text()));
async function waitForCopy(page, expected) {
  const deadline=Date.now()+130000;
  // Playwright 1.58 waitForFunction treats an async predicate's Promise as
  // truthy rather than polling its resolved value. Check completed reads here.
  do {
    const body=await disk(page);
    if(body===expected)return body;
    await page.waitForTimeout(200);
  }while(Date.now()<deadline);
  assert.fail('The automatic file copy did not match the committed full export.');
}
try {
 for(const section of ['backup','guide','data']){
  const loading=await context();
  await loading.addInitScript(()=>{window.showSaveFilePicker=undefined;});
  let releaseModule,requestedModule;
  const moduleRequested=new Promise(resolve=>{requestedModule=resolve;});
  const moduleGate=new Promise(resolve=>{releaseModule=resolve;});
  await loading.route(base+'app.mjs*',async route=>{
    const response=await route.fetch();let source=await response.text();
    const marker='cpuStartup = await prepareDefaultCPU(';
    assert.equal(source.split(marker).length,2);
    source=source.replace(marker,"if(!sessionStorage.getItem('__moduleCPUReleased')){window.__moduleCPUHeld=true;await new Promise(resolve=>{window.__releaseModuleCPU=()=>{sessionStorage.setItem('__moduleCPUReleased','true');resolve();};});} "+marker);
    requestedModule();await moduleGate;
    return route.fulfill({response,body:source});
  });
  const loadingPage=await newPage(loading);
  try{
    await loadingPage.goto(base+'#/setup',{waitUntil:'commit'});
    await moduleRequested;
    for(const id of ['open-backup','open-guide','open-data-settings']){
      const button=loadingPage.locator('#'+id);await button.waitFor();
      assert.equal(await button.isDisabled(),true,'Visible startup controls must not accept clicks before their handlers bind');
      assert.equal(await button.evaluate(node=>node.onclick),null);
    }
    assert.equal(await loadingPage.evaluate(()=>window.__workers),0);
    // A normal action waits for the control to become usable; no forced click.
    const click=loadingPage.locator(section==='backup'?'#open-backup':section==='guide'?'#open-guide':'#open-data-settings').click();
    releaseModule();
    await loadingPage.waitForFunction(()=>window.__moduleCPUHeld===true);
    await click;
    for(const id of ['open-backup','open-guide','open-data-settings'])assert.equal(await loadingPage.locator('#'+id).isEnabled(),true);
    assert.equal(await loadingPage.evaluate(()=>window.__workers),0,'Header controls must work before Python starts');
    assert.equal(await loadingPage.locator('#status').textContent(),section==='backup'?'Opening backup…':section==='guide'?'Opening guide…':'Opening device settings…');
    await loadingPage.evaluate(()=>window.__releaseModuleCPU());
    if(section==='guide')await loadingPage.locator('#quick-guide').waitFor();
    else{
      await loadingPage.locator('#data-settings').waitFor();
      assert.equal(await loadingPage.locator('#browser-data-menu').getAttribute('open'),section==='backup'?'':null);
      if(section==='backup')await loadingPage.locator('#linked-backup-status').filter({hasText:'Download backup and Restore backup'}).waitFor();
    }
    report.checks.push(`Module-loading ${section} control waits for handler binding, accepts intent before Python and survives CPU preparation`);
  }finally{releaseModule();await loading.close();}
 }
 for(const {reload,last} of [{reload:false,last:'data'},{reload:true,last:'backup'},{reload:true,last:'guide'}]){
  const early=await context();
  await early.addInitScript(()=>{window.showSaveFilePicker=undefined;});
  await early.route(base+'__early-seed__',route=>route.fulfill({contentType:'text/html',body:'<title>Isolated early-action fixture</title>'}));
  await early.route(base+'app.mjs*',async route=>{
    const response=await route.fetch();let source=await response.text();
    const marker='cpuStartup = await prepareDefaultCPU(';
    assert.equal(source.split(marker).length,2,'The CPU-preparation gate must instrument exactly one production call');
    source=source.replace(marker,"if(!sessionStorage.getItem('__guideForcedReload')){window.__guideCPUHeld=true; await new Promise(resolve=>{window.__guideReleaseCPU=resolve;});} "+marker);
    if(reload){
      const boundary='isolationRequested = cpuStartup.isolate;';
      assert.equal(source.split(boundary).length,2);
      source=source.replace(boundary,"if(!sessionStorage.getItem('__guideForcedReload')){sessionStorage.setItem('__guideForcedReload','true');cpuStartup.reload=true;} "+boundary);
    }
    return route.fulfill({response,body:source});
  });
  const earlyPage=await newPage(early);
  try {
    await earlyPage.goto(base+'__early-seed__');
    await earlyPage.evaluate(async()=>{
      const request=indexedDB.open('dailydigest:/daily_digests/',1);
      request.onupgradeneeded=()=>request.result.createObjectStore('state');
      const database=await new Promise(resolve=>{request.onsuccess=()=>resolve(request.result);});
      await new Promise(resolve=>{const transaction=database.transaction('state','readwrite');transaction.objectStore('state').put('cpu','compute');transaction.oncomplete=resolve;});database.close();
    });
    await earlyPage.goto(base+'#/setup');
    await earlyPage.waitForFunction(()=>window.__guideCPUHeld===true);
    assert.equal(await earlyPage.evaluate(()=>window.__workers),0);
    await earlyPage.locator('#open-backup').click();
    assert.equal(await earlyPage.locator('#status').textContent(),'Opening backup…','Visible Backup must acknowledge its action while CPU preparation is pending');
    assert.equal(await earlyPage.locator('#data-settings').evaluate(node=>node.open),false,'Controls must be bound before exposing the dialog');
    await earlyPage.locator('#open-guide').click();
    assert.equal(await earlyPage.locator('#status').textContent(),'Opening guide…');
    assert.equal(await earlyPage.locator('#quick-guide').isVisible(),false);
    await earlyPage.locator('#open-data-settings').click();
    assert.equal(await earlyPage.locator('#status').textContent(),'Opening device settings…');
    if(last!=='data')await earlyPage.locator(last==='guide'?'#open-guide':'#open-backup').click();
    await earlyPage.evaluate(()=>window.__guideReleaseCPU());
    if(last==='guide'){
      await earlyPage.locator('#quick-guide').waitFor();
      assert.equal(await earlyPage.locator('#data-settings').evaluate(node=>node.open),false);
      await earlyPage.locator('#close-guide').click();
      await earlyPage.locator('#open-backup').click();
    }else{
      await earlyPage.locator('#data-settings').waitFor();
      assert.equal(await earlyPage.locator('#data-settings').evaluate(node=>node.open),true);
      assert.equal(await earlyPage.locator('#browser-data-menu').getAttribute('open'),last==='backup'?'':null,'The latest queued action must win');
    }
    assert.equal(await earlyPage.evaluate(()=>sessionStorage.getItem('dailydigest:startup-data-intent:'+location.pathname)),null);
    assert.equal(await earlyPage.locator('#import').evaluate(node=>typeof node.onchange),'function');
    if(last==='data'){
      await earlyPage.locator('#close-data-settings').click();
      await earlyPage.locator('#open-backup').click();
    }
    await earlyPage.locator('#linked-backup-status').filter({hasText:'Download backup and Restore backup'}).waitFor();
    await earlyPage.locator('#close-data-settings').click();
    await earlyPage.locator('#open-data-settings').click();
    await earlyPage.locator('#compute-menu > summary').click();
    assert.equal(await earlyPage.locator('#compute-menu').getAttribute('open'),'');
    await earlyPage.locator('#close-data-settings').click();
    if(reload){
      assert.equal(await earlyPage.evaluate(()=>sessionStorage.getItem('__guideForcedReload')),'true');
      await earlyPage.reload();
      await earlyPage.waitForFunction(()=>document.getElementById('linked-backup-status').textContent.includes('Download backup and Restore backup'));
      assert.equal(await earlyPage.locator('#data-settings').evaluate(node=>node.open),false,'Consumed startup intent must not reopen Backup on a later reload');
      assert.equal(await earlyPage.locator('#quick-guide').isVisible(),false,'Consumed Guide intent must not replay on a later reload');
    }
    report.checks.push(`Early Backup/Guide/Data clicks acknowledge immediately; latest ${last} opens fully bound ${reload?'across the automatic CPU reload exactly once':'after held CPU preparation without Python readiness'}`);
  } finally { await early.close(); }
 }
  const owner=await context(),page=await newPage(owner);
  await page.goto(base+'#/setup');
  await page.locator('#welcome-guide').waitFor();
  await page.locator('#welcome-guide-open').click();
  assert.equal(await page.locator('#quick-guide').getAttribute('aria-labelledby'),'guide-title-0');
  await page.locator('#guide-next').click();await page.locator('#guide-next').click();
  await page.getByRole('heading',{name:'Keep a copy you can restore'}).waitFor();
  assert.match(await page.locator('#quick-guide').innerText(),/does not merge|not merge/);
  await page.locator('[data-guide-action="backup"]').click();
  assert.equal(await page.locator('#browser-data-menu').getAttribute('open'),'');
  await page.locator('#close-data-settings').click();
  await page.reload();await ready(page);
  assert.equal(await page.locator('#welcome-guide').isHidden(),true);
  await page.frameLocator('#app').locator('[data-browser-action="open-guide"]').click();
  await page.locator('#quick-guide').waitFor();await page.locator('#close-guide').click();
  report.checks.push('first visit offers guide and restore; guide explains storage and reopens from Setup after dismissal');

  const original=await page.evaluate(()=>window.__rpc({type:'export',includeCredentials:true}));
  assert.equal(original.status,200);
  await page.evaluate(async body=>{
    const db=await new Promise(resolve=>{const request=indexedDB.open('dailydigest:/daily_digests/',1);request.onsuccess=()=>resolve(request.result);});
    await new Promise((resolve,reject)=>{const tx=db.transaction('state','readwrite');for(let i=0;i<4;i++)tx.objectStore('state').put(body,`recovery:${9999999999996+i}:fixture`);tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);});db.close();
  },original.body);
  await page.locator('#import').setInputFiles({name:'fixture.json',mimeType:'application/json',buffer:Buffer.from(fixture)});
  const imported=page.waitForEvent('load');await page.locator('#restore-proceed').click();await imported;await ready(page);
  const recovery=await page.evaluate(async()=>JSON.parse((await window.__rpc({type:'recovery-status'})).body));
  assert.equal(recovery.quarantined.length,3);
  const retained=await page.evaluate(key=>window.__rpc({type:'recovery-export',key}),recovery.quarantined.at(-1));
  assert.equal(retained.body,original.body);
  report.checks.push('normal restore atomically retains the exact original data and prunes recovery copies to three');
  await page.evaluate(()=>location.hash='#/');
  const card=page.frameLocator('#app').locator('.item[data-section="research"]').first();
  await card.waitFor();
  await page.locator('#open-backup').click();await page.locator('#link-backup').click();
  await page.locator('#linked-backup-status').filter({hasText:'last copy saved'}).waitFor();
  assert.equal(JSON.parse(await disk(page)).credentialsIncluded,false);
  assert.equal(await page.locator('#automatic-backup').isChecked(),false);
  await page.locator('#automatic-backup').check();
  await page.locator('#import').setInputFiles({name:'invalid.json',mimeType:'application/json',buffer:Buffer.from('{}')});
  await page.locator('#restore-proceed').click();
  await page.locator('#app-error').waitFor();await page.locator('#dismiss-error').click();
  assert.equal(await page.locator('#automatic-backup').isChecked(),true);
  report.checks.push('a rejected backup preserves the automatic-copy preference and the working desk');
  // Let the requested initial copy finish, then test a real feedback mutation.
  const before=await disk(page);
  await writeFile(resolve(output,'before-vote-fixture.json'),before);
  await page.locator('#close-data-settings').click();
  await card.locator('.vote-btn[data-grade="70"]').click();
  await card.locator('.vote-btn[data-grade="70"][aria-pressed="true"]').waitFor();
  await page.frameLocator('#app').locator('body').evaluate(()=>window.ddFlushRatings?.());
  const afterVote=await page.evaluate(()=>window.__rpc({type:'export',includeCredentials:false}));
  assert.equal(afterVote.status,200,afterVote.body);
  await writeFile(resolve(output,'after-vote-fixture.json'),afterVote.body);
  const copied=await waitForCopy(page,afterVote.body);assert.equal(JSON.parse(copied).credentialsIncluded,false);
  await writeFile(resolve(output,'copied-fixture.json'),copied);
  report.checks.push('real fixture restores; explicit file choice saves a credential-free copy; opt-in background copy captures a real rating');

  // Normal startup checks file metadata without waking Python on a saved view.
  const readingDeadline=Date.now()+30000;
  let readingReady=false;
  while(!readingReady && Date.now()<readingDeadline){
    readingReady=await page.evaluate(async()=>{
      const request=indexedDB.open('dailydigest:/daily_digests/',1);
      const db=await new Promise((done,fail)=>{request.onsuccess=()=>done(request.result);request.onerror=()=>fail(request.error);});
      try {
        const copy=await new Promise((done,fail)=>{const get=db.transaction('state').objectStore('state').get('reading-view');get.onsuccess=()=>done(get.result);get.onerror=()=>fail(get.error);});
        return copy?.version===5 && Array.isArray(copy.items) && copy.items.length>0;
      } finally { db.close(); }
    });
    if(!readingReady)await page.waitForTimeout(200);
  }
  assert.equal(readingReady,true,'A saved reading view must be committed before reloading.');
  await page.reload();
  await page.locator('#reading-view-notice').waitFor();
  await page.locator('#open-backup').click();
  await page.locator('#linked-backup-status').filter({hasText:'last copy saved'}).waitFor();
  assert.equal(await page.evaluate(()=>window.__workers),0);
  assert.equal(await page.locator('#automatic-backup').isChecked(),true);
  report.checks.push('file handle and opt-in survive reload; checking the file does not start Python from a saved reading view');
  await page.locator('#close-data-settings').click();
  await page.locator('#reading-view-open').click();await ready(page);
  await page.locator('#open-backup').click();
  const external=JSON.stringify({...JSON.parse(copied),changedElsewhere:true});
  await page.evaluate(async body=>{
    const handle=await (await navigator.storage.getDirectory()).getFileHandle('guide-test-backup.json');
    const writer=await handle.createWritable({mode:'exclusive'});await writer.write(body);await writer.close();
  },external);
  await page.locator('#write-linked-backup').click();
  await page.locator('#linked-backup-status').filter({hasText:'changed or belongs'}).waitFor();
  assert.equal(await disk(page),external);
  report.checks.push('external file edits stop a stale desk from replacing the backup');

  const second=await context(),other=await newPage(second);
  await other.goto(base+'#/setup');await ready(other);
  assert.equal(await other.frameLocator('#app').locator('#topics').inputValue(),'');
  await other.evaluate(async body=>{
    const handle=await (await navigator.storage.getDirectory()).getFileHandle('guide-test-backup.json',{create:true});
    const writer=await handle.createWritable();await writer.write(body);await writer.close();
  },copied);
  await other.locator('#open-backup').click();await other.locator('#link-backup').click();
  await other.locator('#linked-backup-status').filter({hasText:'backup found'}).waitFor();
  assert.equal(await disk(other),copied,'Linking existing data must not overwrite it');
  assert.equal(await other.locator('#automatic-backup-label').isHidden(),true);
  await other.locator('#restore-linked-backup').click();const restored=other.waitForEvent('load');await other.locator('#restore-proceed').click();await restored;await ready(other);
  await other.evaluate(()=>location.hash='#/');
  await other.frameLocator('#app').locator('.item[data-section="research"]').first().waitFor();
  await writeFile(resolve(output,'restored.html'),await other.frameLocator('#app').locator('body').innerHTML());
  await other.frameLocator('#app').locator('.vote-btn[data-grade="70"][aria-pressed="true"]').first().waitFor({state:'attached'});
  report.checks.push('a separate profile starts empty and can explicitly link and restore the copy, preserving the rating');

  await other.setViewportSize({width:390,height:844});
  await other.locator('#open-guide').click();
  for(let index=0;index<4;index++){
    assert.equal(await other.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    assert.equal(await other.locator('#quick-guide').evaluate(node=>node.scrollWidth<=node.clientWidth),true);
    if(index<3)await other.locator('#guide-next').click();
  }
  await other.screenshot({path:resolve(output,'guide-phone.png')});
  await other.locator('#guide-finish').click();
  assert.equal(await other.locator('#quick-guide').isVisible(),false);
  report.checks.push('all guide steps fit a 390 px viewport and can be dismissed');

  const unsupported=await context();await unsupported.addInitScript(()=>{window.showSaveFilePicker=undefined;});
  const fallback=await newPage(unsupported);await fallback.goto(base+'#/setup');
  await fallback.locator('#open-backup').click();
  await fallback.locator('#linked-backup-status').filter({hasText:'Download backup and Restore backup'}).waitFor();
  assert.equal(await fallback.locator('#link-backup').isHidden(),true);
  assert.equal(await fallback.locator('#import').isEnabled(),true);
  report.checks.push('browsers without the file picker retain clear download and restore actions');
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.blocked,[]);report.passed=true;
}catch(error){report.passed=false;report.failure=error.stack;throw error;}
finally{
  await writeFile(resolve(output,'report.json'),JSON.stringify(report,null,2)+'\n');
  await browser.close();await new Promise(done=>server.close(done));
}
console.log(JSON.stringify(report));
