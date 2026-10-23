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
        return copy?.version===4 && Array.isArray(copy.items) && copy.items.length>0;
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
