// Visual/lifecycle checks for a bounded fictional cached desk. No runtime or providers.
import assert from 'node:assert/strict';
import {mkdir,writeFile,mkdtemp} from 'node:fs/promises';
import {resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';
const base=process.env.DAILYDIGEST_BROWSER_URL||'http://127.0.0.1:4413/';
const output=resolve(process.env.DAILYDIGEST_CACHED_SHELL_REPORT||'.full-review/revision12/cached-shell');
await mkdir(output,{recursive:true});
const scratch=await mkdtemp(resolve(process.env.TMPDIR||tmpdir(),'dd-cached-shell-'));
const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox'],env:{...process.env,TMPDIR:scratch}});
const report={checks:[],errors:[],blocked:[],console:[]};
try{
  const context=await browser.newContext({viewport:{width:1500,height:900},reducedMotion:'reduce'});
  await context.route('**/*',route=>{if(route.request().url().startsWith(base))return route.continue();report.blocked.push(route.request().url());return route.abort();});
  await context.route(base+'__cached_seed__',route=>route.fulfill({contentType:'text/html',body:'<title>Fictional seed</title>'}));
  const page=await context.newPage();page.on('console',message=>{if(message.type()==='error')report.console.push(message.text());});page.on('pageerror',error=>report.errors.push(error.message));
  await page.addInitScript(()=>{window.__workerStarts=0;window.Worker=class {constructor(){window.__workerStarts++;throw Error('Cached entry must never start Python');}};});
  await page.goto(base+'__cached_seed__');
  await page.evaluate(async(basePath)=>{
    const view={version:4,date:'2026-10-22',savedAt:'2026-10-22T12:00:00Z',profileName:'Ada',notes:['Tiny fact — This is a fictional cached note.','Lab joke — A fictional harmless joke.'],items:[{section:'Research',title:'Fictional engineering result',url:'https://example.org/paper',meta:'Example journal',summary:'A small simulated result.',reason:'Text contains: engineering',opportunity:'',uncertainty:'Title only — topic fit needs checking.',details:'Caveat: Only simulated.\n\nAbstract: Not supplied by this source.'}]};
    const request=indexedDB.open('dailydigest:'+basePath,1);request.onupgradeneeded=()=>request.result.createObjectStore('state');
    const db=await new Promise(done=>request.onsuccess=()=>done(request.result));
    await new Promise(done=>{const tx=db.transaction('state','readwrite');tx.objectStore('state').put('{"version":1,"files":{}}','snapshot');tx.objectStore('state').put(view,'reading-view');tx.oncomplete=done;});db.close();
    localStorage.setItem('dailydigest-theme','dark');
  },new URL(base).pathname);
  await page.clock.setFixedTime(new Date('2026-10-22T20:00:00'));
  await page.goto(base+'#/');
  report.buildId=await page.locator('meta[name=dailydigest-build]').getAttribute('content');
  await page.locator('#cached-pip[data-state=ready]').waitFor({state:'attached'});
  await page.locator('#cached-pip .tea-break').waitFor();
  const frame=()=>page.frameLocator('#app');
  assert.equal(await frame().locator('#page-salutation').textContent(),'Good evening, Ada');
  assert.equal(await frame().locator('html').getAttribute('data-theme'),'dark');
  assert.equal(await frame().locator('link[rel=stylesheet]').count(),0,'Shared CSS must be embedded to survive opaque-frame COEP');
  await page.frames().find(frame=>frame.parentFrame()).waitForFunction(()=>getComputedStyle(document.body).backgroundImage.includes('linear-gradient'));
  assert.match(await frame().locator('body').evaluate(node=>getComputedStyle(node).backgroundImage),/linear-gradient/);
  assert.equal(await frame().locator('body').evaluate(node=>getComputedStyle(node).color),'rgb(237, 244, 239)');
  await page.evaluate(()=>window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true})));
  assert.equal(await page.locator('#cached-pip').count(),1,'BFCache suspension must retain the live component for restoration');
  await page.waitForFunction(()=>document.querySelector('#tea-break').getBoundingClientRect().bottom<=document.querySelector('#app').getBoundingClientRect().bottom);
  const dock=await page.locator('#tea-break').boundingBox(),articleFrame=await page.locator('#app').boundingBox();
  assert.ok(dock.y+dock.height<=articleFrame.y+articleFrame.height,'Default Pip must clear the shell warning and controls');
  await page.screenshot({path:output+'/desktop-dark.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.locator('#cached-pip').waitFor({state:'hidden'});
  assert.equal(await page.locator('#cached-pip').isVisible(),false);
  const actionRows=await page.locator('#reading-view-notice > div button').evaluateAll(buttons=>buttons.map(button=>button.getBoundingClientRect().top));
  assert.ok(Math.max(...actionRows)-Math.min(...actionRows)<1,'All cached mobile actions must fit one row at390px');
  report.mobileToolbarHeight=await page.locator('#reading-view-notice').evaluate(node=>node.getBoundingClientRect().height);
  await page.screenshot({path:output+'/mobile-dark.png',fullPage:true});
  const closedHeight=await page.locator('#app').evaluate(node=>node.getBoundingClientRect().height);
  await page.locator('#cached-pip-toggle').click();
  assert.equal(await page.locator('#tea-another').evaluate(node=>node===document.activeElement),true);
  await page.screenshot({path:output+'/mobile-dark-pip.png',fullPage:true});
  assert.equal(await page.locator('#cached-pip .tea-break').count(),1);
  assert.ok((await page.locator('#app').evaluate(node=>node.getBoundingClientRect().height))>300);
  await page.locator('#tea-another').click();
  await page.waitForFunction(()=>document.querySelector('#tea-leaf-count').textContent==='1');
  await page.locator('#cached-pip-toggle').click();
  assert.equal(await page.locator('#app').evaluate(node=>node.getBoundingClientRect().height),closedHeight);
  assert.equal(await page.evaluate(()=>window.__workerStarts),0);
  // A late markup response must never resurrect Pip after disposal.
  await page.route('**/pip.html*',async route=>{await new Promise(done=>setTimeout(done,100));await route.continue().catch(()=>{});});
  await page.evaluate(async()=>{
    const {mountCachedReadingPip}=await import(document.querySelector('script[type=module]').src);
    const mounted=mountCachedReadingPip(document.querySelector('#app'),{date:'2026-10-22',notes:['Late note']});
    mounted.dispose();mounted.dispose();await mounted.ready;
  });
  assert.equal(await page.locator('#cached-pip').count(),1);
  assert.equal(await page.locator('#cached-pip .tea-break').count(),1);
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.blocked,[]);
  report.checks.push('dark current-time named greeting and original shared gradient render with zero workers',
    'mobile Pip stays collapsed until requested; closing returns all article space',
    'real Pip leaf action works in trusted parent with article scripts disabled; default desktop dock clears the shell footer',
    'simulated persisted pagehide retains Pip; disclosure focuses its leaf for keyboard access',
    'disposal before delayed markup completion is idempotent and cannot resurrect a duplicate component');
  report.success=true;
}catch(error){report.success=false;report.error=error.stack;for(const context of browser.contexts())for(const page of context.pages()){await page.screenshot({path:output+'/failure.png',fullPage:true});report.frames=await Promise.all(page.frames().map(frame=>frame.evaluate(()=>({url:location.href,html:document.head.innerHTML,bodyStyle:getComputedStyle(document.body).cssText,styles:[...document.styleSheets].map(sheet=>({href:sheet.href,disabled:sheet.disabled}))})).catch(()=>null)));}throw error;}
finally{await writeFile(output+'/report.json',JSON.stringify(report,null,2)+'\n');await browser.close();}
console.log(JSON.stringify(report,null,2));
