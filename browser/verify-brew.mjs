// Real CPU/Pyodide brew regression matrix. Uses an isolated fictional desk and
// intercepted feeds; never contacts live sources/providers or the user's data.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {pathToFileURL} from 'node:url';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';
const extension=process.env.DAILYDIGEST_BREW_EXTENSION
 ? await import(pathToFileURL(resolve(process.env.DAILYDIGEST_BREW_EXTENSION)).href+'?brew-extension') : null;
const site=resolve(process.env.DAILYDIGEST_BREW_SITE || 'browser/dist');
const out=resolve(process.env.DAILYDIGEST_BREW_REPORT || '.full-review/brew-matrix');
const fixtures=resolve(process.env.DAILYDIGEST_BREW_FIXTURES || '.full-review/tool-assessment-2026-10-16/fixtures');
const computeMode=process.env.DAILYDIGEST_BREW_COMPUTE || 'cpu';
assert.ok(['cpu','multicore'].includes(computeMode));
const snapshot=await readFile(resolve(fixtures,'fixture-backup.json'),'utf8');
const originalFeed=await readFile(resolve(fixtures,'research-feed.xml'),'utf8');
const feed=extension?.prepareFeed ? extension.prepareFeed(originalFeed) : originalFeed;
const fixtureDate = new Date(Math.max(...[...originalFeed.matchAll(/<pubDate>(.*?)<\/pubDate>/g)].map(match=>Date.parse(match[1]))) + 5*86400000).toISOString();
const fixtureClockOffset=Date.parse(fixtureDate)-Date.now();
const entries=feed.match(/<item>[\s\S]*?<\/item>/g);assert.ok(entries?.length);
const expanded=feed.replace('</channel>',Array.from({length:3},(_,copy)=>entries.filter(entry=>!extension?.omitLoadCopy?.(entry)).map((entry,index)=>entry
 .replace(/<guid>.*?<\/guid>/,'<guid>load-'+copy+'-'+index+'</guid>')
 .replace(/<link>.*?<\/link>/,'<link>https://acceptance.invalid/load/'+copy+'/'+index+'</link>')
 .replace('</title>',` — synthetic load case ${copy}-${index}</title>`)
 .replace('</description>',` Synthetic load fixture ${copy}-${index}, not a real publication.</description>`)).join('')).join('')+'</channel>');
await mkdir(out,{recursive:true});
const report={scope:'Actual browser CPU/Pyodide, fictional desk and intercepted feeds. Local unthrottled assets; no production success-rate claim.',computeMode,fixtureDate,checks:[],errors:[],blocked:[]};
let sourceMode='normal',slowStarted;
const types={'.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.json':'application/json','.wasm':'application/wasm','.css':'text/css','.svg':'image/svg+xml'};
const server=createServer(async(req,res)=>{try{if(computeMode==='multicore'){res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');}let path=new URL(req.url,'http://local').pathname;if(path.endsWith('__compute__')){res.end(computeMode);return;}if(path.endsWith('__seed__')){res.end('<title>Seed</title>');return;}if(path.endsWith('__fixture__')){res.setHeader('Content-Type','application/json');res.end(snapshot);return;}assert.ok(path.startsWith('/daily_digests/'));if(path.endsWith('/'))path+='index.html';const file=resolve(site,'.'+path.slice('/daily_digests'.length));assert.ok(file.startsWith(site+sep));res.setHeader('Content-Type',types[extname(file)]||'application/octet-stream');res.end(await readFile(file));}catch{res.writeHead(404).end();}});
await new Promise(done=>server.listen(0,'127.0.0.1',done));const base=`http://127.0.0.1:${server.address().port}/daily_digests/`;
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--enable-experimental-webassembly-jspi']});
try{
 await extension?.beforeDesk?.({browser,base,report,out});
 const context=await browser.newContext({timezoneId:'America/Phoenix',serviceWorkers:extension?.allowServiceWorkers?'allow':'block'});
 await context.route('**/*',async route=>{const url=route.request().url();if(url.startsWith(base)){if(new URL(url).pathname.endsWith('/worker.mjs')){const response=await route.fetch();const clock=`const TestDate=Date;let clockOffset=${fixtureClockOffset};globalThis.Date=class extends TestDate{constructor(...args){super(...(args.length?args:[TestDate.now()+clockOffset]));}static now(){return TestDate.now()+clockOffset;}};\n`;let source=await response.text();if(extension?.instrumentWorker)source=extension.instrumentWorker(source);return route.fulfill({response,body:clock+source});}return route.continue();}if(url.startsWith('https://acceptance.invalid/')){
  if(sourceMode==='failure')return route.fulfill({status:503,body:'Fictional source unavailable',headers:{'access-control-allow-origin':'*'}});
  if(sourceMode==='slow'){slowStarted?.();await new Promise(resolve=>setTimeout(resolve,8000));}
  let body=sourceMode==='large'?expanded:feed;if(extension?.feedForRequest)body=extension.feedForRequest(body);
  return route.fulfill({contentType:'application/rss+xml',body,headers:{'access-control-allow-origin':'*'}}).catch(()=>{});
 }if(await extension?.routeExternal?.(route,report))return;report.blocked.push(url);return route.abort();});
 const page=await context.newPage();await page.clock.install({time:new Date(fixtureDate)});page.setDefaultTimeout(30000);page.on('pageerror',e=>report.errors.push(e.message));
 await page.addInitScript(()=>{
  window.__events=[];let id=900000;const Native=Worker;
  window.Worker=class extends Native{constructor(...args){super(...args);window.__worker=this;this.addEventListener('message',({data})=>{window.__events.push({at:performance.now(),...data});if(data.type==='ready')window.__ready=true;});}};
  window.__rpc=payload=>new Promise((resolve,reject)=>{const request=++id;const timeout=setTimeout(()=>{window.__worker.removeEventListener('message',listener);reject(Error('RPC timeout'));},60000);function listener({data}){if(data.id!==request||data.type!=='response')return;clearTimeout(timeout);window.__worker.removeEventListener('message',listener);resolve(data);}window.__worker.addEventListener('message',listener);window.__worker.postMessage({...payload,id:request});});
 });
 await page.goto(base+'__seed__');await page.evaluate(async()=>{const mode=await(await fetch('./__compute__')).text();const value=await(await fetch('./__fixture__')).text(),r=indexedDB.open('dailydigest:/daily_digests/',1);r.onupgradeneeded=()=>r.result.createObjectStore('state');const db=await new Promise(done=>r.onsuccess=()=>done(r.result));await new Promise(done=>{const tx=db.transaction('state','readwrite');tx.objectStore('state').put(value,'snapshot');tx.objectStore('state').put(mode,'compute');tx.oncomplete=done;});db.close();});
 const startupStarted=performance.now();await page.goto(base+'#/run');await page.waitForFunction(()=>window.__ready,undefined,{timeout:120000});report.runtimeReadyMs=performance.now()-startupStarted;report.browser=await page.evaluate(()=>({hardwareConcurrency:navigator.hardwareConcurrency,deviceMemory:navigator.deviceMemory,userAgent:navigator.userAgent}));const frame=()=>page.frameLocator('#app');
 async function route(path,selector){await page.waitForFunction(()=>typeof document.querySelector('#app')?.contentWindow?.ddNavigate==='function');await page.evaluate(path=>{window.__priorDocument=document.querySelector('#app').contentDocument;document.querySelector('#app').contentWindow.ddNavigate(path)},path);await page.waitForFunction(path=>document.querySelector('#app')?.contentDocument!==window.__priorDocument&&document.querySelector('#app')?.contentWindow?.ddLocation?.endsWith(path),path);await frame().locator(selector).first().waitFor();}
 const titles=()=>frame().locator('.item[data-section="research"] .title-link').allTextContents();
 const stats=async()=>{const r=await page.evaluate(()=>window.__rpc({type:'compute-status'}));assert.equal(r.status,200);return JSON.parse(r.body).stats;};
 async function record(name,work){const started=performance.now();try{const details=await work();report.checks.push({name,passed:true,elapsedMs:performance.now()-started,...details});console.log(JSON.stringify(report.checks.at(-1)));}catch(error){report.checks.push({name,passed:false,error:error.stack,state:await page.evaluate(()=>({hash:location.hash,text:document.querySelector('#app')?.contentDocument?.body?.innerText})).catch(()=>null)});await page.screenshot({path:out+'/failure.png',fullPage:true}).catch(()=>{});await writeFile(out+'/failure-events.json',JSON.stringify(await page.evaluate(()=>window.__events),null,2));throw error;}finally{await writeFile(out+'/report.json',JSON.stringify(report,null,2));}}
 async function brew(mode,{allowEmpty=false}={}){await route('/run?reading_mode='+mode,'#brew-button');await frame().locator('#mode-'+mode).check();await page.evaluate(()=>window.__events=[]);const before=await stats(),started=performance.now();await frame().locator('#brew-button').click();await page.waitForFunction(()=>location.hash.startsWith('#/done')||document.querySelector('#app')?.contentDocument?.querySelector('#err:not([hidden])'),undefined,{timeout:300000});if(!(await page.evaluate(()=>location.hash)).startsWith('#/done'))throw Error(await frame().locator('#err').textContent());const elapsedMs=performance.now()-started;await frame().locator('a.cta').click();await frame().locator(allowEmpty?'.digest-date':'.item[data-item-id]').first().waitFor();const after=await stats(),events=await page.evaluate(()=>window.__events);await writeFile(out+'/events-'+report.checks.length+'.json',JSON.stringify(events,null,2));return{elapsedMs,titles:await titles(),encoded:after.completedInputs-before.completedInputs,reused:(after.reusedInputs||0)-(before.reusedInputs||0)};}
 const extensionContext={page,frame,route,record,brew,titles,out,report,fixtureDate,setSourceMode:value=>{sourceMode=value;}};
 await extension?.beforeMatrix?.(extensionContext);
 let standard,full,edited,edition;
 await record('standard-first',async()=>{standard=await brew('usual');assert.ok(standard.titles.length>0);return standard;});
 await record('standard-repeat',async()=>{const r=await brew('usual');assert.deepEqual(r.titles,standard.titles);assert.equal(r.encoded,0);return r;});
 await record('short',async()=>{const r=await brew('minimal');assert.ok(r.titles.length>0&&r.titles.length<=5);assert.ok(r.titles.every(t=>standard.titles.includes(t)));return r;});
 await record('full',async()=>{full=await brew('full');edition=await page.evaluate(()=>location.hash);assert.ok(full.titles.length>=standard.titles.length);return full;});
 await record('changed-research-context',async()=>{await route('/setup','#interest-editor');await frame().locator('#profile-options>summary').click();const bio=frame().locator('#bio');const current=await bio.inputValue();await bio.fill(current+' Prioritize programmable DNA origami hinges and mechanical nanodevices.');await frame().locator('button[type=submit]').click();await frame().locator('#brew-button').waitFor();edited=await brew('full');assert.equal(await page.evaluate(()=>location.hash),edition,'saving research context must preserve the edition date');assert.ok(edited.encoded>0,'new context must be encoded');return edited;});
 await record('changed-context-repeat',async()=>{const r=await brew('full');assert.deepEqual(r.titles,edited.titles);assert.equal(r.encoded,0);return r;});
 await record('reload-and-brew',async()=>{await page.evaluate(()=>location.hash='#/run');await page.reload();await page.waitForFunction(()=>window.__ready,undefined,{timeout:120000});const r=await brew('full');assert.deepEqual(r.titles,edited.titles);assert.ok(r.encoded<entries.length,'stored article vectors survive reload');return r;});
 async function preserve(before){await route('/','.item[data-item-id]');assert.deepEqual(await titles(),before);}
 await record('source-failure-preserves-digest',async()=>{const before=await titles();sourceMode='failure';await route('/run?reading_mode=usual','#brew-button');await frame().locator('#brew-button').click();await frame().locator('#err').waitFor({state:'visible',timeout:60000});const error=await frame().locator('#err').textContent();assert.match(error,/source|coverage|research/i);await preserve(before);sourceMode='normal';return{preserved:true,error};});
 await record('retry-after-source-failure',async()=>{const r=await brew('full');assert.deepEqual(r.titles,edited.titles);return r;});
 await record('stop-during-fetch',async()=>{const before=await titles();sourceMode='slow';let timer;const requested=new Promise((resolve,reject)=>{slowStarted=resolve;timer=setTimeout(()=>reject(Error('Source not requested')),30000)});await route('/run?reading_mode=usual','#brew-button');await frame().locator('#brew-button').click();await requested;clearTimeout(timer);const started=performance.now();await frame().locator('#cancel-button').click();await frame().locator('#err').waitFor({state:'visible',timeout:60000});const stopMs=performance.now()-started;assert.match(await frame().locator('#err').textContent(),/stop|cancel/i);sourceMode='normal';slowStarted=null;await preserve(before);return{stopMs,preserved:true};});
 await record('stop-during-new-document-inference',async()=>{const before=await titles();sourceMode='large';await route('/run?reading_mode=full','#brew-button');await page.evaluate(()=>window.__events=[]);await frame().locator('#brew-button').click();await page.waitForFunction(()=>window.__events.some(e=>e.compute?.phase==='encoding'&&e.compute.total>100&&e.compute.completed>=4),undefined,{timeout:120000});const started=performance.now();await frame().locator('#cancel-button').click();await frame().locator('#err').waitFor({state:'visible',timeout:60000});const stopMs=performance.now()-started;assert.match(await frame().locator('#err').textContent(),/stop|cancel/i);await preserve(before);return{stopMs,preserved:true};});
 let large;
 await record('larger-pool-retry',async()=>{large=await brew('full');assert.ok(large.encoded>100);return{...large,feedItems:(extension?.feedForRequest?extension.feedForRequest(expanded):expanded).match(/<item>/g).length};});
 await record('larger-pool-repeat',async()=>{const r=await brew('full');assert.deepEqual(r.titles,large.titles);assert.ok(r.encoded<20);return r;});
 await extension?.afterMatrix?.(extensionContext);
 assert.deepEqual(report.errors,[]);assert.deepEqual(report.blocked,[]);report.success=true;
}catch(error){report.failure=error.stack;throw error;}finally{await writeFile(out+'/report.json',JSON.stringify(report,null,2));await browser.close();await new Promise(done=>server.close(done));}
