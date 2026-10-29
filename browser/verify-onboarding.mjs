// Source-shell verification: no model, feed, provider, or personal data access.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir, mkdtemp, writeFile} from 'node:fs/promises';
import {resolve, extname, sep} from 'node:path';
import {tmpdir} from 'node:os';
import {chromium, firefox} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';
const root=resolve('browser'), output=resolve('.full-review/onboarding-2026-10-28');
await mkdir(output,{recursive:true});
const scratch=await mkdtemp(resolve(process.env.TMPDIR||tmpdir(),'dd-onboarding-'));
const server=createServer(async(req,res)=>{
  try {
    let path=new URL(req.url,'http://localhost').pathname;
    if(path==='/')path='/index.html';
    const file=resolve(root,'.'+path);assert.ok(file.startsWith(root+sep));
    const data=await readFile(path==='/design-tokens.css'?resolve('templates/_design_tokens.css.j2'):file);
    res.setHeader('Content-Type',({'.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.css':'text/css'})[extname(file)]||'text/plain');res.end(data);
  }catch{res.writeHead(404).end();}
});
await new Promise(done=>server.listen(0,'127.0.0.1',done));
const base=`http://127.0.0.1:${server.address().port}/`;
const report={checks:[],engines:{},physicalDevices:false};
try {
 for(const [name,engine,executablePath] of [['chromium',chromium,'/usr/bin/google-chrome'],['firefox',firefox,'/usr/bin/firefox']]) {
  let browser;
  try {browser=await engine.launch({executablePath,headless:true,timeout:10000,args:name==='chromium'?['--no-sandbox']:[],env:{...process.env,TMPDIR:scratch}});}
  catch(error){report.engines[name]={unverified:error.message};continue;}
  try {
   const context=await browser.newContext();
   await context.route('**/*',route=>route.request().url().startsWith(base)?route.continue():route.abort());
   // Keep runtime intentionally idle while exercising real shell controls.
   await context.addInitScript(()=>{window.__workerPosts=[];window.Worker=class {postMessage(message){window.__workerPosts.push(message.type);} addEventListener(){} terminate(){}};});
   const page=await context.newPage();
   for(const width of [1280,390]) {
    await page.setViewportSize({width,height:900});
    for(const file of ['example.html','companion.html']) {
     await page.goto(base+file);
     assert.equal(await page.locator('script').count(),0);
     assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
     await page.screenshot({path:`${output}/${name}-${width}-${file}.png`,fullPage:true});
    }
    await page.goto(base);
    const unsupported=await page.evaluate(()=>!WebAssembly.Suspending||!WebAssembly.promising||!navigator.locks);
    if(unsupported){await page.getByText(/Local brewing is unavailable/).first().waitFor();report.checks.push(`${name}: honest unsupported-browser fallback`);continue;}
    // Default CPU preparation may reload once. Wait for this origin’s setup
    // command, which is initialized only after that preparation finishes.
    await page.waitForFunction(()=>document.querySelector('.companion-command').textContent.includes(location.origin));
    await page.locator('#open-more').click();await page.locator('#open-data-settings').click();
    await page.locator('#companion-menu > summary').click();
    assert.doesNotMatch(await page.locator('.companion-command').innerText(),/--schedule/);
    const messagesBefore=await page.evaluate(()=>window.__workerPosts.length);
    await page.locator('#companion-schedule-command').check();
    assert.match(await page.locator('.companion-command').innerText(),/--state-dir .\/data\/companion-state --schedule/);
    await page.locator('#companion-schedule-command').uncheck();
    assert.doesNotMatch(await page.locator('.companion-command').innerText(),/--schedule/);
    assert.equal(await page.evaluate(()=>window.__workerPosts.length),messagesBefore,'Preparing a command must not start work or send state');
    assert.ok(await page.locator('#data-settings').evaluate(el=>el.scrollWidth<=el.clientWidth));
    await page.screenshot({path:`${output}/${name}-${width}-companion-panel.png`,fullPage:true});
    await page.keyboard.press('Escape');assert.equal(await page.locator('#data-settings').isVisible(),false);
    report.checks.push(`${name} ${width}px: static guides, optional scheduling command, dialog Escape, no horizontal overflow`);
   }
   report.engines[name]={success:true};
  }finally{await browser.close();}
 }
 assert.ok(report.engines.chromium?.success);
}finally{await writeFile(output+'/report.json',JSON.stringify(report,null,2)+'\n');server.close();}
console.log(JSON.stringify(report,null,2));
