// Offline real-browser checks for deferred startup and publisher image previews.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {resolve, sep, extname} from 'node:path';
import {spawnSync} from 'node:child_process';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const site=resolve(process.env.DAILYDIGEST_STARTUP_SITE || 'browser/dist');
const output=resolve(process.env.DAILYDIGEST_STARTUP_REPORT || '.full-review/startup-figures');
const baseline=process.env.DAILYDIGEST_STARTUP_BASELINE === '1';
await mkdir(output,{recursive:true});
const types={'.mjs':'text/javascript','.js':'text/javascript','.html':'text/html','.css':'text/css','.json':'application/json','.wasm':'application/wasm'};
const server=createServer(async(request,response)=>{
  try {
    let path=new URL(request.url,'http://localhost').pathname;
    if (!path.startsWith('/daily_digests/')) {response.writeHead(404).end();return;}
    if(path.endsWith('/'))path+='index.html';
    const file=resolve(site,'.'+path.slice('/daily_digests'.length));
    assert.ok(file.startsWith(site+sep));
    response.setHeader('Content-Type',types[extname(file)]||'application/octet-stream');
    if(new URL(request.url,'http://localhost').searchParams.has('isolate')){
      response.setHeader('Cross-Origin-Opener-Policy','same-origin');
      response.setHeader('Cross-Origin-Embedder-Policy','require-corp');
    }
    response.end(await readFile(file));
  } catch {response.writeHead(404).end();}
});
await new Promise(done=>server.listen(0,'127.0.0.1',done));
const base=`http://127.0.0.1:${server.address().port}/daily_digests/`;
const report={baseline,checks:[],samples:[],errors:[],external:[]};
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH || '/usr/bin/google-chrome',headless:true,
  args:['--no-sandbox','--enable-experimental-webassembly-jspi']});
try {
  for(let trial=0;trial<3;trial++) {
    const context=await browser.newContext({viewport:{width:1280,height:900},serviceWorkers:'block'});
    await context.route('**/*',async route=>{
      if(!route.request().url().startsWith(base)){report.external.push(route.request().url());return route.abort();}
      if(new URL(route.request().url()).pathname.endsWith('/worker.mjs')) {
        const response=await route.fetch();
        const source=(await response.text()).replace('self.onmessage = async ({ data }) => {',`self.onmessage = async ({ data }) => {
          if(data.type==='startup-probe'){
            await initialize();
            self.postMessage({id:data.id,type:'response',body:state.py.runPython("import json,sys; json.dumps({'numpy': 'numpy' in sys.modules, 'pipeline': 'dailydigest.pipeline' in sys.modules, 'scipy': 'scipy' in sys.modules, 'sklearn': 'sklearn' in sys.modules})")});return;
          }`);
        return route.fulfill({response,body:source});
      }
      return route.continue();
    });
    const page=await context.newPage();
    page.on('pageerror',error=>report.errors.push(error.message));
    await page.addInitScript(()=>{
      window.__startupEvents=[];
      const Native=Worker;
      window.Worker=class extends Native {
        constructor(...args){super(...args);window.__startupWorker=this;this.addEventListener('message',({data})=>window.__startupEvents.push(data));}
      };
    });
    await page.goto(base+'#/setup');
    let inputMs;
    if(!baseline && trial===0){
      await page.locator('#startup-topics-form').waitFor();
      inputMs=await page.evaluate(()=>performance.now());
      await page.locator('#startup-topics').fill('Induced proximity\nAsymmetric catalysis');
      await page.setViewportSize({width:390,height:740});
      await page.locator('#startup-topics-continue').scrollIntoViewIfNeeded();
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      await page.screenshot({path:resolve(output,'first-topics-phone.png')});
      await page.locator('#startup-topics-continue').click();
      await page.setViewportSize({width:1280,height:900});
    }
    await page.waitForFunction(()=>window.__startupEvents.some(row=>row.type==='ready'),undefined,{timeout:120000});
    const frame=page.frameLocator('#app');
    await frame.locator('#interest-editor').waitFor();
    const timings=await page.evaluate(()=>window.__startupEvents.find(row=>row.type==='startup-timings'));
    const probe=await page.evaluate(()=>new Promise(done=>{
      const worker=window.__startupWorker;
      const listener=({data})=>{if(data.id==='probe'){worker.removeEventListener('message',listener);done(JSON.parse(data.body));}};
      worker.addEventListener('message',listener);worker.postMessage({type:'startup-probe',id:'probe'});
    }));
    if(!baseline){
      assert.equal(probe.pipeline,false,'Opening Setup must not import the brew pipeline');
      assert.equal(probe.numpy,false);assert.equal(probe.scipy,false);assert.equal(probe.sklearn,false);
      if(trial===0){
        assert.deepEqual(await frame.locator('[data-interest-name]').evaluateAll(nodes=>nodes.map(node=>node.value)),['Induced proximity','Asymmetric catalysis']);
        assert.deepEqual(await frame.locator('[data-interest-weight]').evaluateAll(nodes=>nodes.map(node=>node.value)),['50','50']);
        report.checks.push('topic draft is interactive before Python readiness and transfers to equal-weight editable rows');
      }
    }
    report.samples.push({inputMs,totalMs:timings.totalMs,stages:timings.stages,probe});
    await context.close();
  }
  if(!baseline){
    for(const isolated of [false,true]) {
    const context=await browser.newContext({serviceWorkers:'block'});const page=await context.newPage();
    await context.addInitScript(() => {
      const Native = Worker;
      window.Worker = class extends Native {
        constructor(...args) { super(...args); window.__figureWorkerStarted = true; }
      };
    });
    // Exercise the renderer without a background startup replacing its fixture.
    await page.route('**/worker.mjs*',route=>route.fulfill({contentType:'text/javascript',body:'self.onmessage=()=>{};'}));
    console.log('Testing publisher figure',isolated?'isolated':'normal');
    await page.goto(base+(isolated?'?isolate=1':'index.html'));
    // Finish the shell's asynchronous storage decision before replacing its
    // frame; otherwise a late startup continuation can hide this fixture.
    await page.waitForFunction(() => window.__figureWorkerStarted === true);
    assert.equal(await page.evaluate(()=>crossOriginIsolated),isolated);
    // Use the actual renderer and shared template script without brewing.
    const fixture=spawnSync(new URL('../.venv/bin/python',import.meta.url).pathname,['-c', `
import sys,zipfile
from jinja2 import DictLoader,Environment
with zipfile.ZipFile(sys.argv[1]) as archive:
    templates={name.removeprefix('templates/'):archive.read(name).decode() for name in archive.namelist() if name.startswith('templates/') and name.endswith('.j2')}
env=Environment(loader=DictLoader(templates),autoescape=True)
print(env.get_template('saved.html.j2').render(items=[{'id':1,'title':'A molecular method','url':'https://publisher.test/paper','source':'Publisher','section':'Research','current_grade':None,'current_reasons':[],'figure':{'url':'https://publisher.test/toc.png','label':'Graphical abstract','alt':'A reaction scheme'}}],total=1,page=1,pages=1,csrf_token='fixture',query=''))
`,resolve(site,'assets/app.zip')],{encoding:'utf8'});
    assert.equal(fixture.status,0,fixture.stderr);
    const rendered=fixture.stdout;
    let imageRequests=0;
    await page.route('https://publisher.test/**',route=>{
      imageRequests++;
      if(route.request().url().includes('?broken'))return route.fulfill({contentType:'image/png',body:'not an image',headers:{'access-control-allow-origin':'*'}});
      return route.fulfill({contentType:'image/svg+xml',headers:{'access-control-allow-origin':'*'},
        body:'<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80"><rect width="120" height="80" fill="green"/></svg>'});
    });
    await page.evaluate(async html=>{
      const {renderedDocument}=await import(document.querySelector('script[type=module][src]').src);
      document.querySelector('#startup').hidden=true;
      const frame=document.querySelector('#app');frame.hidden=false;
      frame.srcdoc=renderedDocument(html,'figure-test','/',location.href);
    },rendered);
    const frame=page.frameLocator('#app');
    await frame.locator('.article-figure').waitFor();
    assert.equal(imageRequests,0,'Collapsed preview must make no image request');
    await frame.locator('.article-figure summary').click();
    await page.waitForFunction(()=>document.querySelector('#app').contentDocument.querySelector('.article-figure img')?.naturalWidth>0,undefined,{timeout:10000});
    assert.equal(imageRequests,1);
    await frame.locator('.article-figure img').evaluate(image=>{image.src+='?broken';});
    await frame.locator('.figure-unavailable').waitFor({state:'visible'});
    assert.ok(await frame.locator('.figure-link').getAttribute('href'));
    report.checks.push(`actual template and CSP: click-to-load image and failed-image fallback (${isolated?'COEP isolated':'normal'} browser)`);
    await context.close();
    }
  }
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.external,[]);
  report.checks.push('three isolated startup samples; no external services, provider calls or brews');
} finally {
  await browser.close();await new Promise(done=>server.close(done));
  await writeFile(resolve(output,'report.json'),JSON.stringify(report,null,2)+'\n');
}
console.log(JSON.stringify(report));
