// Actual current templates, fictional local store, real Chrome UI; no providers.
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdir,mkdtemp,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath} from 'node:url';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';
const root=fileURLToPath(new URL('..',import.meta.url));
const output=resolve(process.env.DAILYDIGEST_READING_UI_REPORT||'.full-review/reading-ui');await mkdir(output,{recursive:true});
const scratch=await mkdtemp(resolve(process.env.TMPDIR||tmpdir(),'dd-reading-ui-'));
const generated=spawnSync(resolve(root,'.venv/bin/python'),['-c',`
import os,sys,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
root,scratch=map(Path,sys.argv[1:]);os.chdir(scratch);sys.path.insert(0,str(root/'src'))
Path('data').mkdir();Path('data/profile.yaml').write_text('name: Ada\\nkeywords: [DNA origami, Protein design, Synthetic condensates]\\n')
Path('data/sources.yaml').write_text('research: []\\n')
from dailydigest import store,web
from dailydigest.models import Item
from starlette.requests import Request
from jinja2 import FileSystemLoader
web.templates.env.loader=FileSystemLoader(str(root/'templates'))
now=datetime.now(timezone.utc);day=now.date().isoformat()
titles=['Programmable DNA origami assembly design','De novo protein binder engineering','Machine learning for synthetic condensates']
store.upsert_items([Item(source='Fixture Journal',section='research',external_id=str(i),url=f'https://reading.invalid/{i}',title=title,abstract='A fictional assembly and molecular design method.',authors='Sample Author',published_at=now) for i,title in enumerate(titles)])
with store.session_scope() as session:
 rows=session.query(store.ItemRow).order_by(store.ItemRow.id).all();ids=[row.id for row in rows]
 for row in rows:row.summary='Key finding: A programmable assembly method connects molecular building blocks.\\nWhy read: Compare geometry and binding rules.\\nCaveat: Fictional interface fixture.'
store.write_digest(day,[(f'R{i+1}',item_id,.85) for i,item_id in enumerate(ids)])
store.write_digest((now-timedelta(days=2)).date().isoformat(),[('R1',ids[0],.85)])
store.write_digest_features(day,[(f'R{i+1}',item_id,.85,{'primary_facet':topic,'source_bucket':'published_journal'}) for i,(item_id,topic) in enumerate(zip(ids,['DNA origami','Protein design','Synthetic condensates']))])
pages={}
for path,handler in [('/',web.index),('/setup',web.setup_get),('/saved',web.saved_items)]:
 request=Request({'type':'http','method':'GET','path':path,'query_string':b'','headers':[(b'host',b'reading-ui.test')],'scheme':'http','server':('reading-ui.test',80),'client':('test',1),'app':web.app})
 pages[path]=handler(request).body.decode()
print(json.dumps(pages))
`,root,scratch],{encoding:'utf8',maxBuffer:4*1024*1024,env:{...process.env,DD_APP_ROOT:scratch,DD_ENV_PATH:resolve(scratch,'.env'),DB_PATH:'data/digest.db',PROFILE_PATH:'data/profile.yaml',SOURCES_PATH:'data/sources.yaml',OPPORTUNITY_PROFILE_PATH:'data/opportunities.yaml',USER_TZ:'UTC',LLM_BACKEND:'extractive',LLM_API_KEY:'',RESEND_API_KEY:'',SEMANTIC_MATCHING:'false'}});
assert.equal(generated.status,0,generated.stderr);const pages=JSON.parse(generated.stdout);
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox'],env:{...process.env,TMPDIR:scratch}});
const report={checks:[],errors:[],external:[]};
async function verifySurface(page,path){
 const surface=await page.evaluate(()=>({
  font:getComputedStyle(document.body).fontFamily,
  loaded:[...document.fonts].some(face=>face.family.includes('Figtree')&&face.status==='loaded'),
  background:getComputedStyle(document.body).backgroundImage,
  bodySize:parseFloat(getComputedStyle(document.body).fontSize),
  headingSize:parseFloat(getComputedStyle(document.querySelector('h1')).fontSize),
 }));
 assert.match(surface.font,/^Figtree/);assert.equal(surface.loaded,true);assert.match(surface.background,/linear-gradient/);
 assert.ok(surface.bodySize>=16);assert.ok(surface.headingSize>=24);
 if(path==='/'){
  const positions=await page.evaluate(()=>{
   const box=selector=>{const rect=document.querySelector(selector).getBoundingClientRect();return {x:rect.x,y:rect.y,height:rect.height};};
   return {brew:box('.reading-toolbar .primary'),saved:box('.reading-toolbar a[href="/saved"]'),settings:box('.reading-toolbar a[href="/setup"]')};
  });
  assert.ok(positions.brew.height>=48);assert.ok(positions.saved.x<positions.settings.x,'Saved and Settings must remain ordered at the right');
 }else if(path==='/setup'){
  const headings=await page.locator('.settings-row>summary').evaluateAll(nodes=>nodes.map(node=>({size:parseFloat(getComputedStyle(node).fontSize),weight:Number(getComputedStyle(node).fontWeight)})));
  assert.ok(headings.length>=4);assert.ok(headings.every(heading=>heading.size>=18&&heading.weight>=600));
 }
}
try{for(const width of [1440,390,320]){
 const context=await browser.newContext({viewport:{width,height:900}});
 await context.route('**/*',route=>{const url=new URL(route.request().url());if(url.origin!=='http://reading-ui.test'){report.external.push(url.href);return route.abort();}return route.fulfill(pages[url.pathname]?{contentType:'text/html',body:pages[url.pathname]}:{status:204,body:''});});
 const page=await context.newPage();page.on('pageerror',error=>report.errors.push(error.message));
 for(const [path,name] of [['/','digest'],['/setup','settings'],['/saved','saved']]){await page.goto('http://reading-ui.test'+path);await page.locator('h1').waitFor();await page.evaluate(()=>document.fonts.ready);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);assert.equal(await page.locator('#tea-pet').count(),1);await verifySurface(page,path);await page.screenshot({path:resolve(output,`${name}-${width}.png`)});}
 report.checks.push({width,name:'Three actual pages share type, gradient and Pip without overflow'});
 await page.goto('http://reading-ui.test/');assert.equal(await page.locator('.item:not([hidden])').count(),3);
 await page.locator('#digest-search').fill('binder');assert.equal(await page.locator('.item:not([hidden])').count(),1);await page.reload();assert.equal(await page.locator('.item:not([hidden])').count(),1);
 await page.locator('[data-filter=all]').click();await page.locator('.more-filters>summary').click();await page.locator('#topic-filter').selectOption('DNA origami');assert.equal(await page.locator('.item:not([hidden])').count(),1);await page.locator('[data-filter=preprints]').click();assert.equal(await page.locator('.item:not([hidden])').count(),0);await page.keyboard.press('Escape');await page.locator('[data-filter=all]').click();assert.equal(await page.locator('.item:not([hidden])').count(),3);
 report.checks.push({width,name:'Search URL/reload, combined topic/source filtering and reset'});
 await page.locator('.contents-links a').first().click();assert.equal(await page.evaluate(()=>document.activeElement.classList.contains('section-toggle')),true);
 await page.evaluate(()=>{window.__copied='';Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>window.__copied=text}});});await page.locator('.copy-summary').first().click();assert.match(await page.evaluate(()=>window.__copied),/Why read: Compare geometry/);assert.match(await page.evaluate(()=>window.__copied),/Caveat: Fictional/);
 await page.evaluate(()=>Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async()=>{throw Error('denied');}}}));await page.locator('.copy-summary').first().click();await page.locator('.copy-fallback').waitFor();assert.equal(await page.locator('.copy-fallback').evaluate(node=>node===document.activeElement),true);
 report.checks.push({width,name:'Contents keyboard focus, Copy summary and denied-clipboard fallback'});
 for(const path of ['/','/setup','/saved']){await page.goto('http://reading-ui.test'+path);await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.evaluate(()=>document.fonts.ready);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await page.screenshot({path:resolve(output,`${path==='/'?'digest':path.slice(1)}-dark-${width}.png`)});}
 report.checks.push({width,name:'All pages remain usable in dark mode'});await context.close();
}assert.deepEqual(report.errors,[]);assert.deepEqual(report.external,[]);report.success=true;
}finally{await browser.close();await writeFile(resolve(output,'report.json'),JSON.stringify(report,null,2));}
console.log(JSON.stringify(report));
