// Actual Saved template and controls; fictional metadata and intercepted mutation responses.
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdtemp,mkdir,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath} from 'node:url';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const root=fileURLToPath(new URL('..',import.meta.url));
const output=resolve(process.env.DAILYDIGEST_SAVED_UI_REPORT||'.full-review/saved-ui');
const scratch=await mkdtemp(resolve(process.env.TMPDIR||tmpdir(),'dd-saved-ui-'));
await mkdir(output,{recursive:true});
const rendered=spawnSync(resolve(root,'.venv/bin/python'),['-c',`
import os,sys,json
from pathlib import Path
from starlette.requests import Request
from jinja2 import FileSystemLoader
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
os.chdir(sys.argv[2])
Path('data').mkdir()
Path('data/profile.yaml').write_text('name: Fictional saved reader\\nkeywords: [assembly]\\n')
from dailydigest import web
web.templates.env.loader=FileSystemLoader(str(Path(sys.argv[1])/'templates'))
request=Request({'type':'http','method':'GET','path':'/saved','query_string':b'',
 'headers':[(b'host',b'saved.test')],'scheme':'http','server':('saved.test',80),'client':('test',1)})
items=[dict(id=i,label=f'Fictional paper {i}',title=f'Fictional paper {i}',url=f'https://example.invalid/{i}',
 source='Fixture Journal',section='Research',publication_label='Research article',figure=None,related_commentaries=[],
 summary='Key finding: Fixture finding.\\nWhy read: Fixture transfer.\\nCaveat: Fixture limitation.',
 summary_fields={'Key finding':'Fixture finding.','Why read':'Fixture transfer.','Caveat':'Fixture limitation.'},
 published='Oct 21, 2026',saved='Oct 23, 2026',current_grade=70,current_reasons=[]) for i in [1,2]]
print(web.templates.TemplateResponse(request,'saved.html.j2',dict(items=items,query='',csrf_token='fictional-token',
 total=2,page=1,pages=1,feedback_warning='',previous_page='',next_page='')).body.decode())
`,root,scratch],{encoding:'utf8',maxBuffer:4*1024*1024,env:{...process.env,
 DD_APP_ROOT:scratch,DD_ENV_PATH:resolve(scratch,'.env'),DB_PATH:'data/digest.db',PROFILE_PATH:'data/profile.yaml',
 OPPORTUNITY_PROFILE_PATH:'data/opportunities.yaml',LLM_BACKEND:'extractive',LLM_API_KEY:'',RESEND_API_KEY:'',SEMANTIC_MATCHING:'false'}});
assert.equal(rendered.status,0,rendered.stderr);
const html=rendered.stdout;
const report={checks:[],errors:[],external:[],scope:'Actual Saved UI with fictional metadata and mocked persistence responses; no real providers/user data.'};
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox'],env:{...process.env,TMPDIR:scratch}});
try{
 for(const width of [1280,390,320]){
  const context=await browser.newContext({viewport:{width,height:900}});
  let mode='durable';const requests=[];
  await context.route('**/*',route=>{
   const url=new URL(route.request().url());
   if(url.origin!=='http://saved.test'){report.external.push(url.href);return route.abort();}
   if(route.request().method()==='GET')return route.fulfill({contentType:'text/html',body:html});
   const method=route.request().method();requests.push({path:url.pathname,method});
   const fail=mode==='fail-remove'&&method==='DELETE'||mode==='fail-restore'&&method==='POST';
   const pending=mode==='pending-remove'&&method==='DELETE'||mode==='pending-restore'&&method==='POST';
   return route.fulfill({status:fail?503:200,headers:{'content-type':'application/json',...(pending?{'x-dailydigest-storage':'pending'}:{})},body:JSON.stringify({ok:!fail,saved:method==='POST'})});
  });
  const page=await context.newPage();page.on('pageerror',error=>report.errors.push({width,error:error.message}));
  await page.goto('http://saved.test/saved');await page.evaluate(()=>document.fonts.ready);
  const first=page.locator('.saved-card').first();
  assert.equal(await first.locator('.saved-title').innerText(),'Fictional paper 1');
  assert.equal(await first.locator('.reading-context-field').count(),1,'Only Why read sits beside the finding');
  assert.equal(await first.locator('.item-details [data-summary-label="Caveat"]').count(),1,'The caveat stays available in Details');
  assert.equal(await page.locator('.saved-header a').last().innerText(),'Settings');
  await first.locator('.remove-saved').click();
  await first.locator('.saved-removal').waitFor();assert.equal(await first.locator('.saved-content').isVisible(),false);
  assert.equal(await page.locator('#archive-count').innerText(),'1 saved item');
  await first.locator('.undo-saved').click();await first.locator('.saved-content').waitFor();
  assert.equal(await page.locator('#archive-count').innerText(),'2 saved items');
  assert.equal(await first.locator('[data-grade="70"]').getAttribute('aria-pressed'),'true');
  assert.deepEqual(requests.map(row=>row.method),['DELETE','POST']);
  report.checks.push({width,check:'Durable Remove/Undo use DELETE/POST, keep the existing rating and update counts without reloading'});

  mode='pending-remove';await first.locator('.remove-saved').click();
  await first.locator('.remove-saved',{hasText:'Retry removal'}).waitFor();
  assert.equal(await first.locator('.saved-content').isVisible(),true);assert.equal(await page.locator('#archive-count').innerText(),'2 saved items');
  await page.evaluate(()=>window.dispatchEvent(new Event('dd-storage-committed')));
  await first.locator('.saved-removal').waitFor();assert.equal(await page.locator('#archive-count').innerText(),'1 saved item');
  mode='pending-restore';await first.locator('.undo-saved').click();
  await first.locator('.undo-saved',{hasText:'Retry restore'}).waitFor();
  assert.equal(await first.locator('.saved-content').isVisible(),false);assert.equal(await page.locator('#archive-count').innerText(),'1 saved item');
  await page.evaluate(()=>window.dispatchEvent(new Event('dd-storage-committed')));await first.locator('.saved-content').waitFor();
  assert.equal(await page.locator('#archive-count').innerText(),'2 saved items');
  report.checks.push({width,check:'Pending Remove and Undo preserve the last durable presentation until checkpoint acknowledgement'});

  mode='fail-remove';await first.locator('.remove-saved').click();await first.locator('.save-state',{hasText:'Could not remove'}).waitFor();
  assert.equal(await first.locator('.saved-content').isVisible(),true);assert.equal(await first.locator('.remove-saved').isEnabled(),true);
  mode='durable';await first.locator('.remove-saved').click();await first.locator('.saved-removal').waitFor();
  mode='fail-restore';await first.locator('.undo-saved').click();await first.locator('.removal-state',{hasText:'Could not restore'}).waitFor();
  assert.equal(await first.locator('.saved-content').isVisible(),false);assert.equal(await first.locator('.undo-saved').isEnabled(),true);
  mode='durable';await first.locator('.undo-saved').click();await first.locator('.saved-content').waitFor();
  report.checks.push({width,check:'Rejected Remove/Undo remain retryable and preserve the correct durable saved state'});

  await page.evaluate(()=>{Object.defineProperty(navigator,'clipboard',{value:{writeText:async text=>{window.__copiedSummary=text;}},configurable:true});});
  await first.locator('.copy-summary').click();
  await first.locator('.copy-status',{hasText:'Summary copied'}).waitFor();
  assert.match(await page.evaluate(()=>window.__copiedSummary),/Caveat: Fixture limitation\./);
  await page.evaluate(()=>{Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('Fixture denied');}},configurable:true});});
  await first.locator('.copy-summary').click();await first.locator('.copy-fallback').waitFor();
  assert.match(await first.locator('.copy-fallback').inputValue(),/Fixture finding.*\nWhy read: Fixture transfer\./s);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.screenshot({path:resolve(output,`saved-${width}-light.png`)});
  await page.evaluate(()=>document.documentElement.dataset.theme='dark');
  assert.equal(await page.locator('.saved-header a').first().evaluate(node=>getComputedStyle(node).color),await page.locator('.saved-header').evaluate(node=>getComputedStyle(node).color));
  await page.screenshot({path:resolve(output,`saved-${width}-dark.png`)});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  report.checks.push({width,check:'Copy success/fallback, structured context and light/dark layout remain readable without overflow'});
  await context.close();
 }
 assert.deepEqual(report.errors,[]);assert.deepEqual(report.external,[]);report.passed=true;
}catch(error){report.passed=false;report.failure=error.stack;throw error;}
finally{await browser.close();await writeFile(resolve(output,'report.json'),JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify(report));
