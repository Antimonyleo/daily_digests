// Extend the real CPU/Pyodide Brew matrix with publication and daily-use checks.
// Every profile, article and API response below is fictional; no live provider
// request is allowed. Test instrumentation only reads state and advances time.
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';

const prefix='10.9999/dailydigest-acceptance.';
const papers={
 primary:{doi:prefix+'hinges',pmid:'990001',title:'Geometry-controlled hinges in DNA origami',types:['research-article']},
 perspective:{doi:prefix+'perspective',pmid:'990002',title:'Design rules for programmable DNA origami assemblies',types:['Perspective'],
  abstract:'Programmable DNA origami engineering connects connector geometry, mechanical response and robust self assembly. We present a framework for choosing DNA nanotechnology architectures and compare reusable design methods for molecular devices, with worked examples of geometry-controlled hinges and experimentally tested constraints.'},
 today:{doi:prefix+'comment-today',pmid:'990003',title:'Molecular hinges connect geometry with programmable motion',types:['News & Views'],
  abstract:'DNA origami hinges provide a modular method for controlling molecular motion. This article discusses how connector geometry and measured mechanical behavior guide programmable DNA nanotechnology design, explains the accompanying assembly experiments and describes reusable principles for molecular devices.',parent:'primary'},
 later:{doi:prefix+'comment-later',pmid:'990004',title:'A second look at engineering molecular motion',types:['News & Views'],
  abstract:'Programmable DNA origami devices connect local connector architecture with global mechanical motion. This article discusses the same primary hinge study and its assembly methods, considers the measurements of nanoscale response and explains implications for molecular engineering.',parent:'primary'},
};
const evidence={metadataRequests:[],articleDois:Object.values(papers).map(p=>p.doi)};
let metadataMode='failure',laterDay=false;
const xml=value=>value.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

export function prepareFeed(feed){
 const published=feed.match(/<pubDate>(.*?)<\/pubDate>/)?.[1];
 assert.ok(published,'The fictional feed needs a publication date');
 feed=feed.replace('<rss version="2.0">','<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:atom="http://www.w3.org/2005/Atom">');
 feed=feed.replace(/<item>([\s\S]*?)<\/item>/,(entry)=>entry.replace('</item>',`<dc:identifier>${papers.primary.doi}</dc:identifier></item>`));
 const additional=['perspective','today','later'].map(key=>{
  const paper=papers[key];
  const publicationDate=key==='later'?new Date(Date.parse(published)+6*86400000).toUTCString():published;
  // Same-day tests API discovery. Across days the acceptance boundary is a
  // known directed relation, supplied here by the publisher's RSS metadata.
  // Unknown lookalikes may legitimately be deduplicated before API lookup.
  const genre=key==='perspective'?'<dc:type>Perspective</dc:type>':key==='later'?'<dc:type>News &amp; Views</dc:type>':'';
  const relation=key==='later'?`<atom:link rel="is-commentary-on" href="https://doi.org/${papers.primary.doi}"/><category>daily-release-later</category>`:'';
  return `<item><guid>${paper.doi}</guid><link>https://acceptance.invalid/publication/${key}</link><title>${xml(paper.title)}</title><description>${xml(paper.abstract)}</description><pubDate>${publicationDate}</pubDate><author>Fictional Acceptance Author</author><dc:identifier>${paper.doi}</dc:identifier>${genre}${relation}</item>`;
 }).join('');
 return feed.replace('</channel>',additional+'</channel>');
}

export function feedForRequest(feed){
 return laterDay?feed:feed.replace(/<item>[\s\S]*?<\/item>/g,entry=>entry.includes('<category>daily-release-later</category>')?'':entry);
}

export const omitLoadCopy=entry=>entry.includes(prefix);
// A blocked-service-worker Playwright init script throws in opaque read-only
// frames. Explicitly seeded CPU mode avoids the isolation/reload branch here.
export const allowServiceWorkers=true;

export function instrumentWorker(source){
 const marker='self.onmessage = async ({ data }) => {';
 assert.ok(source.includes(marker),'Worker instrumentation must match the released handler');
 const audit=`import json
from sqlalchemy import select
from dailydigest.store import ItemRow, session_scope
with session_scope() as _daily_release_session:
    _daily_release_rows = _daily_release_session.execute(select(ItemRow)).scalars().all()
    _daily_release_audit = [{'id': row.id, 'title': row.title, 'section': row.section, 'published_at': str(row.published_at), 'fetched_at': str(row.fetched_at), 'metadata': json.loads(row.metadata_json or '{}')} for row in _daily_release_rows if 'dailydigest-acceptance.' in (row.metadata_json or '')]
json.dumps(_daily_release_audit)`;
 const funnel=`import json
from sqlalchemy import select
from dailydigest.store import DigestRow, session_scope, load_digest_audit
with session_scope() as _daily_release_session:
    _daily_release_latest = _daily_release_session.execute(select(DigestRow.id).order_by(DigestRow.id.desc())).scalars().first()
json.dumps({'digest_id': _daily_release_latest, 'candidate_funnel': load_digest_audit(_daily_release_latest, 'candidate_funnel'), 'source_health': load_digest_audit(_daily_release_latest, 'source_health')})`;
 return source.replace(marker,marker+`
    if (data.type === 'daily-release-audit') {
      try { self.postMessage(response(data.id, state.py.runPython(${JSON.stringify(audit)}))); }
      catch (error) { self.postMessage(response(data.id, String(error), 500)); }
      return;
    }
    if (data.type === 'daily-release-funnel') {
      try { self.postMessage(response(data.id, state.py.runPython(${JSON.stringify(funnel)}))); }
      catch (error) { self.postMessage(response(data.id, String(error), 500)); }
      return;
    }
    if (data.type === 'daily-release-clock') {
      clockOffset += Number(data.advanceMs);
      self.postMessage(response(data.id, JSON.stringify({now: new Date().toISOString()})));
      return;
    }
 `);
}

export async function routeExternal(route){
 const url=new URL(route.request().url());
 if(url.origin==='https://api.crossref.org'){
  assert.ok(url.pathname.startsWith('/works/'));
  const doi=decodeURIComponent(url.pathname.slice('/works/'.length));
  const paper=Object.values(papers).find(p=>p.doi===doi);
  assert.ok(paper,'Only fictional publication identifiers may reach the metadata fixture');
  evidence.metadataRequests.push({provider:'Crossref',doi,mode:metadataMode});
  if(metadataMode==='failure'){
   await route.fulfill({status:503,body:'Fictional metadata service unavailable',headers:{'access-control-allow-origin':'*'}});
  }else{
   const relation=paper.parent?{'is-commentary-on':[{'id-type':'doi',id:papers[paper.parent].doi}]}:{};
   await route.fulfill({contentType:'application/json',headers:{'access-control-allow-origin':'*'},body:JSON.stringify({message:{DOI:doi,type:'journal-article',relation}})});
  }
  return true;
 }
 if(url.origin!=='https://www.ebi.ac.uk')return false;
 assert.equal(url.pathname,'/europepmc/webservices/rest/search');
 const query=url.searchParams.get('query');
 assert.ok(query&&(/DOI:/.test(query)||/EXT_ID:/.test(query)));
 evidence.metadataRequests.push({query,mode:metadataMode});
 if(metadataMode==='failure'){
  await route.fulfill({status:503,body:'Fictional metadata service unavailable',headers:{'access-control-allow-origin':'*'}});
  return true;
 }
 const selected=Object.values(papers).filter(p=>query.includes(`DOI:"${p.doi}"`)||query.includes(`EXT_ID:${p.pmid}`));
 const records=selected.map(p=>({doi:p.doi,id:p.pmid,source:'MED',pubTypeList:{pubType:p.types},
  ...(p.parent?{commentCorrectionList:{commentCorrection:[{source:'MED',id:papers[p.parent].pmid,type:'Comment on'}]}}:{})}));
 await route.fulfill({contentType:'application/json',headers:{'access-control-allow-origin':'*'},body:JSON.stringify({hitCount:records.length,resultList:{result:records}})});
 return true;
}

async function rpc(page,payload){
 if(payload.type==='request'&&['POST','DELETE','PUT','PATCH'].includes(payload.method)){
  const token=await page.evaluate(()=>{
   const scripts=[...document.querySelector('#app').contentDocument.scripts].map(node=>node.textContent).join('\n');
   return JSON.parse(scripts.match(/const CSRF_TOKEN\s*=\s*("[^"\n]+")/)[1]);
  });
  payload={...payload,headers:{...payload.headers,'X-CSRF-Token':token}};
 }
 const result=await page.evaluate(payload=>window.__rpc(payload),payload);
 assert.equal(result.status,200,result.body);
 return JSON.parse(result.body);
}
const audit=page=>rpc(page,{type:'daily-release-audit'});
async function advance(page,advanceMs){
 const clock=await rpc(page,{type:'daily-release-clock',advanceMs});
 await page.clock.fastForward(advanceMs);
 return clock;
}
const cardFor=(frame,paper)=>frame().locator('.item[data-section="research"]').filter({has:frame().locator('.title-link').filter({hasText:paper.title})});

export async function beforeDesk({browser,base,report,out}){
 const context=await browser.newContext({timezoneId:'America/Phoenix',serviceWorkers:'block'});
 const errors=[],external=[];
 await context.route('**/*',route=>{
  if(route.request().url().startsWith(base))return route.continue();
  external.push(route.request().url());return route.abort();
 });
 const page=await context.newPage();
 page.on('pageerror',error=>errors.push(error.message));
 await page.addInitScript(()=>{
  const Native=Worker;
  window.Worker=class extends Native{constructor(...args){super(...args);this.addEventListener('message',({data})=>{if(data.type==='ready')window.__ready=true;});}};
 });
 try{
  await page.goto(base+'__seed__');
  await page.evaluate(async()=>{
   const request=indexedDB.open('dailydigest:/daily_digests/',1);
   request.onupgradeneeded=()=>request.result.createObjectStore('state');
   const database=await new Promise(resolve=>request.onsuccess=()=>resolve(request.result));
   await new Promise(resolve=>{const transaction=database.transaction('state','readwrite');transaction.objectStore('state').put('cpu','compute');transaction.oncomplete=resolve;});
   database.close();
  });
  const started=performance.now();
  await page.goto(base+'#/setup');
  await page.locator('#startup-topics').waitFor({timeout:5000});
  await page.locator('#startup-topics').fill('DNA origami assembly\nProtein design');
  const firstInteractiveMs=performance.now()-started;
  assert.ok(firstInteractiveMs<=5000,`First topic entry took ${Math.round(firstInteractiveMs)} ms`);
  await page.locator('#startup-topics-continue').click();
  await page.waitForFunction(()=>window.__ready,undefined,{timeout:120000});
  const frame=page.frameLocator('#app');
  await frame.locator('#interest-editor').waitFor();
  assert.deepEqual(await frame.locator('[data-interest-name]').evaluateAll(nodes=>nodes.map(node=>node.value)),['DNA origami assembly','Protein design']);
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  const check={name:'first-visit-topic-entry',passed:true,firstInteractiveMs,runtimeAndSetupReadyMs:performance.now()-started,topicDraftTransferred:true};
  report.checks.push(check);console.log(JSON.stringify(check));
  await writeFile(out+'/report.json',JSON.stringify(report,null,2));
 }finally{await context.close();}
}

export async function beforeMatrix({page,frame,route,record,report}){
 report.scope+=' Daily-use extension verifies opt-in public metadata, type labels and directed commentary using fixtures.';
 report.publication=evidence;
 await record('publication-setting-save-and-reload',async()=>{
  await route('/setup','#interest-editor');
  const setting=()=>frame().locator('input[type="checkbox"][name="publication_enrichment"]');
  assert.equal(await setting().isChecked(),false,'Public metadata must remain opt-in');
  const sections=frame().locator('#digest-sections');
  if(await sections.getAttribute('open')===null)await sections.locator(':scope>summary').click();
  await setting().check();
  await frame().locator('form[action="/setup"] button[type="submit"]').click();
  await frame().locator('#brew-button').waitFor();
  await page.reload();await page.waitForFunction(()=>window.__ready,undefined,{timeout:120000});
  await route('/setup','#interest-editor');
  assert.equal(await setting().isChecked(),true,'The explicit setting must survive reload');
  return{enabledExplicitly:true,persisted:true};
 });
}

export async function afterMatrix({page,frame,route,record,brew,titles,out,report,setSourceMode}){
 let firstEdition,parentId;
 await record('metadata-failure-is-unknown-and-retries-are-bounded',async()=>{
  const rows=await audit(page);
  const primary=rows.find(row=>row.metadata.doi===papers.primary.doi);
  assert.ok(primary,'Primary paper must enter the actual pipeline');
  assert.equal(primary.metadata.publication_enrichment?.status,'failed');
  assert.equal(primary.metadata.publication_types,undefined,'Failure must not invent a research genre');
  assert.ok((await cardFor(frame,papers.primary).locator('.publication-label').allTextContents()).includes('Publication'));
  assert.equal(evidence.metadataRequests.length,1,'Repeated brews within the failed-lookup TTL must reuse the backoff');
  return{cachedStatus:'failed',requests:evidence.metadataRequests.length,brews:report.checks.filter(row=>row.name.includes('repeat')).length};
 });
 setSourceMode('normal');metadataMode='success';
 await record('metadata-retry-types-and-same-day-commentary',async()=>{
  await advance(page,20*60*1000);
  const result=await brew('full');
  firstEdition=await page.evaluate(()=>location.hash.slice(1));
  const primary=cardFor(frame,papers.primary);
  await primary.waitFor();parentId=await primary.getAttribute('data-item-id');
  assert.ok((await primary.locator('.publication-label').allTextContents()).includes('Research article'));
  assert.equal(await primary.locator('.related-commentaries a').allTextContents().then(xs=>xs.includes(papers.today.title)),true);
  assert.ok((await primary.locator('.related-commentaries .publication-label').allTextContents()).includes('News & Views'));
  assert.equal(await cardFor(frame,papers.today).count(),0,'Linked commentary must not occupy an independent recommendation');
  const rows=await audit(page),perspective=rows.find(row=>row.metadata.doi===papers.perspective.doi);
  assert.ok(perspective?.metadata.publication_types?.some(value=>value.toLowerCase()==='perspective'));
  const perspectiveCard=cardFor(frame,papers.perspective);
  let perspectiveSurface='digest';
  if(await perspectiveCard.count()){
   assert.ok((await perspectiveCard.locator('.publication-label').allTextContents()).includes('Perspective'));
  }else{
   // Ranking may reasonably omit a perspective from this finite digest. Exercise
   // the real bookmark API and Saved rendering without forcing a ranking result.
   await rpc(page,{type:'request',method:'POST',path:`/bookmark/${perspective.id}`});
   await route('/saved','.saved-card');
   assert.match(await frame().locator(`.saved-card[data-item-id="${perspective.id}"]`).innerText(),/Perspective/);
   perspectiveSurface='saved';
   await route(firstEdition,'.item[data-item-id]');
  }
  await page.screenshot({path:out+'/research-with-commentary.png',fullPage:true});
  return{...result,parentId,firstEdition,perspectiveSurface,requests:evidence.metadataRequests.length};
 });
 await record('rating-and-save-survive-reload',async()=>{
  const primary=cardFor(frame,papers.primary);
  await primary.locator('.vote-btn[data-grade="100"]').click();
  await primary.locator('.vote-btn[data-grade="100"][aria-pressed="true"]').waitFor();
  const save=primary.locator('.bookmark-btn:not(.known-btn)');
  if(await save.getAttribute('aria-pressed')!=='true')await save.click();
  await primary.locator('.bookmark-btn:not(.known-btn)[aria-pressed="true"]').waitFor();
  await page.reload();await page.waitForFunction(()=>window.__ready,undefined,{timeout:120000});
  await route('/saved','.saved-card');
  const saved=frame().locator(`.saved-card[data-item-id="${parentId}"]`);
  await saved.locator('.vote-btn[data-grade="100"][aria-pressed="true"]').waitFor();
  assert.ok((await saved.locator('.related-commentaries a').allTextContents()).includes(papers.today.title));
  return{parentId,rating:100,saved:true,commentaryPreserved:true};
 });
 await record('fast-home-cache-preserves-companion-without-python',async()=>{
  await page.evaluate(()=>window.__events=[]);
  await route('/','.item[data-item-id]');
  // An older projection can still exist while this home visit checkpoints its
  // reading history. Observe the app's completion for this visit before timing
  // a cached reload, instead of racing a projection about to be invalidated.
  await page.waitForFunction(()=>window.__events.some(event=>event.type==='response'&&event.body==='{"cached":true}'));
  await page.waitForFunction(async title=>{
   const request=indexedDB.open('dailydigest:/daily_digests/',1);
   const database=await new Promise(resolve=>request.onsuccess=()=>resolve(request.result));
   const value=await new Promise(resolve=>{const get=database.transaction('state').objectStore('state').get('reading-view');get.onsuccess=()=>resolve(get.result);});
   database.close();
   return value?.items?.some(item=>item.companions?.some(child=>child.title===title));
  },papers.today.title,{timeout:30000});
  await writeFile(out+'/home-cache-events.json',JSON.stringify(await page.evaluate(()=>window.__events),null,2));
  const started=performance.now();
  await page.reload({waitUntil:'domcontentloaded'});
  await page.locator('#reading-view-notice').waitFor({state:'visible'});
  const cached=frame().locator('article').filter({has:frame().locator('h3').filter({hasText:papers.primary.title})});
  await cached.locator('.related-commentaries a').filter({hasText:papers.today.title}).waitFor();
  const cacheReadyMs=performance.now()-started;
  assert.equal(await page.evaluate(()=>Boolean(window.__worker)),false,'Reading the cached digest must not initialize Python');
  assert.ok((await cached.locator('.related-commentaries .publication-label').allTextContents()).includes('News & Views'));
  assert.match(await cached.innerText(),/Research article/);
  await page.screenshot({path:out+'/cached-companion.png',fullPage:true});
  await page.locator('#reading-view-open').click();
  await page.waitForFunction(()=>window.__ready,undefined,{timeout:120000});
  await cardFor(frame,papers.primary).waitFor();
  return{cacheReadyMs,pythonStarted:false,companionPreserved:true};
 });
 await record('next-day-commentary-attaches-to-retained-parent',async()=>{
  laterDay=true;
  await advance(page,24*60*60*1000);
  const result=await brew('full',{allowEmpty:true});
  const nextEdition=await page.evaluate(()=>location.hash.slice(1));
  evidence.nextDayRows=await audit(page);
  evidence.nextDayFunnel=await rpc(page,{type:'daily-release-funnel'});
  const later=evidence.nextDayRows.find(row=>row.metadata.doi===papers.later.doi);
  assert.ok(later?.metadata.companion_of_dois?.includes(papers.primary.doi),'Publisher relation must survive ingest');
  assert.notEqual(nextEdition,firstEdition,'A new day must create a new edition');
  assert.equal((await titles()).includes(papers.today.title),false);
  assert.equal((await titles()).includes(papers.later.title),false);
  await route('/saved','.saved-card');
  const saved=frame().locator(`.saved-card[data-item-id="${parentId}"]`);
  const links=await saved.locator('.related-commentaries a').allTextContents();
  assert.ok(links.includes(papers.today.title));assert.ok(links.includes(papers.later.title));
  await route(firstEdition,'.item[data-item-id]');
  const oldParent=cardFor(frame,papers.primary);
  assert.ok((await oldParent.locator('.related-commentaries a').allTextContents()).includes(papers.later.title),'Old digest must expose the newly linked follow-up without republishing it');
  await page.screenshot({path:out+'/retained-parent-next-day.png',fullPage:true});
  return{...result,firstEdition,nextEdition,commentaryCount:links.length,relationEvidence:'publisher is-commentary-on'};
 });
 evidence.finalMetadata=await audit(page);
 report.publication=evidence;
 await writeFile(out+'/publication-evidence.json',JSON.stringify(evidence,null,2));
}

if(!new URL(import.meta.url).searchParams.has('brew-extension')){
 process.env.DAILYDIGEST_BREW_EXTENSION=fileURLToPath(import.meta.url);
 process.env.DAILYDIGEST_BREW_REPORT||='.full-review/daily-release-2026-10-22/browser';
 await import('./verify-brew.mjs');
}
