// Real DOM extraction/rendering; fictional metadata only and no runtime/providers.
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,mkdtemp} from 'node:fs/promises';
import {resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';
const output=resolve(process.env.DAILYDIGEST_READING_FIDELITY_REPORT||'.full-review/reading-fidelity');
await mkdir(output,{recursive:true});
const scratch=await mkdtemp(resolve(process.env.TMPDIR||tmpdir(),'dd-reading-fidelity-'));
const moduleURL='data:text/javascript;base64,'+Buffer.from(await readFile(new URL('./reading-view.mjs',import.meta.url))).toString('base64');
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox'],env:{...process.env,TMPDIR:scratch}});
const report={checks:[],blocked:[]};
try{
  const context=await browser.newContext({viewport:{width:390,height:844}});
  await context.route('**/*',route=>{report.blocked.push(route.request().url());return route.abort();});
  const shellURL='http://127.0.0.1/reading-fidelity/';
  for (const [name,source] of [['design-tokens.css','_design_tokens.css.j2'],['reading-surface.css','_reading_surface.css.j2']]) {
    const body=await readFile(new URL('../templates/'+source,import.meta.url),'utf8');
    await context.route(shellURL+name,route=>route.fulfill({contentType:'text/css',body}));
  }
  await context.route(shellURL,route=>route.fulfill({contentType:'text/html',body:'<title>Fictional reading shell</title>'}));
  const page=await context.newPage();
  await page.goto(shellURL);
  await page.setContent(`<script type="application/json" id="reading-personalization">{"name":"Ada","notes":["A fictional note."]}</script><div class="digest-date">2026-10-15</div><section class="section-block"><h2 class="section-heading">Research</h2><article class="item" data-section="research">
    <a class="title-link" href="https://example.org/paper">Fictional result</a><p class="meta">Example Journal</p>
    <p class="match-topic">Shown for a nearby topic</p><p class="topic-support" data-reading-uncertainty>Title only — relevance uncertain.</p>
    <div class="summary-primary"><span class="summary-text">A promising result.</span></div>
    <details><summary>Details</summary><div class="item-details-body"><div class="summary-field"><div class="summary-label">Why read</div><div class="summary-text">Compare a transferable method.</div></div><div class="summary-field"><div class="summary-label">Caveat</div><div class="summary-text">Only simulated.</div></div><div class="summary-field article-abstract"><div class="summary-label">Abstract</div><div class="summary-text">Supplied fictional abstract.</div></div><div class="summary-field"><div class="summary-label">Limitations</div><div class="summary-text">No field validation.</div></div><p>Author-reported claims; not independently verified.</p></div></details>
  </article></section><section id="related-reading"><h2>Related reading</h2><p data-reading-context>Outside your core interests; fit needs a closer look.</p><a href="https://example.org/related">Adjacent paper</a><a href="javascript:alert(1)">Unsafe link</a></section>`);
  const result=await page.evaluate(async moduleURL=>{
    window.__workerStarts=0;window.Worker=class{constructor(){window.__workerStarts++;throw Error('A reading copy must not create a worker');}};
    const module=await import(moduleURL);
    const value=module.extractReadingView(document,new Date('2026-10-15T12:00:00Z'));
    const legacyRejected=module.validateReadingView({...value,version:1})===null;
    value.items[0].details+='\n<script>window.__injected=true</script>';
    return {value,legacyRejected,html:module.readingViewDocument(value,'light',location.href)};
  },moduleURL);
  assert.equal(result.value.version,5);assert.equal(result.legacyRejected,true);
  assert.equal(result.value.items[0].caveat,'Only simulated.');
  assert.equal(result.value.items[0].why,'Compare a transferable method.');
  assert.doesNotMatch(result.value.items[0].details,/Only simulated/);
  assert.match(result.value.items[0].details,/Abstract\s+Supplied fictional abstract/);
  assert.match(result.value.items[0].details,/Limitations\s+No field validation/);
  assert.match(result.value.items[0].uncertainty,/Title only/);
  assert.deepEqual(result.value.related.links,[{title:'Adjacent paper',url:'https://example.org/related'}]);
  await page.setContent(result.html);
  await page.getByText('Title only — relevance uncertain.',{exact:true}).waitFor();
  assert.equal(await page.locator('article details').getAttribute('open'),null);
  await page.getByText('Details',{exact:true}).click();
  await page.getByText(/Only simulated/).waitFor();
  assert.match(await page.locator('article').innerText(),/Author-reported claims; not independently verified/);
  assert.equal(await page.getByRole('link',{name:'Adjacent paper'}).getAttribute('href'),'https://example.org/related');
  assert.equal(await page.evaluate(()=>window.__workerStarts),0);
  assert.equal(await page.evaluate(()=>Boolean(window.__injected)),false);
  assert.equal(await page.locator('script,iframe,img,form').count(),0);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.deepEqual(report.blocked,[]);
  await page.screenshot({path:output+'/mobile-fidelity.png',fullPage:true});
  report.checks.push('real DOM retains finding, caveat, abstract, limitations, uncertainty and attribution',
    'related links keep their outside-core context and reject javascript URLs',
    'Details opens without any worker or external request; shared styles stay local and script-like article text stays escaped',
    'version 1 copies are rejected and 390px reading has no horizontal overflow');
  // An ordinary #anchor in srcdoc resolves against the parent URL and can
  // replace the reading frame with the app. Target its current document explicitly.
  await page.goto(shellURL);
  await page.evaluate(html=>{document.body.replaceChildren();const frame=document.createElement('iframe');frame.id='contents';frame.setAttribute('sandbox','allow-popups allow-popups-to-escape-sandbox');frame.srcdoc=html;document.body.append(frame);},result.html);
  await page.frameLocator('#contents').locator('.contents-nav a').first().click();
  const contentsFrame=page.frames().find(frame=>frame.parentFrame());
  assert.equal(page.url(),shellURL);
  assert.match(contentsFrame.url(),/^about:srcdoc#cached-section-0$/);
  assert.equal(await contentsFrame.evaluate(()=>document.activeElement.id),'cached-section-0');
  await page.keyboard.press('Tab');
  assert.equal(await page.frameLocator('#contents').getByRole('link',{name:'Fictional result',exact:true}).evaluate(node=>node===document.activeElement),true);
  report.checks.push('cached contents anchors stay inside the script-free reading frame and focus the section before its article');
  // A cached iframe cannot navigate its parent. Its explicit controls link
  // must instead open the same edition in a permitted new tab.
  await page.goto(shellURL);
  const relatedOnlyHTML=await page.evaluate(async moduleURL=>{
    const module=await import(moduleURL);
    document.body.innerHTML='<script type="application/json" id="reading-personalization">{"name":"Ada","notes":[]}</script><p class="digest-date">2026-10-15</p><section id="related-reading"><h2>Related reading</h2><p data-reading-context>Check the topic connection.</p></section>';
    const shelf=document.querySelector('#related-reading');
    for(let i=0;i<73;i++){const a=document.createElement('a');a.href=`https://example.org/${i}`;a.textContent=`Related ${i}`;shelf.append(a);}
    const view=module.extractReadingView(document,new Date('2026-10-15T12:00:00Z'));
    return module.readingViewDocument(view,'light',location.href);
  },moduleURL);
  await page.evaluate(html=>{document.body.replaceChildren();const frame=document.createElement('iframe');frame.id='saved';frame.setAttribute('sandbox','allow-popups allow-popups-to-escape-sandbox');frame.srcdoc=html;document.body.append(frame);},relatedOnlyHTML);
  await page.frameLocator('#saved').getByText(/Showing 50 of 73 related papers/).waitFor();
  const popup=page.waitForEvent('popup');
  await page.frameLocator('#saved').getByRole('link',{name:'Open controls',exact:true}).click();
  const opened=await popup;await opened.waitForURL(shellURL+'#/?digest_id=2026-10-15');
  assert.equal(opened.url(),shellURL+'#/?digest_id=2026-10-15');
  assert.equal(page.url(),shellURL);
  await opened.close();
  assert.deepEqual(report.blocked,[]);
  report.checks.push('shelf-only reading remains cached; 73 links become 50 plus an explicit count and sandbox-safe controls link to the same edition');
  report.success=true;
}catch(error){report.success=false;report.error=error.stack;throw error;}
finally{await writeFile(output+'/report.json',JSON.stringify(report,null,2)+'\n');await browser.close();}
console.log(JSON.stringify(report,null,2));
