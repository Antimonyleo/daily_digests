// Isolated file-page verification. Uses public study fixtures, never the app DB.
import assert from 'node:assert/strict';
import {readFile, writeFile, mkdir, mkdtemp} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {tmpdir} from 'node:os';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';
const site=resolve(process.env.DAILYDIGEST_STUDY || '.full-review/first-shot-next/study/index.html');
const out=resolve(process.env.DAILYDIGEST_STUDY_CHECK || '.full-review/first-shot-next/study-browser');
await mkdir(out,{recursive:true});
const scratch=await mkdtemp(resolve(tmpdir(),'reader-study-'));
const context=await chromium.launchPersistentContext(resolve(scratch,'profile'),{executablePath:process.env.CHROME_PATH || '/usr/bin/google-chrome',headless:true,args:['--no-sandbox'],acceptDownloads:true,downloadsPath:resolve(scratch,'downloads')});
const report={site,layouts:[],errors:[],requests:[]};
try{
for(const width of [1280,390,320]){
 const page=await context.newPage();
 await page.setViewportSize({width,height:844});
 page.on('pageerror',e=>report.errors.push(e.message));
 page.on('request',r=>{if(/^https?:/.test(r.url()))report.requests.push(r.url())});
 await page.goto(pathToFileURL(site).href);
 assert.equal(await page.locator('#paper article').count(),1);
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.locator('#reader').fill('fixture-reader');
 await page.locator('#independent').check();
 await page.locator('[data-label=useful]').click();
 assert.equal(await page.locator('[data-label=useful]').getAttribute('aria-pressed'),'true');
 await page.locator('#next').click();
 await page.locator('[data-label=unsure]').click();
 const downloadPromise=page.waitForEvent('download');
 await page.locator('#export').click();
 const download=await downloadPromise;
 const data=JSON.parse(await readFile(await download.path(),'utf8'));
 assert.equal(data.judgments.length,2);
 assert.equal(data.independent,true);
 await page.getByRole('button',{name:'清除本篇评价'}).click();
 await page.locator('#import').setInputFiles({name:'response.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(data))});
 await page.waitForFunction(()=>document.getElementById('status').textContent.includes('已载入 2'));
 assert.match(await page.locator('#status').textContent(),/已载入 2/);
 assert.equal(await page.locator('[data-label=unsure]').getAttribute('aria-pressed'),'true');
 const bad=structuredClone(data);bad.study_id='wrong';
 await page.locator('#import').setInputFiles({name:'wrong.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(bad))});
 await page.waitForFunction(()=>document.getElementById('status').textContent.includes('不属于'));
 assert.match(await page.locator('#status').textContent(),/不属于/);
 assert.equal(await page.locator('[data-label=unsure]').getAttribute('aria-pressed'),'true');
 if(await page.locator('#profile option').count()>1) await page.locator('#profile').selectOption({index:1});
 else await page.locator('#prev').click();
 assert.match(await page.locator('#progress').textContent(),/^1 \/ /);
 await page.screenshot({path:resolve(out,`study-${width}.png`),fullPage:true});
 report.layouts.push({width,overflow:false,roundtrip:true,invalidImportPreservesAnswers:true});
 await page.close();
}
assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);
await writeFile(resolve(out,'report.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
}finally{await context.close()}
