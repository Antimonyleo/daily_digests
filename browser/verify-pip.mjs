// Exercise the production Pip template in an isolated browser and empty desk.
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, readFile, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve, extname} from 'node:path';
import {spawnSync} from 'node:child_process';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const root = resolve(new URL('..', import.meta.url).pathname);
const output = resolve(process.env.DAILYDIGEST_PIP_REPORT || '.full-review/pip');
const scratch = await mkdtemp(resolve(process.env.TMPDIR || tmpdir(), 'dd-pip-'));
await mkdir(output, {recursive: true});
const rendered = spawnSync(resolve(root, '.venv/bin/python'), ['-c', `
import os,sys
from pathlib import Path
from jinja2 import FileSystemLoader
from starlette.requests import Request
sys.path.insert(0, str(Path(sys.argv[1])/'src'))
os.chdir(sys.argv[2])
Path('data').mkdir()
Path('data/profile.yaml').write_text('name: Fictional reader\\nbio: Research\\nkeywords: []\\ndownweight: []\\n')
from dailydigest import web
from dailydigest.tea_break import TEA_NOTE_BANK
failed_joke=next(note for note in TEA_NOTE_BANK if 'The assay was robust right up until' in note)
long_note=max(TEA_NOTE_BANK,key=len)
deck=(failed_joke,long_note,failed_joke,*TEA_NOTE_BANK[:12])
web.daily_tea_deck=lambda day: deck
web.preview_tea_deck=lambda day: deck
web.templates.env.loader = FileSystemLoader(str(Path(sys.argv[1])/'templates'))
def request(path): return Request({'type':'http','method':'GET','path':path,'query_string':b'',
 'headers':[(b'host',b'pip.test')],'scheme':'http','server':('pip.test',80),'client':('test',1)})
import json
sys.stdout.write(json.dumps({'/':web.index(request('/')).body.decode(),'/saved':web.saved_items(request('/saved')).body.decode()}))
`, root, scratch], {encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
  env: {...process.env, DD_APP_ROOT: scratch, DD_ENV_PATH: resolve(scratch, '.env'),
    DB_PATH: 'data/digest.db', PROFILE_PATH: 'data/profile.yaml', OPPORTUNITY_PROFILE_PATH: 'data/opportunities.yaml',
    LLM_BACKEND: 'extractive', LLM_API_KEY: '', RESEND_API_KEY: '', SEMANTIC_MATCHING: 'false'}});
assert.equal(rendered.status, 0, rendered.stderr);
const pages = JSON.parse(rendered.stdout);
await writeFile(resolve(output, 'pip.html'), pages['/']);
const report = {scope: 'Production template, fictional empty desk; no Python browser worker or live sources.', checks: [], errors: []};
const browser = await chromium.launch({executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true, args: ['--no-sandbox'], env: {...process.env, TMPDIR: scratch}});
try {
  const context = await browser.newContext({viewport: {width: 1600, height: 1000}, hasTouch: true});
  await context.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== 'http://pip.test') return route.abort();
    if (pages[url.pathname]) return route.fulfill({contentType: 'text/html', body: pages[url.pathname]});
    if (url.pathname.startsWith('/static/') && !url.pathname.includes('..')) {
      try {
        const file = resolve(root, '.' + url.pathname);
        const types = {'.css': 'text/css', '.js': 'text/javascript', '.woff2': 'font/woff2', '.svg': 'image/svg+xml'};
        return route.fulfill({contentType: types[extname(file)] || 'application/octet-stream', body: await readFile(file)});
      } catch { /* Missing assets are recorded by the browser. */ }
    }
    return route.fulfill({status: 404, body: 'No live services in this test'});
  });
  const page = await context.newPage();
  page.on('pageerror', error => report.errors.push(String(error)));
  await page.goto('http://pip.test/',{waitUntil:process.env.DAILYDIGEST_PIP_FONT_READY==='false'?'domcontentloaded':'load'});
  report.fontStateAtLoad=await page.evaluate(()=>document.fonts.status);
  if(process.env.DAILYDIGEST_PIP_FONT_READY!=='false')await page.evaluate(()=>document.fonts.ready);
  const pet = page.locator('#tea-pet'), note = page.locator('#tea-note-text');
  const start = await page.locator('#tea-break').boundingBox();
  // A first display greets with one short note; a reader's collapse is kept.
  assert.equal(await note.isVisible(), true, 'A first display shows one fact or joke');
  assert.equal(await pet.getAttribute('aria-expanded'), 'true');
  const greeting = await note.innerText();
  await pet.click();
  assert.equal(await note.isVisible(), false);
  assert.equal(await page.evaluate(() => localStorage.getItem('dailydigest-pip-chat')), 'closed');
  await page.reload();
  assert.equal(await note.isVisible(), false, 'A collapsed Pip stays collapsed on the next page');
  assert.equal(await note.textContent(), greeting, 'Remounting keeps the current note');
  report.checks.push('first display greets with one note; an explicit collapse survives reload without rewriting the note');
  await pet.click();
  assert.equal(await pet.getAttribute('aria-expanded'), 'true');
  const fact = await note.innerText();
  assert.ok(fact.length > 20);
  const box = await pet.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 250, box.y + box.height / 2 - 200, {steps: 5});
  assert.equal(await note.isVisible(), true, 'The current fact must stay readable DURING dragging');
  assert.equal(await note.innerText(), fact);
  assert.equal(await page.locator('#tea-another').isVisible(), true);
  await page.mouse.up();
  const moved = await page.locator('#tea-break').boundingBox();
  assert.ok(moved.x > start.x + 100);
  assert.equal(await note.innerText(), fact);
  assert.equal(await note.isVisible(), true);
  report.checks.push('cup opens its note; drag retains the same visible fact DURING movement and after release');
  await page.locator('#tea-another').click();
  await page.waitForFunction(previous => document.querySelector('#tea-note-text').textContent !== previous, fact);
  await page.waitForFunction(() => !document.querySelector('#tea-another').hasAttribute('aria-disabled'));
  assert.equal(await page.locator('#tea-leaf-count').innerText(), '1');
  const nextFact = await note.innerText();
  await page.reload();
  const restored = await page.locator('#tea-break').boundingBox();
  assert.ok(Math.abs(restored.x - moved.x) <= 1);
  assert.equal(await note.innerText(), nextFact);
  assert.equal(await page.locator('#tea-leaf-count').innerText(), '1');
  report.checks.push('leaf advances fact or joke and persists with position across reload');
  await pet.focus();
  await page.keyboard.press('ArrowRight');
  assert.ok((await page.locator('#tea-break').boundingBox()).x > restored.x);
  await page.keyboard.press('Home');
  assert.ok(Math.abs((await page.locator('#tea-break').boundingBox()).x - start.x) <= 1);
  await page.keyboard.press('Enter');
  assert.equal(await note.isVisible(),false);
  await page.keyboard.press('Enter');
  assert.equal(await note.isVisible(),true);
  await page.keyboard.press('Escape');
  assert.equal(await note.isVisible(),false);
  await page.keyboard.press('Enter');
  report.checks.push('keyboard movement, Home reset, Enter note toggle and Escape close work');
  assert.equal(await page.getByRole('button', {name: 'Move Pip to another preset position'}).isVisible(), false,
    'The removed Move label must not reappear on wide screens');
  report.checks.push('wide-screen Pip has no redundant Move button');
  await page.screenshot({path: resolve(output, 'desktop.png')});
  await page.emulateMedia({reducedMotion: 'reduce'});
  for (const width of [390, 320]) {
    await page.setViewportSize({width, height: 844});
    await page.locator('#tea-break').scrollIntoViewIfNeeded();
    assert.equal(await pet.getAttribute('role'), 'button');
    assert.equal(await note.isVisible(), true);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    const beforeTouch=await page.locator('#tea-pet .pet-cup').boundingBox(),currentNote=await note.innerText();
    (report.touchStarts||=[]).push(await pet.evaluate(node=>{
      const pet=node.getBoundingClientRect(),cup=node.querySelector('.pet-cup').getBoundingClientRect();
      const target=(box,x,y)=>{const hit=document.elementFromPoint(box.x+box.width*x,box.y+box.height*y);return{id:hit?.id,className:hit?.className,isCup:node.contains(hit)};};
      return{wholePetCenter:target(pet,.5,.5),visibleCupCenter:target(cup,.5,.5),visibleCupGrip:target(cup,.25,.5)};
    }));
    assert.equal(report.touchStarts.at(-1).visibleCupGrip.isCup,true,JSON.stringify(report.touchStarts.at(-1)));
    const touch=await context.newCDPSession(page);
    await touch.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:beforeTouch.x+beforeTouch.width*.25,y:beforeTouch.y+beforeTouch.height/2}]});
    await touch.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:20,y:200}]});
    assert.equal(await page.locator('#tea-break').evaluate(node=>node.classList.contains('is-dragging')),true);
    assert.equal(await note.isVisible(),true,'Touch dragging also retains the current note');
    assert.equal(await note.innerText(),currentNote);
    const bubble=await page.locator('.tea-dialog').boundingBox();
    assert.ok(bubble.x>=7 && bubble.y>=7 && bubble.x+bubble.width<=width-7);
    await touch.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
    await touch.detach();
    const hit=await page.locator('#tea-another').evaluate(button=>{
      const leaf=button.getBoundingClientRect(),dialog=document.querySelector('.tea-dialog').getBoundingClientRect();
      const target=document.elementFromPoint(leaf.x+leaf.width/2,leaf.y+leaf.height/2);
      return{width:innerWidth,leaf:leaf.toJSON(),dialog:dialog.toJSON(),style:document.querySelector('.tea-dialog').getAttribute('style'),target:target?.id||target?.className,clickable:button.contains(target),overlap:dialog.left<leaf.right&&dialog.right>leaf.left&&dialog.top<leaf.bottom&&dialog.bottom>leaf.top};
    });
    (report.leafHits||=[]).push(hit);
    assert.equal(hit.clickable,true,JSON.stringify(hit));
    assert.equal(hit.overlap,false,'The dialog must leave the entire leaf target clear: '+JSON.stringify(hit));
    await page.locator('#tea-another').click();
    await page.screenshot({path: resolve(output, `mobile-${width}.png`)});
  }
  assert.equal(await page.locator('#tea-leaf-count').innerText(), '3');
  report.checks.push('320/390px inline layout and reduced-motion leaf interaction work');
  const metadata = await page.locator('#reading-personalization').textContent();
  assert.equal(JSON.parse(metadata).name, 'Fictional reader');
  assert.equal(JSON.parse(metadata).notes.length, 15);
  for (let leaf = 3; leaf < 10; leaf++) await page.locator('#tea-another').click();
  assert.equal(await page.locator('#tea-leaf-count').innerText(), '10');
  assert.equal(await page.locator('#tea-another').isDisabled(), true);
  await page.reload();
  assert.equal(await page.locator('#tea-leaf-count').innerText(), '10');
  assert.match(await note.innerText(), /Your cup is full/);
  report.checks.push('ten-leaf cap and complete state survive reload');

  await page.setViewportSize({width: 1600, height: 1000});
  await page.emulateMedia({reducedMotion: 'no-preference'});
  const lifecycle = await page.evaluate(() => {
    const host = document.querySelector('#tea-break');
    const button = host.querySelector('#tea-another');
    const notes = ['<img src=x onerror="window.pipInjected=true">', 'Second fictional note'];
    const nativeSet = window.setTimeout, nativeClear = window.clearTimeout;
    const NativeObserver = window.IntersectionObserver;
    const NativeResizeObserver = window.ResizeObserver;
    const pending = new Set();
    let disconnected = 0;
    let resizeDisconnected = 0;
    window.setTimeout = (callback, delay, ...args) => {
      const id = nativeSet(() => { pending.delete(id); callback(...args); }, delay);
      pending.add(id); return id;
    };
    window.clearTimeout = id => { pending.delete(id); nativeClear(id); };
    window.IntersectionObserver = class extends NativeObserver {
      disconnect() { disconnected++; super.disconnect(); }
    };
    window.ResizeObserver = class extends NativeResizeObserver {
      disconnect() { resizeDisconnected++; super.disconnect(); }
    };
    localStorage.removeItem('dailydigest-tea-leaves:shared-lifetime-test');
    // Mount twice: one accepted click must still count only once.
    const old = mountPip(host, {digestId: 'shared-lifetime-test', notes});
    const cleanup = mountPip(host, {digestId: 'shared-lifetime-test', notes});
    old(); // A stale cleanup must not destroy the new instance.
    const newDayReset = !button.disabled
      && button.getAttribute('title') === 'Drop a tea leaf into Pip’s cup'
      && button.getAttribute('aria-label').includes('another science fact or joke');
    const literal = host.querySelector('#tea-note-text').textContent === notes[0]
      && !host.querySelector('#tea-note-text img') && !window.pipInjected;
    button.click();
    const accepted = localStorage.getItem('dailydigest-tea-leaves:shared-lifetime-test');
    const wasAnimating = host.classList.contains('is-feeding');
    cleanup(); cleanup();
    const remaining = pending.size;
    button.click();
    const afterCleanup = localStorage.getItem('dailydigest-tea-leaves:shared-lifetime-test');
    const settled = !host.matches('.is-feeding,.is-sipping,.is-regrowing,.is-dropping');
    const cleanAgain = mountPip(host, {digestId: 'shared-lifetime-test', notes});
    const restoredCount = host.querySelector('#tea-leaf-count').textContent;
    cleanAgain();
    window.setTimeout = nativeSet; window.clearTimeout = nativeClear;
    window.IntersectionObserver = NativeObserver;
    window.ResizeObserver = NativeResizeObserver;
    return {literal, newDayReset, accepted, wasAnimating, remaining, afterCleanup, settled, restoredCount, disconnected, resizeDisconnected};
  });
  assert.deepEqual(lifecycle, {literal: true, newDayReset: true, accepted: '1', wasAnimating: true, remaining: 0,
    afterCleanup: '1', settled: true, restoredCount: '1', disconnected: 3, resizeDisconnected: 3});
  report.checks.push('shared mount resets an exhausted prior day, escapes notes, replaces prior instance, cleans timers/observer/listeners during feeding, and restores accepted leaf');
  await page.locator('#display-menu-toggle').click();
  await page.getByRole('button', {name: 'Show Pip', exact: true}).click();
  assert.equal(await page.locator('#tea-break').isVisible(), false);
  await page.reload();
  assert.equal(await page.locator('#tea-break').isVisible(), false);
  await page.locator('#display-menu-toggle').click();
  await page.getByRole('button', {name: 'Show Pip', exact: true}).click();
  assert.equal(await page.locator('#tea-break').isVisible(), true);
  report.checks.push('Show Pip preference persists and remains reversible');
  const openDisplay = async () => {
    if (await page.locator('#display-menu').getAttribute('open') === null) await page.locator('#display-menu-toggle').click();
  };
  await openDisplay(); await page.locator('#display-toggle').click();
  assert.equal(await page.locator('html').getAttribute('data-display'), 'compact');
  await openDisplay(); await page.locator('#pip-toggle').click();
  const hiddenCup = () => page.locator('#tea-break').evaluate(node => ({display:getComputedStyle(node).display, rectangles:node.getClientRects().length}));
  assert.equal(await page.locator('html').getAttribute('data-pip'), 'hidden');
  assert.deepEqual(await hiddenCup(), {display:'none',rectangles:0});
  await page.reload();
  assert.equal(await page.locator('html').getAttribute('data-display'), 'compact');
  assert.deepEqual(await hiddenCup(), {display:'none',rectangles:0});
  await openDisplay(); await page.locator('#pip-toggle').click();
  assert.equal(await page.locator('#tea-break').isVisible(), true);
  report.checks.push('Compact view respects the real Pip visibility toggle and restores its saved hidden preference');
  report.dragHitTargets=[];
  await page.keyboard.press('Escape');
  for(const motion of ['reduce','no-preference'])for(const width of [390,320]){
    await page.emulateMedia({reducedMotion:motion});
    await page.setViewportSize({width,height:844});
    await page.evaluate(({width,motion})=>{
      const notes=JSON.parse(document.querySelector('#reading-personalization').textContent).notes;
      const digestId=`hit-target-${width}-${motion}`;
      localStorage.removeItem('dailydigest-tea-leaves:'+digestId);
      localStorage.setItem('dailydigest-pip-chat','open');
      mountPip(document.querySelector('#tea-break'),{digestId,notes});
    },{width,motion});
    await page.locator('#tea-pet .pet-cup').click({trial:true,position:{x:8,y:16}});
    const touch=await context.newCDPSession(page);
    for(const [index,point] of [{x:width/2,y:200},{x:width/2,y:270},{x:45,y:700},{x:width-45,y:110}].entries()){
      const grip=await pet.evaluate(node=>{
        const box=node.querySelector('.pet-cup').getBoundingClientRect(),x=box.x+box.width*.25,y=box.y+box.height/2,target=document.elementFromPoint(x,y);
        return{x,y,isCup:node.contains(target),target:target?.className};
      }),currentNote=await note.innerText();
      assert.equal(grip.isCup,true,JSON.stringify({width,motion,point,grip}));
      await touch.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:grip.x,y:grip.y}]});
      await touch.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[point]});
      const dragState=await page.locator('#tea-break').evaluate(node=>({dragging:node.classList.contains('is-dragging'),rect:node.getBoundingClientRect().toJSON()}));
      assert.equal(dragState.dragging,true,JSON.stringify({width,motion,point,grip,dragState}));
      assert.equal(await note.isVisible(),true);
      assert.equal(await note.innerText(),currentNote);
      await touch.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
      const hit=await page.locator('#tea-another').evaluate(button=>{
        const leaf=button.getBoundingClientRect(),dialog=document.querySelector('.tea-dialog').getBoundingClientRect();
        const hits=[.2,.5,.8].flatMap(x=>[.2,.5,.8].map(y=>{const target=document.elementFromPoint(leaf.x+leaf.width*x,leaf.y+leaf.height*y);return button.contains(target);}));
        return{leaf:leaf.toJSON(),dialog:dialog.toJSON(),hits,rootTransition:getComputedStyle(document.querySelector('#tea-break')).transitionProperty,dialogTransition:getComputedStyle(document.querySelector('.tea-dialog')).transitionProperty,overlap:dialog.left<leaf.right&&dialog.right>leaf.left&&dialog.top<leaf.bottom&&dialog.bottom>leaf.top};
      });
      report.dragHitTargets.push({width,motion,point,...hit});
      assert.equal(hit.rootTransition,'none');assert.equal(hit.dialogTransition,'none');
      assert.equal(hit.hits.every(Boolean),true,JSON.stringify(report.dragHitTargets.at(-1)));
      assert.equal(hit.overlap,false,JSON.stringify(report.dragHitTargets.at(-1)));
      if(index%2===1){
        await page.locator('#tea-another').click();
        await page.waitForFunction(()=>!document.querySelector('#tea-another').hasAttribute('aria-disabled'));
      }
    }
    await touch.detach();
  }
  report.checks.push('Long supplied notes leave all nine leaf hit points clear after real touch moves at390/320px with reduced and normal motion; real leaf clicks advance normally');

  // Display is fixed to the viewport corner and never deadlocks with Pip.
  await page.emulateMedia({reducedMotion:'no-preference'});
  report.displayOffsets=[];
  for(const [width,height] of [[320,640],[390,844],[1600,1000]]){
    await page.setViewportSize({width,height});
    await page.evaluate(()=>{document.body.style.minHeight='3000px';});
    for(const where of [0,.5,1]){
      await page.evaluate(where=>scrollTo(0,(document.documentElement.scrollHeight-innerHeight)*where),where);
      report.displayOffsets.push({width,where,...await page.locator('#display-menu-toggle').evaluate(node=>{const box=node.getBoundingClientRect();return{right:Math.round(innerWidth-box.right),bottom:Math.round(innerHeight-box.bottom),position:getComputedStyle(node.parentElement).position};})});
    }
    const mine=report.displayOffsets.filter(entry=>entry.width===width);
    assert.ok(mine.every(entry=>entry.position==='fixed'&&Math.abs(entry.right-mine[0].right)<=1&&Math.abs(entry.bottom-mine[0].bottom)<=1&&entry.right>=8&&entry.bottom>=8),JSON.stringify(mine));
  }
  await page.evaluate(()=>{document.body.style.minHeight='';scrollTo(0,0);});
  await page.locator('#display-menu-toggle').focus();
  await page.keyboard.press('Enter');
  assert.notEqual(await page.locator('#display-menu').getAttribute('open'),null);
  const panel=await page.locator('#display-options').boundingBox();
  assert.ok(panel.y>=0&&panel.x>=0&&panel.x+panel.width<=1600,'The open menu is not clipped: '+JSON.stringify(panel));
  await page.keyboard.press('Tab');
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#display-menu').getAttribute('open'),null);
  assert.equal(await page.locator('#display-menu-toggle').evaluate(node=>node===document.activeElement),true,'Escape returns focus to Display');
  await page.locator('#display-menu-toggle').click();
  await page.locator('h1').first().click();
  assert.equal(await page.locator('#display-menu').getAttribute('open'),null,'An outside click closes Display');
  // Drop Pip onto the Display control: it lands clear, and both stay usable.
  const toggleBox=await page.locator('#display-menu-toggle').boundingBox(),cup=await pet.boundingBox();
  await page.mouse.move(cup.x+cup.width/2,cup.y+cup.height/2);await page.mouse.down();
  await page.mouse.move(toggleBox.x+toggleBox.width/2,toggleBox.y+toggleBox.height/2,{steps:6});await page.mouse.up();
  const dropped=await page.locator('#tea-break').boundingBox();
  assert.ok(dropped.y+dropped.height<=toggleBox.y||dropped.x>=toggleBox.x+toggleBox.width||dropped.x+dropped.width<=toggleBox.x,'Pip must not cover Display: '+JSON.stringify({dropped,toggleBox}));
  assert.equal(await page.evaluate(({x,y})=>document.elementFromPoint(x,y)?.closest('#display-menu')!==null,{x:toggleBox.x+toggleBox.width/2,y:toggleBox.y+toggleBox.height/2}),true);
  await page.locator('#display-menu-toggle').click();
  const openPanel=await page.locator('#display-options').boundingBox();
  assert.equal(await page.evaluate(({x,y})=>document.elementFromPoint(x,y)?.closest('#display-options')!==null,{x:openPanel.x+openPanel.width/2,y:openPanel.y+openPanel.height/2}),true,'The open menu sits above Pip and its note');
  await pet.focus();
  assert.equal(await page.locator('#display-menu').getAttribute('open'),null,'Moving to Pip closes Display');
  await page.keyboard.press('Home');
  report.checks.push('Display is fixed bottom-right at top/middle/bottom for 320/390/1600px; keyboard, Escape focus return, outside click; Pip drops clear of Display and the open menu stays on top');

  // Fake clock: after each accepted leaf, one new note is visible at once and
  // never rotates at 0, 100, 480, 900, 1200 or 2500 ms.
  const SAMPLE_MS=[0,100,480,900,1200,2500];
  const clockContext=await browser.newContext({viewport:{width:1600,height:1000}});
  await clockContext.route('**/*',route=>{const url=new URL(route.request().url());if(url.origin==='http://pip.test'&&pages[url.pathname])return route.fulfill({contentType:'text/html',body:pages[url.pathname]});if(url.origin!=='http://pip.test')return route.abort();return route.fulfill({status:404,body:'No live services in this test'});});
  await clockContext.clock.install();
  const clocked=await clockContext.newPage();
  clocked.on('pageerror',error=>report.errors.push(String(error)));
  const watch=()=>clocked.evaluate(()=>{window.__noteObserver?.disconnect();window.__noteWrites=[];window.__noteObserver=new MutationObserver(()=>window.__noteWrites.push(document.querySelector('#tea-note-text').textContent));window.__noteObserver.observe(document.querySelector('#tea-note-text'),{childList:true,characterData:true,subtree:true});});
  const pause=()=>clocked.evaluate(()=>Date.now()).then(now=>clocked.clock.pauseAt(now+20000));
  const sample=()=>clocked.evaluate(()=>({text:document.querySelector('#tea-note-text').textContent,visible:document.querySelector('#tea-note-text').checkVisibility(),count:document.querySelector('#tea-leaf-count').textContent}));
  const timeline=async(start=0)=>{const seen={};let at=start;for(const ms of SAMPLE_MS){if(ms<at)continue;if(ms>at)await clocked.clock.runFor(ms-at);at=ms;seen[ms]=await sample();}return seen;};
  const stable=(seen,text,label)=>{for(const [ms,state] of Object.entries(seen)){assert.equal(state.text,text,`${label}: text at ${ms} ms`);assert.equal(state.visible,true,`${label}: visible at ${ms} ms`);}};
  const leafCount=()=>clocked.locator('#tea-leaf-count').textContent();
  const notes=JSON.parse(pages['/'].match(/id="reading-personalization">(.*?)<\/script>/s)[1]).notes;
  await clocked.goto('http://pip.test/');
  await pause();
  assert.equal(await clocked.locator('#tea-note-text').isVisible(),true,'first display greets');
  assert.equal(await clocked.locator('#tea-note-text').textContent(),notes[0]);
  await watch();
  report.clock={};
  // Single click.
  await clocked.locator('#tea-another').click();
  report.clock.single=await timeline();
  stable(report.clock.single,notes[1],'single');
  assert.equal(report.clock.single[0].count,'0');assert.equal(report.clock.single[480].count,'1');assert.equal(report.clock.single[2500].count,'1');
  assert.deepEqual(await clocked.evaluate(()=>window.__noteWrites),[notes[1]],'one text write per accepted leaf');
  // Fast double click.
  await watch();
  await clocked.locator('#tea-another').dblclick();
  report.clock.double=await timeline();
  stable(report.clock.double,notes[2],'double');
  assert.equal(await leafCount(),'2');
  assert.deepEqual(await clocked.evaluate(()=>window.__noteWrites),[notes[2]]);
  // Drag during the feed keeps the note and the single count.
  await watch();
  await clocked.locator('#tea-another').click();
  await clocked.clock.runFor(100);
  const clockCup=await clocked.locator('#tea-pet').boundingBox();
  await clocked.mouse.move(clockCup.x+clockCup.width/2,clockCup.y+clockCup.height/2);await clocked.mouse.down();
  await clocked.mouse.move(clockCup.x+clockCup.width/2+200,clockCup.y-150,{steps:4});
  assert.equal(await clocked.locator('#tea-note-text').textContent(),notes[3]);
  assert.equal(await clocked.locator('#tea-note-text').isVisible(),true,'readable while dragging');
  await clocked.mouse.up();
  report.clock.drag=await timeline(100);
  stable(report.clock.drag,notes[3],'drag');
  assert.equal(await leafCount(),'3');
  assert.deepEqual(await clocked.evaluate(()=>window.__noteWrites),[notes[3]]);
  await clocked.locator('#tea-pet').focus();await clocked.keyboard.press('Home');
  // Hide and re-show Pip mid-feed.
  await watch();
  await clocked.locator('#tea-another').click();
  await clocked.clock.runFor(100);
  await clocked.locator('#display-menu-toggle').click();await clocked.locator('#pip-toggle').click();
  assert.equal(await clocked.locator('#tea-break').isVisible(),false);
  await clocked.locator('#pip-toggle').click();
  report.clock.reshow=await timeline(100);
  stable(report.clock.reshow,notes[4],'hide and re-show');
  assert.equal(await leafCount(),'4');
  assert.deepEqual(await clocked.evaluate(()=>window.__noteWrites),[notes[4]]);
  await clocked.keyboard.press('Escape');
  // Cross-page: Saved keeps the same note, count and open bubble, without a new note.
  await clocked.goto('http://pip.test/saved');
  await pause();
  await watch();
  report.clock.crossPage=await timeline();
  stable(report.clock.crossPage,notes[4],'cross-page');
  assert.equal(await leafCount(),'4');
  assert.deepEqual(await clocked.evaluate(()=>window.__noteWrites),[]);
  // A collapsed bubble stays collapsed across pages, then opens with its new note.
  await clocked.locator('#tea-pet').click();
  await clocked.goto('http://pip.test/');
  await pause();
  assert.equal(await clocked.locator('#tea-note-text').isVisible(),false,'collapsed stays collapsed');
  await watch();
  await clocked.locator('#tea-another').click();
  report.clock.fromCollapsed=await timeline();
  stable(report.clock.fromCollapsed,notes[5],'open from collapsed');
  assert.deepEqual(await clocked.evaluate(()=>window.__noteWrites),[notes[5]]);
  // Reduced motion: note and count at once; a double-click is one leaf.
  await clocked.emulateMedia({reducedMotion:'reduce'});
  await watch();
  await clocked.locator('#tea-another').dblclick();
  report.clock.reduced=await timeline();
  stable(report.clock.reduced,notes[6],'reduced motion');
  assert.ok(Object.values(report.clock.reduced).every(state=>state.count==='6'));
  assert.deepEqual(await clocked.evaluate(()=>window.__noteWrites),[notes[6]]);
  await clockContext.close();
  report.checks.push('fake clock at 0/100/480/900/1200/2500 ms: single, double click, drag, hide/re-show, cross-page, collapsed and reduced motion each show one stable new note and count one leaf');
  assert.deepEqual(report.errors, []);
  report.success = true;
} catch (error) {
  report.failure = String(error.stack); process.exitCode = 1;
} finally {
  await writeFile(resolve(output, 'report.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify(report));
