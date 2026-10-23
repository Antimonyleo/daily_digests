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
web.templates.env.loader = FileSystemLoader(str(Path(sys.argv[1])/'templates'))
request=Request({'type':'http','method':'GET','path':'/','query_string':b'',
 'headers':[(b'host',b'pip.test')],'scheme':'http','server':('pip.test',80),'client':('test',1)})
sys.stdout.write(web.index(request).body.decode())
`, root, scratch], {encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
  env: {...process.env, DD_APP_ROOT: scratch, DD_ENV_PATH: resolve(scratch, '.env'),
    DB_PATH: 'data/digest.db', PROFILE_PATH: 'data/profile.yaml', OPPORTUNITY_PROFILE_PATH: 'data/opportunities.yaml',
    LLM_BACKEND: 'extractive', LLM_API_KEY: '', RESEND_API_KEY: '', SEMANTIC_MATCHING: 'false'}});
assert.equal(rendered.status, 0, rendered.stderr);
await writeFile(resolve(output, 'pip.html'), rendered.stdout);
const report = {scope: 'Production template, fictional empty desk; no Python browser worker or live sources.', checks: [], errors: []};
const browser = await chromium.launch({executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true, args: ['--no-sandbox'], env: {...process.env, TMPDIR: scratch}});
try {
  const context = await browser.newContext({viewport: {width: 1600, height: 1000}});
  await context.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== 'http://pip.test') return route.abort();
    if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: rendered.stdout});
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
  await page.goto('http://pip.test/');
  const pet = page.locator('#tea-pet'), note = page.locator('#tea-note-text');
  const start = await page.locator('#tea-break').boundingBox();
  const fact = await note.innerText();
  assert.ok(fact.length > 20);
  const box = await pet.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 250, box.y + box.height / 2 - 200, {steps: 5});
  assert.equal(await page.locator('.pip-drag-copy').isVisible(), true);
  assert.equal(await page.locator('#tea-another').isVisible(), true);
  await page.mouse.up();
  const moved = await page.locator('#tea-break').boundingBox();
  assert.ok(moved.x > start.x + 100);
  assert.equal(await note.innerText(), fact);
  assert.equal(await note.isVisible(), true);
  report.checks.push('drag retains a visible conversation and leaf; original fact returns on release');
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
  report.checks.push('keyboard movement and Home reset work');
  assert.equal(await page.getByRole('button', {name: 'Move Pip to another preset position'}).isVisible(), false,
    'The removed Move label must not reappear on wide screens');
  report.checks.push('wide-screen Pip has no redundant Move button');
  await page.screenshot({path: resolve(output, 'desktop.png')});
  await page.emulateMedia({reducedMotion: 'reduce'});
  for (const width of [390, 320]) {
    await page.setViewportSize({width, height: 844});
    await page.locator('#tea-break').scrollIntoViewIfNeeded();
    assert.equal(await pet.getAttribute('role'), 'img');
    assert.equal(await note.isVisible(), true);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
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
  assert.match(await note.innerText(), /Tea break over/);
  report.checks.push('ten-leaf cap and complete state survive reload');

  await page.setViewportSize({width: 1600, height: 1000});
  await page.emulateMedia({reducedMotion: 'no-preference'});
  const lifecycle = await page.evaluate(() => {
    const host = document.querySelector('#tea-break');
    const button = host.querySelector('#tea-another');
    const notes = ['<img src=x onerror="window.pipInjected=true">', 'Second fictional note'];
    const nativeSet = window.setTimeout, nativeClear = window.clearTimeout;
    const NativeObserver = window.IntersectionObserver;
    const pending = new Set();
    let disconnected = 0;
    window.setTimeout = (callback, delay, ...args) => {
      const id = nativeSet(() => { pending.delete(id); callback(...args); }, delay);
      pending.add(id); return id;
    };
    window.clearTimeout = id => { pending.delete(id); nativeClear(id); };
    window.IntersectionObserver = class extends NativeObserver {
      disconnect() { disconnected++; super.disconnect(); }
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
    return {literal, newDayReset, accepted, wasAnimating, remaining, afterCleanup, settled, restoredCount, disconnected};
  });
  assert.deepEqual(lifecycle, {literal: true, newDayReset: true, accepted: '1', wasAnimating: true, remaining: 0,
    afterCleanup: '1', settled: true, restoredCount: '1', disconnected: 3});
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
  assert.deepEqual(report.errors, []);
  report.success = true;
} catch (error) {
  report.failure = String(error.stack); process.exitCode = 1;
} finally {
  await writeFile(resolve(output, 'report.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify(report));
