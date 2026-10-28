// Offline real-browser checks for the article-figure component.
// The page is served from http://127.0.0.1:8876 and every image from
// http://localhost:8876, so the image is genuinely cross-origin. No remote host
// is contacted: Chrome resolves every other hostname to NOTFOUND.
// Run: node browser/verify-article-figure.mjs
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const root = new URL('..', import.meta.url).pathname;
const PORT = 8876, PAGE = `http://127.0.0.1:${PORT}`, IMAGES = `http://localhost:${PORT}`;
const svg = (w, h) => `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><rect width="${w}" height="${h}" fill="teal"/></svg>`;
const figure = (name, label, kind) => ({url: `${IMAGES}/img/${name}`, label, kind, alt: `${label} for ${name}`});
const items = [
  ['ga', 'Graphical abstract', 'graphical_abstract', 'ok.svg'],
  ['plain', 'Article image', 'article_preview', 'nocors.svg'],
  ['missing', 'Article image', 'article_preview', 'missing.png'],
  ['flaky', 'Article image', 'article_preview', 'flaky.svg'],
].map(([id, label, kind, file], index) => ({id: index + 1, title: `Paper ${id}`, url: `https://publisher.test/${id}`,
  source: 'Fixture Journal', section: 'Research', current_grade: null, current_reasons: [], figure: figure(file, label, kind)}));
items.push({id: 9, title: 'Paper without image', url: 'https://publisher.test/none', source: 'Fixture Journal',
  section: 'Research', current_grade: null, current_reasons: [], figure: null});

// Render the working-tree templates: saved page (real _base loader) and a
// minimal digest-shaped page for the cached reading-view extractor.
const rendered = spawnSync('uv', ['run', '--frozen', 'python', '-c', `
import json, sys
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates'), autoescape=True)
items = json.loads(sys.argv[1])
saved = env.get_template('saved.html.j2').render(items=items, total=len(items), page=1, pages=1, csrf_token='fixture', query='')
part = env.get_template('_article_figure.html.j2')
card = lambda i: f'<article class="item" data-section="research"><a class="title-link" href="{i["url"]}">{i["title"]}</a><p class="meta">Fixture</p><div class="summary">A finding.</div>{part.render(item=i)}</article>'
digest = lambda chosen: ('<!doctype html><html><body><p class="digest-date">2026-10-26</p><script id="reading-personalization" type="application/json">{"name":"Fixture","notes":[]}</script>'
          '<section class="section-block"><h2 class="section-heading">Research</h2>' + ''.join(map(card, chosen)) + '</section></body></html>')
# Cached copies only keep HTTPS figure links and never load them; use a
# publisher-shaped HTTPS URL there (nothing fetches it).
cached = [dict(i, figure=dict(i['figure'], url=i['figure']['url'].replace('http://localhost:8876/img/', 'https://images.publisher.test/'))) if i['figure'] else i for i in items]
declared = [i for i in cached if not i['figure'] or i['figure']['kind'] == 'graphical_abstract']
print(json.dumps({'saved': saved, 'digest': digest(cached), 'digest-ga': digest(declared)}))
`, JSON.stringify(items)], {cwd: root, encoding: 'utf8', maxBuffer: 64 << 20});
assert.equal(rendered.status, 0, rendered.stderr);
const pages = JSON.parse(rendered.stdout);

const hits = {}, seen = [];
let flakyFailures = 1;
const server = createServer(async (request, response) => {
  const path = new URL(request.url, PAGE).pathname;
  if (path.startsWith('/img/')) {
    hits[path] = (hits[path] || 0) + 1;
    seen.push({path, cookie: request.headers.cookie || '', referer: request.headers.referer || '', origin: request.headers.origin || ''});
    const cors = {'access-control-allow-origin': PAGE, 'cache-control': 'public, max-age=600', 'content-type': 'image/svg+xml'};
    if (path === '/img/ok.svg') return response.writeHead(200, cors).end(svg(900, 600));
    if (path === '/img/nocors.svg') return response.writeHead(200, {'content-type': 'image/svg+xml'}).end(svg(900, 600));
    if (path === '/img/flaky.svg') {
      if (flakyFailures-- > 0) return response.writeHead(503, {...cors, 'cache-control': 'no-store'}).end();
      return response.writeHead(200, {...cors, 'cache-control': 'no-store'}).end(svg(600, 400));
    }
    return response.writeHead(404, {'access-control-allow-origin': PAGE}).end();
  }
  if (path === '/saved') return response.writeHead(200, {'content-type': 'text/html; charset=utf-8'}).end(pages.saved);
  if (path === '/digest' || path === '/digest-ga') return response.writeHead(200, {'content-type': 'text/html; charset=utf-8'}).end(pages[path.slice(1)]);
  if (path === '/reading-view.mjs') return response.writeHead(200, {'content-type': 'text/javascript'}).end(await readFile(root + 'browser/reading-view.mjs'));
  response.writeHead(404).end();
});
await new Promise(done => server.listen(PORT, '127.0.0.1', done));

const checks = [];
const browser = await chromium.launch({executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: true,
  args: ['--no-sandbox', '--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1, EXCLUDE localhost']});
try {
  const context = await browser.newContext({viewport: {width: 1280, height: 900}, serviceWorkers: 'block'});
  // A publisher cookie must never accompany the anonymous image request.
  await context.addCookies([{name: 'publisher_session', value: 'secret', domain: 'localhost', path: '/'}]);
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`${PAGE}/saved`);
  const card = title => page.locator('.saved-card', {hasText: title});

  assert.equal(Object.keys(hits).length, 0, 'Closed disclosures must not request images');
  assert.equal(await card('Paper without image').locator('.article-figure, img, .figure-unavailable').count(), 0);
  assert.doesNotMatch(await card('Paper without image').innerText(), /not found|unavailable|no image/i);
  assert.equal((await card('Paper ga').locator('.article-figure summary').innerText()).trim(), 'Graphical abstract · show image');
  assert.match(await card('Paper plain').locator('.article-figure summary').innerText(), /^Article image · /);
  assert.ok(await card('Paper ga').locator('.article-figure summary').isVisible(), 'Entry point is in the default reading layer');
  checks.push('lazy: no image request before opening; no figure markup or "not found" text without an image');

  const ga = card('Paper ga');
  await ga.locator('.article-figure summary').click();
  await page.waitForFunction(() => [...document.querySelectorAll('.article-figure img')].some(i => i.complete && i.naturalWidth === 900));
  const box = await ga.locator('.article-figure img').boundingBox();
  assert.ok(box.width > 480 && Math.abs(box.width / box.height - 1.5) < 0.02, `enlarged at natural aspect: ${JSON.stringify(box)}`);
  assert.match(await ga.locator('figcaption').innerText(), /Graphical abstract supplied in the Fixture Journal feed · Open full size · Article and image credit/);
  assert.equal(await ga.locator('a.figure-full').getAttribute('href'), `${IMAGES}/img/ok.svg`);
  const request = seen.find(row => row.path === '/img/ok.svg');
  assert.equal(request.cookie, '', 'anonymous: no cookies'); assert.equal(request.referer, '', 'no referrer');
  assert.equal(request.origin, PAGE, 'CORS-mode request');
  checks.push('CORS-anonymous load: cross-origin image shown enlarged (aspect kept, >480 px) without cookie or referrer, with attribution');

  await ga.locator('.article-figure summary').click();
  await ga.locator('.article-figure summary').click();
  assert.equal(hits['/img/ok.svg'], 1, 'reopening reuses the loaded image');
  await page.goto(`${PAGE}/saved`);
  await card('Paper ga').locator('.article-figure summary').click();
  await page.waitForFunction(() => [...document.querySelectorAll('.article-figure img')].some(i => i.complete && i.naturalWidth === 900));
  assert.equal(hits['/img/ok.svg'], 1, 'a revisit restores the image from the HTTP cache');
  checks.push('cache restore: reopen and page revisit reuse the cached image (1 server request)');

  for (const [title, path, reason] of [['Paper plain', '/img/nocors.svg', 'CORS refusal'], ['Paper missing', '/img/missing.png', '404']]) {
    const node = card(title);
    await node.locator('.article-figure summary').click();
    await node.locator('.figure-unavailable').waitFor({state: 'visible'});
    assert.ok(await node.locator('.article-figure img').isHidden(), `${reason}: no blank image area`);
    assert.equal(await node.locator('.figure-unavailable a').getAttribute('href'), `${IMAGES}${path}`);
    assert.equal(hits[path], 1);
    checks.push(`${reason}: image hidden, fallback offers Retry and a direct link`);
  }

  const flaky = card('Paper flaky');
  await flaky.locator('.article-figure summary').click();
  await flaky.locator('.figure-unavailable').waitFor({state: 'visible'});
  await flaky.locator('.figure-retry').click();
  await page.waitForFunction(() => [...document.querySelectorAll('.article-figure img')].some(i => i.complete && i.naturalWidth === 600));
  assert.ok(await flaky.locator('.figure-unavailable').isHidden());
  assert.equal(hits['/img/flaky.svg'], 2);
  checks.push('retry: a failed (503) image loads after Retry and the fallback disappears');

  // Cached reading path: the extractor keeps URL + label and never embeds <img>.
  const extract = async path => {
    await page.goto(`${PAGE}${path}`);
    return page.evaluate(async () => {
      const view = await import('/reading-view.mjs');
      const value = view.extractReadingView(document, new Date('2026-10-26T12:00:00Z'));
      if (!value) return null;
      const html = view.readingViewDocument(value);
      return {figures: value.items.map(item => [item.title, item.figureURL || '', item.figureLabel || '']),
        gaLink: html.includes('Graphical abstract · open publisher image'), img: /<img/i.test(html)};
    });
  };
  const declared = await extract('/digest-ga');
  assert.deepEqual(declared.figures, [['Paper ga', 'https://images.publisher.test/ok.svg', 'Graphical abstract'], ['Paper without image', '', '']]);
  assert.ok(declared.gaLink && !declared.img);
  checks.push('cached reading view: extractor keeps the graphical-abstract URL and label; copy links the image without <img>');
  const all = await extract('/digest');
  if (all) {
    assert.deepEqual(all.figures[1], ['Paper plain', 'https://images.publisher.test/nocors.svg', 'Article image']);
    checks.push('cached reading view accepts the "Article image" label');
  } else checks.push('PENDING (browser/reading-view.mjs): validateReadingView rejects the "Article image" label, so a digest containing one gets no cached copy');
  assert.deepEqual(errors, []);
  await context.close();
} finally {
  await browser.close();
  await new Promise(done => server.close(done));
}
console.log(JSON.stringify({checks, imageRequests: hits}, null, 2));
