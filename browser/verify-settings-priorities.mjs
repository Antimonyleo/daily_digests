// Real Settings controls, isolated fictional forms, and no live services.
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdtemp, mkdir, readFile, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {extname, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const output = resolve(process.env.DAILYDIGEST_SETTINGS_REPORT || '.full-review/settings-priorities');
const scratch = await mkdtemp(resolve(process.env.TMPDIR || tmpdir(), 'dd-settings-priorities-'));
await mkdir(output, {recursive: true});
const fixtures = {
  custom: 'Battery materials | 90\nThermal storage | 10',
  equal: 'Battery materials\nThermal storage',
  rounded: 'Battery materials | 34\nThermal storage | 33\nPolymer recycling | 33',
};
const rendered = spawnSync(resolve(root, '.venv/bin/python'), ['-c', `
import json, os, sys
from pathlib import Path
from jinja2 import FileSystemLoader
from starlette.requests import Request
sys.path.insert(0, str(Path(sys.argv[1])/'src'))
os.chdir(sys.argv[2])
Path('data').mkdir()
Path('data/profile.yaml').write_text('name: Fictional settings reader\\nkeywords: []\\nbio: Research\\n')
from dailydigest import web
web.templates.env.loader = FileSystemLoader(str(Path(sys.argv[1])/'templates'))
request = Request({'type':'http', 'method':'GET', 'path':'/setup', 'query_string':b'',
 'headers':[(b'host', b'settings.test')], 'scheme':'http', 'server':('settings.test',80), 'client':('test',1)})
pages = {}
for name, topics in json.loads(sys.argv[3]).items():
    form = web._load_existing_form_defaults()
    form.update(topics=topics, llm_backend='extractive', llm_api_key='')
    pages[name] = web.templates.TemplateResponse(request, 'setup.html.j2',
        {'form':form, 'errors':[], 'csrf_token':'fictional-test-token'}).body.decode()
print(json.dumps(pages))
`, root, scratch, JSON.stringify(fixtures)], {
  encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
  env: {...process.env, DD_APP_ROOT: scratch, DD_ENV_PATH: resolve(scratch, '.env'),
    DB_PATH: 'data/digest.db', PROFILE_PATH: 'data/profile.yaml', OPPORTUNITY_PROFILE_PATH: 'data/opportunities.yaml',
    LLM_BACKEND: 'extractive', LLM_API_KEY: '', RESEND_API_KEY: '', SEMANTIC_MATCHING: 'false'},
});
assert.equal(rendered.status, 0, rendered.stderr);
const pages = JSON.parse(rendered.stdout);
const report = {scope: 'Actual production Settings template and Chrome controls; fictional local forms, no live services.', checks: [], external: [], errors: []};
const browser = await chromium.launch({executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true, args: ['--no-sandbox'], env: {...process.env, TMPDIR: scratch}});

async function check(name, fixture, run, width = 1280) {
  const context = await browser.newContext({viewport: {width, height: 900}});
  const submissions = [], errors = [];
  await context.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== 'http://settings.test') {
      report.external.push(url.origin);
      return route.abort();
    }
    if (route.request().method() === 'POST' && url.pathname === '/setup') {
      submissions.push(new URLSearchParams(route.request().postData()));
      return route.fulfill({contentType: 'text/html', body: '<p>Fictional settings submitted</p>'});
    }
    if (url.pathname === '/setup') return route.fulfill({contentType: 'text/html', body: pages[fixture]});
    if (url.pathname.startsWith('/static/')) {
      const path = resolve(root, '.' + url.pathname);
      if (path.startsWith(resolve(root, 'static') + sep)) {
        try {
          const types = {'.css': 'text/css', '.js': 'text/javascript', '.woff2': 'font/woff2', '.svg': 'image/svg+xml'};
          return route.fulfill({contentType: types[extname(path)] || 'application/octet-stream', body: await readFile(path)});
        } catch { /* Unused optional assets do not contact any service. */ }
      }
    }
    return route.fulfill({status: 204, body: ''});
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
  try {
    await page.goto('http://settings.test/setup');
    await page.locator('#interest-editor').waitFor({state: 'visible'});
    const weights = () => page.locator('[data-interest-weight]').evaluateAll(inputs => inputs.map(input => Number(input.value)));
    await run({page, weights, submissions});
    assert.deepEqual(errors, []);
    report.checks.push({name, width, passed: true});
  } catch (error) {
    report.checks.push({name, width, passed: false, error: error.stack});
    await page.screenshot({path: resolve(output, `${name}-${width}-failure.png`)});
  } finally {
    report.errors.push(...errors.map(error => ({name, error})));
    await context.close();
  }
}

try {
  await check('custom-add-remove', 'custom', async ({page, weights}) => {
    assert.deepEqual(await weights(), [90, 10]);
    await page.locator('#add-interest').click();
    assert.deepEqual(await weights(), [89, 10, 1]);
    await page.getByRole('button', {name: 'Remove research topic 3', exact: true}).click();
    assert.deepEqual(await weights(), [90, 10]);
    assert.equal(await page.locator('#topics').inputValue(), fixtures.custom);
  });
  await check('equal-add-remove', 'equal', async ({page, weights}) => {
    assert.deepEqual(await weights(), [50, 50]);
    await page.locator('#add-interest').click();
    assert.deepEqual(await weights(), [34, 33, 33]);
    await page.getByRole('button', {name: 'Remove research topic 3', exact: true}).click();
    assert.deepEqual(await weights(), [50, 50]);
  });
  await check('rounded-equal-add', 'rounded', async ({page, weights}) => {
    assert.deepEqual(await weights(), [34, 33, 33]);
    await page.locator('#add-interest').click();
    assert.deepEqual(await weights(), [25, 25, 25, 25]);
  });
  await check('explicit-even-resets-automatic-shares', 'custom', async ({page, weights}) => {
    await page.locator('#priority-options > summary').click();
    await page.locator('[data-interest-weight]').first().fill('80');
    await page.locator('#even-interests').click();
    assert.deepEqual(await weights(), [50, 50]);
    await page.locator('#add-interest').click();
    assert.deepEqual(await weights(), [34, 33, 33]);
  });
  for (const width of [1280, 390, 320]) {
    await check('collapsed-invalid-priority', 'custom', async ({page, weights, submissions}) => {
      const first = page.locator('[data-interest-weight]').first();
      await page.locator('#priority-options > summary').click();
      await first.fill('60');
      await page.locator('#priority-options > summary').click();
      await first.waitFor({state: 'hidden'});
      assert.equal(await first.isVisible(), false);
      await page.locator('button[type="submit"]').click();
      assert.equal(submissions.length, 0, 'Invalid values must not submit');
      assert.equal(await page.locator('#priority-options').evaluate(node => node.open), true);
      assert.equal(await first.isVisible(), true);
      assert.equal(await first.evaluate(node => document.activeElement === node), true);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({path: resolve(output, `validation-${width}.png`)});
      await page.locator('#normalize-interests').click();
      assert.deepEqual(await weights(), [86, 14]);
      await page.locator('button[type="submit"]').click();
      await page.getByText('Fictional settings submitted').waitFor();
      assert.equal(submissions.length, 1);
      assert.deepEqual(submissions[0].get('topics').split(/\r?\n/), ['Battery materials | 86', 'Thermal storage | 14']);
      assert.equal(submissions[0].get('topic_weight_mode'), 'percentage');
    }, width);
  }
  report.success = report.checks.every(check => check.passed) && !report.errors.length && !report.external.length;
} finally {
  await writeFile(resolve(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  await browser.close();
}
console.log(JSON.stringify({success: report.success, checks: report.checks.map(({name, width, passed}) => ({name, width, passed})), errors: report.errors}, null, 2));
assert.equal(report.success, true, `See ${resolve(output, 'report.json')}`);
