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
  empty: '',
  custom: 'Battery materials | 90\nThermal storage | 10',
  equal: 'Battery materials\nThermal storage',
  rounded: 'Battery materials | 34\nThermal storage | 33\nPolymer recycling | 33',
  legacy: 'RNA delivery | 17.5\nHydrogels | 1',
  many: 'DNA nanotechnology\nDNA origami\nProtein design\nDe novo protein design\nSynthetic condensates\nMolecular machine learning',
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
    if name == 'legacy':
        form['topic_details'] = json.dumps({'RNA delivery': {'aliases':['RNA transport'], 'description':'Compare delivery methods.'}})
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
  const submissions = [], errors = [], wordingRequests = [];
  const expectedFailedURLs = new Set();
  let wordingMode = 'normal', releaseWording;
  await context.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== 'http://settings.test') {
      report.external.push(url.origin);
      return route.abort();
    }
    if (url.pathname === '/setup/check-wording') {
      const topic = url.searchParams.get('topic');
      wordingRequests.push(topic);
      if (wordingMode === 'slow') await new Promise(resolve => {releaseWording = resolve;});
      if (wordingMode === 'failure') {
        expectedFailedURLs.add(route.request().url());
        return route.fulfill({status: 503, json: {detail: 'Unavailable'}});
      }
      const suggestion = {'induced provimity': 'induced proximity', 'assymetric catalysis': 'asymmetric catalysis'}[topic] || null;
      return route.fulfill({json: {topic, suggestion}}).catch(() => {});
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
  page.on('console', message => {
    if (message.type() !== 'error') return;
    // The dependency-failure case intentionally returns this one HTTP error.
    if (expectedFailedURLs.has(message.location().url) && message.text() === 'Failed to load resource: the server responded with a status of 503 (Service Unavailable)') return;
    errors.push(message.text());
  });
  try {
    await page.goto('http://settings.test/setup');
    await page.locator('#interest-editor').waitFor({state: 'visible'});
    const weights = () => page.locator('[data-interest-weight]').evaluateAll(inputs => inputs.map(input => Number(input.value)));
    await run({page, weights, submissions, wordingRequests,
      setWordingMode: mode => {wordingMode = mode;}, releaseWording: () => releaseWording?.()});
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
  for (const width of [1280, 390, 320]) {
    await check('first-use-one-commit-and-no-single-priority', 'empty', async ({page, submissions}) => {
      const row = page.locator('.interest-row').first();
      assert.equal(await row.locator('[data-interest-weight]').isVisible(), false);
      assert.equal(await row.locator('[data-interest-done], [data-interest-cancel]').count(), 0);
      assert.equal(await page.locator('#priority-options').isVisible(), false);
      assert.equal(await page.getByRole('button', {name: 'Brew', exact: true}).count(), 1);
      const input = row.locator('[data-interest-name]');
      await input.fill('DNA origami assembly');
      const button = page.getByRole('button', {name: 'Brew', exact: true});
      await page.evaluate(() => scrollTo(0, 0));
      const box = await button.boundingBox();
      assert.ok(box.y + box.height <= 740, 'The primary first-use action must fit the phone first view');
      await page.screenshot({path: resolve(output, `first-use-${width}.png`)});
      await button.click();await page.getByText('Fictional settings submitted').waitFor();
      assert.equal(submissions[0].get('after_save'), 'brew');
      assert.match(submissions[0].get('topics'), /^DNA origami assembly \|/);
    }, width);
  }
  await check('single-priority-restores-after-discarding-new-blank', 'legacy', async ({page}) => {
    await page.locator('#add-interest').click();
    await page.getByRole('button', {name:'Remove research topic 3', exact:true}).click();
    assert.equal(await page.locator('#topics').inputValue(), fixtures.legacy);
    assert.equal(await page.locator('#topic-weight-mode').inputValue(), 'relative');
    assert.equal(await page.locator('button[type="submit"]').innerText(), 'Save changes');
    assert.equal(await page.locator('[data-interest-weight]:visible').count(), 2);
    await page.locator('#topic-undo button').click();
    assert.equal(await page.locator('[data-interest-name]').last().evaluate(node => node === node.ownerDocument.activeElement), true, 'Undoing a blank topic must focus its editable name');
  });
  await check('draft-removal-preserves-raw-weights-and-undo', 'legacy', async ({page}) => {
    await page.locator('#add-interest').click();
    await page.locator('[data-interest-name]').last().fill('Protein design');
    await page.locator('[data-interest-name]').last().press('Enter');
    await page.getByRole('button', {name:'Edit research topic 3',exact:true}).click();
    await page.getByRole('button', {name:'Remove research topic 3',exact:true}).click();
    assert.equal(await page.locator('#topics').inputValue(), fixtures.legacy);
    assert.equal(await page.locator('#topic-weight-mode').inputValue(), 'relative');
    await page.locator('#topic-undo button').click();
    assert.equal(await page.locator('[data-interest-name]').last().inputValue(), 'Protein design');
    assert.equal(await page.locator('#topic-weight-mode').inputValue(), 'percentage');
  });
  await check('draft-removal-keeps-explicit-priority-edits', 'legacy', async ({page}) => {
    await page.locator('#add-interest').click();
    await page.locator('[data-interest-weight]').first().fill('80');
    await page.locator('[data-interest-weight]').first().press('Enter');
    await page.getByRole('button', {name:'Remove research topic 3',exact:true}).click();
    assert.equal(await page.locator('#topic-weight-mode').inputValue(), 'percentage');
    assert.notEqual(await page.locator('#topics').inputValue(), fixtures.legacy);
  });
  await check('undo-last-empty-topic-restores-working-editor', 'empty', async ({page, submissions}) => {
    await page.getByRole('button', {name:'Remove research topic 1',exact:true}).click();
    await page.locator('#topic-undo button').click();
    const name = page.locator('[data-interest-name]');
    assert.equal(await name.count(),1);
    assert.equal(await name.evaluate(node => node === node.ownerDocument.activeElement),true);
    await name.fill('DNA origami assembly');
    await page.getByRole('button',{name:'Brew',exact:true}).click();
    await page.getByText('Fictional settings submitted').waitFor();
    assert.match(submissions[0].get('topics'),/^DNA origami assembly \|/);
  });
  await check('automatic-spelling-debounce-and-reader-control', 'custom', async ({page, wordingRequests}) => {
    await page.clock.install();
    await page.clock.runFor(1500);
    assert.deepEqual(wordingRequests, [], 'Loaded topics must not be sent automatically');
    await page.locator('[data-interest-edit]').first().click();
    const input = page.locator('[data-interest-name]').first();
    await input.fill('induced prov');
    await page.clock.runFor(500);
    await input.fill('induced provimity');
    await page.clock.runFor(699);
    assert.deepEqual(wordingRequests, [], 'Do not check unfinished typing');
    await page.clock.runFor(1);
    const use = page.getByRole('button', {name: 'Use suggestion', exact: true});
    await use.waitFor();
    assert.deepEqual(wordingRequests, ['induced provimity']);
    assert.equal(await input.inputValue(), 'induced provimity');
    await use.click();
    assert.equal(await input.inputValue(), 'induced proximity');
    await page.clock.runFor(1500);
    assert.deepEqual(wordingRequests, ['induced provimity'], 'Accepting must not cause another check');
    assert.equal(JSON.parse(await page.locator('#topic-details').inputValue())['induced proximity'].previous_name, 'Battery materials');
  });
  await check('automatic-spelling-composition-blur-and-deduplication', 'custom', async ({page, wordingRequests}) => {
    await page.clock.install();
    await page.locator('[data-interest-edit]').first().click();
    const input = page.locator('[data-interest-name]').first();
    await input.dispatchEvent('compositionstart');
    await input.fill('assymetric catalysis');
    await page.clock.runFor(1500);
    assert.deepEqual(wordingRequests, [], 'Do not query while an IME composition is unfinished');
    await input.dispatchEvent('compositionend');
    await page.locator('[data-interest-weight]').first().focus();
    await page.getByRole('button', {name: 'Keep wording', exact: true}).waitFor();
    assert.deepEqual(wordingRequests, ['assymetric catalysis']);
    await page.clock.runFor(1500);
    assert.deepEqual(wordingRequests, ['assymetric catalysis'], 'Blur and debounce must coalesce');
    await page.getByRole('button', {name: 'Keep wording', exact: true}).click();
    assert.equal(await input.inputValue(), 'assymetric catalysis');
  });
  await check('automatic-spelling-failure-retry-and-stale-result', 'custom', async ({page, wordingRequests, setWordingMode, releaseWording}) => {
    await page.locator('[data-interest-edit]').first().click();
    const row = page.locator('.interest-row').first(), input = row.locator('[data-interest-name]');
    setWordingMode('failure');
    await input.fill('induced provimity');
    await row.locator('[data-interest-weight]').focus();
    await row.getByRole('button', {name: 'Retry', exact: true}).waitFor();
    assert.equal(await input.inputValue(), 'induced provimity');
    setWordingMode('slow');
    const retryStarted = page.waitForRequest(request => request.url().includes('/setup/check-wording'));
    await row.getByRole('button', {name: 'Retry', exact: true}).click();
    await retryStarted;
    await input.fill('New specialist topic');
    setWordingMode('normal'); releaseWording();
    const latest = page.waitForResponse(response => response.url().includes('New+specialist+topic'));
    await row.locator('[data-interest-weight]').focus();
    await latest;
    assert.equal(await input.inputValue(), 'New specialist topic');
    assert.equal(await row.getByRole('button', {name: 'Use suggestion', exact: true}).isHidden(), true);
    assert.equal(wordingRequests.filter(topic => topic === 'induced provimity').length, 2);
  });
  await check('optional-other-names-retain-context-and-precision', 'legacy', async ({page, submissions}) => {
    await page.locator('[data-interest-edit]').first().click();
    const row = page.locator('.interest-row').first();
    await row.locator('.topic-aliases > summary').click();
    await row.locator('[data-interest-aliases]').fill('RNA transport\nRNA transfer');
    await page.locator('button[type="submit"]').click();
    await page.getByText('Fictional settings submitted').waitFor();
    assert.equal(submissions[0].get('topics').replace(/\r\n/g, '\n'), fixtures.legacy);
    assert.deepEqual(JSON.parse(submissions[0].get('topic_details'))['RNA delivery'],
      {aliases: ['RNA transport', 'RNA transfer'], description: 'Compare delivery methods.'});
  });
  await check('saved-topic-focus-remains-editable-without-new-blanks', 'legacy', async ({page, submissions}) => {
    await page.locator('[data-interest-edit]').first().click();
    const row = page.locator('.interest-row').first();
    await row.locator('.topic-aliases > summary').click();
    assert.equal(await row.locator('[data-interest-description]').inputValue(), 'Compare delivery methods.');
    await row.locator('[data-interest-description]').fill('Compare delivery mechanisms.');
    await page.locator('button[type="submit"]').click();
    await page.getByText('Fictional settings submitted').waitFor();
    assert.equal(JSON.parse(submissions[0].get('topic_details'))['RNA delivery'].description, 'Compare delivery mechanisms.');
  });
  for (const width of [1280, 390, 320]) {
    await check('compact-topic-editor-and-done-spelling', 'legacy', async ({page, submissions, wordingRequests}) => {
      await page.locator('[data-interest-edit]').first().click();
      const row = page.locator('.interest-row').first();
      assert.equal(await row.getByText('Matching & terminology', {exact: true}).count(), 0);
      assert.equal(await row.getByRole('button', {name: 'Suggest wording variants'}).count(), 0);
      assert.equal(await row.locator('[data-interest-description]').isVisible(), false);
      assert.equal(await row.locator('[data-interest-aliases]').isVisible(), false);
      await row.locator('[data-interest-name]').fill('induced provimity');
      await row.locator('[data-interest-name]').press('Enter');
      await row.getByRole('button', {name: 'Use suggestion', exact: true}).waitFor();
      assert.equal(await row.locator('[data-interest-title]').innerText(), 'induced provimity');
      await row.getByRole('button', {name: 'Keep wording', exact: true}).click();
      await page.locator('button[type="submit"]').click();
      await page.getByText('Fictional settings submitted').waitFor();
      assert.deepEqual(JSON.parse(submissions[0].get('topic_details'))['induced provimity'],
        {aliases: ['RNA transport'], description: 'Compare delivery methods.', previous_name: 'RNA delivery'});
      assert.deepEqual(wordingRequests, ['induced provimity']);
    }, width);
  }
  for (const width of [1280, 390, 320]) {
    for (const fixture of ['empty', 'custom']) {
      await check('usable-topic-editor-' + fixture, fixture, async ({page}) => {
        if (fixture === 'custom') await page.locator('[data-interest-edit]').first().click();
        const input = page.locator('[data-interest-name]').first();
        await input.waitFor({state: 'visible'});
        const geometry = await input.evaluate(node => {
          const box = node.getBoundingClientRect();
          const column = node.closest('.interest-heading').firstElementChild.getBoundingClientRect();
          return {input: box.width, column: column.width};
        });
        assert.ok(geometry.input >= geometry.column - 2,
          `Topic input must fill its name column: ${JSON.stringify(geometry)}`);
        assert.ok(geometry.input >= 125, 'A mobile topic needs usable typing space');
        await page.screenshot({path: resolve(output, `topic-input-${fixture}-${width}.png`)});
        await input.fill('DNA origami lattice assembly');
        await page.locator('[data-interest-name]').first().press('Enter');
        assert.equal(await page.locator('[data-interest-title]').first().innerText(), 'DNA origami lattice assembly');
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        await page.screenshot({path: resolve(output, `topic-editor-${fixture}-${width}.png`)});
      }, width);
    }
  }
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
    await check('invalid-priority', 'custom', async ({page, weights, submissions}) => {
      const first = page.locator('[data-interest-weight]').first();
      await page.locator('#priority-options > summary').click();
      await first.fill('0');
      await page.locator('#priority-options > summary').click();
      assert.equal(await first.isVisible(), true, 'Inline priorities remain visible when balancing help is closed');
      await page.locator('button[type="submit"]').click();
      assert.equal(submissions.length, 0, 'Invalid values must not submit');
      assert.equal(await page.locator('#priority-options').evaluate(node => node.open), true);
      assert.equal(await first.isVisible(), true);
      assert.equal(await first.evaluate(node => document.activeElement === node), true);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({path: resolve(output, `validation-${width}.png`)});
      await page.locator('#normalize-interests').click();
      assert.deepEqual(await weights(), [1, 99]);
      await page.locator('button[type="submit"]').click();
      await page.getByText('Fictional settings submitted').waitFor();
      assert.equal(submissions.length, 1);
      assert.deepEqual(submissions[0].get('topics').split(/\r?\n/), ['Battery materials | 1', 'Thermal storage | 99']);
      assert.equal(submissions[0].get('topic_weight_mode'), 'percentage');
    }, width);
  }
  await check('compact-edit-cancel-undo', 'custom', async ({page, weights}) => {
    assert.equal(await page.locator('.interest-name-field').first().isVisible(), false);
    await page.getByRole('button', {name:'Edit research topic 1', exact:true}).click();
    await page.locator('[data-interest-name]').first().fill('Unsaved rename');
    await page.locator('[data-interest-name]').first().press('Escape');
    assert.equal(await page.locator('[data-interest-title]').first().innerText(), 'Battery materials');
    assert.equal(await page.locator('form[action="/setup"]').getAttribute('data-dirty'), 'false');
    await page.locator('#add-interest').click();
    assert.equal(await page.locator('.interest-edit-panel:visible').count(), 1);
    await page.locator('.interest-row').last().locator('[data-interest-name]').press('Escape');
    assert.equal(await page.locator('.interest-row').count(), 2);
    assert.deepEqual(await weights(), [90, 10]);
    assert.equal(await page.locator('form[action="/setup"]').getAttribute('data-dirty'), 'false');
    await page.getByRole('button', {name:'Edit research topic 2', exact:true}).click();
    await page.getByRole('button', {name:'Remove research topic 2', exact:true}).click();
    assert.equal(await page.locator('.interest-row').count(), 1);
    await page.locator('#topic-undo button').click();
    assert.deepEqual(await weights(), [90, 10]);
    assert.equal(await page.locator('.interest-edit-panel:visible').count(), 0);
  });
  await check('proportional-valid-priority-commit', 'rounded', async ({page, weights}) => {
    const first = page.locator('[data-interest-weight]').first();
    await first.fill('50');
    await first.press('Enter');
    assert.deepEqual(await weights(), [50, 25, 25]);
    await page.locator('[data-interest-weight]').nth(1).fill('20');
    await page.locator('#add-interest').focus();
    assert.deepEqual(await weights(), [53, 20, 27]);
  });
  await check('legacy-precision-and-metadata-save', 'legacy', async ({page, weights, submissions}) => {
    assert.deepEqual(await weights(), [95, 5]);
    await page.locator('#profile-options > summary').click();
    await page.locator('#bio').fill('An updated research goal.');
    await page.locator('button[type="submit"]').click();
    await page.getByText('Fictional settings submitted').waitFor();
    assert.equal(submissions[0].get('topics').replace(/\r\n/g, '\n'), fixtures.legacy);
    assert.equal(submissions[0].get('topic_weight_mode'), 'relative');
    assert.deepEqual(JSON.parse(submissions[0].get('topic_details'))['RNA delivery'], {aliases:['RNA transport'],description:'Compare delivery methods.'});
  });
  await check('topic-rename-preserves-origin', 'legacy', async ({page, submissions}) => {
    await page.getByRole('button', {name:'Edit research topic 1', exact:true}).click();
    await page.locator('[data-interest-name]').first().fill('RNA delivery platforms');
    await page.locator('[data-interest-name]').first().press('Enter');
    await page.locator('button[type="submit"]').click();
    await page.getByText('Fictional settings submitted').waitFor();
    assert.equal(submissions[0].get('topics').replace(/\r\n/g, '\n'), 'RNA delivery platforms | 17.5\nHydrogels | 1');
    const detail = JSON.parse(submissions[0].get('topic_details'))['RNA delivery platforms'];
    assert.equal(detail.previous_name, 'RNA delivery');
    assert.deepEqual(detail.aliases, ['RNA transport']);
  });
  await check('form-cancel-restores-all-values', 'legacy', async ({page, weights}) => {
    await page.locator('#profile-options > summary').click();
    const originalBio = await page.locator('#bio').inputValue();
    await page.locator('#bio').fill('Unsaved goal');
    await page.locator('[data-interest-weight]').first().fill('50');
    await page.locator('[data-interest-weight]').first().press('Enter');
    await page.locator('#cancel-settings').click();
    assert.equal(await page.locator('#bio').inputValue(), originalBio);
    assert.deepEqual(await weights(), [95, 5]);
    assert.equal(await page.locator('#topics').inputValue(), fixtures.legacy);
    assert.equal(await page.locator('#topic-weight-mode').inputValue(), 'relative');
  });
  for (const width of [1280, 390, 320]) {
    await check('collapsed-required-opportunity-reveals-focus', 'custom', async ({page, submissions}) => {
      await page.locator('#digest-sections > summary').click();
      await page.locator('#include_events').check();
      await page.locator('[data-opportunity-required]').evaluateAll(nodes => nodes.forEach(node => {node.value = 'Fictional value';}));
      const country = page.locator('#opportunity-country');
      await country.fill('');
      await page.locator('#digest-sections > summary').click();
      assert.equal(await country.isVisible(), false);
      await page.locator('button[type="submit"]').click();
      assert.equal(submissions.length, 0);
      assert.equal(await country.isVisible(), true);
      assert.equal(await country.evaluate(node => node === document.activeElement), true);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    }, width);
  }
  for (const width of [1280, 390, 320]) {
    await check('compact-settings-presentation', 'many', async ({page}) => {
      assert.equal(await page.locator('[data-interest-name]:visible').count(), 0);
      assert.equal(await page.locator('[data-interest-weight]:visible').count(), 6);
      assert.ok(await page.locator('#interest-rows').evaluate(node => node.getBoundingClientRect().height < 400));
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({path:resolve(output, `settings-${width}-light.png`),fullPage:true});
      await page.evaluate(() => document.documentElement.dataset.theme = 'dark');
      await page.screenshot({path:resolve(output, `settings-${width}-dark.png`),fullPage:true});
    }, width);
  }
  report.success = report.checks.every(check => check.passed) && !report.errors.length && !report.external.length;
} finally {
  await writeFile(resolve(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  await browser.close();
}
console.log(JSON.stringify({success: report.success, checks: report.checks.map(({name, width, passed}) => ({name, width, passed})), errors: report.errors}, null, 2));
assert.equal(report.success, true, `See ${resolve(output, 'report.json')}`);
