// Actual browser acceptance check. The server is static: forbid remote requests
// and assert that a genuine browser model run reproduces native fixture policy.
import { chromium } from 'playwright-core';
import { writeFile } from 'node:fs/promises';

const output = process.argv[2];
if (!output) throw new Error('Supply a new local evaluation-report path');
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true });
const requests = [], errors = [];
try {
  const context = await browser.newContext();
  await context.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.protocol === 'http:' && url.hostname === '127.0.0.1' && url.port === '8766' && route.request().method() === 'GET') {
      requests.push({ path: url.pathname, method: 'GET' });
      await route.continue();
    } else {
      errors.push('Unexpected network request: '+url.origin+url.pathname);
      await route.abort();
    }
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto('http://127.0.0.1:8766/', { waitUntil: 'domcontentloaded' });
  await page.locator('#brew').click();
  await page.waitForFunction(() => window.prototypeResult || window.prototypeError, null, { timeout: 240000 });
  const result = await page.evaluate(() => ({ result: window.prototypeResult, error: window.prototypeError }));
  if (result.error) throw new Error(result.error+'; '+errors.join('; '));
  const report = { ...result.result.report, network: { externalRequests: errors.filter(value => value.startsWith('Unexpected network request:')), requestCount: requests.length,
    onlyStaticGets: requests.every(row => row.method === 'GET'), paths: [...new Set(requests.map(row => row.path))] }, errors };
  await page.screenshot({ path: output + '.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: output + '.mobile.png', fullPage: true });
  report.mobileLayoutOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  await writeFile(output, JSON.stringify(report,null,2)+'\n', { flag:'wx' });
  console.log(JSON.stringify({ passed:report.verdict.passed, timings:report.timings, browser:report.browser, policy:report.policy,
    externalRequests:report.network.externalRequests.length, errors:errors.length, mobileLayoutOverflow:report.mobileLayoutOverflow }));
  if (!report.verdict.passed || errors.length || report.mobileLayoutOverflow) process.exitCode=1;
} catch(error) {
  console.error(JSON.stringify({ error:error.message, errors, requests }));
  process.exitCode=1;
} finally { await browser.close(); }
