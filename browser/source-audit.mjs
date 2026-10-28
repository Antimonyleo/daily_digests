// Public endpoint feasibility audit. No profile, credentials, model, or email.
// Run from repo root: node browser/source-audit.mjs [origin] [output.json]
import { chromium } from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';
import { execFileSync } from 'node:child_process';
import { writeFile } from 'node:fs/promises';

const origin = process.argv[2] || 'http://127.0.0.1:8766/browser/';
const output = process.argv[3] || 'browser/source-access.json';
const config = JSON.parse(execFileSync('.venv/bin/python', ['-c',
  'import json,yaml; print(json.dumps(yaml.safe_load(open("config/sources.yaml"))))'], { encoding: 'utf8' }));
const day = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
const endpoint = spec => {
  switch (spec.kind) {
    case 'rss': case 'events_rss': return { url: spec.url };
    case 'arxiv': return { url: 'https://export.arxiv.org/api/query?' + new URLSearchParams({ search_query: `cat:${spec.category}`, max_results: '1' }) };
    case 'biorxiv': return { url: `https://api.biorxiv.org/details/${spec.server}/${day}/${day}/0` };
    case 'pubmed': return { url: 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=clinical%20trial&retmode=json&retmax=1' };
    case 'openalex': return { url: 'https://api.openalex.org/works?search=biotechnology&per-page=1' };
    case 'openalex_authors': return { url: 'https://api.openalex.org/authors?search=Smith&per-page=1' };
    case 'openalex_venues': return { url: 'https://api.openalex.org/works?' + new URLSearchParams({ filter: `primary_location.source.id:${spec.venue_ids[0]}`, 'per-page': '1' }) };
    case 'fda_api': return { url: 'https://api.fda.gov/' + spec.endpoint + '?limit=1&search=submissions.submission_status:AP' };
    case 'clinicaltrials': return { url: 'https://clinicaltrials.gov/api/v2/studies?pageSize=1&format=json' };
    case 'grants_gov': return { url: 'https://api.grants.gov/v1/api/search2', method: 'POST', headers: { 'Content-Type': 'text/plain;charset=UTF-8' }, body: JSON.stringify({ rows: 1, keyword: 'research', oppStatuses: 'forecasted|posted' }) };
    case 'eu_funding': return { url: 'https://api.tech.ec.europa.eu/search-api/prod/rest/search?apiKey=SEDIA&text=***&pageSize=1&pageNumber=1', method: 'POST', multipart: { query: JSON.stringify({ bool: { must: [{ terms: { type: ['1'] } }, { terms: { status: ['31094501', '31094502'] } }] } }), languages: '["en"]' } };
    case 'nih_guide': return { url: 'https://search.grants.nih.gov/guide/api/data?' + new URLSearchParams({ perpage: '1', sort: 'reldate:desc', from: '0', query: 'nanoparticle', type: 'active,activenosis', parentic: 'all', primaryic: 'all', activitycodes: 'all', doctype: 'all', parentfoa: 'all', clinicaltrials: 'all', fields: 'all', spons: 'true', daterange: '01011991-' + new Date().toISOString().slice(5, 7) + new Date().toISOString().slice(8, 10) + new Date().toISOString().slice(0, 4) }) };
    case 'embl_events': return { url: spec.url + '?per_page=1&_fields=id,link' };
    case 'embo_events': return { url: spec.url };
    default: throw new Error('Unknown source kind: ' + spec.kind);
  }
};
const specs = Object.entries(config).flatMap(([section, rows]) => rows.map(spec => ({ section, name: spec.name, kind: spec.kind, ...endpoint(spec) })));
const supplementary = [
  { section: 'supplementary', name: 'PubMed abstract endpoint', kind: 'pubmed', url: 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=31452104&retmode=xml&rettype=abstract' },
  { section: 'supplementary', name: 'ACS Crossref fallback', kind: 'openalex_venues', url: 'https://api.crossref.org/prefixes/10.1021/works?rows=1' },
];
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: true });
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const diagnostics = [], failures = [], responses = [], requestUrls = new Map();
  page.on('console', event => { if (event.type() === 'error') diagnostics.push(event.text()); });
  const cdp = await context.newCDPSession(page);
  await cdp.send('Network.enable');
  cdp.on('Network.requestWillBeSent', event => requestUrls.set(event.requestId, event.request.url));
  cdp.on('Network.responseReceivedExtraInfo', event => responses.push({ url: requestUrls.get(event.requestId), status: event.statusCode }));
  cdp.on('Network.loadingFailed', event => failures.push({ url: requestUrls.get(event.requestId), error: event.errorText, cors: event.corsErrorStatus || null, blocked: event.blockedReason || null }));
  // Serve an inert audit document at the deployment origin. Application scripts
  // never load, so a source audit cannot accidentally trigger a production brew.
  await page.route(origin, route => route.fulfill({ contentType: 'text/html', body: '<title>Public source CORS audit</title>' }));
  await page.goto(origin, { waitUntil: 'domcontentloaded' });
  const results = [];
  async function probe(spec) {
    const result = await page.evaluate(async spec => {
      const start = performance.now();
      try {
        let requestBody = spec.body;
        if (spec.multipart) {
          // Same shape the adapter sends: JSON parts in a multipart form, a CORS simple request.
          requestBody = new FormData();
          for (const [name, value] of Object.entries(spec.multipart)) requestBody.append(name, new Blob([value], { type: 'application/json' }), 'blob');
        }
        const response = await fetch(spec.url, { method: spec.method || 'GET', headers: spec.headers || {}, body: requestBody, credentials: 'omit', signal: AbortSignal.timeout(15000) });
        const body = await response.text();
        let opportunityId = null;
        if (spec.kind === 'grants_gov') {
          try { opportunityId = JSON.parse(body).data?.oppHits?.[0]?.id || null; } catch {}
        }
        return { readable: true, status: response.status, finalUrl: response.url, contentType: response.headers.get('content-type'), bodyBytes: new TextEncoder().encode(body).length, sample: body.slice(0, 100), opportunityId, elapsedMs: Math.round(performance.now() - start) };
      } catch (error) { return { readable: false, exception: error.name, error: error.message, elapsedMs: Math.round(performance.now() - start) }; }
    }, spec);
    const evidence = failures.filter(row => row.url === spec.url);
    const corsEvidence = evidence.find(row => row.cors);
    const consoleEvidence = diagnostics.filter(message => message.includes(spec.url) && /CORS|Access-Control/i.test(message));
    const outcome = result.readable ? (result.status < 400 ? 'readable_http_success' : result.status === 429 ? 'readable_rate_limit' : 'readable_http_error') :
      result.exception === 'TimeoutError' ? 'timeout' : corsEvidence || consoleEvidence.length ? 'cors_blocked' : 'network_failed';
    const underlyingHttp = responses.filter(row => row.url === spec.url);
    results.push({ ...spec, ...result, outcome, evidence, underlyingHttp, consoleEvidence });
    process.stdout.write(`${spec.name}: ${outcome}${result.status ? ' ' + result.status : ''}\n`);
    return result;
  }
  // Keep publisher traffic bounded; no proxy, no opaque no-cors response.
  for (let i = 0; i < specs.length; i += 4) await Promise.all(specs.slice(i, i + 4).map(probe));
  for (const spec of supplementary) await probe(spec);
  const id = results.find(row => row.kind === 'grants_gov' && row.opportunityId)?.opportunityId;
  if (id) await probe({ section: 'supplementary', name: 'Grants.gov official detail', kind: 'grants_gov', url: 'https://api.grants.gov/v1/api/fetchOpportunity', method: 'POST', headers: { 'Content-Type': 'text/plain;charset=UTF-8' }, body: JSON.stringify({ opportunityId: id }) });
  const summary = {};
  for (const row of results) { summary[row.kind] ||= {}; summary[row.kind][row.outcome] = (summary[row.kind][row.outcome] || 0) + 1; }
  const configuredSummary = {};
  for (const row of results.filter(row => row.section !== 'supplementary')) configuredSummary[row.outcome] = (configuredSummary[row.outcome] || 0) + 1;
  const report = { generatedAt: new Date().toISOString(), origin: new URL(origin).origin, sourceConfig: 'config/sources.yaml', configuredSources: specs.length, configuredSummary, browser: await browser.version(), method: 'Real isolated Chrome CORS fetch, credentials omitted; public neutral samples only', limits: ['Origin-specific observations are not a guarantee for a future HTTPS host.', 'Readable HTTP is transport evidence, not complete pipeline/eligibility validation.', 'Timeout/network errors do not establish that a source is unavailable everywhere.', 'HTTP-only feeds will also face mixed-content restrictions on HTTPS deployments.', 'No public proxy or no-cors fallback; sources lacking CORS need an explicitly configured trusted companion.'], summary, results };
  await writeFile(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ configuredSources: specs.length, summary, output }));
} finally { await browser.close(); }
