import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

// Runs the real release-check code from app.mjs against fake manifests.
function run(options) {
  const source = readFileSync(new URL('./app.mjs', import.meta.url), 'utf8');
  const start = source.indexOf('  let computeBusy = false, dismissedRelease');
  const end = source.indexOf('  releaseReload.onclick =', start);
  const element = () => ({ hidden: true, disabled: false, textContent: '', dataset: {}, children: [],
    replaceChildren(...nodes) { this.children = nodes; }, append(...nodes) { this.children.push(...nodes); } });
  const nodes = Object.fromEntries(['release-notice', 'release-reload', 'release-dismiss', 'release-notes',
    'release-message', 'source-freshness'].map(id => [id, element()]));
  const storage = new Map(options.seen == null ? [] : [['dailydigest-version-seen', options.seen]]);
  const sandbox = {
    document: { hidden: false, getElementById: id => nodes[id], createElement: element,
      querySelector: selector => { const name = /name="([^"]+)"/.exec(selector)?.[1]; return name in options.metas ? { content: options.metas[name] } : null; } },
    fetch: async () => ({ ok: true, json: async () => structuredClone(options.manifest) }),
    localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
    publicSourceFreshness: () => null, runtimeStarted: false, ready: true, URL, importMetaUrl: 'https://example.test/app.mjs',
  };
  const body = source.slice(start, end).replace(/import\.meta\.url/g, 'importMetaUrl');
  runInNewContext(body + '\nglobalThis.__check = checkRelease;', sandbox);
  return { nodes, storage, check: sandbox.__check };
}

const notes = [
  { version: '2026.11.02', title: 'Faster funding search', notes: ['EU calls are included.'] },
  { version: '2026.10.27', title: 'One-click updates', notes: ['Updates say what changed.'] },
];
const metas = { 'dailydigest-build': 'build-a', 'dailydigest-app-release': 'app-a', 'dailydigest-version': '2026.10.27' };

test('a public-data rebuild with the same app release shows no update notice', async () => {
  const { nodes, check } = run({ metas, seen: '2026.10.27',
    manifest: { build_id: 'build-b', app_release: 'app-a', release_notes: notes.slice(1) } });
  await check();
  assert.equal(nodes['release-notice'].hidden, true);
});

test('an application release says what changed since the open version', async () => {
  const { nodes, check } = run({ metas, seen: '2026.10.27',
    manifest: { build_id: 'build-c', app_release: 'app-b', release_notes: notes } });
  await check();
  assert.equal(nodes['release-notice'].hidden, false);
  assert.equal(nodes['release-reload'].hidden, false);
  assert.match(nodes['release-message'].textContent, /new version of DailyDigest is ready/);
  const [section] = nodes['release-notes'].children;
  assert.equal(nodes['release-notes'].children.length, 1);
  assert.equal(section.children[0].textContent, 'Faster funding search');
  assert.equal(section.children[1].children[0].textContent, 'EU calls are included.');
});

test('after updating, the notes for the new version appear once', async () => {
  const opened = { ...metas, 'dailydigest-version': '2026.11.02', 'dailydigest-app-release': 'app-b' };
  const manifest = { build_id: 'build-c', app_release: 'app-b', release_notes: notes };
  const first = run({ metas: opened, seen: '2026.10.27', manifest });
  await first.check();
  assert.equal(first.nodes['release-notice'].hidden, false);
  assert.equal(first.nodes['release-reload'].hidden, true, 'Nothing to install after updating');
  assert.equal(first.nodes['release-dismiss'].textContent, 'Got it');
  assert.match(first.nodes['release-message'].textContent, /was updated/);
  assert.equal(first.storage.get('dailydigest-version-seen'), '2026.11.02');
  const again = run({ metas: opened, seen: '2026.11.02', manifest });
  await again.check();
  assert.equal(again.nodes['release-notice'].hidden, true);
});

test('a first visit records the version without announcing it', async () => {
  const { nodes, storage, check } = run({ metas, seen: null,
    manifest: { build_id: 'build-a', app_release: 'app-a', release_notes: notes.slice(1) } });
  await check();
  assert.equal(nodes['release-notice'].hidden, true);
  assert.equal(storage.get('dailydigest-version-seen'), '2026.10.27');
});

test('a page from before app releases still compares whole builds', async () => {
  const { nodes, check } = run({ metas: { 'dailydigest-build': 'build-a' },
    manifest: { build_id: 'build-b', app_release: 'app-b', release_notes: notes } });
  await check();
  assert.equal(nodes['release-notice'].hidden, false);
});
