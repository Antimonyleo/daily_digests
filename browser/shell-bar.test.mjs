import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { sharedFontFace } from './app.mjs';
import { initializeQuickGuide } from './guide.mjs';

const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');

test('one top bar holds Guide and a More menu with every former bar control', () => {
  const bar = html.split('<header class="browser-bar">', 2)[1]?.split('</header>', 1)[0];
  assert.ok(bar, 'The shell has a single header bar');
  assert.equal(html.includes('<footer class="browser-bar">'), false, 'No second, bottom bar');
  assert.equal(html.includes('id="welcome-guide"'), false, 'No separate getting-started band');
  assert.equal((html.match(/>Guide</g) || []).length, 1, 'Guide appears once');
  for (const id of ['status', 'announcements', 'install', 'open-guide', 'open-more', 'storage-warning']) assert.match(bar, new RegExp(`id="${id}"`));
  const menu = bar.split('id="more-menu"', 2)[1];
  for (const [id, label] of [['open-backup', 'Backup &amp; transfer'], ['open-restore', 'Restore backup'], ['open-data-settings', 'Data &amp; compute']]) {
    assert.match(menu, new RegExp(`id="${id}" type="button" disabled>${label}<`), `${id} waits for its handler`);
  }
  assert.match(menu, /Saved in this browser profile/);
  assert.match(bar, /id="open-more" type="button" aria-expanded="false" aria-controls="more-menu" disabled>More/);
  assert.match(bar, /id="announcements" class="visually-hidden" role="status" aria-live="polite"/);
  assert.match(bar, /id="storage-warning" class="storage-warning" role="alert" hidden/);
});

test('startup steps use reader words and keep the tea wording', () => {
  const stages = [...html.matchAll(/data-startup-stage="(\w+)">([^<]+)</g)].map(match => [match[1], match[2]]);
  assert.deepEqual(stages, [['python', 'Loading the app'], ['data', 'Opening your saved reading'], ['ready', 'Ready to read']]);
  assert.match(html, /Putting the kettle on…/);
  assert.doesNotMatch(html.split('id="startup"', 2)[1].split('</main>', 1)[0], /Python|CPU model|libraries/);
  assert.match(html, /<span id="download-size">[^<]+<\/span>/, 'The build still stamps the measured size');
});

test('the build stamps the measured size in plain words', () => {
  const build = readFileSync(new URL('./build.py', import.meta.url), 'utf8');
  assert.match(build, /f"up to \{math\.ceil\(initial_bytes \/ 1_000_000\)\} MB"/);
});

test('the shell reuses the shipped Figtree face without a new request', () => {
  const styles = ':root{--x:1}\n/* Figtree */\n@font-face { font-family: Figtree; src: url(data:font/woff2;base64,AAAA) format("woff2"); }\nbody{font-family:Figtree}';
  assert.equal(sharedFontFace(styles), '@font-face { font-family: Figtree; src: url(data:font/woff2;base64,AAAA) format("woff2"); }');
  assert.equal(sharedFontFace(null), '', 'Unbuilt sources fall back to the system stack');
  const reading = readFileSync(new URL('../templates/_reading_surface.css.j2', import.meta.url), 'utf8');
  assert.match(sharedFontFace(reading), /^@font-face \{ font-family: Figtree;[^}]*url\(data:font\/woff2;base64,/);
  assert.match(readFileSync(new URL('./style.css', import.meta.url), 'utf8').split('\n', 1)[0], /font-family: Figtree,/);
});

function guideDocument() {
  const node = () => ({ hidden: false, disabled: false, dataset: {}, textContent: '', id: '', focus() {}, setAttribute() {}, addEventListener() {} });
  const nodes = Object.fromEntries(['open-guide', 'guide-position', 'guide-previous', 'guide-next', 'guide-finish', 'close-guide'].map(id => [id, node()]));
  const listeners = {};
  const steps = [0, 1].map(() => ({ ...node(), querySelector: () => node() }));
  nodes['quick-guide'] = { ...node(), open: false, querySelectorAll: selector => selector === '[data-guide-step]' ? steps : [],
    showModal() { this.open = true; }, close() { this.open = false; listeners.close?.(); },
    addEventListener(name, handler) { listeners[name] = handler; } };
  return { document: { getElementById: id => nodes[id] }, nodes };
}

test('first visit marks Guide as the starting point until the guide closes', () => {
  const { document, nodes } = guideDocument();
  let seen = false;
  const guide = initializeQuickGuide({ document, hasSeen: () => seen, remember: () => { seen = true; }, openSection() {} });
  guide.showWelcome(false);
  assert.equal('firstRun' in nodes['open-guide'].dataset, true);
  nodes['open-guide'].onclick();
  assert.equal(nodes['quick-guide'].open, true);
  nodes['close-guide'].onclick();
  assert.equal('firstRun' in nodes['open-guide'].dataset, false);
  assert.equal(seen, true);
  guide.showWelcome(false);
  assert.equal('firstRun' in nodes['open-guide'].dataset, false, 'Seen guide stays quiet');
  const returning = guideDocument();
  initializeQuickGuide({ document: returning.document, hasSeen: () => false, remember() {}, openSection() {} }).showWelcome(true);
  assert.equal('firstRun' in returning.nodes['open-guide'].dataset, false, 'A saved desk needs no first-visit cue');
});
