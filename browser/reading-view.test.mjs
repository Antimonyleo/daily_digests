import test from 'node:test';
import assert from 'node:assert/strict';
import { DeferredWorker, validateReadingView, readingViewDocument } from './reading-view.mjs';

const fixture = () => ({ version: 1, date: '2026-10-15', savedAt: '2026-10-15T12:00:00.000Z',
  items: [{ section: 'Research', title: 'A <new> result', url: 'https://example.org/paper',
    meta: 'Nature · 2026-10-15', summary: 'A finding & its limits.', reason: 'Matches your interests',
    opportunity: '' }] });

test('reading a cached desk does not create a Python worker; first action creates just one', () => {
  let created = 0;
  class Native extends EventTarget { postMessage(data) { this.sent = data; } }
  const worker = new DeferredWorker(() => { created++; return new Native(); });
  let received;
  worker.addEventListener('message', event => { received = event.data; });
  assert.equal(created, 0);
  worker.postMessage({type:'init'});
  worker.postMessage({type:'request'});
  assert.equal(created, 1);
  worker.instance.dispatchEvent(new MessageEvent('message', {data:'ready'}));
  assert.equal(received, 'ready');
});

test('reading cache is bounded structured content, never HTML or a private snapshot', () => {
  assert.deepEqual(validateReadingView(fixture()), fixture());
  for (const value of [null, {}, {...fixture(), version:2}, {...fixture(), date:'today'},
    {...fixture(), items:[]}, {...fixture(), credentials:'secret'},
    {...fixture(), items:[{...fixture().items[0], html:'<script>bad()</script>'}]},
    {...fixture(), items:[{...fixture().items[0], constructor:'unexpected private value'}]},
    {...fixture(), items:Array(501).fill(fixture().items[0])},
    {...fixture(), items:[{...fixture().items[0], summary:'a'.repeat(1000001)}]}]) {
    assert.equal(validateReadingView(value), null);
  }
});

test('a moved desk cannot restart its obsolete worker or write stale state', () => {
  let created = 0, stopped = 0;
  class Native extends EventTarget { postMessage() {} terminate() { stopped++; } }
  const worker = new DeferredWorker(() => { created++; return new Native(); });
  worker.postMessage({type:'init'});
  worker.terminate();
  assert.throws(() => worker.postMessage({type:'request'}), /moved to another tab/);
  assert.equal(created, 1); assert.equal(stopped, 1);
});

test('read-only document escapes content and permits only HTTP article links', () => {
  const data=fixture();data.items[0].title='<script>steal()</script>';
  const html=readingViewDocument(data, 'dark');
  assert.match(html, /&lt;script&gt;steal\(\)&lt;\/script&gt;/);
  assert.match(html, /script-src 'none'/);
  assert.match(html, /data-theme="dark"/);
  assert.match(html, /noopener noreferrer/);
  assert.doesNotMatch(html, /<script|<form|<button|<iframe|onerror=/);
  for (const url of ['javascript:alert(1)','data:text/html,bad','/setup','https://user:secret@example.org/paper']) {
    const bad=fixture();bad.items[0].url=url;
    assert.equal(validateReadingView(bad), null);
  }
});

test('saved opportunity facts are explicitly historical rather than rechecked eligibility', () => {
  const data=fixture();data.items[0].opportunity='Deadline: 2026-10-20 · Needs verification';
  assert.match(readingViewDocument(data), /Verify current deadlines and eligibility at the official source/);
});

test('saved reading views retain optional publisher image links without contacting image hosts', () => {
  const value=fixture();Object.assign(value.items[0],{figureURL:'https://publisher.test/a.png',figureLabel:'Graphical abstract'});
  assert.ok(validateReadingView(value));
  const html=readingViewDocument(value);
  assert.match(html,/Graphical abstract · open publisher image/);
  assert.doesNotMatch(html,/<img|<script/);
  value.items[0].figureURL='javascript:alert(1)';assert.equal(validateReadingView(value),null);
});
