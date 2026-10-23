import test from 'node:test';
import assert from 'node:assert/strict';
import { DeferredWorker, validateReadingView, readingViewDocument, readingViewRoute, extractReadingView } from './reading-view.mjs';

const fixture = () => ({ version: 5, date: '2026-10-15', savedAt: '2026-10-15T12:00:00.000Z', profileName:'Ada', notes:['A fictional note.'],
  items: [{ section: 'Research', title: 'A <new> result', url: 'https://example.org/paper',
    meta: 'Nature · 2026-10-15', summary: 'A finding & its limits.', reason: 'Matches your interests',
    opportunity: '', uncertainty: '', details: '', why:'', caveat:'' }] });

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
  for (const value of [null, {}, {...fixture(), version:1}, {...fixture(), version:2}, {...fixture(), date:'today'},
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
  data.items[0].details='<img src=x onerror="steal()">';
  data.items[0].uncertainty='<iframe>uncertain</iframe>';
  const html=readingViewDocument(data, 'dark');
  assert.match(html, /&lt;script&gt;steal\(\)&lt;\/script&gt;/);
  assert.match(html, /script-src 'none'/);
  assert.match(html, /data-theme="dark"/);
  assert.match(html, /noopener noreferrer/);
  assert.doesNotMatch(html, /<script|<form|<button|<iframe|<img/);
  assert.match(html, /&lt;img src=x onerror=&quot;steal\(\)&quot;&gt;/);
  assert.match(html, /&lt;iframe&gt;uncertain&lt;\/iframe&gt;/);
  for (const url of ['javascript:alert(1)','data:text/html,bad','/setup','https://user:secret@example.org/paper']) {
    const bad=fixture();bad.items[0].url=url;
    assert.equal(validateReadingView(bad), null);
  }
});

test('related reading remains optional bounded links, not executable markup or authoritative state', () => {
  const value=fixture();
  value.related={heading:'Related reading',context:'Outside the core selection; relevance needs checking.',
    links:[{title:'An <adjacent> paper',url:'https://example.org/related'}]};
  assert.ok(validateReadingView(value));
  const html=readingViewDocument(value);
  assert.match(html,/Outside the core selection; relevance needs checking/);
  assert.match(html,/An &lt;adjacent&gt; paper/);
  assert.match(html,/href="https:\/\/example.org\/related"/);
  for(const related of [null, {...value.related,snapshot:'private'}, {...value.related,links:[]},
    {...value.related,links:Array(51).fill(value.related.links[0])},
    {...value.related,links:[{...value.related.links[0],grade:70}]},
    {...value.related,links:[{title:'Unsafe',url:'javascript:alert(1)'}]},
    {...value.related,links:[{title:'Credentials',url:'https://user:secret@example.org'}]}]) {
    assert.equal(validateReadingView({...value,related}),null);
  }
  for(const key of ['details','uncertainty','why','caveat']){
    const invalid=fixture();delete invalid.items[0][key];
    assert.equal(validateReadingView(invalid),null,'Missing fidelity fields must regenerate from the saved desk');
    invalid.items[0][key]='x'.repeat(60001);assert.equal(validateReadingView(invalid),null);
  }
});

test('current cached layout promotes supplied context once and keeps section/date navigation inert', () => {
  const value=fixture();
  value.items[0].why='Compare assembly methods.';
  value.items[0].caveat='Only simulated <not validated>.';
  value.items[0].details='Abstract\nThe source abstract.';
  const html=readingViewDocument(value,'light','https://example.org/desk/reading-view.mjs');
  assert.match(html,/aria-label="In this digest"/);
  assert.match(html,/href="about:srcdoc#cached-section-0">Research <span>1<\/span>/);
  assert.match(html,/<time class="digest-date" datetime="2026-10-15">2026-10-15/);
  assert.equal((html.match(/Compare assembly methods\./g)||[]).length,1);
  assert.match(html,/Only simulated &lt;not validated&gt;/);
  assert.ok(html.indexOf('Compare assembly methods.')<html.indexOf('<details>'));
  assert.match(html,/font-src data:/);
  assert.match(html,/script-src 'none'/);
  assert.equal(validateReadingView({...value,version:4}),null,'The older disposable copy regenerates from SQLite');
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

test('opening controls preserves the edition in a saved reading view after midnight', () => {
  assert.equal(readingViewRoute(fixture()), '/?digest_id=2026-10-15');
  assert.equal(readingViewRoute(null), '/');
  assert.equal(readingViewRoute({...fixture(), date:'bad&redirect=/setup'}), '/');
});


test('reading extraction retains caveats, attribution and uncertainty rather than just a finding', () => {
  const content = {
    '.title-link': 'A fictional paper', '.meta': 'Example journal',
    '.summary-primary .summary-text, .summary': 'A promising result.',
    '.match-topic, .reason-line': 'Shown for engineering',
    '.topic-support, [data-reading-uncertainty]': 'Title only — relevance uncertain.',
    '.item-details-body': 'Caveat\nOnly evaluated in simulation.\nAbstract\nA fictional supplied abstract.\nLimitations\nNo field validation.\nAuthor-reported claims; not independently verified.',
  };
  const item = {dataset:{section:'Research'}, closest:()=>null,
    querySelector: selector => selector in content ? {textContent:content[selector],getAttribute:()=> 'https://example.org/paper'} : null};
  const document = {querySelector:selector=>selector==='.digest-date'?{textContent:'2026-10-15'}:selector==='#reading-personalization'?{textContent:'{"name":"Ada","notes":[]}'}:null,
    querySelectorAll:selector=>selector==='.item[data-section]'?[item]:[]};
  const view=extractReadingView(document,new Date('2026-10-15T12:00:00Z'));
  assert.equal(view.items[0].details,content['.item-details-body']);
  assert.equal(view.items[0].uncertainty,content['.topic-support, [data-reading-uncertainty]']);
  const html=readingViewDocument(view);
  assert.match(html,/Only evaluated in simulation/);
  assert.match(html,/No field validation/);
  assert.match(html,/Author-reported claims; not independently verified/);
  assert.match(html,/Title only — relevance uncertain/);
  assert.match(html,/<details><summary>Details<\/summary>/);
});


test('related-only reading and long related shelves keep a bounded usable cache', () => {
  const links=Array.from({length:73},(_,i)=>({textContent:`Related paper ${i}`,getAttribute:()=>`https://example.org/${i}`}));
  const related={querySelector:selector=>({textContent:selector==='h2, h3'?'Related reading':'Uncertain topic fit.'}),querySelectorAll:()=>links};
  const document={querySelector:selector=>selector==='.digest-date'?{textContent:'2026-10-15'}:selector==='#related-reading'?related:selector==='#reading-personalization'?{textContent:'{"name":"Ada","notes":[]}'}:null,querySelectorAll:()=>[]};
  const view=extractReadingView(document,new Date('2026-10-15T12:00:00Z'));
  assert.ok(view,'A shelf-only digest must still open without Python');
  assert.deepEqual(view.items,[]);
  assert.equal(view.related.links.length,50);
  assert.equal(view.related.total,73);
  const html=readingViewDocument(view,'light','https://example.org/desk/reading-view.mjs');
  assert.match(html,/Showing 50 of 73 related papers/);
  assert.match(html,/href="https:\/\/example.org\/desk\/#\/\?digest_id=2026-10-15" target="_blank" rel="noopener noreferrer">Open controls/);
  assert.equal(validateReadingView({...view,related:{...view.related,total:49}}),null);
  assert.equal(validateReadingView({...view,related:{...view.related,total:Infinity}}),null);
});


test('cached entry uses the reader name and current local greeting with shared surface styles', () => {
  const value=fixture();value.profileName='Ada <script>';value.notes=['Tiny fact — Fictional safe note.'];
  const html=readingViewDocument(value,'dark','https://example.org/desk/reading-view.mjs',new Date(2026,9,22,19));
  assert.match(html,/Good evening, Ada &lt;script&gt;/);
  assert.match(html,/design-tokens\.css/);
  assert.match(html,/reading-surface\.css/);
  assert.doesNotMatch(html,/Your reading desk<\/h1>|background:#f6f7f4/);
});


test('cached personalization rejects missing, unbounded and non-string data', () => {
  for (const change of [{profileName:undefined},{profileName:'x'.repeat(201)},
    {notes:undefined},{notes:Array(33).fill('x')},{notes:['x'.repeat(2001)]},{notes:[{}]}]) {
    assert.equal(validateReadingView({...fixture(),...change}),null);
  }
  const value=fixture();value.profileName='<img src=x onerror=bad()>';value.notes=['<script>bad()</script>'];
  const html=readingViewDocument(value);
  assert.match(html,/&lt;img src=x onerror=bad\(\)&gt;/);
  assert.doesNotMatch(html,/<script|<img/);
});


test('shared CSS is inlined without opaque-frame requests or weaker script isolation', () => {
  const css=':root{--bg:green}';
  const html=readingViewDocument(fixture(),'dark','https://example.test/desk/reading-view.mjs',new Date(),css);
  assert.doesNotMatch(html,/<link/);assert.match(html,/script-src 'none'/);assert.match(html,/--bg:green/);
  assert.doesNotMatch(readingViewDocument(fixture(),'light',undefined,undefined,'</style><script>bad()</script>'),/<script>/);
});
