import test from 'node:test';
import assert from 'node:assert/strict';
import {extractReadingView,readingViewDocument,validateReadingView} from './reading-view.mjs';

const companion={title:'How the molecular hinge works',url:'https://example.org/commentary',publicationLabel:'News & Views'};
const fixture=()=>({version:4,date:'2026-10-22',savedAt:'2026-10-22T12:00:00Z',profileName:'Reader',notes:[],
 items:[{section:'Research',title:'A molecular hinge',url:'https://example.org/research',meta:'Research article',
  summary:'A measured result.',reason:'DNA origami',opportunity:'',uncertainty:'',details:'',companions:[companion]}]});

test('cached reading preserves each companion label and link beneath its research paper',()=>{
 const text={'.title-link':'A molecular hinge','.meta':'Research article','.summary-primary .summary-text, .summary':'A measured result.'};
 const child={querySelector:selector=>selector==='.publication-label'?{textContent:companion.publicationLabel}:selector==='a[href]'?{textContent:companion.title,getAttribute:()=>companion.url}:null};
 const item={dataset:{section:'Research'},closest:()=>null,
  querySelector:selector=>selector in text?{textContent:text[selector],getAttribute:()=> 'https://example.org/research'}:null,
  querySelectorAll:selector=>selector==='.related-commentaries li'?[child]:[]};
 const document={querySelector:selector=>selector==='.digest-date'?{textContent:'2026-10-22'}:selector==='#reading-personalization'?{textContent:'{"name":"Reader","notes":[]}'}:null,
  querySelectorAll:selector=>selector==='.item[data-section]'?[item]:[]};
 const view=extractReadingView(document,new Date('2026-10-22T12:00:00Z'));
 assert.deepEqual(view.items[0].companions,[companion]);
 const html=readingViewDocument(view);
 assert.match(html,/News &amp; Views/);
 assert.match(html,/href="https:\/\/example.org\/commentary"/);
 assert.equal((html.match(/<article>/g)||[]).length,1,'A companion must remain inside its primary paper, not become another article');
 assert.ok(html.indexOf(companion.title)<html.indexOf('</article>'));
});

test('older reading copies regenerate from SQLite instead of silently hiding new companion evidence',()=>{
 const legacy=fixture();legacy.version=3;delete legacy.items[0].companions;
 assert.equal(validateReadingView(legacy),null);
});

test('cached companion links are bounded inert display data',()=>{
 assert.ok(validateReadingView(fixture()));
 for(const children of [null,{},Array(51).fill(companion),[{...companion,url:'javascript:alert(1)'}],
  [{...companion,url:'https://user:secret@example.org/commentary'}],[{...companion,html:'<script>bad()</script>'}],
  [{...companion,title:''}],[{...companion,publicationLabel:'x'.repeat(81)}]]){
  const value=fixture();value.items[0].companions=children;
  assert.equal(validateReadingView(value),null);
 }
 const value=fixture();value.items[0].companions=[{...companion,title:'<script>bad()</script>',publicationLabel:'<img src=x>'}];
 const html=readingViewDocument(value);
 assert.match(html,/&lt;script&gt;bad\(\)&lt;\/script&gt;/);
 assert.match(html,/&lt;img src=x&gt;/);
 assert.doesNotMatch(html,/<script|<img/);
});
