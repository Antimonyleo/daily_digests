import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(process.env.DAILYDIGEST_CACHED_SHELL_SOURCE||new URL('./app.mjs',import.meta.url),'utf8').match(/export function mountCachedReadingPip\([^]*?\n\}/)[0]
  .replace('export function','function').replaceAll('import.meta.url',"'https://example.test/app.mjs'").replace('import(new URL(', 'loadModule(new URL(');
function setup({holdMarkup=false}={}) {
  const nodes=[],window=new EventTarget(),document={documentElement:{dataset:{}}};
  let mounts=0,unmounts=0,release,observers=0;
  class Element extends EventTarget {
    constructor(){super();nodes.push(this);this.removed=false;this.dataset={};this.style={setProperty:(name,value)=>{this[name]=value;}};}
    setAttribute(){} remove(){this.removed=true;} append(){} querySelector(){return {focus(){}};}
  }
  Object.assign(window,{AbortController,Event,innerHeight:900,ResizeObserver:class{constructor(callback){this.callback=callback;}observe(){observers++;this.callback();}disconnect(){observers--;}},localStorage:{getItem:()=>null,setItem(){}},matchMedia:()=>Object.assign(new EventTarget(),{matches:true})});
  Object.assign(document,{defaultView:window,createElement:()=>new Element(),head:new Element(),querySelector:()=>new Element()});
  const context=vm.createContext({URL,Error,Promise,fetch:(_url,{signal})=>new Promise((done,fail)=>{
    release=()=>done({ok:true,text:async()=>'<trusted-component>'});
    signal.addEventListener('abort',()=>fail(Object.assign(Error('aborted'),{name:'AbortError'})),{once:true});
    if(!holdMarkup)release();
  }),loadModule:async()=>({mountPip:()=>{mounts++;return()=>unmounts++;}})});
  vm.runInContext(source,context);
  const mounted=context.mountCachedReadingPip({ownerDocument:document,after(){},getBoundingClientRect:()=>({bottom:800})},{date:'2026-10-22',notes:[]});
  return {mounted,nodes,window,style:nodes.find(node=>node.rel==='stylesheet'),stats:()=>({mounts,unmounts}),observerCount:()=>observers,release:()=>release()};
}
test('Pip mounts only after its shared stylesheet loads',async()=>{
  const state=setup();await new Promise(done=>setImmediate(done));
  assert.equal(state.stats().mounts,0,'Unstyled dimensions must not alter the saved dock position');
  state.style.dispatchEvent(new Event('load'));await state.mounted.ready;
  assert.equal(state.stats().mounts,1);state.mounted.dispose();
});
test('BFCache suspension retains Pip and a real page exit disposes it once',async()=>{
  const state=setup();state.style.dispatchEvent(new Event('load'));await state.mounted.ready;
  const suspended=new Event('pagehide');Object.defineProperty(suspended,'persisted',{value:true});
  state.window.dispatchEvent(suspended);
  assert.equal(state.stats().unmounts,0,'BFCache must preserve Pip for pageshow');
  state.window.dispatchEvent(new Event('pagehide'));state.mounted.dispose();
  assert.deepEqual(state.stats(),{mounts:1,unmounts:1});
  assert.equal(state.observerCount(),0);
});
test('early disposal settles stylesheet/markup waits and cannot resurrect Pip',async()=>{
  const state=setup({holdMarkup:true});state.mounted.dispose();state.mounted.dispose();
  await state.mounted.ready;state.release();state.style.dispatchEvent(new Event('load'));
  assert.deepEqual(state.stats(),{mounts:0,unmounts:0});
  assert.equal(state.style.removed,true);assert.equal(state.observerCount(),0);
});
test('failed shared stylesheet removes the incomplete Pip controls',async()=>{
  const state=setup();state.style.dispatchEvent(new Event('error'));
  await assert.rejects(state.mounted.ready,/Pip styles unavailable/);
  assert.equal(state.stats().mounts,0);assert.equal(state.style.removed,true);
  assert.equal(state.nodes.find(node=>node.id==='cached-pip').removed,true);
  assert.equal(state.nodes.find(node=>node.id==='cached-pip-toggle').removed,true);
});

test('default cached dock clears the shell footer without changing saved position',async()=>{
  const state=setup();const host=state.nodes.find(node=>node.id==='cached-pip');
  assert.equal(host['--cached-pip-bottom'],'118px');
  state.window.innerHeight=1000;state.window.dispatchEvent(new Event('resize'));
  assert.equal(host['--cached-pip-bottom'],'218px');
  state.mounted.dispose();await state.mounted.ready;
  assert.equal(state.observerCount(),0);
});
