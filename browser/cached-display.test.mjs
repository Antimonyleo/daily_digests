import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(process.env.DAILYDIGEST_CACHED_SHELL_SOURCE||new URL('./app.mjs',import.meta.url),'utf8').match(/export function mountCachedDisplayMenu\([^]*?\n\}/)[0]
  .replace('export function','function');
function setup({stored={}, storage=true, loaded=true}={}) {
  const nodes=[],window=new EventTarget(),documentTarget=new EventTarget(),saved={...stored};
  let observers=0,renders=0;
  class Element extends EventTarget {
    constructor(tag){super();this.tag=tag;nodes.push(this);this.children=[];this.attributes={};this.removed=false;this.open=false;this.style={setProperty:(name,value)=>{this[name]=value;}};}
    setAttribute(name,value){this.attributes[name]=value;} getAttribute(name){return this.attributes[name];}
    append(...children){this.children.push(...children);} remove(){this.removed=true;} focus(){this.focused=true;}
    contains(target){return target===this||this.children.some(child=>child===target||child.contains?.(target));}
  }
  const root={dataset:{theme:'light'}};
  const document=Object.assign(documentTarget,{documentElement:root,defaultView:window,createElement:tag=>new Element(tag)});
  const localStorage=storage?{getItem:key=>saved[key]??null,setItem:(key,value)=>{saved[key]=value;}}
    :{getItem(){throw Error('blocked');},setItem(){throw Error('blocked');}};
  Object.assign(window,{AbortController,innerHeight:900,localStorage,ResizeObserver:class{constructor(callback){this.callback=callback;}observe(){observers++;this.callback();}disconnect(){observers--;}}});
  const context=vm.createContext({String,Math});
  vm.runInContext(source,context);
  let placed=null;
  // The open saved page; a frame still loading has no reading document yet.
  const scrolled=[];
  const page={dataset:{theme:'light',display:'comfortable'},hasAttribute:name=>name in {'data-display':1},
    ownerDocument:{defaultView:{scrollY:1000,scrollTo:(x,y)=>scrolled.push(y)}},querySelectorAll:()=>articles};
  // Halfway through a 200px card that compact view shortens to 100px.
  const articles=[{getBoundingClientRect:()=>page.dataset.display==='compact'?{top:-120,bottom:-20,height:100}:{top:-100,bottom:100,height:200}}];
  const frame={ownerDocument:document,after(node){placed=node;},getBoundingClientRect:()=>({bottom:852}),
    get contentDocument(){return loaded?{documentElement:page}:{documentElement:{dataset:{},hasAttribute:()=>false}};}};
  const mounted=context.mountCachedDisplayMenu(frame,()=>{renders++;});
  const byId=id=>nodes.find(node=>node.id===id);
  return {mounted,root,saved,page,scrolled,placed:()=>placed,byId,document,renders:()=>renders,observerCount:()=>observers};
}
test('saved page gets a Display menu with the native theme and density preferences',()=>{
  const state=setup({stored:{'dailydigest-display':'compact'}});
  assert.equal(state.placed().id,'cached-display-menu');
  assert.equal(state.byId('cached-display-toggle').textContent,'Display');
  assert.equal(state.root.dataset.display,'compact');
  assert.equal(state.byId('cached-theme-toggle').getAttribute('aria-pressed'),'false');
  assert.equal(state.byId('cached-display-toggle-compact').getAttribute('aria-pressed'),'true');
  assert.equal(state.renders(),0,'mounting alone must not re-render the saved page');
});
test('changing theme or density saves the native keys and restyles the open page in place',()=>{
  const state=setup();
  state.byId('cached-theme-toggle').dispatchEvent(new Event('click'));
  assert.equal(state.root.dataset.theme,'dark');assert.equal(state.saved['dailydigest-theme'],'dark');
  assert.equal(state.byId('cached-theme-toggle').getAttribute('aria-pressed'),'true');
  state.byId('cached-display-toggle-compact').dispatchEvent(new Event('click'));
  assert.equal(state.root.dataset.display,'compact');assert.equal(state.saved['dailydigest-display'],'compact');
  assert.deepEqual({...state.page.dataset},{theme:'dark',display:'compact'});
  assert.equal(state.renders(),0,'re-rendering would lose the scroll position and open Details');
});
test('compact view keeps the reader halfway through the same card',()=>{
  const state=setup();
  state.byId('cached-display-toggle-compact').dispatchEvent(new Event('click'));
  assert.deepEqual(state.scrolled,[930],'card top moves from -120 to -50, half of its new height');
  state.byId('cached-theme-toggle').dispatchEvent(new Event('click'));
  assert.ok(state.scrolled.slice(1).every(y=>y===0),'a theme change leaves the position alone');
});
test('a change before the saved page has loaded renders it with the new preference',()=>{
  const state=setup({loaded:false});
  state.byId('cached-theme-toggle').dispatchEvent(new Event('click'));
  assert.equal(state.root.dataset.theme,'dark');assert.equal(state.renders(),1);
});
test('blocked storage still changes this view without throwing',()=>{
  const state=setup({storage:false});
  assert.equal(state.root.dataset.display,'comfortable');
  state.byId('cached-display-toggle-compact').dispatchEvent(new Event('click'));
  assert.equal(state.root.dataset.display,'compact');assert.equal(state.page.dataset.display,'compact');
});
test('the menu sits above the shell footer and Escape closes it',()=>{
  const state=setup();const menu=state.placed();
  assert.equal(menu['--cached-display-bottom'],'64px');
  menu.open=true;
  const escape=Object.assign(new Event('keydown'),{key:'Escape'});
  menu.dispatchEvent(escape);
  assert.equal(menu.open,false);assert.equal(state.byId('cached-display-toggle').focused,true);
});
test('dispose removes the menu once and stops listening',()=>{
  const state=setup();state.mounted.dispose();state.mounted.dispose();
  assert.equal(state.placed().removed,true);assert.equal(state.observerCount(),0);
  state.byId('cached-theme-toggle').dispatchEvent(new Event('click'));
  assert.equal(state.renders(),0,'a disposed menu must not re-render over the full desk');
  assert.equal(state.page.dataset.theme,'light','a disposed menu must not restyle the full desk');
});
