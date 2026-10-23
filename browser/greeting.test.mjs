import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const source=readFileSync(new URL('../templates/_greeting.js.j2',import.meta.url),'utf8');
function setup(saved){
  let now=Date.UTC(2026,9,23,9),sequence=0;const timers=new Map(),values=new Map(saved||[]);
  class Clock extends Date {constructor(value=now){super(value);}static now(){return now;}getHours(){return this.getUTCHours();}getFullYear(){return this.getUTCFullYear();}getMonth(){return this.getUTCMonth();}getDate(){return this.getUTCDate();}}
  const heading={dataset:{profileName:'Ada'},textContent:''},document=new EventTarget(),window=new EventTarget();
  Object.assign(document,{hidden:false,getElementById:()=>heading});
  runInNewContext(source,{document,window,Date:Clock,JSON,Number,Math,localStorage:{getItem:key=>values.get(key)||null,setItem:(key,value)=>values.set(key,value)},setInterval:fn=>{timers.set(++sequence,fn);return sequence;},clearInterval:key=>timers.delete(key)});
  return {heading,window,values,timers,advance(ms){now+=ms;for(const fn of timers.values())fn();}};
}
test('fresh reading greets by local time and refreshes do not inflate visits',()=>{
  const first=setup();assert.equal(first.heading.textContent,'Good morning, Ada');
  const next=setup([...first.values]);assert.equal(JSON.parse(next.values.get('dailydigest-reading-visits')).todayVisits,1);
});
test('active reading changes the greeting after time spent reading',()=>{
  const state=setup();for(let i=0;i<10;i++)state.advance(60000);assert.equal(state.heading.textContent,'Enjoying the deep dive, Ada');
});
test('BFCache return resumes the greeting clock without adding refresh visits',()=>{
  const state=setup();state.window.dispatchEvent(new Event('pagehide'));assert.equal(state.timers.size,0);
  const restored=new Event('pageshow');Object.defineProperty(restored,'persisted',{value:true});state.window.dispatchEvent(restored);assert.equal(state.timers.size,1);
  for(let i=0;i<10;i++)state.advance(60000);assert.equal(state.heading.textContent,'Enjoying the deep dive, Ada');
  assert.equal(JSON.parse(state.values.get('dailydigest-reading-visits')).todayVisits,1);
});
