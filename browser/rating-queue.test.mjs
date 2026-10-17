import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const create=runInNewContext(readFileSync(new URL('../templates/_rating_queue.js.j2',import.meta.url),'utf8')+';createRatingQueue');
test('rapid ratings paint immediately, send in order and ignore stale completions',async()=>{
  const sent=[],paint=[],done=[];let release;
  const queue=create({initial:null,send:grade=>{sent.push(grade);return grade===70?new Promise(r=>{release=r;}):Promise.resolve({});},select:grade=>paint.push(grade),success:(_r,g)=>done.push(g),failure:()=>assert.fail('unexpected error'),busy(){}});
  const a=queue(70),b=queue(100);assert.deepEqual(paint,[70,100]);
  await new Promise(r=>setImmediate(r));assert.deepEqual(sent,[70]);
  release({});await Promise.all([a,b]);assert.deepEqual(sent,[70,100]);assert.deepEqual(done,[100]);
});
test('a failed latest rating restores the last confirmed rating, not a stale optimistic state',async()=>{
  const paint=[],errors=[];
  const queue=create({initial:10,send:grade=>grade===70?Promise.resolve({}):Promise.reject(Error('offline')),select:g=>paint.push(g),success(){},failure:e=>errors.push(e.message),busy(){}});
  await Promise.all([queue(70),queue(100)]);assert.deepEqual(paint,[70,100,70]);assert.deepEqual(errors,['offline']);
});


test('the frame drain includes intents still queued behind an in-flight rating',async()=>{
  const window={addEventListener(){}};
  const make=runInNewContext(readFileSync(new URL('../templates/_rating_queue.js.j2',import.meta.url),'utf8')+';createRatingQueue',{window});
  let first,second;
  const queue=make({initial:null,send:g=>new Promise(resolve=>{if(g===70)first=resolve;else second=resolve;}),select(){},success(){},failure(){},busy(){}});
  const a=queue(70),b=queue(100);let drained=false;
  const flushing=window.ddFlushRatings().then(()=>{drained=true;});
  await new Promise(resolve=>setImmediate(resolve));assert.equal(window.ddHasPendingRatings(),true);
  first({});await a;await new Promise(resolve=>setImmediate(resolve));assert.equal(drained,false);
  second({});await Promise.all([b,flushing]);assert.equal(window.ddHasPendingRatings(),false);
});
