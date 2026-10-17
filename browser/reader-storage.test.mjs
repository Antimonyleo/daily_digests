import test from 'node:test';
import assert from 'node:assert/strict';
import {createSnapshotWriter, capturePrivateSnapshot} from './storage.mjs';

test('scheduled saves coalesce without delaying the action, and explicit flush is durable', async()=>{
  let generation=1,saves=0;
  const writer=createSnapshotWriter({generation:()=>generation,capture:()=>String(generation),save:async()=>{saves++;}});
  writer.schedule({delayMs:1000});generation++;
  writer.schedule({delayMs:1000});
  assert.equal(saves,0);assert.equal(writer.status().dirty,true);
  assert.equal(await writer.flush({force:true}),true);
  assert.equal(saves,1);assert.equal(writer.status().durable,true);
});

test('unchanged file versions reuse encoding while changed/deleted files match a full export',()=>{
  const cache=new Map();let reads=[];
  const bytes={'data/db':new Uint8Array([1]),'data/vectors':new Uint8Array([2,3])};
  const read=name=>{reads.push(name);return bytes[name];};
  const names=Object.keys(bytes),versions={'data/db':'0:1','data/vectors':'0:0'};
  capturePrivateSnapshot(names,read,{cache,versions});reads=[];
  bytes['data/db']=new Uint8Array([4]);versions['data/db']='0:2';
  const incremental=capturePrivateSnapshot(names,read,{cache,versions}).payload;
  assert.deepEqual(reads,['data/db']);
  assert.equal(incremental,capturePrivateSnapshot(names,read).payload);
  const deleted=capturePrivateSnapshot(['data/db'],read,{cache,versions}).payload;
  assert.equal(deleted,capturePrivateSnapshot(['data/db'],read).payload);
  assert.equal(cache.has('data/vectors'),false);
});
