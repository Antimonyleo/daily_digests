import test from 'node:test';
import assert from 'node:assert/strict';
import {createTabHandoff} from './tab-handoff.mjs';

function channels() {
  const clients=[];
  return () => {
    const channel=new EventTarget();
    channel.postMessage=data=>{for(const peer of clients)if(peer!==channel)queueMicrotask(()=>peer.dispatchEvent(new MessageEvent('message',{data})));};
    clients.push(channel);return channel;
  };
}
const turn=()=>new Promise(done=>setImmediate(done));

test('ownership changes only after the first tab has saved and released', async () => {
  const open=channels();let release,owned=true;
  const saved=new Promise(done=>{release=done;});
  createTabHandoff({channel:open(),id:'owner',owns:()=>owned,release:async()=>{await saved;owned=false;}});
  const second=createTabHandoff({channel:open(),id:'second',owns:()=>false,timeoutMs:1000});
  let result;
  const switching=second.request().then(value=>{result=value;});
  await turn();assert.equal(result,undefined);assert.equal(owned,true);
  release();await switching;assert.equal(result.ok,true);assert.equal(owned,false);
});

test('failed saving or an active brew keeps ownership and returns actionable feedback', async () => {
  const open=channels();let owned=true;
  createTabHandoff({channel:open(),id:'owner',owns:()=>owned,release:async()=>{throw Error('Finish or stop the brew in the other tab, then try again.');}});
  const second=createTabHandoff({channel:open(),id:'second',owns:()=>false,timeoutMs:1000});
  const result=await second.request();assert.equal(result.ok,false);assert.match(result.message,/Finish or stop/);assert.equal(owned,true);
});

test('simultaneous requests cannot run two release operations', async () => {
  const open=channels();let count=0,release;
  const saved=new Promise(done=>{release=done;});
  createTabHandoff({channel:open(),id:'owner',owns:()=>true,release:async()=>{count++;await saved;}});
  const a=createTabHandoff({channel:open(),id:'a',owns:()=>false,timeoutMs:1000});
  const b=createTabHandoff({channel:open(),id:'b',owns:()=>false,timeoutMs:1000});
  const first=a.request(),second=b.request();await turn();assert.equal(count,1);
  assert.equal((await second).ok,false);release();assert.equal((await first).ok,true);
});

test('unresponsive or older tabs time out without forcing the storage lock', async () => {
  const open=channels();const second=createTabHandoff({channel:open(),id:'second',owns:()=>false,timeoutMs:10});
  const result=await second.request();assert.equal(result.ok,false);assert.match(result.message,/other.*tab/i);
});
