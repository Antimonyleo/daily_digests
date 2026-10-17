import test from 'node:test';
import assert from 'node:assert/strict';
import {createLinkedBackup, createBackupScheduler} from './linked-backup.mjs';

test('a committed file remains writable if remembering its revision fails', async () => {
  let content = '', saved, fail = false;
  const handle = {name:'desk.json',queryPermission:async()=> 'granted',getFile:async()=>({size:content.length,text:async()=>content}),
    createWritable:async()=>{let pending;return {write:async body=>{pending=body;},close:async()=>{content=pending;},abort:async()=>{}};}};
  const backup = createLinkedBackup({load:async()=>saved,store:async value=>{if(fail)throw Error('quota');saved=value;}});
  await backup.connect(handle);fail=true;
  await assert.rejects(backup.write('first copy'), /saved.*remember/i);
  assert.equal(content,'first copy');
  fail=false;await backup.write('second copy');
  assert.equal(content,'second copy');
});

function fixture(content = '') {
  let permission = 'granted', saved, prompts = 0, writes = 0, reads = 0, failWrite = false, locked = false;
  const handle = {name:'desk.json', queryPermission:async()=>permission,
    requestPermission:async()=>{ prompts++; return permission; },
    getFile:async()=>({size:content.length,lastModified:1,text:async()=>{reads++;return content;}}),
    createWritable:async options=>{
      assert.equal(options.mode,'exclusive');
      if (locked) throw new Error('File is already open');
      locked = true; let next;
      return {write:async body=>{if(failWrite)throw new Error('Disk full');next=body;},
        close:async()=>{content=next;writes++;locked=false;},abort:async()=>{locked=false;}};
    }};
  const backup=createLinkedBackup({load:async()=>saved,store:async value=>{saved=value;}});
  return {backup,handle,get content(){return content;},set content(value){content=value;},
    get prompts(){return prompts;},get writes(){return writes;},get reads(){return reads;},
    set permission(value){permission=value;},set failWrite(value){failWrite=value;},
    reload:()=>createLinkedBackup({load:async()=>saved,store:async value=>{saved=value;}})};
}

test('linking and startup checks never prompt, write, or read the full backup',async()=>{
  const f=fixture('existing reader data');
  await f.backup.connect(f.handle);
  const next=f.reload();await next.open();
  const info=await next.inspect();
  assert.equal(info.name,'desk.json');assert.equal(info.exists,true);
  assert.equal(f.writes,0);assert.equal(f.prompts,0);assert.equal(f.reads,0);
  f.permission='prompt';
  assert.equal((await next.inspect()).permission,'prompt');assert.equal(f.prompts,0);
  assert.equal(await next.requestAccess(),'prompt');assert.equal(f.prompts,1);
});

test('an empty file can be saved and reopened; failures keep the last good backup',async()=>{
  const f=fixture();await f.backup.connect(f.handle);
  await f.backup.write('first copy');assert.equal(f.content,'first copy');
  const next=f.reload();await next.open();await next.write('second copy');
  assert.equal(f.content,'second copy');f.failWrite=true;
  await assert.rejects(next.write('broken copy'),/Disk full/);
  assert.equal(f.content,'second copy');assert.equal(f.writes,2);
});

test('an existing file and edits from another browser cannot be silently overwritten',async()=>{
  const f=fixture('other browser');await f.backup.connect(f.handle);
  await assert.rejects(f.backup.write('my desk'),/changed|existing/i);
  assert.equal(f.content,'other browser');
  const copy=await f.backup.read();await f.backup.accept(copy.revision);
  await f.backup.write('accepted edit');assert.equal(f.content,'accepted edit');
  f.content='new changes elsewhere';
  await assert.rejects(f.backup.write('stale desk'),/changed/i);
  assert.equal(f.content,'new changes elsewhere');
});

test('revoked permission and missing files leave browser data and backup untouched',async()=>{
  const f=fixture();await f.backup.connect(f.handle);await f.backup.write('good copy');
  f.permission='denied';
  await assert.rejects(f.backup.write('bad copy'),/permission|access/i);
  assert.equal(f.content,'good copy');assert.equal(f.prompts,0);
  f.permission='granted';f.handle.getFile=async()=>{throw Object.assign(new Error('missing'),{name:'NotFoundError'});};
  assert.equal((await f.backup.inspect()).exists,false);
  await assert.rejects(f.backup.write('bad copy'),/missing/);
});

test('disconnect forgets the handle without deleting the file',async()=>{
  const f=fixture('keep me');await f.backup.connect(f.handle);await f.backup.disconnect();
  assert.equal((await f.backup.inspect()).linked,false);assert.equal(f.content,'keep me');
  const next=f.reload();await next.open();assert.equal((await next.inspect()).linked,false);
});

test('a late startup read cannot forget a file the reader just selected',async()=>{
  let release;
  const loaded=new Promise(resolve=>{release=resolve;});
  const f=fixture();
  const backup=createLinkedBackup({load:()=>loaded,store:async()=>{}});
  const opening=backup.open();
  await backup.connect(f.handle);
  release(null);await opening;
  assert.equal((await backup.inspect()).linked,true);
});

test('changing the linked file blocks a copy until selection has finished', async () => {
  const old = fixture(), next = fixture();
  await old.backup.connect(old.handle);
  let release;
  const permission = new Promise(resolve => {release = resolve;});
  next.handle.name = 'new-desk.json';
  next.handle.queryPermission = () => permission;
  const connecting = old.backup.connect(next.handle);
  const blocked = assert.rejects(old.backup.write('wrong destination'), /changing|finish/i);
  release('granted'); await Promise.all([connecting, blocked]);
  await old.backup.write('new desk');
  assert.equal(old.content, '');
  assert.equal(next.content, 'new desk');
  assert.equal((await old.backup.inspect()).name, 'new-desk.json');
});

test('a queued file selection blocks writes while its settings commit is pending', async () => {
  const old = fixture(), next = fixture();
  let release, entered;
  const pending = new Promise(resolve => {release = resolve;});
  const saving = new Promise(resolve => {entered = resolve;});
  let calls = 0;
  const backup = createLinkedBackup({load:async()=>null,store:async()=>{
    if (++calls === 2) {entered(); await pending;}
  }});
  await backup.connect(old.handle);
  const connecting = backup.connect(next.handle); await saving;
  const blocked = assert.rejects(backup.write('old desk'), /changing|finish/i);
  await assert.rejects(backup.disconnect(), /changing|finish/i);
  release(); await Promise.all([connecting, blocked]);
  await backup.write('new desk');
  assert.equal(old.content, ''); assert.equal(next.content, 'new desk');
});

test('disconnecting blocks a copy until forgetting the handle commits', async () => {
  const f = fixture();
  let release, entered;
  const pending = new Promise(resolve => {release = resolve;});
  const saving = new Promise(resolve => {entered = resolve;});
  const backup = createLinkedBackup({load:async()=>null,store:async value=>{
    if (value === null) {entered(); await pending;}
  }});
  await backup.connect(f.handle);
  const disconnecting = backup.disconnect(); await saving;
  const blocked = assert.rejects(backup.write('orphan copy'), /changing|finish/i);
  release(); await Promise.all([disconnecting, blocked]);
  assert.equal((await backup.inspect()).linked, false); assert.equal(f.content, '');
});

test('turning automatic copies off wins over an earlier slow settings write',async()=>{
  let release, started, calls=0;
  const slow=new Promise(resolve=>{release=resolve;});
  const saving=new Promise(resolve=>{started=resolve;});
  const f=fixture();
  const backup=createLinkedBackup({load:async()=>null,store:async()=>{
    if(++calls===2){started();await slow;}
  }});
  await backup.connect(f.handle);
  const on=backup.setAutomatic(true);await saving;
  const off=backup.setAutomatic(false);release();
  await Promise.all([on,off]);assert.equal((await backup.inspect()).auto,false);
});

test('two readers of the same revision cannot write the file concurrently',async()=>{
  const f=fixture();await f.backup.connect(f.handle);await f.backup.write('original');
  const other=f.reload();await other.open();
  const outcomes=await Promise.allSettled([f.backup.write('first'),other.write('second')]);
  assert.equal(outcomes.filter(value=>value.status==='fulfilled').length,1);
  assert.equal(outcomes.filter(value=>value.status==='rejected').length,1);
  assert.ok(['first','second'].includes(f.content));
});

test('automatic copies coalesce, wait for a brew to finish, and stop after an error',async()=>{
  let timer, busy=true,saves=0,errors=0;
  const scheduler=createBackupScheduler({save:async()=>{saves++;},canSave:()=>!busy,
    setTimer:fn=>{timer=fn;return 1;},clearTimer:()=>{timer=null;},onError:()=>{errors++;}});
  scheduler.changed();scheduler.changed();assert.equal(saves,0);
  await timer();assert.equal(saves,0);
  busy=false;await timer();assert.equal(saves,1);
  scheduler.changed();scheduler.cancel();assert.equal(timer,null);
  const broken=createBackupScheduler({save:async()=>{throw new Error('denied');},canSave:()=>true,
    setTimer:fn=>{timer=fn;return 1;},clearTimer:()=>{timer=null;},onError:()=>{errors++;}});
  broken.changed();const once=timer;timer=null;await once();assert.equal(errors,1);assert.equal(timer,null);
});
