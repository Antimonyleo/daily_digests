import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const template=readFileSync(new URL('../templates/run.html.j2',import.meta.url),'utf8');
function runControls(fetch) {
  const context=vm.createContext({fetch,URL,JSON,startPending:false,stopBeforeStart:false,
    brewButton:{disabled:false},choicePanel:{},progressPanel:{},cancelButton:{},detailEl:{},
    form:{},newBrewLink:{},RESTART_URL:'/run',RESUME_URL:'/run?resume=1',RUN_ID:'fixture',CSRF_TOKEN:'fixture',
    FormData:class {get(){return 'usual';}},window:{location:{href:'http://localhost/run'},history:{replaceState(){}}},
    connectStream:()=>{},showError:message=>{context.error=message;}});
  for(const name of ['startBrew','stopBrew']) {
    const match=template.match(new RegExp(`  async function ${name}\\(\\) {[^]*?\\n  }`));
    assert.ok(match,`actual template ${name}`);vm.runInContext(match[0],context);
  }
  return context;
}

test('Stop during a pending start is sent after confirmation, never as an unknown run', async () => {
  let confirm;
  const started=new Promise(resolve=>{confirm=resolve;});const paths=[];
  const controls=runControls(async path=>{
    paths.push(path);
    return path==='/run/start'?started:{ok:true,json:async()=>({message:'Stopping safely'})};
  });
  const pending=controls.startBrew();await controls.stopBrew();
  assert.deepEqual(paths,['/run/start']);assert.equal(controls.stopBeforeStart,true);
  confirm({ok:true});await pending;
  assert.deepEqual(paths,['/run/start','/run/cancel?run_id=fixture']);assert.equal(controls.error,undefined);
  assert.equal(controls.cancelButton.textContent,'Stop requested');
});

test('a failed start does not send a cancel request or claim the brew was stopped', async () => {
  let reject;const paths=[];const started=new Promise((_resolve,fail)=>{reject=fail;});
  const controls=runControls(async path=>{paths.push(path);return started;});
  const pending=controls.startBrew();await controls.stopBrew();reject(Error('offline'));await pending;
  assert.deepEqual(paths,['/run/start']);assert.match(controls.error,/Could not confirm.*offline/);
  assert.equal(controls.startPending,false);
});
