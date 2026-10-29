import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';

const digest=readFileSync(new URL('../templates/digest_web.html.j2',import.meta.url),'utf8');
const saved=readFileSync(new URL('../templates/saved.html.j2',import.meta.url),'utf8')
  .replace('{% include "_countdown.js.j2" %}',readFileSync(new URL('../templates/_countdown.js.j2',import.meta.url),'utf8'));
function between(source,start,end) {
  const first=source.indexOf(start),last=source.indexOf(end,first+start.length);
  assert.ok(first>=0&&last>first,`Missing actual handler boundaries: ${start}`);
  return source.slice(first,last);
}
function control(dataset={}) {
  const classes=new Set();
  return {dataset,textContent:'',disabled:false,hidden:false,listeners:{},
    classList:{contains:name=>classes.has(name),add:(...names)=>names.forEach(name=>classes.add(name)),
      remove:(...names)=>names.forEach(name=>classes.delete(name)),toggle(name,value){
        const selected=value??!classes.has(name);if(selected)classes.add(name);else classes.delete(name);return selected;
      }},
    addEventListener(name,callback){this.listeners[name]=callback;},
    setAttribute(name,value){this[name]=String(value);},focus(){}};
}
function response({pending=false,status=200,...payload}={}) {
  return new Response(JSON.stringify(payload),{status,headers:{'content-type':'application/json',
    ...(pending?{'x-dailydigest-storage':'pending'}:{})}});
}

// These tests execute the shipped handlers and durability notification listener.
// The DOM double only provides controls; responses simulate a failed checkpoint
// independently of a successful in-memory mutation.
function digestControls({respond=async()=>response(),bookmarked=false,known=false,reason=false}={}) {
  const status=control(),bookmark=control(),bookmarkLabel=control(),bookmarkIcon=control();
  const knownButton=control(),knownLabel=control(),knownIcon=control(),chip=control({reason:'off_topic'});
  const overflowState=control(),overflowSave=control(),votes=['10','70','100'].map(grade=>control({grade}));
  bookmarkLabel.textContent=bookmarked?'Saved':'Save';knownLabel.textContent=known?'Known':'Mark known';
  overflowSave.textContent='Save for tomorrow';chip.classList.toggle('active',reason);
  bookmark.querySelector=selector=>({'.bookmark-label':bookmarkLabel,'.bookmark-icon':bookmarkIcon}[selector]);
  knownButton.querySelector=selector=>({'.known-label':knownLabel,'.known-icon':knownIcon}[selector]);
  const itemEl={dataset:{itemId:'42',currentGrade:'10',bookmarked:String(bookmarked),known:String(known)},
    classList:control().classList,
    querySelector:selector=>({'.save-state':status,'.bookmark-btn:not(.known-btn)':bookmark,'.known-btn':knownButton}[selector]),
    querySelectorAll:selector=>selector.includes('.reason-chip')?[chip]:[]};
  const overflowRow={dataset:{itemId:'43'},querySelector:()=>overflowState,querySelectorAll:()=>votes};
  const events={},toasts=[],requests=[];
  const allStatuses=[status,overflowState],allLabels=[bookmarkLabel,knownLabel,overflowSave];
  const context={itemEl,CSRF_TOKEN:'fixture',OVERFLOW_QUERY:'digest_id=fixture',Date,
    NEGATIVE_GRADES:new Set(['10','40']),
    fetch:async(url,options)=>{requests.push({url,method:options.method});return respond(url,options);},
    showToast:message=>toasts.push(message),updateRankingStatus(){},flash(){},
    window:{addEventListener:(name,callback)=>{events[name]=callback;}},
    document:{getElementById:id=>id==='overflow-save-btn'?overflowSave:null,
      querySelectorAll:selector=>{
        if(selector==='.overflow-row')return [overflowRow];
        if(selector.includes('data-storage-saved-label'))return allLabels.filter(label=>label.dataset.storageSavedLabel);
        if(selector.includes('.pending'))return allStatuses.filter(node=>node.classList.contains('pending'));
        return [];
      }}};
  const helpers=between(digest,'  function reasonFeedbackEnabled(', '  const pendingKey =');
  const commit=between(digest,'  window.addEventListener("dd-storage-committed"','  function flash(');
  const overflow=between(digest,'  const overflowSaveBtn =','  document.querySelectorAll(".item").forEach((itemEl) => {');
  const item=between(digest,'    const bookmarkBtn = itemEl.querySelector','    const rate = createRatingQueue(');
  const reasons=between(digest.slice(digest.indexOf('    const rate = createRatingQueue(')),
    '    itemEl.querySelectorAll(".reason-chip").forEach((chip) => {','\n  });\n\n  if (trainBtn)');
  runInNewContext(helpers+commit+overflow+item+reasons,context);
  return {status,itemEl,chip,bookmark,bookmarkLabel,knownButton,knownLabel,overflowState,overflowSave,votes,toasts,requests,
    commit:()=>events['dd-storage-committed'](),
    click:async node=>node.listeners.click()};
}

for(const removing of [false,true]) {
  test(`digest reason ${removing?'removal':'addition'} waits for the durable checkpoint`,async()=>{
    const ui=digestControls({reason:removing,respond:async()=>response({pending:true})});
    await ui.click(ui.chip);
    assert.equal(ui.chip.classList.contains('active'),!removing);
    assert.equal(ui.status.classList.contains('pending'),true);
    assert.match(ui.status.textContent,/saving browser copy/i);
    assert.doesNotMatch(ui.toasts.at(-1),/saved/i);
    ui.commit();assert.equal(ui.status.classList.contains('pending'),false);
    assert.equal(ui.status.textContent,'Saved in this browser');
  });
}

for(const kind of ['bookmark','known']) for(const initial of [false,true]) {
  test(`digest ${kind} ${initial?'clear':'set'} reports pending until persistence succeeds`,async()=>{
    const ui=digestControls({bookmarked:initial,known:initial,respond:async()=>response({pending:true})});
    const button=kind==='bookmark'?ui.bookmark:ui.knownButton,label=kind==='bookmark'?ui.bookmarkLabel:ui.knownLabel;
    await ui.click(button);
    assert.equal(button.disabled,false);
    assert.equal(ui.itemEl.dataset[kind==='bookmark'?'bookmarked':'known'],String(!initial));
    assert.equal(ui.status.classList.contains('pending'),true);
    assert.match(label.textContent,/saving/i);
    ui.commit();assert.equal(ui.status.textContent,'Saved in this browser');
    assert.equal(label.textContent,kind==='bookmark'?(initial?'Save':'Saved'):(initial?'Mark known':'Known'));
  });
}

test('failed bookmark/known/reason requests preserve the last confirmed state and remain retryable',async()=>{
  for(const kind of ['bookmark','knownButton','chip']) {
    const ui=digestControls({respond:async()=>response({status:503,detail:'Storage unavailable'})});
    await ui.click(ui[kind]);
    assert.equal(ui.itemEl.dataset.bookmarked,'false');assert.equal(ui.itemEl.dataset.known,'false');
    assert.equal(ui.chip.classList.contains('active'),false);assert.equal(ui[kind].disabled,false);
    assert.equal(ui.status.classList.contains('error'),true);
    const error=ui.status.textContent;ui.commit();assert.equal(ui.status.textContent,error);
  }
});

test('bulk save for tomorrow does not claim durable success before a checkpoint',async()=>{
  const ui=digestControls({respond:async()=>response({saved:4,pending:true})});
  await ui.click(ui.overflowSave);
  assert.match(ui.overflowSave.textContent,/saving/i);
  assert.doesNotMatch(ui.toasts.at(-1),/will compete|saved/i);
  ui.commit();assert.match(ui.overflowSave.textContent,/Saved for tomorrow/);
});

test('a rejected bulk save is retryable and does not show a success label',async()=>{
  const ui=digestControls({respond:async()=>response({status:503,detail:'Try again'})});
  await ui.click(ui.overflowSave);assert.equal(ui.overflowSave.disabled,false);
  assert.equal(ui.overflowSave.textContent,'Save for tomorrow');
});

for(const grade of ['10','70']) {
  test(`overflow grade ${grade} waits for its final mutation checkpoint`,async()=>{
    const ui=digestControls({respond:async()=>response({pending:true})});
    await ui.click(ui.votes.find(button=>button.dataset.grade===grade));
    assert.equal(ui.overflowState.classList.contains('pending'),true);
    assert.match(ui.overflowState.textContent,/saving browser copy/i);
    ui.commit();assert.equal(ui.overflowState.classList.contains('pending'),false);
    assert.equal(ui.overflowState.textContent,grade==='70'?'Saved · kept for tomorrow':'Saved');
    assert.equal(ui.requests.length,grade==='70'?2:1);
  });
}

test('a later durable pin also confirms the earlier pending overflow rating',async()=>{
  const ui=digestControls({respond:async url=>response({pending:url.startsWith('/vote/')})});
  await ui.click(ui.votes[1]);
  assert.equal(ui.overflowState.classList.contains('pending'),false);
  assert.equal(ui.overflowState.textContent,'Saved · kept for tomorrow');
});

test('overflow pin failure retains the rating and explains what can be retried',async()=>{
  const ui=digestControls({respond:async url=>url.startsWith('/vote/')?response({pending:true}):response({status:503})});
  await ui.click(ui.votes[1]);
  assert.equal(ui.votes[1].classList.contains('active'),true);
  assert.equal(ui.overflowState.classList.contains('pending'),true);
  assert.match(ui.overflowState.textContent,/could not keep.*Try again/i);
  ui.commit();assert.match(ui.overflowState.textContent,/Rating saved.*could not keep.*Try again/i);
  assert.equal(ui.votes.every(button=>!button.disabled),true);
});

test('an older checkpoint does not overwrite an in-flight or failed overflow rating',async()=>{
  let reject;
  const ui=digestControls({respond:async url=>url.endsWith('/10')?response({pending:true}):new Promise((_done,fail)=>{reject=fail;})});
  await ui.click(ui.votes[0]);
  const next=ui.click(ui.votes[1]);await new Promise(done=>setImmediate(done));
  ui.commit();assert.match(ui.overflowState.textContent,/Recording response/);
  reject(Error('offline'));await next;
  const failed=ui.overflowState.textContent;ui.commit();assert.equal(ui.overflowState.textContent,failed);
  assert.match(failed,/Could not save/);
});

function savedRemoval({respond=async()=>response({pending:true})}={}) {
  const button=control(),undo=control(),status=control(),removalStatus=control(),content=control(),removal=control();
  const count=control({total:'2',query:''});
  const nodes={'.save-state':status,'.undo-saved':undo,'.saved-content':content,'.saved-removal':removal,'.removal-state':removalStatus};
  const card={dataset:{itemId:'7',savedState:'saved'},querySelector:selector=>nodes[selector]};
  const events={},reloads=[],requests=[];button.textContent='Remove';button.closest=()=>card;removal.hidden=true;undo.textContent='Undo';
  const context={CSRF_TOKEN:'fixture',fetch:async(url,options)=>{requests.push({url,method:options.method});return respond(url,options);},
    window:{addEventListener:(name,callback)=>{events[name]=callback;},location:{reload:()=>reloads.push(true)}},
    document:{getElementById:id=>id==='archive-count'?count:null,querySelectorAll:selector=>{
      if(selector==='.remove-saved')return [button];
      if(selector==='.save-state.pending')return status.classList.contains('pending')?[status]:[];
      return [];
    }}};
  const prefix=saved.slice(saved.indexOf('\n',saved.indexOf('  const CSRF_TOKEN ='))+1,saved.indexOf('  document.querySelectorAll(".saved-card")'));
  const remove=between(saved,'  document.querySelectorAll(".remove-saved")','\n})();');
  runInNewContext(prefix+remove,context);
  return {button,undo,status,removalStatus,content,removal,card,count,reloads,requests,
    click:()=>button.listeners.click(),restore:()=>undo.listeners.click(),commit:()=>events['dd-storage-committed']()};
}

test('Saved removal waits for durable storage, then offers Undo without a reload',async()=>{
  const ui=savedRemoval();await ui.click();
  assert.equal(ui.reloads.length,0);assert.match(ui.status.textContent,/saving browser copy/i);
  assert.equal(ui.content.hidden,false);assert.equal(ui.button.disabled,false,'A failed checkpoint must allow an explicit retry');
  ui.commit();ui.commit();assert.equal(ui.reloads.length,0);
  assert.equal(ui.content.hidden,true);assert.equal(ui.removal.hidden,false);assert.equal(ui.undo.textContent,'Undo');
  assert.equal(ui.count.textContent,'1 saved item');
});

test('Saved removal can retry a failed checkpoint and offers Undo after durable success',async()=>{
  let attempts=0;const ui=savedRemoval({respond:async()=>response({pending:attempts++===0})});
  await ui.click();assert.equal(ui.content.hidden,false);await ui.click();assert.equal(ui.content.hidden,true);
  assert.equal(ui.removal.hidden,false);ui.commit();assert.equal(ui.count.textContent,'1 saved item');assert.equal(ui.reloads.length,0);
});

test('rejected Saved removal keeps the card available without changing its count',async()=>{
  const ui=savedRemoval({respond:async()=>response({status:503})});
  await ui.click();ui.commit();assert.equal(ui.reloads.length,0);assert.equal(ui.button.disabled,false);
  assert.equal(ui.content.hidden,false);assert.equal(ui.button.textContent,'Try again');assert.equal(ui.removal.hidden,true);
  assert.match(ui.status.textContent,/could not remove/i);
});

test('Saved Undo restores through POST and preserves the existing card and rating controls',async()=>{
  const ui=savedRemoval({respond:async()=>response()});await ui.click();await ui.restore();
  assert.deepEqual(ui.requests.map(row=>row.method),['DELETE','POST']);
  assert.equal(ui.content.hidden,false);assert.equal(ui.removal.hidden,true);assert.equal(ui.card.dataset.savedState,'saved');
  assert.equal(ui.count.textContent,'2 saved items');assert.match(ui.status.textContent,/Restored/);assert.equal(ui.reloads.length,0);
});

test('Saved Undo reports pending restoration without claiming durable success',async()=>{
  const ui=savedRemoval({respond:async(_url,options)=>response({pending:options.method==='POST'})});
  await ui.click();await ui.restore();assert.equal(ui.content.hidden,true);assert.equal(ui.count.textContent,'1 saved item');
  assert.match(ui.removalStatus.textContent,/Restore recorded.*saving browser copy/);assert.equal(ui.undo.textContent,'Retry restore');
  ui.commit();assert.equal(ui.content.hidden,false);assert.equal(ui.count.textContent,'2 saved items');assert.equal(ui.removal.hidden,true);
});

test('failed Saved Undo keeps a retryable removed state and the correct count',async()=>{
  const ui=savedRemoval({respond:async(_url,options)=>response({status:options.method==='POST'?503:200})});
  await ui.click();await ui.restore();ui.commit();assert.equal(ui.content.hidden,true);assert.equal(ui.count.textContent,'1 saved item');
  assert.equal(ui.undo.disabled,false);assert.equal(ui.undo.textContent,'Retry restore');assert.match(ui.removalStatus.textContent,/Could not restore/);
});

test('a failed retry does not discard a pending removal or a later durable acknowledgement',async()=>{
  let attempts=0;const ui=savedRemoval({respond:async()=>attempts++===0?response({pending:true}):response({status:503})});
  await ui.click();await ui.click();assert.equal(ui.content.hidden,false);assert.match(ui.status.textContent,/still waiting for browser storage/);
  ui.commit();assert.equal(ui.content.hidden,true);assert.equal(ui.removal.hidden,false);
});

test('a checkpoint that completes a pending removal invalidates a stale in-flight retry failure',async()=>{
  let attempts=0,reject;const ui=savedRemoval({respond:async()=>attempts++===0?response({pending:true}):new Promise((_done,fail)=>{reject=fail;})});
  await ui.click();const retry=ui.click();await new Promise(done=>setImmediate(done));ui.commit();
  reject(Error('late failure'));await retry;assert.equal(ui.content.hidden,true);assert.equal(ui.undo.disabled,false);assert.equal(ui.undo.textContent,'Undo');assert.equal(ui.count.textContent,'1 saved item');
});

test('completed calibration failure is visible and leaves the action retryable',async()=>{
  const button=control(),message=control(),toasts=[];button.disabled=true;
  const context={trainBtn:button,updateRankingStatus(){},showToast:text=>toasts.push(text),
    document:{getElementById:id=>id==='ranking-message'?message:null},
    fetch:async()=>new Response(JSON.stringify({status:{can_train:true},training_job:{running:false,last_result:{
      ok:false,trained:false,reason:'training_error',message:'Ranking training failed: PermissionError: fixture',
    }}}),{headers:{'content-type':'application/json'}})};
  const poll=between(digest,'  async function pollTrainingStatus()','  const searchInput =');
  const run=runInNewContext(poll+';pollTrainingStatus',context);
  await run();assert.equal(button.disabled,false);assert.equal(button.textContent,'Refresh calibration');
  assert.match(toasts.at(-1)||'',/calibration.*(failed|could not)|could not.*calibration/i);
  assert.match(message.textContent,/Try again/);
});
