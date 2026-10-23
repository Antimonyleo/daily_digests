import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';

function control(dataset={}) {
  const classes=new Set(),listeners={};
  return {dataset,disabled:false,hidden:false,textContent:'',listeners,
    classList:{add:(...names)=>names.forEach(name=>classes.add(name)),remove:(...names)=>names.forEach(name=>classes.delete(name)),
      contains:name=>classes.has(name),toggle(name,value){const next=value??!classes.has(name);if(next)classes.add(name);else classes.delete(name);return next;}},
    setAttribute(name,value){this[name]=String(value);},addEventListener(name,callback){listeners[name]=callback;},focus(){}};
}

// Execute the templates' real selection, queue and clear handlers with a small
// DOM double; only the HTTP outcome is controlled here.
function feedbackControls(page,{initial='10',reason=true,pending=false,respond=async()=>({})}={}) {
  const source=readFileSync(new URL(`../templates/${page}.html.j2`,import.meta.url),'utf8');
  const votes=['10','40','70','100'].map(grade=>control({grade}));
  const chip=control({reason:'off_topic'}),clear=control(),status=control(),reasonBlock=control(),done=control();
  chip.classList.toggle('active',reason);chip.setAttribute('aria-pressed',reason);
  const groups={'.vote-btn':votes,'.reason-chip':[chip],'.clear-rating':[clear],'.feedback-done':[done]};
  const card={dataset:{itemId:'7',currentGrade:initial},
    querySelectorAll(selector){return selector.split(',').flatMap(value=>{
      const part=value.trim();return part==='.reason-chip.active'?(chip.classList.contains('active')?[chip]:[]):groups[part]||[];
    });},
    querySelector(selector){return {'.save-state':status,'.clear-rating':clear,'.reason-block':reasonBlock,'.feedback-done':done,'.vote-btn':votes[0]}[selector];}};
  const events={},window={addEventListener:(name,handler)=>{events[name]=handler;}};
  const context={window,card,itemEl:card,status,clear,reasonBlock,CSRF_TOKEN:'fixture',
    document:{querySelectorAll:()=>status.classList.contains('pending')?[status]:[]},
    NEGATIVE_GRADES:new Set(['10','40']),GRADE_CLASS:{},GRADE_LABEL:{10:'Not for me',40:'Hmmm',70:'Relevant',100:'Must read'},
    fetch:async(url,{method})=>({ok:true,json:async()=>respond(url,method),headers:{get:()=>pending?'pending':null}}),
    responsePayload:async response=>response.json(),
    setFeedbackPending(){},applyReaderFilters(){},updateRankingStatus(){},showToast(){},flash(){},
    setStatus:(_card,message)=>{status.textContent=message;}};
  const queue=readFileSync(new URL('../templates/_rating_queue.js.j2',import.meta.url),'utf8');
  let code;
  if(page==='digest_web') {
    const select=source.slice(source.indexOf('  function setActive('),source.indexOf('  function setStatus('));
    const start=source.indexOf('    const rate = createRatingQueue(');
    code=select+source.slice(start,source.indexOf('    itemEl.querySelectorAll(".reason-chip").forEach',start));
  } else {
    // This lane exercises ratings only; bookmark durability is covered by the
    // Saved Remove/Undo tests using the real controller and card states.
    context.bookmarkCards = new Map();
    const start=source.indexOf('    function busy(');
    code=source.slice(start,source.indexOf('    card.querySelectorAll(".reason-chip").forEach',start));
    const committedEvent=source.indexOf("  window.addEventListener('dd-storage-committed'");
    if(committedEvent>=0)code=source.slice(committedEvent,source.indexOf('  document.querySelectorAll(".saved-card")',committedEvent))+code;
  }
  runInNewContext(queue+code,context);
  return {card,chip,status,
    commit:()=>events['dd-storage-committed']?.(),
    async rate(grade){await votes.find(button=>button.dataset.grade===grade).listeners.click();await window.ddFlushRatings();},
    async clear(){await clear.listeners.click();await window.ddFlushRatings();},
  };
}

test('saved: durable checkpoint replaces the pending rating indicator',async()=>{
  const ui=feedbackControls('saved',{pending:true});await ui.rate('70');
  assert.match(ui.status.textContent,/saving browser copy/);
  ui.commit();assert.equal(ui.status.textContent,'Saved in this browser');
});

test('saved: an old checkpoint cannot replace a newer action or failure message',async()=>{
  let fail;
  const ui=feedbackControls('saved',{pending:true,respond:url=>url.endsWith('/70')?Promise.resolve({}):new Promise((_done,reject)=>{fail=reject;})});
  await ui.rate('70');const latest=ui.rate('40');await new Promise(done=>setImmediate(done));
  ui.commit();assert.equal(ui.status.textContent,'Recording response…');
  fail(Error('offline'));await latest;const failed=ui.status.textContent;
  ui.commit();assert.equal(ui.status.textContent,failed);assert.match(failed,/not saved/);
});

for(const page of ['digest_web','saved']) {
  test(`${page}: a failed rating after clearing stays unrated`,async()=>{
    const ui=feedbackControls(page,{initial:'70',reason:false,respond:async(_url,method)=>{
      if(method==='POST')throw Error('offline');return {};
    }});
    await ui.clear();assert.equal(ui.card.dataset.currentGrade,'');
    await ui.rate('100');assert.equal(ui.card.dataset.currentGrade,'');
    assert.match(ui.status.textContent,/not saved/);
  });

  test(`${page}: a rejected positive rating preserves confirmed negative reasons`,async()=>{
    let fail;
    const ui=feedbackControls(page,{respond:()=>new Promise((_resolve,reject)=>{fail=reject;})});
    const rating=ui.rate('70');await new Promise(done=>setImmediate(done));
    const preserved=ui.chip.classList.contains('active');
    fail(Error('offline'));await rating;
    assert.equal(preserved,true,'An unconfirmed choice must not erase reasons');
    assert.equal(ui.card.dataset.currentGrade,'10');
    assert.equal(ui.chip.classList.contains('active'),true);
    assert.equal(ui.chip['aria-pressed'],'true');
  });

  test(`${page}: a confirmed intermediate positive clears reasons before a later negative finishes`,async()=>{
    let finish;
    const ui=feedbackControls(page,{respond:(_url,method)=>method==='POST'&&!finish?new Promise(resolve=>{finish=resolve;}):Promise.resolve({})});
    const positive=ui.rate('70');await new Promise(done=>setImmediate(done));
    const negative=ui.rate('40');finish({});await Promise.all([positive,negative]);
    assert.equal(ui.card.dataset.currentGrade,'40');
    assert.equal(ui.chip.classList.contains('active'),false,'The positive server commit removed the old reason');
    assert.equal(ui.chip['aria-pressed'],'false');
  });

  test(`${page}: a failed clear keeps the confirmed grade and reasons`,async()=>{
    const ui=feedbackControls(page,{respond:async()=>{throw Error('offline');}});
    await ui.clear();assert.equal(ui.card.dataset.currentGrade,'10');
    assert.equal(ui.chip.classList.contains('active'),true);
  });

  test(`${page}: rollback after an intermediate positive commit does not resurrect old reasons`,async()=>{
    let finish;
    const ui=feedbackControls(page,{respond:async url=>{
      if(url.endsWith('/70'))await new Promise(done=>{finish=done;});else throw Error('offline');
      return {};
    }});
    const positive=ui.rate('70');await new Promise(done=>setImmediate(done));
    const negative=ui.rate('40');finish();await Promise.all([positive,negative]);
    assert.equal(ui.card.dataset.currentGrade,'70');
    assert.equal(ui.chip.classList.contains('active'),false);
  });
}
