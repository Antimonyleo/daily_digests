/** Control messages only: private reading state never crosses this channel. */
export function createTabHandoff({channel, id, owns, release, timeoutMs = 10000}) {
  let pending = null, releasing = false;
  const finish = result => {
    if (!pending) return;
    clearTimeout(pending.timer);
    pending.resolve(result); pending = null;
  };
  channel.addEventListener('message', async ({data}) => {
    if (!data || typeof data.requestId !== 'string') return;
    if (data.type === 'handoff-result' && data.receiver === id && data.requestId === pending?.id) {
      finish({ok:data.ok === true, message:typeof data.message === 'string' ? data.message : ''});
    } else if (data.type === 'handoff-request' && data.requester !== id && owns()) {
      const reply = (ok, message = '') => channel.postMessage({type:'handoff-result', requestId:data.requestId, receiver:data.requester, ok, message});
      if (releasing) { reply(false, 'Another tab switch is already finishing. Try again in a moment.'); return; }
      releasing = true;
      try { await release(); reply(true); }
      catch (error) { reply(false, error.message || 'The other tab could not save its changes.'); }
      finally { releasing = false; }
    }
  });
  return {
    request() {
      if (pending) return pending.promise;
      let resolve;
      const promise = new Promise(done => {resolve = done;});
      pending = {id:crypto.randomUUID(), resolve, promise,
        timer:setTimeout(() => finish({ok:false, message:'Could not reach the other DailyDigest tab. Switch to it, or close it and retry here.'}), timeoutMs)};
      channel.postMessage({type:'handoff-request', requestId:pending.id, requester:id});
      return promise;
    },
  };
}
