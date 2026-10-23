// Pick a CPU default without overriding an explicit per-device choice.
export function defaultComputeMode(saved, capabilities, memory = globalThis.navigator?.deviceMemory) {
  return ['cpu', 'gpu', 'multicore'].includes(saved) ? saved
    : capabilities.multicore && capabilities.threads >= 4 && (memory ?? 4) >= 4 ? 'multicore' : 'cpu';
}

export function readComputeChoice(indexedDB, name) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, 1);
    request.onupgradeneeded = () => request.result.createObjectStore('state');
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      try {
        const transaction = db.transaction('state');
        const value = transaction.objectStore('state').get('compute');
        transaction.oncomplete = () => { db.close(); resolve(value.result); };
        transaction.onabort = transaction.onerror = () => { db.close(); reject(transaction.error); };
      } catch (error) { db.close(); reject(error); }
    };
  });
}

function waitFor(target, event, predicate, signal) {
  return new Promise((resolve, reject) => {
    const cleanup = () => { target.removeEventListener(event, check); signal.removeEventListener('abort', abort); };
    const check = () => { if (predicate()) { cleanup(); resolve(); } };
    const abort = () => { cleanup(); reject(Error('CPU preparation timed out')); };
    target.addEventListener(event, check);
    signal.addEventListener('abort', abort, {once: true});
    if (signal.aborted) abort(); else check();
  });
}

export async function prepareDefaultCPU({navigator, storage, session, key, isolated, secure,
  readChoice, register, timeoutMs = 1000}) {
  let isolate = false;
  try {
    const saved = storage.getItem(key);
    isolate = saved === 'true';
    // Keep existing choices and low-memory devices on their current path.
    if (saved !== null || isolated || !secure || !navigator.serviceWorker
      || Number(navigator.hardwareConcurrency || 0) < 4 || (navigator.deviceMemory ?? 4) < 4
      || session.getItem(key + ':attempted') === 'true') return {isolate, reload: false};
    const controller = new AbortController();
    let timer;
    try {
      return await Promise.race([
        (async () => {
          const choice = await readChoice();
          if (controller.signal.aborted || choice !== undefined) return {isolate, reload: false};
          session.setItem(key + ':attempted', 'true');
          storage.setItem(key, 'true');
          isolate = true;
          const registration = await register(true);
          const candidate = registration.installing || registration.waiting || registration.active;
          if (!candidate) throw Error('Service worker not available');
          await waitFor(candidate, 'statechange', () => candidate.state === 'activated', controller.signal);
          await waitFor(navigator.serviceWorker, 'controllerchange', () => {
            const url = navigator.serviceWorker.controller?.scriptURL;
            return url && new URL(url).searchParams.get('isolate') === '1';
          }, controller.signal);
          return {isolate, reload: true};
        })(),
        new Promise(resolve => { timer = setTimeout(() => {
          controller.abort(); resolve({isolate, reload: false});
        }, timeoutMs); }),
      ]);
    } finally { clearTimeout(timer); controller.abort(); }
  } catch { return {isolate, reload: false}; }
}
