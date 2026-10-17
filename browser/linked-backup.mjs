/** An optional disk copy. The browser database remains the working reading desk. */
async function revision(body) {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(body));
  return Array.from(new Uint8Array(bytes), value => value.toString(16).padStart(2, '0')).join('');
}

export function linkedBackupStore(indexedDB, name) {
  async function access(write, value) {
    const db = await new Promise((resolve, reject) => {
      const request = indexedDB.open(name, 1);
      request.onupgradeneeded = () => request.result.createObjectStore('settings');
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    try {
      return await new Promise((resolve, reject) => {
        const transaction = db.transaction('settings', write ? 'readwrite' : 'readonly');
        const store = transaction.objectStore('settings');
        const request = write ? store.put(value, 'linked-backup') : store.get('linked-backup');
        transaction.oncomplete = () => resolve(request.result);
        transaction.onerror = () => reject(transaction.error);
        transaction.onabort = () => reject(transaction.error || new Error('Could not remember the backup file.'));
      });
    } finally { db.close(); }
  }
  return {load: () => access(false), store: value => access(true, value)};
}

export function createLinkedBackup({load, store}) {
  let record = null, writing = false, changingFile = false, configured = false, changes = Promise.resolve();
  function remember(update) {
    const pending = changes.then(async () => {
      const next = typeof update === 'function' ? update(record) : update;
      await store(next); record = next; configured = true;
    });
    changes = pending.catch(() => {});
    return pending;
  }
  async function permission(selected = record) { return selected ? selected.handle.queryPermission({mode:'readwrite'}) : 'denied'; }
  async function requireAccess(selected = record) {
    if (!selected || await permission(selected) !== 'granted') throw new Error('Reconnect your backup file to allow access.');
  }
  return {
    async open() { const previous = await load(); if (!configured) record = previous || null; },
    async connect(handle) {
      if (writing || changingFile) throw new Error('Wait for the backup copy or file change to finish.');
      changingFile = true;
      try {
        let expected = null;
        if (await handle.queryPermission({mode:'readwrite'}) === 'granted' && (await handle.getFile()).size === 0) expected = await revision('');
        await remember({handle, revision:expected, auto:false, updatedAt:null});
      } finally { changingFile = false; }
    },
    async inspect() {
      if (!record) return {linked:false};
      const access = await permission();
      const info = {linked:true,name:record.handle.name,permission:access,auto:record.auto,updatedAt:record.updatedAt,writable:record.revision !== null};
      if (access !== 'granted') return info;
      try {
        const file = await record.handle.getFile();
        return {...info,exists:file.size > 0,size:file.size,modified:file.lastModified};
      } catch (error) {
        if (error.name !== 'NotFoundError') throw error;
        return {...info,exists:false,missing:true};
      }
    },
    requestAccess: () => record ? record.handle.requestPermission({mode:'readwrite'}) : Promise.resolve('denied'),
    async setAutomatic(auto) {
      if (!record) throw new Error('Choose a backup file first.');
      await remember(current => ({...current,auto:Boolean(auto)}));
    },
    async read() {
      await requireAccess();
      const file = await record.handle.getFile();
      if (file.size > 128 * 1024 * 1024) throw new Error('Backup exceeds the 128 MB restore limit.');
      const body = await file.text();
      return {body,revision:await revision(body)};
    },
    async accept(value) { await remember(current => ({...current,revision:value})); },
    async write(body) {
      if (writing) throw new Error('A backup copy is already being saved.');
      if (changingFile) throw new Error('Wait for the backup file change to finish.');
      writing = true;
      const selected = record;
      let stream;
      try {
        await requireAccess(selected);
        // Hold the file lock while checking the revision and committing. A
        // different browser's changes must never be replaced by an older desk.
        stream = await selected.handle.createWritable({mode:'exclusive'});
        const current = await revision(await (await selected.handle.getFile()).text());
        if (selected.revision === null || current !== selected.revision) {
          throw new Error('The existing backup changed or belongs to another reading desk. Restore it, or choose a new backup file.');
        }
        const next = await revision(body);
        await stream.write(body);
        await stream.close(); stream = null;
        const updatedAt = new Date().toISOString();
        // Disk commit succeeded even if IndexedDB is full. Keep its revision
        // in memory so another save in this tab does not conflict with itself.
        record = {...record,revision:next,updatedAt}; configured = true;
        try { await remember(current => ({...current,revision:next,updatedAt})); }
        catch (error) { throw new Error('Backup file saved, but this browser could not remember its revision. Reconnect it after reopening the site.', {cause:error}); }
        return {updatedAt};
      } finally {
        if (stream) await stream.abort().catch(() => {});
        writing = false;
      }
    },
    async disconnect() {
      if (writing || changingFile) throw new Error('Wait for the backup copy or file change to finish.');
      changingFile = true;
      try { await remember(null); }
      finally { changingFile = false; }
    },
  };
}

export function createBackupScheduler({save, canSave, onError, delayMs = 60000, setTimer = setTimeout, clearTimer = clearTimeout}) {
  let timer = null, changed = 0, saved = 0, stopped = false, running = false;
  function schedule() {
    if (stopped || timer !== null || running || saved === changed) return;
    timer = setTimer(async () => {
      timer = null;
      if (!canSave()) { schedule(); return; }
      running = true;
      const version = changed;
      try { await save(); saved = version; }
      catch (error) { stopped = true; onError(error); }
      finally { running = false; schedule(); }
    }, delayMs);
  }
  return {
    changed() { changed++; stopped = false; schedule(); },
    cancel() { stopped = true; clearTimer(timer); timer = null; },
  };
}
