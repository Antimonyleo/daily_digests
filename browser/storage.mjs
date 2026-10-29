/** Serialize private checkpoints; unchanged page reads never encode the database. */
export function createSnapshotWriter({ generation, capture, save, ready = () => true, warning = () => {}, now = Date.now }) {
  let savedGeneration = 0, pending = null, lastError = null, retryAt = 0, timer;
  const status = () => ({ dirty: generation() !== savedGeneration, saving: Boolean(pending),
    durable: ready() && generation() === savedGeneration && !lastError, error: lastError?.message || null });
  function flush({ force = false } = {}) {
    clearTimeout(timer); timer = null;
    if (pending) return pending;
    if (!ready()) return Promise.resolve(false);
    if (generation() === savedGeneration) return Promise.resolve(true);
    if (!force && now() < retryAt) return Promise.resolve(false);
    pending = Promise.resolve().then(async () => {
      try {
        while (generation() !== savedGeneration) {
          if (!ready()) return false;
          const capturedGeneration = generation();
          const snapshot = capture();
          await save(snapshot);
          savedGeneration = capturedGeneration;
        }
        lastError = null; retryAt = 0;
        warning(null);
        return true;
      } catch (error) {
        lastError = error; retryAt = now() + 5000;
        warning(`Changes are available in this tab but could not be saved to browser storage: ${error.message || error}. Export a backup before closing this tab.`);
        return false;
      } finally { pending = null; }
    });
    return pending;
  }
  function schedule({delayMs = 350} = {}) {
    if (timer || pending || !status().dirty) return;
    timer = setTimeout(() => { timer = null; void flush(); }, delayMs);
  }
  return { flush, status, schedule };
}

/** Validate the envelope without Python, including paths, conflicts and base64. */
const MAX_IMPORT_FILES = 10000, MAX_IMPORT_CHARACTERS = 128 * 1024 * 1024;
const exceedsImportLimit = (payload, names) => names.length > MAX_IMPORT_FILES || payload.length > MAX_IMPORT_CHARACTERS;
const storedSizeWarning = (payload, names) => exceedsImportLimit(payload, names)
  ? 'Your saved reading state exceeds the portable import limit. It was preserved; importing a backup of this size is currently unsupported.' : null;

function validatePrivatePaths(names, { allowSettingsJournal = false } = {}) {
  const paths = new Set(names);
  for (const name of names) {
    if (typeof name !== 'string') throw new Error('Backup contains an invalid private path.');
    const parts = name.split('/');
    if (name !== '.env' && !(allowSettingsJournal && name === '.settings-transaction.json') && (parts.length < 2 || parts[0] !== 'data') ||
        parts.some(part => !part || part === '.' || part === '..' || part === '.compute.lock') ||
        name.includes('\\')) throw new Error('Backup contains an invalid private path.');
    if (parts.some((_, index) => index > 0 && paths.has(parts.slice(0, index).join('/')))) throw new Error('Backup contains conflicting paths.');
  }
}

/** Match export_snapshot byte-for-byte, including Python's ASCII JSON keys. */
export function capturePrivateSnapshot(names, readFile, {cache = new Map(), versions = {}} = {}) {
  validatePrivatePaths(names, {allowSettingsJournal:true});
  const wanted = new Set(names);
  for (const name of cache.keys()) if (!wanted.has(name)) cache.delete(name);
  const quote = value => JSON.stringify(value).replace(/[\u007f-\uffff]/g,
    char => '\\u' + char.charCodeAt(0).toString(16).padStart(4, '0'));
  const files = names.map(name => {
    const previous = cache.get(name), version = versions[name];
    let encoded = version !== undefined && previous?.version === version ? previous.encoded : undefined;
    if (encoded === undefined) {
      const bytes = readFile(name);
      if (typeof bytes.toBase64 === 'function') encoded = bytes.toBase64();
      else {
        // Older browsers need bounded chunks: spreading a complete database
        // into fromCharCode recreates the startup call-stack overflow.
        let binary = '';
        for (let offset = 0; offset < bytes.length; offset += 32768)
          binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
        encoded = btoa(binary);
      }
      if (version !== undefined) cache.set(name, {version, encoded});
    }
    return quote(name) + ': "' + encoded + '"';
  });
  const payload = '{"format": "' + SNAPSHOT_FORMAT + '", "files": {' + files.join(', ') + '}}';
  return { payload, warning: storedSizeWarning(payload, names) };
}

function validBase64(value) {
  if (value.length % 4 !== 0) return false;
  const end = value.length - (value.endsWith('==') ? 2 : value.endsWith('=') ? 1 : 0);
  // A quantified group over an entire database can overflow the JS regexp
  // stack. Scan the alphabet instead, leaving the supplied bytes untouched.
  for (let index = 0; index < end; index++) {
    const code = value.charCodeAt(index);
    if (!(code >= 65 && code <= 90 || code >= 97 && code <= 122
      || code >= 48 && code <= 57 || code === 43 || code === 47)) return false;
  }
  return true;
}

export const SNAPSHOT_FORMAT = 'dailydigest-browser-v1';
// Formats the runtime can read: the current one plus each older format that
// has an upgrade step in runtime.py (SNAPSHOT_UPGRADES). Add both together.
export const READABLE_SNAPSHOT_FORMATS = new Set([SNAPSHOT_FORMAT]);

export function snapshotFormatError(format) {
  const found = /^dailydigest-browser-v(\d+)$/.exec(String(format ?? ''));
  return found && Number(found[1]) > Number(SNAPSHOT_FORMAT.split('-v').pop())
    ? 'This backup comes from a newer version of DailyDigest. Reload the page to get the newest version, then try again.'
    : 'Unsupported DailyDigest backup format.';
}

export function validateRecoverySnapshot(payload, { enforceLimits = true, allowSettingsJournal = false } = {}) {
  const snapshot = JSON.parse(payload);
  if (!READABLE_SNAPSHOT_FORMATS.has(snapshot?.format)) throw new Error(snapshotFormatError(snapshot?.format));
  if (!snapshot.files || Array.isArray(snapshot.files) || typeof snapshot.files !== 'object') {
    throw new Error('Unsupported DailyDigest backup format.');
  }
  const names = Object.keys(snapshot.files);
  if (enforceLimits && exceedsImportLimit(payload, names)) throw new Error('Backup exceeds the recovery limit.');
  validatePrivatePaths(names, {allowSettingsJournal});
  for (const name of names) {
    if (typeof snapshot.files[name] !== 'string') throw new Error('Backup contains an invalid private path.');
    if (!validBase64(snapshot.files[name])) throw new Error('Backup contains invalid encoded data.');
  }
  return snapshot;
}

/** Existing checkpoints may grow beyond external import limits; preserve them. */
export function validateStoredSnapshot(payload) {
  const snapshot = validateRecoverySnapshot(payload, { enforceLimits: false, allowSettingsJournal: true });
  return { snapshot, warning: storedSizeWarning(payload, Object.keys(snapshot.files)) };
}
