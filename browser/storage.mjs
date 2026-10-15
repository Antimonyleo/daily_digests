/** Serialize private checkpoints; unchanged page reads never encode the database. */
export function createSnapshotWriter({ generation, capture, save, ready = () => true, warning = () => {}, now = Date.now }) {
  let savedGeneration = 0, pending = null, lastError = null, retryAt = 0;
  const status = () => ({ dirty: generation() !== savedGeneration, saving: Boolean(pending),
    durable: ready() && generation() === savedGeneration && !lastError, error: lastError?.message || null });
  function flush({ force = false } = {}) {
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
  return { flush, status };
}

/** Validate the envelope without Python, including paths, conflicts and base64. */
const MAX_IMPORT_FILES = 10000, MAX_IMPORT_CHARACTERS = 128 * 1024 * 1024;
const exceedsImportLimit = (payload, names) => names.length > MAX_IMPORT_FILES || payload.length > MAX_IMPORT_CHARACTERS;

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

export function validateRecoverySnapshot(payload, { enforceLimits = true } = {}) {
  const snapshot = JSON.parse(payload);
  if (snapshot?.format !== 'dailydigest-browser-v1' || !snapshot.files ||
      Array.isArray(snapshot.files) || typeof snapshot.files !== 'object') {
    throw new Error('Unsupported DailyDigest backup format.');
  }
  const names = Object.keys(snapshot.files);
  if (enforceLimits && exceedsImportLimit(payload, names)) throw new Error('Backup exceeds the recovery limit.');
  const paths = new Set(names);
  for (const name of names) {
    const parts = name.split('/');
    if (name !== '.env' && (parts.length < 2 || parts[0] !== 'data') ||
        parts.some(part => !part || part === '.' || part === '..' || part === '.compute.lock') ||
        name.includes('\\') || typeof snapshot.files[name] !== 'string') throw new Error('Backup contains an invalid private path.');
    if (!validBase64(snapshot.files[name])) throw new Error('Backup contains invalid encoded data.');
    if (parts.some((_, index) => index > 0 && paths.has(parts.slice(0, index).join('/')))) throw new Error('Backup contains conflicting paths.');
  }
  return snapshot;
}

/** Existing checkpoints may grow beyond external import limits; preserve them. */
export function validateStoredSnapshot(payload) {
  const snapshot = validateRecoverySnapshot(payload, { enforceLimits: false });
  return { snapshot, warning: exceedsImportLimit(payload, Object.keys(snapshot.files))
    ? 'Your saved reading state exceeds the portable import limit. It was preserved; importing a backup of this size is currently unsupported.' : null };
}
