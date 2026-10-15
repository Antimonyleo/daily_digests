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
export function validateRecoverySnapshot(payload) {
  const snapshot = JSON.parse(payload);
  if (snapshot?.format !== 'dailydigest-browser-v1' || !snapshot.files ||
      Array.isArray(snapshot.files) || typeof snapshot.files !== 'object') {
    throw new Error('Unsupported DailyDigest backup format.');
  }
  const names = Object.keys(snapshot.files);
  if (names.length > 10000 || payload.length > 128 * 1024 * 1024) throw new Error('Backup exceeds the recovery limit.');
  for (const name of names) {
    const parts = name.split('/');
    if (name !== '.env' && (parts.length < 2 || parts[0] !== 'data') ||
        parts.some(part => !part || part === '.' || part === '..' || part === '.compute.lock') ||
        name.includes('\\') || typeof snapshot.files[name] !== 'string') throw new Error('Backup contains an invalid private path.');
    if (!/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(snapshot.files[name])) throw new Error('Backup contains invalid encoded data.');
    if (names.some(other => other !== name && other.startsWith(name + '/'))) throw new Error('Backup contains conflicting paths.');
  }
  return snapshot;
}
