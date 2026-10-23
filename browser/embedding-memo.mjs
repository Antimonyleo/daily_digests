// Reuse exact text within a single model/worker without changing the encoder
// identity used by already-published public comparison vectors.
export function withEmbeddingMemo(engine, {shouldCancel = () => false,
  yieldTask = () => new Promise(resolve => setTimeout(resolve, 0))} = {}) {
  // Keep the original engine getters (including GPU fallback state) live.
  const encode = engine.embed.bind(engine);
  const memoized = Object.create(engine);
  memoized.stats.reusedInputs = 0;
  // Exact input text includes the caller's query/document prefix. This memo
  // belongs to one fixed model export and worker; no private text is persisted.
  const memo = new Map();
  async function embed(input) {
    if (shouldCancel()) throw new Error('Brew stopped before publication');
    const texts = Array.from(input);
    if (!texts.length) return [];
    const resolved = new Map(), missing = [];
    for (const text of texts) {
      if (resolved.has(text)) continue;
      if (memo.has(text)) {
        const vector = memo.get(text);
        memo.delete(text);memo.set(text, vector);
        resolved.set(text, vector);
      } else {
        resolved.set(text, null);missing.push(text);
      }
    }
    if (missing.length) {
      const fresh = await encode(missing);
      if (shouldCancel()) throw new Error('Brew stopped before publication');
      missing.forEach((text, index) => resolved.set(text, fresh[index]));
      // Retain only complete, successful calls. Bound both entry count and text
      // size; large article vectors already have a durable SQLite cache.
      for (const text of missing) {
        if (text.length > 8192) continue;
        memo.set(text, resolved.get(text));
        if (memo.size > 256) memo.delete(memo.keys().next().value);
      }
    } else {
      await yieldTask();
      if (shouldCancel()) throw new Error('Brew stopped before publication');
    }
    memoized.stats.reusedInputs += texts.length - missing.length;
    return texts.map(text => resolved.get(text).slice());
  }
  memoized.embed = embed;
  return memoized;
}
