// Raw cross-encoder relevance logits; these are not calibrated probabilities.
// Keep this token window protocol aligned with the native semantic scorer.
export function prepareSemanticRecords(tokenizer, records) {
  if (!Array.isArray(records)) throw new Error('Semantic records must be an array');
  const cls = tokenizer.cls_token_id ?? 101, sep = tokenizer.sep_token_id ?? 102;
  const encode = text => Array.from(tokenizer.encode(text, { add_special_tokens: false }));
  const windows = [], summaries = [];
  for (const [recordIndex, record] of records.entries()) {
    if (!record || !['query', 'title', 'abstract'].every(key => typeof record[key] === 'string')) {
      throw new Error('Semantic query, title, and abstract must be strings');
    }
    if (!record.query.trim()) throw new Error('Semantic query must be nonempty');
    let query = encode(record.query);
    const fullTitle = encode(record.title), abstract = encode(record.abstract);
    if (!query.length) throw new Error('Semantic query must contain between 1 and 192 tokens');
    const queryTruncated = query.length > 192;
    if (queryTruncated) {
      if (record.topic !== undefined && typeof record.topic !== 'string') throw new Error('Semantic topic must be a string');
      const topic = encode(record.topic || '').slice(0, 96);
      query = [...topic, ...query.slice(0, 192 - topic.length)];
    }
    const title = fullTitle.slice(0, 128), budget = 512 - query.length - 3 - title.length;
    const overlap = Math.min(32, Math.floor(budget / 4)), step = budget - overlap;
    const count = Math.max(1, Math.ceil((abstract.length - budget) / step) + 1);
    const indices = count > 3 ? [0, Math.floor(count / 2), count - 1] : Array.from({ length: count }, (_, i) => i);
    summaries.push({ windows: indices.length, complete: fullTitle.length <= 128 && count <= 3 && !queryTruncated,
      ...(queryTruncated ? { query_truncated: true } : {}) });
    for (const index of indices) {
      const body = [...title, ...abstract.slice(index * step, index * step + budget)];
      const input_ids = [cls, ...query, sep, ...body, sep];
      windows.push({ recordIndex, input_ids, attention_mask: input_ids.map(() => 1),
        token_type_ids: [...Array(query.length + 2).fill(0), ...Array(body.length + 1).fill(1)] });
    }
  }
  return { windows, records: summaries };
}

export function createSemanticEngine({ loadTokenizer, loadModel, makeTensor,
  progress = () => {}, shouldCancel = () => false,
  yieldTask = () => new Promise(resolve => setTimeout(resolve, 0)) }) {
  let tokenizer, model;
  const checkCancelled = () => {
    if (shouldCancel()) throw new Error('Brew stopped before semantic scoring completed');
  };
  async function score(records) {
    checkCancelled();
    if (Array.isArray(records) && !records.length) return [];
    if (!tokenizer) {
      progress({ phase: 'loading' });
      tokenizer = await loadTokenizer();
    }
    const prepared = prepareSemanticRecords(tokenizer, records);
    checkCancelled();
    model ||= await loadModel({ device: 'wasm', dtype: 'fp32' });
    const values = prepared.records.map(record => ({ ...record, logit: -Infinity }));
    const total = prepared.windows.length;
    progress({ phase: 'scoring', completed: 0, total });
    for (let start = 0; start < total; start += 4) {
      checkCancelled();
      const batch = prepared.windows.slice(start, start + 4);
      const length = Math.max(...batch.map(row => row.input_ids.length));
      const inputs = {};
      let output;
      try {
        for (const key of ['input_ids', 'attention_mask', 'token_type_ids']) {
          const pad = key === 'input_ids' ? (tokenizer.pad_token_id ?? 0) : 0;
          const flat = batch.flatMap(row => [...row[key], ...Array(length - row[key].length).fill(pad)]);
          inputs[key] = makeTensor(BigInt64Array.from(flat.map(BigInt)), [batch.length, length]);
        }
        output = await model(inputs);
        const logits = output.logits;
        if (!logits || !((logits.dims.length === 1 && logits.dims[0] === batch.length)
          || (logits.dims.length === 2 && logits.dims[0] === batch.length && logits.dims[1] === 1))
          || logits.data.length !== batch.length || !Array.from(logits.data).every(Number.isFinite)) {
          throw new Error('Expected one finite scalar logit per semantic window');
        }
        for (let i = 0; i < batch.length; i++) {
          const record = values[batch[i].recordIndex];
          record.logit = Math.max(record.logit, logits.data[i]);
        }
      } finally {
        for (const tensor of Object.values(output || {})) tensor?.dispose?.();
        for (const tensor of Object.values(inputs)) tensor?.dispose?.();
      }
      progress({ phase: 'scoring', completed: start + batch.length, total });
      await yieldTask();
    }
    checkCancelled();
    progress({ phase: 'complete', completed: total, total });
    return values;
  }
  return { score };
}
