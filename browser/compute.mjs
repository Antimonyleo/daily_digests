export function computeCapabilities(environment = globalThis) {
  const cores = Number(environment.navigator?.hardwareConcurrency) || 1;
  const multicore = Boolean(environment.crossOriginIsolated && environment.SharedArrayBuffer && cores > 1);
  return { gpu: Boolean(environment.navigator?.gpu), multicore, threads: multicore ? Math.min(4, cores) : 1 };
}

export function computeSelection(mode, capabilities) {
  if (!['cpu', 'gpu', 'multicore'].includes(mode)) throw new Error('Unknown compute mode');
  return {
    device: mode === 'gpu' && capabilities.gpu ? 'webgpu' : 'wasm',
    threads: mode === 'multicore' && capabilities.multicore ? capabilities.threads : 1,
    fallback: mode === 'gpu' && !capabilities.gpu ? 'GPU unavailable; using one CPU core.'
      : mode === 'multicore' && !capabilities.multicore ? 'Multiple CPU cores require an isolated host; using one CPU core.' : '',
  };
}

function duration(ms) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return seconds >= 60 ? `${Math.floor(seconds / 60)}m ${seconds % 60}s` : `${seconds}s`;
}

export function formatComputeProgress(info) {
  if (info.phase === 'fallback') return info.message;
  if (info.phase === 'loading') return `Preparing the local model${info.file ? ` · ${info.file}` : ''}${Number.isFinite(info.percent) ? ` · ${Math.floor(info.percent)}%` : ''}…`;
  const count = `${info.completed} / ${info.total} completed`;
  const elapsed = `${duration(info.elapsed_ms || 0)} elapsed`;
  const remaining = Number.isFinite(info.remaining_ms) && info.remaining_ms > 0
    ? ` · about ${Math.max(1, Math.round(info.remaining_ms / 60000))}m remaining` : '';
  return `${info.phase === 'complete' ? 'Text comparison completed' : 'Comparing research text'}: ${count} · ${elapsed}${remaining}${info.backend ? ` · ${info.backend}` : ''}`;
}

// One model per browser worker. BGE tokenization, pooling and float32
// normalization are used for every backend; ranking rules remain unchanged.
export function createEmbeddingEngine({ mode, capabilities, env, loadTokenizer, loadModel, prepareBatch,
  makeTensor, progress, shouldCancel = () => false, now = () => performance.now(),
  yieldTask = () => new Promise(resolve => setTimeout(resolve, 0)) }) {
  const selection = computeSelection(mode, capabilities);
  env.wasm.numThreads = selection.threads;
  let device = selection.device, tokenizer, model, task = 0;
  const stats = { calls: 0, completedInputs: 0, encodingMs: 0, cpuFallbacks: 0 };
  const backend = () => device === 'webgpu' ? 'GPU' : selection.threads > 1 ? `CPU · ${selection.threads} cores` : 'CPU · one core';
  const report = (phase, fields = {}) => progress({ phase, task, backend: backend(), ...fields });
  const downloadProgress = event => {
    if (event.status === 'progress') report('loading', { file: event.file?.split('/').at(-1), percent: event.progress });
  };
  async function useCPU() {
    // A lost GPU can also reject disposal. CPU initialization must still run;
    // an error from the CPU path itself remains fatal.
    try { await model?.dispose?.(); } catch { /* failed GPU session */ }
    model = null;
    device = 'wasm';
    stats.cpuFallbacks++;
    report('fallback', { message: 'GPU acceleration was unavailable or stopped. Continuing locally on CPU.' });
    model = await loadModel({ device, dtype: 'fp32', progress_callback: downloadProgress });
  }
  async function embed(texts) {
    task++;
    const started = now();
    if (!model) {
      if (selection.fallback) report('fallback', { message: selection.fallback });
      report('loading');
      tokenizer ||= await loadTokenizer({ progress_callback: downloadProgress });
      try { model = await loadModel({ device, dtype: 'fp32', progress_callback: downloadProgress }); }
      catch (error) { if (device !== 'webgpu') throw error; await useCPU(); }
    }
    stats.calls++;
    const ordered = Array.from(texts).map((text, index) => ({ text, index })).sort((a, b) => Array.from(a.text).length - Array.from(b.text).length);
    const vectors = Array(ordered.length), timings = [];
    report('encoding', { completed: 0, total: ordered.length, elapsed_ms: now() - started });
    for (let start = 0; start < ordered.length; start += 4) {
      if (shouldCancel()) throw new Error('Brew stopped before publication');
      const batchStarted = now(), batch = ordered.slice(start, start + 4);
      const raw = await tokenizer(batch.map(row => row.text), { padding: false, truncation: false, return_tensor: false });
      const prepared = prepareBatch(raw, { padId: tokenizer.pad_token_id, sepId: tokenizer.sep_token_id });
      const inputs = Object.fromEntries(Object.entries(prepared.encoded).map(([key, rows]) =>
        [key, makeTensor(BigInt64Array.from(rows.flat().map(BigInt)), prepared.dims)]));
      let output;
      try {
        try { output = await model(inputs); }
        catch (error) {
          if (device !== 'webgpu') throw error;
          await useCPU();
          // Discard this call's GPU vectors; don't return a partially computed matrix.
          return await embed(texts);
        }
        const hidden = output.last_hidden_state;
        if (!hidden || hidden.dims.length !== 3 || hidden.dims[2] !== 384) throw new Error('Unexpected BGE model output; ranking cannot continue.');
        const [, sequence, dimension] = hidden.dims;
        for (let row = 0; row < batch.length; row++) {
          const offset = row * sequence * dimension;
          const vector = Array.from(hidden.data.slice(offset, offset + dimension));
          const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
          if (!Number.isFinite(norm) || !norm) throw new Error('Invalid BGE embedding; ranking cannot continue.');
          vectors[batch[row].index] = vector.map(value => value / norm);
        }
      } finally {
        for (const tensor of Object.values(output || {})) tensor?.dispose?.();
        for (const tensor of Object.values(inputs)) tensor?.dispose?.();
      }
      const completed = start + batch.length;
      timings.push((now() - batchStarted) / batch.length);
      if (timings.length > 5) timings.shift();
      report('encoding', { completed, total: ordered.length, elapsed_ms: now() - started,
        remaining_ms: completed >= 12 ? (ordered.length - completed) * timings.reduce((a, b) => a + b, 0) / timings.length : null });
      // Resolved inference promises alone need not let posted cancellation or
      // Python stream timers run. Give the worker a real event-loop turn.
      await yieldTask();
    }
    stats.completedInputs += ordered.length;
    stats.encodingMs += now() - started;
    report('complete', { completed: ordered.length, total: ordered.length, elapsed_ms: now() - started });
    return vectors;
  }
  return { embed, stats, get device() { return device; }, get threads() { return selection.threads; } };
}
