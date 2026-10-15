// Async fetch keeps a suspended Python request interruptible by Stop and timers.
const encode = bytes => {
  let value = '';
  for (let offset = 0; offset < bytes.length; offset += 32768) value += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
  return btoa(value);
};
const decode = value => Uint8Array.from(atob(value), char => char.charCodeAt(0));

export function createBrowserHttpBridge({ fetch = globalThis.fetch, shouldCancel = () => false,
  pollMs = 100, now = () => Date.now() } = {}) {
  // Browsers cannot distinguish CORS denial from a disconnected host. Suppress
  // only this exact failing URL for a short period, never a whole provider.
  const unavailable = new Map();
  const cancelled = () => {
    if (shouldCancel()) throw new Error('DailyDigest request cancelled');
  };
  async function wait(milliseconds) {
    cancelled();
    const end = now() + Math.max(0, milliseconds);
    do {
      await new Promise(resolve => setTimeout(resolve, Math.min(pollMs, Math.max(0, end - now()))));
      cancelled();
    } while (now() < end);
  }
  async function request(payload) {
    const input = JSON.parse(payload);
    cancelled();
    if ((unavailable.get(input.url) || 0) > now()) throw new Error('Recent CORS or network failure; skipping repeated request');
    const controller = new AbortController();
    let reason = '';
    const cancel = setInterval(() => {
      if (shouldCancel()) { reason = 'DailyDigest request cancelled'; controller.abort(); }
    }, pollMs);
    const timeout = input.timeoutMs == null ? null : setTimeout(() => {
      reason = 'TimeoutError: Browser request timed out'; controller.abort();
    }, input.timeoutMs);
    try {
      const response = await fetch(input.url, { method: input.method, headers: input.headers,
        body: ['GET', 'HEAD'].includes(input.method) || !input.body ? undefined : decode(input.body),
        credentials: 'omit', mode: 'cors', redirect: input.followRedirects === false ? 'error' : 'follow',
        signal: controller.signal });
      const body = new Uint8Array(await response.arrayBuffer());
      cancelled();
      return JSON.stringify({ status: response.status, url: response.url || input.url,
        headers: [...response.headers], body: encode(body) });
    } catch (error) {
      if (reason) throw new Error(reason);
      if (error instanceof TypeError) unavailable.set(input.url, now() + 5 * 60 * 1000);
      throw error;
    } finally {
      clearInterval(cancel);
      if (timeout !== null) clearTimeout(timeout);
    }
  }
  return { request, wait };
}
