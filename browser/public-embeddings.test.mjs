import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash, webcrypto} from 'node:crypto';
import {createPublicEmbeddingCache} from './public-embeddings.mjs';

const sha = value => createHash('sha256').update(value).digest('hex');
const vector = offset => Array.from({length:384}, (_, index) => Number(index === offset));
function fixture(overrides={}) {
  const bytes=Buffer.from(new Float32Array([...vector(0),...vector(1)]).buffer);
  const manifest={version:1,public_only:true,signature:'fixture-cpu-v1',backend:'browser-onnx',dimension:384,
    dtype:'float32-le',hashes:[sha('Public paper one'),sha('Public paper two')],
    file:'public-embeddings-fixture.f32',bytes:bytes.length,sha256:sha(bytes),...overrides};
  const calls=[],requests=[];
  const cache=createPublicEmbeddingCache({signature:'fixture-cpu-v1',backend:()=> 'browser-onnx',
    url:new URL('https://example.test/project/assets/public-embeddings.json'),crypto:webcrypto,
    embed:async texts=>{calls.push(texts);return texts.map(()=>vector(2));},
    fetcher:async url=>{requests.push(String(url));return String(url).endsWith('.json')
      ? new Response(JSON.stringify(manifest)) : new Response(bytes);}});
  return {cache,calls,requests,bytes,manifest};
}

test('public vectors preserve order and encode only missing exact text, without sending private text',async()=>{
  const {cache,calls,requests}=fixture();
  assert.deepEqual(await cache.embed(['Public paper two','Private reader profile','Public paper one']),[vector(1),vector(2),vector(0)]);
  assert.deepEqual(calls,[['Private reader profile']]);
  assert.equal(cache.stats.hits,2);assert.equal(cache.stats.misses,1);
  assert.equal(requests.length,2);assert.ok(requests.every(url=>!url.includes('Private')));
  await cache.embed(['Public paper one']);assert.equal(requests.length,2);assert.equal(calls.length,1);
});

for(const [label,change] of Object.entries({signature:{signature:'another-model'},dimension:{dimension:768},
  checksum:{sha256:'0'.repeat(64)},duplicate:{hashes:[sha('Public paper one'),sha('Public paper one')]},
  remote:{file:'https://other.invalid/vector.f32'},oversize:{bytes:16_000_001},private:{public_only:false}})) {
  test(`invalid public ${label} falls back to local embedding`,async()=>{
    const {cache,calls}=fixture(change);
    assert.deepEqual(await cache.embed(['Public paper one']),[vector(2)]);
    assert.deepEqual(calls,[['Public paper one']]);assert.equal(cache.stats.hits,0);
  });
}

test('GPU unfused export skips public CPU vectors even after CPU fallback',async()=>{
  const {manifest,bytes}=fixture();let fetched=0;
  const cache=createPublicEmbeddingCache({signature:manifest.signature,backend:()=> 'browser-onnx-unfused-v1',
    url:new URL('https://example.test/assets/public-embeddings.json'),crypto:webcrypto,
    fetcher:async()=>{fetched++;return new Response(bytes);},embed:async()=>[vector(2)]});
  assert.deepEqual(await cache.embed(['Public paper one']),[vector(2)]);assert.equal(fetched,0);
});

test('missing cache is a local-compute fallback and empty input does not fetch',async()=>{
  let fetched=0;
  const cache=createPublicEmbeddingCache({signature:'x',backend:()=> 'browser-onnx',
    url:new URL('https://example.test/assets/public-embeddings.json'),crypto:webcrypto,
    fetcher:async()=>{fetched++;return new Response('',{status:404});},embed:async()=>[vector(2)]});
  assert.deepEqual(await cache.embed([]),[]);assert.equal(fetched,0);
  assert.deepEqual(await cache.embed(['Public paper one']),[vector(2)]);assert.equal(fetched,1);
});

test('missing optional crypto support cannot disable local ranking',async()=>{
  const cache=createPublicEmbeddingCache({signature:'x',backend:()=> 'browser-onnx',crypto:null,
    url:new URL('https://example.test/assets/public-embeddings.json'),
    fetcher:async()=>new Response('',{status:404}),embed:async()=>[vector(2)]});
  assert.deepEqual(await cache.embed(['Public']),[vector(2)]);
});

test('nonfinite vectors cannot enter ranking even with a matching transport checksum',async()=>{
  const bytes=Buffer.from(new Float32Array(384).fill(NaN).buffer);
  const manifest={version:1,public_only:true,signature:'x',backend:'browser-onnx',dimension:384,
    dtype:'float32-le',hashes:[sha('Public')],file:'public-embeddings-invalid.f32',bytes:bytes.length,sha256:sha(bytes)};
  const cache=createPublicEmbeddingCache({signature:'x',backend:()=> 'browser-onnx',crypto:webcrypto,
    url:new URL('https://example.test/assets/public-embeddings.json'),
    fetcher:async url=>new Response(String(url).endsWith('.json')?JSON.stringify(manifest):bytes),
    embed:async()=>[vector(2)]});
  assert.deepEqual(await cache.embed(['Public']),[vector(2)]);assert.equal(cache.stats.hits,0);
});

for(const phase of ['request','body']) {
  test(`hung optional ${phase} times out and falls back to local work`,async()=>{
    let signal,local=0;
    const cache=createPublicEmbeddingCache({signature:'x',backend:()=> 'browser-onnx',crypto:webcrypto,
      url:new URL('https://example.test/assets/public-embeddings.json'),timeoutMs:20,
      fetcher:async(_url,options)=>{signal=options.signal;return phase==='request'
        ? new Promise(()=>{}) : {ok:true,json:()=>new Promise(()=>{})};},
      embed:async()=>{local++;return [vector(2)];}});
    assert.deepEqual(await cache.embed(['Public']),[vector(2)]);
    assert.equal(signal.aborted,true);assert.equal(local,1);
  });
}

test('Stop cancels a stalled public download before local inference and a later brew can retry',async()=>{
  let cancel=false,signal,local=0,retry=false;
  const cache=createPublicEmbeddingCache({signature:'x',backend:()=> 'browser-onnx',crypto:webcrypto,
    url:new URL('https://example.test/assets/public-embeddings.json'),timeoutMs:1000,cancelPollMs:5,
    shouldCancel:()=>cancel,fetcher:async(_url,options)=>{signal=options.signal;return retry
      ? new Response('',{status:404}) : new Promise(()=>{});},
    embed:async()=>{local++;return [vector(2)];}});
  const pending=cache.embed(['Public']);setTimeout(()=>{cancel=true;},10);
  await assert.rejects(pending,/stopped/i);assert.equal(signal.aborted,true);assert.equal(local,0);
  cancel=false;retry=true;assert.deepEqual(await cache.embed(['Public']),[vector(2)]);assert.equal(local,1);
});

for(const failure of ['503','429','network','body-network','vector-body-network','vector-body-abort','timeout']) {
  test(`transient public ${failure} retries once after cooldown, sharing concurrent recovery`,async()=>{
    const {manifest,bytes}=fixture();let now=0,requests=0,local=0;
    const vectorBodyFailure=failure.startsWith('vector-body-');
    const cache=createPublicEmbeddingCache({signature:manifest.signature,backend:()=> 'browser-onnx',crypto:webcrypto,
      url:new URL('https://example.test/assets/public-embeddings.json'),now:()=>now,retryDelayMs:100,
      timeoutMs:failure==='timeout'?50:1000,
      fetcher:async url=>{
        requests++;
        if(requests===2&&vectorBodyFailure)
          return {ok:true,arrayBuffer:async()=>{throw failure==='vector-body-abort'
            ?new DOMException('Connection aborted','AbortError'):new TypeError('Connection interrupted');}};
        if(requests===1&&!vectorBodyFailure) {
          if(failure==='network') throw new TypeError('Network unavailable');
          if(failure==='body-network') return {ok:true,json:async()=>{throw new TypeError('Connection interrupted');}};
          if(failure==='timeout') return new Promise(()=>{});
          return new Response('',{status:Number(failure)});
        }
        return new Response(String(url).endsWith('.json')?JSON.stringify(manifest):bytes);
      },embed:async texts=>{local++;return texts.map(()=>vector(2));}});
    assert.deepEqual(await cache.embed(['Public paper one']),[vector(2)]);
    now=99;
    assert.deepEqual(await cache.embed(['Public paper one']),[vector(2)]);
    assert.equal(requests,vectorBodyFailure?2:1);
    now=100;
    const recovered=await Promise.all([cache.embed(['Public paper one']),cache.embed(['Public paper two'])]);
    assert.deepEqual(recovered,[[vector(0)],[vector(1)]]);
    assert.equal(requests,vectorBodyFailure?4:3);assert.equal(local,2);
    assert.equal(cache.stats.status,'ready');assert.equal(cache.stats.reason,'');
  });
}

for(const failure of ['404','410','signature','malformed-json']) {
  test(`permanent public ${failure} does not retry after cooldown`,async()=>{
    const {manifest}=fixture({signature:'different-model'});let now=0,requests=0;
    const cache=createPublicEmbeddingCache({signature:'expected',backend:()=> 'browser-onnx',crypto:webcrypto,
      url:new URL('https://example.test/assets/public-embeddings.json'),now:()=>now,retryDelayMs:100,
      fetcher:async()=>{requests++;return failure==='malformed-json'?new Response('{')
        :failure==='signature'?new Response(JSON.stringify(manifest))
        :new Response('',{status:Number(failure)});},embed:async()=>[vector(2)]});
    assert.deepEqual(await cache.embed(['Public']),[vector(2)]);
    now=1000;assert.deepEqual(await cache.embed(['Public']),[vector(2)]);
    assert.equal(requests,1);
  });
}

test('a continuing outage exhausts the single retry without stalling later batches',async()=>{
  let now=0,requests=0;
  const cache=createPublicEmbeddingCache({signature:'x',backend:()=> 'browser-onnx',crypto:webcrypto,
    url:new URL('https://example.test/assets/public-embeddings.json'),now:()=>now,retryDelayMs:100,
    fetcher:async()=>{requests++;return new Response('',{status:503});},embed:async()=>[vector(2)]});
  for(now of [0,50,100,200,1000]) assert.deepEqual(await cache.embed(['Public']),[vector(2)]);
  assert.equal(requests,2);
});
