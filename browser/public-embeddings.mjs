// Public metadata only. Profile text is hashed locally and is never requested.
// The unfused GPU export has a separate vector space, including CPU fallback.
export function createPublicEmbeddingCache({signature, backend, embed, url,
  fetcher=globalThis.fetch, crypto=globalThis.crypto, shouldCancel=()=>false,
  timeoutMs=45000, cancelPollMs=50, retryDelayMs=300000, now=()=>performance.now()}) {
  const stats={hits:0,misses:0,status:'unused',reason:''};
  const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),
    byte=>byte.toString(16).padStart(2,'0')).join('');
  let pending,retryAt=Infinity,retried=false;
  const cancelled=()=>{if(shouldCancel())throw new Error('Brew stopped before publication');};
  async function load() {
    retryAt=Infinity;
    const controller=new AbortController();
    let timer,poll,transientFailure=false;
    const deadline=new Promise((_resolve,reject)=>{
      timer=setTimeout(()=>{transientFailure=true;controller.abort();reject(new Error('Public comparisons download timed out'));},timeoutMs);
      poll=setInterval(()=>{
        if(shouldCancel()){controller.abort();reject(new Error('Brew stopped before publication'));}
      },cancelPollMs);
    });
    const bounded=value=>Promise.race([value,deadline]);
    const request=async(target,label)=>{
      let response;
      try {response=await bounded(fetcher(target,{credentials:'omit',signal:controller.signal}));}
      catch(error){transientFailure=true;throw error;}
      if(!response.ok) {
        transientFailure=[408,425,429].includes(response.status)||response.status>=500;
        throw new Error(`${label} (${response.status})`);
      }
      return response;
    };
    const readBody=async value=>{
      try {return await bounded(value);}
      catch(error){if(error instanceof TypeError||error?.name==='AbortError') transientFailure=true;throw error;}
    };
    try {
      cancelled();
      const response=await request(url,'Public comparisons unavailable');
      const manifest=await readBody(response.json());
      if(manifest.version!==1||manifest.public_only!==true||manifest.signature!==signature
        ||manifest.backend!=='browser-onnx'||manifest.dimension!==384||manifest.dtype!=='float32-le'
        ||!Array.isArray(manifest.hashes)||manifest.hashes.length>10000
        ||new Set(manifest.hashes).size!==manifest.hashes.length
        ||manifest.hashes.some(hash=>!/^[a-f0-9]{64}$/.test(hash))
        ||manifest.bytes!==manifest.hashes.length*384*4||manifest.bytes>16_000_000
        ||!/^public-embeddings-[a-zA-Z0-9-]+\.f32$/.test(manifest.file)
        ||!/^[a-f0-9]{64}$/.test(manifest.sha256)) throw new Error('Public comparisons do not match this model');
      const data=await request(new URL(manifest.file,url),'Public comparison file unavailable');
      const bytes=await readBody(data.arrayBuffer());
      if(bytes.byteLength!==manifest.bytes||await bounded(digest(bytes))!==manifest.sha256)
        throw new Error('Public comparison checksum mismatch');
      const view=new DataView(bytes),vectors=new Map();
      for(let row=0;row<manifest.hashes.length;row++) {
        const vector=Array.from({length:384},(_,column)=>view.getFloat32((row*384+column)*4,true));
        const norm=vector.reduce((sum,value)=>sum+value*value,0);
        if(!Number.isFinite(norm)||Math.abs(norm-1)>.001) throw new Error('Invalid public comparison vector');
        vectors.set(manifest.hashes[row],vector);
      }
      stats.status='ready';stats.reason='';return vectors;
    } catch(error) {
      if(shouldCancel()){stats.status='cancelled';throw new Error('Brew stopped before publication');}
      // Feed or cache failures must never prevent an ordinary local brew.
      if(transientFailure&&!retried) retryAt=now()+retryDelayMs;
      stats.status='unavailable';stats.reason=error.message;return new Map();
    } finally {clearTimeout(timer);clearInterval(poll);controller.abort();}
  }
  async function cachedEmbed(input) {
    const texts=Array.from(input);
    if(!texts.length) return [];
    cancelled();
    if(backend()!=='browser-onnx') {
      stats.status='incompatible-backend';stats.misses+=texts.length;return embed(texts);
    }
    // One recovery attempt per tab, after a quiet period. Missing or invalid
    // assets stay on local inference; a long outage cannot stall every batch.
    if(pending&&now()>=retryAt) {pending=undefined;retried=true;}
    const vectors=await (pending ||= load().catch(error=>{pending=undefined;throw error;}));
    if(!vectors.size){cancelled();stats.misses+=texts.length;return embed(texts);}
    const hashes=await Promise.all(texts.map(text=>digest(new TextEncoder().encode(text))));
    cancelled();
    const output=hashes.map(hash=>vectors.get(hash)?.slice()),missing=[];
    for(let index=0;index<output.length;index++) if(!output[index]) missing.push(index);
    stats.hits+=texts.length-missing.length;stats.misses+=missing.length;
    if(missing.length) {
      const fresh=await embed(missing.map(index=>texts[index]));
      if(fresh.length!==missing.length) throw new Error('Local comparison count mismatch');
      missing.forEach((index,offset)=>{output[index]=fresh[offset];});
    }
    return output;
  }
  return {embed:cachedEmbed,stats};
}
