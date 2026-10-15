// Build-time inference uses the production CPU graph, tokenizer and WASM engine.
// This localhost page never starts Python, reads a profile or contacts a provider.
import {createServer} from 'node:http';
import {readFile,writeFile,appendFile} from 'node:fs/promises';
import {resolve,sep,extname} from 'node:path';
import {chromium} from '../prototypes/browser-compute/node_modules/playwright-core/index.mjs';

const [rootArgument,planPath,resultPath]=process.argv.slice(2);
if(!rootArgument||!planPath||!resultPath) throw new Error('Usage: node precompute-public-embeddings.mjs SITE PLAN_JSON RESULT_F32');
const root=resolve(rootArgument),plan=JSON.parse(await readFile(planPath,'utf8'));
if(!Array.isArray(plan.texts)||plan.texts.length>10000) throw new Error('Invalid public embedding plan');
const mime={'.mjs':'text/javascript','.js':'text/javascript','.json':'application/json','.wasm':'application/wasm','.onnx':'application/octet-stream'};
const server=createServer(async(request,response)=>{
  response.setHeader('Cross-Origin-Opener-Policy','same-origin');
  response.setHeader('Cross-Origin-Embedder-Policy','require-corp');
  const path=new URL(request.url,'http://localhost').pathname;
  if(path==='/__public_embedding_builder'){
    response.setHeader('Content-Type','text/html');response.end('<!doctype html><title>Public embedding build</title>');return;
  }
  try {
    const file=resolve(root,'.'+decodeURIComponent(path));
    if(!file.startsWith(root+sep)) throw new Error('Outside build');
    const bytes=await readFile(file);response.setHeader('Content-Type',mime[extname(file)]||'application/octet-stream');response.end(bytes);
  } catch {response.statusCode=404;response.end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
let shuttingDown=false;
async function shutdown() {
  if(shuttingDown)return;
  shuttingDown=true;
  // Playwright owns a detached Chromium group. A normal Node exit also runs
  // Playwright's synchronous group-kill hook if graceful close cannot finish.
  const force=setTimeout(()=>process.exit(124),1500);
  try {await browser?.close();} finally {clearTimeout(force);server.close();process.exit(124);}
}
process.on('SIGTERM',shutdown);process.on('SIGINT',shutdown);
const external=[];
try {
  browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
  const context=await browser.newContext();
  const base=`http://127.0.0.1:${server.address().port}`;
  await context.route('**/*',route=>{
    if(!route.request().url().startsWith(base+'/')){external.push(route.request().url());return route.abort();}
    return route.continue();
  });
  const page=await context.newPage();
  await writeFile(resultPath,new Uint8Array());
  let checkpointed=0;
  await page.exposeFunction('__savePublicEmbeddingChunk',async rows=>{
    await appendFile(resultPath,new Uint8Array(new Float32Array(rows.flat()).buffer));
    checkpointed+=rows.length;
    if(checkpointed%512===0||checkpointed===plan.texts.length)
      console.error(JSON.stringify({checkpointed,total:plan.texts.length}));
  });
  await page.goto(base+'/__public_embedding_builder');
  const started=performance.now();
  const result=await page.evaluate(async({texts,runtimePath,modelPath,verify,mode})=>{
    const {AutoTokenizer,AutoModel,Tensor,env}=await import('./'+runtimePath+'@huggingface/transformers/dist/transformers.min.js');
    const {createEmbeddingEngine}=await import('./compute.mjs');
    const {prepareBertBatch}=await import('./bge-tokenizer.mjs');
    env.allowRemoteModels=false;env.allowLocalModels=true;env.useBrowserCache=false;
    env.localModelPath=new URL('./'+modelPath,location.href).href;
    env.backends.onnx.wasm.wasmPaths=new URL('./'+runtimePath+'onnxruntime-web/dist/',location.href).href;
    env.backends.onnx.wasm.proxy=false;
    const engine=createEmbeddingEngine({mode:mode||'multicore',capabilities:{gpu:false,multicore:crossOriginIsolated,threads:4},env:env.backends.onnx,
      loadTokenizer:options=>AutoTokenizer.from_pretrained('bge-small',{local_files_only:true,...options}),
      loadModel:({exportKind,...options})=>AutoModel.from_pretrained('bge-small',{local_files_only:true,...options}),
      prepareBatch:prepareBertBatch,makeTensor:(data,dims)=>new Tensor('int64',data,dims),progress:()=>{}});
    const local=[],inferenceStarted=performance.now();
    for(let start=0;start<texts.length;start+=128){
      const chunk=await engine.embed(texts.slice(start,start+128));
      await globalThis.__savePublicEmbeddingChunk(chunk);local.push(...chunk);
    }
    const report={threads:engine.threads,device:engine.device,externalRequests:0,localEncodingMs:performance.now()-inferenceStarted};
    if(verify){
      const {createPublicEmbeddingCache}=await import('./public-embeddings.mjs');
      const signature=(await (await fetch('./assets/public-embeddings.json')).json()).signature;
      const cache=createPublicEmbeddingCache({signature,backend:()=>engine.cacheBackend,embed:items=>engine.embed(items),url:new URL('./assets/public-embeddings.json',location.href)});
      // Reverse batching changes padding; verify vector tolerance and top order.
      const cacheStarted=performance.now(),before=engine.stats.completedInputs;
      const cached=(await cache.embed([...texts].reverse())).reverse();
      report.publicCacheMs=performance.now()-cacheStarted;
      report.locallyEncodedDuringCache=engine.stats.completedInputs-before;
      let maximum=0;
      for(let row=0;row<local.length;row++)for(let column=0;column<384;column++)maximum=Math.max(maximum,Math.abs(local[row][column]-cached[row][column]));
      const queries=await engine.embed(['Represent this sentence for searching relevant passages: targeted protein degradation',
        'Represent this sentence for searching relevant passages: asymmetric catalysis',
        'Represent this sentence for searching relevant passages: quantum error correction']);
      const rank=(query,matrix)=>matrix.map((vector,index)=>({index,score:vector.reduce((sum,value,column)=>sum+value*query[column],0)})).sort((a,b)=>b.score-a.score).map(row=>row.index);
      report.maxVectorDifference=maximum;report.rankingsEqual=queries.every(query=>JSON.stringify(rank(query,local))===JSON.stringify(rank(query,cached)));
      report.cache=cache.stats;
      if(maximum>.00002||!report.rankingsEqual)throw new Error('Public vector parity failed: '+JSON.stringify(report));
    }
    return {report};
  },plan);
  if(external.length)throw new Error('Public embedding builder attempted external access');
  console.log(JSON.stringify({...result.report,inputs:plan.texts.length,bytes:checkpointed*384*4,elapsedMs:performance.now()-started,externalRequests:external.length}));
} finally {
  await browser?.close();await new Promise(resolve=>server.close(resolve));
}
