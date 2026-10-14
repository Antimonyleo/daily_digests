// BGE uses single-sequence BERT inputs. Native tokenizers truncate the body
// before post-processing; Transformers.js 3.8.1 slices the final [SEP] off.
// Keep the native terminal token, then pad the batch with the native pad ID.
export function prepareBertBatch(encoded, { maxLength=512, padId=0, sepId=102 }={}) {
  const ids=encoded.input_ids;
  if(!ids?.length || maxLength<2 || ids.some(row=>row[0]!==101 || row.at(-1)!==sepId)) throw new Error('Expected single-sequence BGE/BERT tokenization');
  const length=Math.min(maxLength,Math.max(...ids.map(row=>row.length)));
  const result={};
  for(const [key,rows] of Object.entries(encoded)) {
    if(rows.length!==ids.length || rows.some((row,i)=>row.length!==ids[i].length)) throw new Error('Tokenizer fields are misaligned');
    result[key]=rows.map(row=>{
      const trimmed=row.length>length?[...row.slice(0,length-1),row.at(-1)]:row.slice();
      while(trimmed.length<length)trimmed.push(key==='input_ids'?padId:0);
      return trimmed;
    });
  }
  return { encoded:result, dims:[ids.length,length] };
}
