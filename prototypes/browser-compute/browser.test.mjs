import test from 'node:test';
import assert from 'node:assert/strict';
import { compareVectors, compareRankings, makeVerdict } from './parity.mjs';
import { prepareBertBatch } from './bge-tokenizer.mjs';

test('long BERT input retains the native separator instead of an extra body token',()=>{
  const raw={input_ids:[[101,11,12,13,14,102],[101,22,102]],attention_mask:[[1,1,1,1,1,1],[1,1,1]],token_type_ids:[[0,0,0,0,0,0],[0,0,0]]};
  const before=structuredClone(raw),result=prepareBertBatch(raw,{maxLength:5});
  assert.deepEqual(result.encoded.input_ids,[[101,11,12,13,102],[101,22,102,0,0]]);
  assert.deepEqual(result.encoded.attention_mask,[[1,1,1,1,1],[1,1,1,0,0]]);
  assert.deepEqual(raw,before);
});
test('exact-boundary BERT input is not shortened',()=>{
  const result=prepareBertBatch({input_ids:[[101,11,12,102]],attention_mask:[[1,1,1,1]]},{maxLength:4});
  assert.deepEqual(result.encoded.input_ids,[[101,11,12,102]]);
  assert.throws(()=>prepareBertBatch({input_ids:[[101,11,12]],attention_mask:[[1,1,1]]}));
});

test('a selected slate match cannot hide changed quality decisions',()=>{
  const ref={selected_ids:[1],candidates:[{id:1,topic_score:.8,final_score:.9,status:'selected',source_bucket:'published_journal'},{id:2,topic_score:.7,final_score:.6,status:'quality',source_bucket:'low_impact_journal'}]};
  const changed=structuredClone(ref);changed.candidates[1].status='budget_or_source_policy';
  const policy=compareRankings(changed,ref);
  const browser={documents:{minCosine:1},queries:{minCosine:1},ranking:policy};
  assert.equal(policy.selectionMatches,true);assert.deepEqual(policy.decisionMismatches,[2]);
  assert.equal(makeVerdict(browser,policy).passed,false);
});
test('similar embeddings alone cannot establish selection parity',()=>{
  const policy={selectionMatches:true,decisionMismatches:[],maxScoreDifference:0};
  const browser={documents:{minCosine:1},queries:{minCosine:1},ranking:{...policy,maxTopicDifference:0,selectionMatches:false}};
  assert.equal(makeVerdict(browser,policy).inferencePass,false);
});
test('failed or missing vectors are never silently replaced',()=>{
  for(const bad of [[[0,0]],[[NaN,1]],[[1]],[]])assert.throws(()=>compareVectors(bad,[[1,0]]));
  assert.equal(compareVectors([[1,0]],[[1,0]]).minCosine,1);
});
