// Pure diagnostics: parity is measured, never inferred from a successful brew.
export function compareVectors(actual, expected) {
  if (actual.length !== expected.length || !actual.length) throw new Error('Vector counts differ');
  let minCosine = 1, maxElementDifference = 0;
  for (let row = 0; row < actual.length; row++) {
    const a = actual[row], b = expected[row];
    if (!a.length || a.length !== b.length) throw new Error('Embedding dimensions differ');
    let dot = 0, aa = 0, bb = 0;
    for (let col = 0; col < a.length; col++) {
      if (!Number.isFinite(a[col]) || !Number.isFinite(b[col])) throw new Error('Nonfinite embedding');
      dot += a[col] * b[col]; aa += a[col] ** 2; bb += b[col] ** 2;
      maxElementDifference = Math.max(maxElementDifference, Math.abs(a[col] - b[col]));
    }
    if (!aa || !bb) throw new Error('Zero embedding');
    minCosine = Math.min(minCosine, dot / Math.sqrt(aa * bb));
  }
  return { minCosine, maxElementDifference };
}

export function compareRankings(actual, expected) {
  const rows = new Map(expected.candidates.map(row => [row.id, row]));
  if (rows.size !== expected.candidates.length || actual.candidates.length !== rows.size ||
      new Set(actual.candidates.map(row => row.id)).size !== rows.size) throw new Error('Candidate identities differ');
  let maxTopicDifference = 0, maxScoreDifference = 0;
  const decisionMismatches = [];
  for (const row of actual.candidates) {
    const ref = rows.get(row.id);
    if (!ref) throw new Error('Unknown candidate');
    if (![row.topic_score, ref.topic_score, row.final_score, ref.final_score].every(Number.isFinite)) throw new Error('Nonfinite score');
    maxTopicDifference = Math.max(maxTopicDifference, Math.abs(row.topic_score - ref.topic_score));
    maxScoreDifference = Math.max(maxScoreDifference, Math.abs(row.final_score - ref.final_score));
    if (row.status !== ref.status || row.source_bucket !== ref.source_bucket) decisionMismatches.push(row.id);
  }
  return { maxTopicDifference, maxScoreDifference, decisionMismatches,
    selectionMatches: JSON.stringify(actual.selected_ids) === JSON.stringify(expected.selected_ids) };
}

export function makeVerdict(browser, policy) {
  const policyPass = policy.selectionMatches && !policy.decisionMismatches.length && policy.maxScoreDifference <= 1e-5;
  const inferencePass = browser.documents.minCosine >= .9999 && browser.queries.minCosine >= .9999 &&
    browser.ranking.maxTopicDifference <= .002 && browser.ranking.maxScoreDifference <= .002 &&
    browser.ranking.selectionMatches && !browser.ranking.decisionMismatches.length;
  return { policyPass, inferencePass, passed: policyPass && inferencePass,
    limits: { minEmbeddingCosine: .9999, maxInferenceScoreDifference: .002, maxPolicyScoreDifference: 1e-5 } };
}
