from copy import deepcopy

import pytest

from dailydigest.rank.contribution_claims import FEATURE_PROTOCOL, contribution_text
from dailydigest.rank.semantic_rerank import LocalContributionReranker
from tests.test_contribution_serving import bundle as bundle, _row, _write


def test_claim_feature_protocol_is_applied_by_local_serving(bundle):
    artifact = deepcopy(bundle.artifact)
    artifact['feature_protocol'] = FEATURE_PROTOCOL
    _write(bundle, artifact)
    serving = LocalContributionReranker(bundle.path, '/local', bundle.profile)
    abstract = 'Protein methods are common. Here we introduce a transferable protein design model.'
    row = _row(1, abstract=abstract)
    features = {}
    serving.rerank([(row, .8)], features)
    pairs = bundle.calls[-1]
    assert all(text == contribution_text(row.title, abstract) for _, text in pairs)
    assert 'methods are common' not in pairs[0][1]
    assert features[1]['contribution_rerank']['feature_protocol'] == FEATURE_PROTOCOL


def test_unknown_claim_feature_protocol_is_rejected_before_encoder_loading(bundle):
    artifact = deepcopy(bundle.artifact)
    artifact['feature_protocol'] = {**FEATURE_PROTOCOL, 'version': 'unknown'}
    _write(bundle, artifact)
    with pytest.raises(ValueError, match='feature protocol'):
        LocalContributionReranker(bundle.path, '/local', bundle.profile)
    assert not bundle.calls
