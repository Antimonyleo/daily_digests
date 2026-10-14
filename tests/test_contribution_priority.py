"""Persisted contribution scores must reach the actual web priority filter."""
import re

import pytest

from tests.test_web import _request


@pytest.mark.parametrize("active,score,confidence,expected", [
    (True, .72, .95, "read"),
    (True, .90, .41, "spotlight"),
    (False, .90, .41, "skim"),
    (False, .41, .95, "spotlight"),
])
def test_web_priority_uses_adjusted_score_only_for_contribution_models(
    tmp_path, monkeypatch, active, score, confidence, expected,
):
    from dailydigest import store, web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")
    store.init_db()
    with store.session_scope() as session:
        session.add(store.DigestRow(id="2026-05-05", item_count=1))
        item = store.ItemRow(source="OpenAlex", section="research", external_id="priority-probe",
                             url="https://example.com/priority", title="Contribution priority probe",
                             abstract="A sufficiently detailed abstract about a new engineering method.",
                             score=score, digest_id="2026-05-05", item_label="R1")
        session.add(item)
        session.flush()
        iid = int(item.id)
    payload = {"score": score, "rank_score": score, "confidence_score": confidence,
               "source_bucket": "aggregator", "tags": [], "why_shown": []}
    if active:
        payload["contribution_rerank"] = {
            "original_final_score": confidence, "rank_signal": .7, "weight": .7,
            "model_fingerprint": "frozen-model", "score_meaning": "ranking signal; not probability",
        }
    store.write_digest_features("2026-05-05", [("R1", iid, score, payload)])
    response = web.index(_request())
    assert response.status_code == 200
    match = re.search(rf'<div class="item" data-item-id="{iid}"[^>]*data-priority="([^"]+)"',
                      response.body.decode())
    assert match is not None
    assert match.group(1) == expected
    persisted = store.load_digest_features("2026-05-05")[iid]
    assert persisted["confidence_score"] == confidence
    if active:
        assert persisted["contribution_rerank"]["model_fingerprint"] == "frozen-model"
