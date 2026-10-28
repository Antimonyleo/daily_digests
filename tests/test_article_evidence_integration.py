"""Article assertions remain quoted evidence, separate from ranking and quality."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from dailydigest.rank.article_evidence import assess_article_evidence
from dailydigest.rank.contribution_claims import contribution_claims
from dailydigest.rank.source_quality import breakdown_payload
from tests.test_web import _request


REPORTED = (
    "We introduce a forest risk predictor named <script>CLAIM_MARKER</script> for regional events. "
    "We validated the predictor at independent sites with held-out measurements. "
    "Data and code are available at https://example.com/research?a=1&b=2. "
    "A limitation is the small sample and performance outside the tested region was not assessed."
)
WEAK_CORE = (
    "We design new enzyme variants with specified molecular binding properties. "
    "Only a preliminary computational demonstration is supplied; independent validation was not performed."
)


def _row(iid, abstract, *, section="research"):
    return SimpleNamespace(id=iid, title=f"Fictional evidence probe {iid}", abstract=abstract,
                           section=section, source="Test journal", metadata_json="{}",
                           url=f"https://example.com/evidence/{iid}", external_id=f"probe-{iid}")


def test_selected_research_annotation_preserves_scores_scope_and_quality():
    from dailydigest import pipeline

    rows = [_row(1, REPORTED), _row(2, WEAK_CORE), _row(3, ""),
            _row(4, REPORTED, section="ai"), _row(5, REPORTED)]
    features = {row.id: {"topic_score": .2 if row.id == 1 else .9, "quality_score": .61,
                         "confidence_score": .62, "rank_score": .8, "final_score": .8,
                         "provisional_scope_label": "off_interest" if row.id == 1 else "core"}
                for row in rows}
    before = deepcopy(features)
    picked = [(row, .8) for row in rows[:4]]
    assert pipeline._annotate_research_evidence(picked, features) is None
    assert picked == [(row, .8) for row in rows[:4]]
    for row in rows[:3]:
        assert {key: value for key, value in features[row.id].items()
                if key not in {"contribution_claims", "article_evidence"}} == before[row.id]
        claims = features[row.id]["contribution_claims"]
        evidence = features[row.id]["article_evidence"]
        assert claims["scope_inferred"] is False
        assert evidence["assessment_basis"] in {"abstract", "insufficient"}
        assert "quality_score" not in evidence and "probability" not in evidence
        for dimension in evidence["dimensions"].values():
            for claim in dimension["claims"]:
                assert claim["source"] == "abstract"
                assert row.abstract[claim["start"]:claim["end"]] == claim["quote"]
                assert claim["verification"] == "author_reported_unverified"
    assert features[4] == before[4]  # Selected nonresearch remains untouched.
    assert features[5] == before[5]  # Unselected research remains untouched.
    assert features[1]["article_evidence"]["dimensions"]["validation"]["status"] == "reported"
    assert features[2]["article_evidence"]["dimensions"]["validation"]["status"] == "unknown"
    assert features[3]["contribution_claims"]["status"] == "uncertain"
    assert all(dimension["status"] == "unknown" for dimension in features[3]["article_evidence"]["dimensions"].values())


@pytest.mark.parametrize("abstract,reported", [(REPORTED, True), ("", False)])
def test_evidence_roundtrips_through_real_store_and_web_details(tmp_path, monkeypatch, abstract, reported):
    from dailydigest import store, web

    profile = tmp_path / "profile.yaml"
    profile.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")
    store.init_db()
    with store.session_scope() as session:
        session.add(store.DigestRow(id="2026-05-05", item_count=1))
        row = store.ItemRow(source="Test journal", section="research", external_id="evidence-web",
                            title="Fictional evidence probe", abstract=abstract,
                            url="https://example.com/evidence", score=.8,
                            digest_id="2026-05-05", item_label="R1")
        session.add(row)
        session.flush()
        iid = int(row.id)
        session.expunge(row)
    claims, evidence = contribution_claims(abstract), assess_article_evidence(row.title, abstract)
    payload = breakdown_payload(row, .9, final_score=.8, rank_score=.8, confidence_score=.61,
                                contribution_claims=claims, article_evidence=evidence)
    store.write_digest_features("2026-05-05", [("R1", iid, .8, payload)])
    persisted = store.load_digest_features("2026-05-05")[iid]
    assert persisted["contribution_claims"] == claims
    assert persisted["article_evidence"] == evidence
    assert persisted["confidence_score"] == .61 and persisted["rank_score"] == .8
    sections, _votes = web._load_today("2026-05-05")
    rendered = sections[0]["entries"][0]["ranking"]
    assert rendered["contribution_claims"] == claims
    assert rendered["article_evidence"] == evidence
    assert rendered["confidence_score"] == .61
    response = web.index(_request())
    assert response.status_code == 200
    html = response.body.decode()
    for label in ("Contribution evidence", "Validation evidence", "Data/code availability", "Limitations"):
        assert label in html
    assert "Author-reported claims; not independently verified." in html
    if reported:
        assert "&lt;script&gt;CLAIM_MARKER&lt;/script&gt;" in html
        assert "<script>CLAIM_MARKER</script>" not in html
        assert "We validated the predictor at independent sites with held-out measurements." in html
        assert "Data and code are available at https://example.com/research?a=1&amp;b=2." in html
        assert "A limitation is the small sample" in html
    else:
        # Unknown dimensions collapse into one line instead of repeating per row.
        assert html.count("Unknown from supplied text") == 1
        assert "quality_score" not in rendered["article_evidence"]


def test_normal_brew_annotates_after_reading_mode_and_captures_evidence(tmp_path, monkeypatch):
    from dailydigest import pipeline, store
    from dailydigest.models import Profile
    from dailydigest.rank import snapshot

    store.init_db()
    with store.session_scope() as session:
        rows = [store.ItemRow(source="Nature", section="research", external_id=f"evidence-{i}",
                              title=f"Fictional evidence research {i}", abstract=WEAK_CORE,
                              url=f"https://example.com/evidence/{i}",
                              published_at=datetime.now(timezone.utc))
                for i in range(pipeline.MINIMAL_RESEARCH_LIMIT + 1)]
        session.add_all(rows)
        session.flush()
        ids = [int(row.id) for row in rows]

    def recent(**_kwargs):
        with store.session_scope() as session:
            rows = [session.get(store.ItemRow, iid) for iid in ids]
            for row in rows:
                session.expunge(row)
            return rows

    def score(rows, _profile, _downweight, **_kwargs):
        pairs = [(row, .94 - i * .01) for i, row in enumerate(rows)]
        return pairs, {row.id: {"topic_score": .9, "quality_score": .61, "confidence_score": .61,
                               "final_score": value, "rank_score": value} for row, value in pairs}

    observed = []
    original_mode, original_annotate = pipeline.apply_reading_mode, pipeline._annotate_research_evidence
    original_snapshot = snapshot.build_ranking_snapshot

    def reading_mode(picked, mode, **kwargs):
        selected = original_mode(picked, mode, **kwargs)
        observed.append(("reading_mode", [row.id for row, _score in selected]))
        return selected

    def annotate(picked, features):
        observed.append(("annotation", [row.id for row, _score in picked]))
        return original_annotate(picked, features)

    def capture(**kwargs):
        observed.append(("snapshot", [row.id for row, _score, _label in kwargs["labeled"]]))
        return original_snapshot(**kwargs)

    monkeypatch.setenv("WITHIN_DAY_DEDUPE", "false")
    monkeypatch.setenv("CONTRIBUTION_RERANKER_PATH", "")
    monkeypatch.setenv("CONTRIBUTION_ENCODER_DIR", "")
    from dailydigest.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setattr(pipeline, "_digest_id", lambda: "2026-05-05")
    monkeypatch.setattr(pipeline, "ingest_all", lambda **_kwargs: 0)
    monkeypatch.setattr(pipeline, "load_profile", lambda: Profile(bio="Engineering", keywords=[]))
    monkeypatch.setattr(pipeline, "build_profile_matrix", lambda _profile: np.zeros((1, 3)))
    monkeypatch.setattr(pipeline, "_build_profile_with_rocchio", None)
    monkeypatch.setattr(pipeline, "_build_neg_centroid", None)
    monkeypatch.setattr(pipeline, "recent_items", recent)
    monkeypatch.setattr(pipeline, "_score_items_for_pipeline", score)
    # Start this integration seam with a prequalified slate larger than the
    # reading budget; venue quota behavior has separate selector regressions.
    monkeypatch.setattr(pipeline, "pick_top_per_section", lambda scored, _caps, **_kwargs: scored)
    monkeypatch.setattr(pipeline, "apply_reading_mode", reading_mode)
    monkeypatch.setattr(pipeline, "_annotate_research_evidence", annotate)
    monkeypatch.setattr(snapshot, "build_ranking_snapshot", capture)
    monkeypatch.setattr(pipeline, "summarize_items", lambda rows, **_kwargs: {int(row.id): "Summary." for row in rows})
    monkeypatch.setattr(pipeline, "send_digest", lambda *_args, **_kwargs: False)
    pipeline.run_all(dry_run=True, reading_mode="minimal")
    stages = [stage for stage, _ids in observed]
    assert stages.index("reading_mode") < stages.index("annotation") < stages.index("snapshot")
    selected = dict(observed)["annotation"]
    assert len(selected) == pipeline.MINIMAL_RESEARCH_LIMIT
    features = store.load_digest_features("2026-05-05")
    assert set(features) == set(selected)
    for payload in features.values():
        assert payload["contribution_claims"]["scope_inferred"] is False
        assert payload["article_evidence"]["dimensions"]["validation"]["status"] == "unknown"
        assert payload["confidence_score"] == .61
    runs = store.list_ranking_snapshots(digest_id="2026-05-05")
    captured = store.load_ranking_snapshot(runs[0]["run_id"])
    assert captured["profile_static_rows"] == 1  # static-only matrix: every row is static
    for candidate in captured["candidates"]:
        if candidate["selected"]:
            assert "article_evidence" in candidate["features"]
            assert candidate["features"]["quality_score"] == .61
        else:
            assert "article_evidence" not in candidate["features"]
