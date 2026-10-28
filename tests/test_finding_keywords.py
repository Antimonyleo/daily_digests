"""Reader keywords are marked in the finding, so the important part stands out."""
from tests.test_top_picks import _request, _text_payload


def test_several_terms_are_marked_whole_word_and_capped():
    from dailydigest.web import _highlight_segments
    text = "DNA origami cages carry mRNA; origami folding, DNA-origami and DNA origamis differ."
    marked = [s["text"] for s in _highlight_segments(text, ["origami", "DNA origami", "mRNA"])]
    assert "".join(marked) == text
    assert [s["text"] for s in _highlight_segments(text, ["origami", "DNA origami", "mRNA"]) if s["match"]] == [
        "DNA origami", "mRNA", "origami", "origami"]
    assert sum(s["match"] for s in _highlight_segments(text, ["origami", "mRNA"], limit=2)) == 2
    assert _highlight_segments(text, []) == [{"text": text, "match": False}]


def test_finding_marks_reader_keywords_and_topic_aliases(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: [mRNA delivery]\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")
    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-05", item_count=1))
        item = store_mod.ItemRow(
            source="Example", section="research", external_id="k1", url="https://example.com/k1",
            title="Carriers", abstract="A fictional abstract.", score=0.9, digest_id="2026-05-05",
            item_label="R1", summary="Key finding: Lipid carriers improved mRNA delivery to <lung> cells.\n"
                                     "Why read: Compare carrier chemistry.")
        s.add(item)
        s.flush()
        item_id = int(item.id)
    store_mod.write_digest_features("2026-05-05", [("R1", item_id, 0.9, {
        "score": 0.9, "rank_score": 0.9, "confidence_score": 0.9, "why_shown": [], "tags": []})])
    text = _text_payload(web.index(_request("GET", "/")))
    assert 'Lipid carriers improved <mark class="finding-term">mRNA delivery</mark> to &lt;lung&gt; cells.' in text
