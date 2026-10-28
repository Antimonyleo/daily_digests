"""Top picks are the strongest of every section, not only high-scoring research."""
import re

from tests.test_web import _request, _text_payload


def _digest(tmp_path, monkeypatch, sections):
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
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")
    store_mod.init_db()
    features = []
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-05", item_count=sum(map(len, sections.values()))))
        for section, scores in sections.items():
            for rank, score in enumerate(scores, start=1):
                label = f"{section[0].upper()}{rank}"
                item = store_mod.ItemRow(
                    source="Example", section=section, external_id=f"{section}-{rank}",
                    url=f"https://example.com/{section}/{rank}", title=f"{section} pick {rank}",
                    abstract="A fictional abstract.", score=score, digest_id="2026-05-05", item_label=label,
                )
                s.add(item)
                s.flush()
                features.append((label, int(item.id), score, {"score": score, "rank_score": score,
                                 "confidence_score": score, "why_shown": [], "tags": []}))
    store_mod.write_digest_features("2026-05-05", features)
    text = _text_payload(web.index(_request("GET", "/")))
    pattern = r'class="item"[^>]*data-top-pick="(true|false)"[^>]*>.*?class="title-link"[^>]*>([^<]+)<'
    return {title: flag == "true" for flag, title in re.findall(pattern, text, re.S)}


def test_top_picks_include_the_best_of_every_section(tmp_path, monkeypatch):
    picks = _digest(tmp_path, monkeypatch, {
        "research": [0.95, 0.9, 0.88, 0.86, 0.8, 0.79, 0.75, 0.7, 0.68, 0.66],
        "ai": [0.42], "world": [0.35, 0.3],
    })
    top = sorted(title for title, flag in picks.items() if flag)
    assert top == ["ai pick 1", "research pick 1", "research pick 2", "research pick 3", "world pick 1"]
    assert len(picks) == 13


def test_top_picks_filter_uses_the_per_section_mark(tmp_path, monkeypatch):
    from dailydigest import web
    _digest(tmp_path, monkeypatch, {"research": [0.9], "world": [0.3]})
    text = _text_payload(web.index(_request("GET", "/")))
    assert 'itemEl.dataset.topPick === "true"' in text
    assert "Show the strongest picks from every section" in text
    assert 'data-priority="skim"' in text, "the confidence border is unchanged"
