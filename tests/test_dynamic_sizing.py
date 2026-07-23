"""Tests for adaptive (supply-driven) section sizing.

A section shows as many items as clear the absolute topic-relevance floor
(`min_topic_relevance`, a true profile cosine), clamped to [min_*, top_*].
Off-topic items (low cosine) are excluded outright rather than filling slots.
"""

from __future__ import annotations

from dailydigest.config import get_settings
from dailydigest.pipeline import _dynamic_section_caps
from dailydigest.rank.ranker import _row_feature_key


class _Row:
    def __init__(self, rid: int, section: str) -> None:
        self.id = rid
        self.section = section
        self.source = "Test"


def _scored_and_features(section: str, topics: list[float]):
    """Build a (scored, features) pair for one section with given topic cosines."""
    scored = []
    features = {}
    for i, t in enumerate(topics):
        row = _Row(1000 + i, section)
        scored.append((row, t))  # fused score == topic for simplicity
        features[_row_feature_key(row)] = {"topic_score": t}
    return scored, features


def _cfg(monkeypatch, **kw):
    s = get_settings()
    monkeypatch.setattr(s, "adaptive_section_sizes", True)
    monkeypatch.setattr(s, "min_topic_relevance", 0.64)
    for k, v in kw.items():
        monkeypatch.setattr(s, k, v)
    return s


def test_many_on_topic_grows_toward_ceiling(monkeypatch):
    _cfg(monkeypatch, top_research=12, min_research=5)
    # 20 clearly on-topic items (>= 0.64), clamped to the ceiling of 12.
    scored, feats = _scored_and_features("research", [0.72] * 20)
    assert _dynamic_section_caps(scored, feats)["research"] == 12


def test_few_on_topic_shrinks_to_floor(monkeypatch):
    _cfg(monkeypatch, top_research=12, min_research=5)
    # Only 2 on-topic; the rest are off-topic (below the floor) -> floor wins.
    scored, feats = _scored_and_features("research", [0.71, 0.68] + [0.50] * 15)
    assert _dynamic_section_caps(scored, feats)["research"] == 5


def test_sizes_to_on_topic_supply(monkeypatch):
    _cfg(monkeypatch, top_research=12, min_research=5)
    # 8 on-topic items above the floor, between floor and ceiling.
    scored, feats = _scored_and_features("research", [0.70] * 8 + [0.55] * 6)
    assert _dynamic_section_caps(scored, feats)["research"] == 8


def test_off_topic_prestige_is_excluded(monkeypatch):
    """Items just below the relevance floor never count, however many there are
    (the bug where 30 prestigious-but-off-topic items filled the section)."""
    _cfg(monkeypatch, top_research=12, min_research=5)
    scored, feats = _scored_and_features("research", [0.60] * 25 + [0.70] * 3)
    # Only the 3 genuinely on-topic count -> floor of 5 (never empties).
    assert _dynamic_section_caps(scored, feats)["research"] == 5


def test_floor_is_tunable(monkeypatch):
    _cfg(monkeypatch, top_research=12, min_research=2, min_topic_relevance=0.69)
    # Raising the floor to 0.69 admits fewer items.
    scored, feats = _scored_and_features("research", [0.72, 0.70, 0.66, 0.66, 0.50])
    assert _dynamic_section_caps(scored, feats)["research"] == 2


def test_disabled_returns_fixed_caps(monkeypatch):
    _cfg(monkeypatch, top_research=12)
    monkeypatch.setattr(get_settings(), "adaptive_section_sizes", False)
    scored, feats = _scored_and_features("research", [0.72] * 3 + [0.50] * 5)
    assert _dynamic_section_caps(scored, feats)["research"] == 12


def test_falls_back_to_fused_score_when_no_snapshot(monkeypatch):
    _cfg(monkeypatch, top_research=6, min_research=1)
    # No features map -> uses the fused score in the tuple as the relevance proxy.
    scored = [(_Row(1, "research"), 0.72), (_Row(2, "research"), 0.70), (_Row(3, "research"), 0.40)]
    assert _dynamic_section_caps(scored, {})["research"] == 2


def test_news_sections_use_fixed_caps_not_topic_floor(monkeypatch):
    # Industry/regulatory/world carry lower-cosine news; they keep their fixed
    # caps rather than being gated by the research topic floor (which would empty
    # them). Here industry items are all below the floor yet the cap stays at 6.
    _cfg(monkeypatch, top_industry=6, min_industry=3)
    scored = [(_Row(i, "industry"), 0.50) for i in range(10)]
    feats = {_row_feature_key(r): {"topic_score": 0.50} for r, _ in scored}
    assert _dynamic_section_caps(scored, feats)["industry"] == 6


def test_filter_off_topic_gates_research_only(monkeypatch):
    from dailydigest.pipeline import _filter_off_topic

    _cfg(monkeypatch, min_topic_relevance=0.65)
    # research: on-topic kept, off-topic dropped. industry/regulatory: kept even
    # at low cosine (news scale differs; FDA/biotech news wanted regardless).
    r_on = _Row(1, "research"); r_off = _Row(2, "research")
    i_low = _Row(3, "industry"); g_low = _Row(4, "regulatory")
    feats = {
        _row_feature_key(r_on): {"topic_score": 0.72},
        _row_feature_key(r_off): {"topic_score": 0.60},
        _row_feature_key(i_low): {"topic_score": 0.40},
        _row_feature_key(g_low): {"topic_score": 0.20},
    }
    scored = [(r_on, 0.9), (r_off, 0.95), (i_low, 0.8), (g_low, 0.5)]
    kept = {row.id for row, _ in _filter_off_topic(scored, feats)}
    assert kept == {1, 3, 4}  # only off-topic research(2) dropped


def test_filter_off_topic_keeps_items_without_topic_snapshot(monkeypatch):
    from dailydigest.pipeline import _filter_off_topic

    _cfg(monkeypatch, min_topic_relevance=0.65)
    r = _Row(1, "research")
    scored = [(r, 0.9)]
    # No topic_score in features -> not dropped (avoids emptying on a fallback).
    assert _filter_off_topic(scored, {}) == scored


def test_news_section_caps_single_source_share():
    from dailydigest.rank.ranker import _pick_news_balanced

    class _SrcRow:
        def __init__(self, rid, source):
            self.id = rid; self.source = source; self.section = "industry"
    # 8 STAT items (highest scores) + 2 others; cap 6 -> STAT limited to ceil(6/3)=2,
    # diversity fills with the others, then backfills remaining slots.
    stat = [(_SrcRow(i, "STAT News"), 0.9 - i * 0.01) for i in range(8)]
    other = [(_SrcRow(100, "FierceBiotech"), 0.5), (_SrcRow(101, "BioPharma Dive"), 0.49)]
    picked = _pick_news_balanced(stat + other, 6)
    from collections import Counter
    by_src = Counter(r.source for r, _ in picked)
    assert len(picked) == 6
    assert by_src["STAT News"] <= 4  # capped on first pass (2), backfilled only as needed
    assert {"FierceBiotech", "BioPharma Dive"} <= set(by_src)  # diversity enforced


def test_news_quality_gate_drops_low_confidence_industry(monkeypatch):
    from dailydigest.pipeline import _filter_off_topic

    _cfg(monkeypatch, min_news_quality=0.45)
    good = _Row(1, "industry"); junk = _Row(2, "industry"); reg = _Row(3, "regulatory")
    feats = {
        _row_feature_key(good): {"confidence_score": 0.64},  # real news -> kept
        _row_feature_key(junk): {"confidence_score": 0.30},  # opinion/teaser -> dropped
        _row_feature_key(reg): {"confidence_score": 0.30},   # regulatory exempt -> kept
    }
    scored = [(good, 0.64), (junk, 0.30), (reg, 0.30)]
    kept = {row.id for row, _ in _filter_off_topic(scored, feats)}
    assert kept == {1, 3}


def test_industry_shrinks_to_quality_supply(monkeypatch):
    _cfg(monkeypatch, top_industry=6, min_industry=1, min_news_quality=0.45)
    # 1 quality item, 5 low-confidence opinion/teasers -> sized to ~1, not 6.
    rows = [_Row(1, "industry")] + [_Row(i, "industry") for i in range(2, 7)]
    feats = {_row_feature_key(rows[0]): {"confidence_score": 0.64}}
    feats.update({_row_feature_key(r): {"confidence_score": 0.30} for r in rows[1:]})
    scored = [(r, feats[_row_feature_key(r)]["confidence_score"]) for r in rows]
    assert _dynamic_section_caps(scored, feats)["industry"] == 1


def test_venue_credit_rescues_borderline_top_journal(monkeypatch):
    """A high-tier journal item just under the floor is rescued by venue credit;
    the same cosine from an unknown venue is not."""
    from dailydigest.pipeline import _filter_off_topic

    _cfg(monkeypatch, min_topic_relevance=0.68, venue_relevance_credit=0.10)

    class Row:
        def __init__(self, rid, source):
            self.id = rid
            self.section = "research"
            self.source = source
            self.title = ""
            self.abstract = ""

    nature = Row(1, "Nature Materials")   # high-tier, prestige 0.90 -> +0.04 credit
    unknown = Row(2, "Some Unknown Journal")  # no credit
    scored = [(nature, 0.66), (unknown, 0.66)]
    feats = {
        _row_feature_key(nature): {"topic_score": 0.66},
        _row_feature_key(unknown): {"topic_score": 0.66},
    }
    kept = {row.id for row, _ in _filter_off_topic(scored, feats)}
    assert nature.id in kept          # 0.66 + 0.04 >= 0.68
    assert unknown.id not in kept     # 0.66 < 0.68, no credit


def test_venue_credit_does_not_rescue_off_topic_prestige(monkeypatch):
    """Venue credit is small: a genuinely off-topic top-journal paper still fails."""
    from dailydigest.pipeline import _filter_off_topic

    _cfg(monkeypatch, min_topic_relevance=0.68, venue_relevance_credit=0.10)

    class Row:
        def __init__(self, rid, source):
            self.id = rid
            self.section = "research"
            self.source = source
            self.title = ""
            self.abstract = ""

    off_topic = Row(1, "Nature Materials")
    scored = [(off_topic, 0.55)]
    feats = {_row_feature_key(off_topic): {"topic_score": 0.55}}
    kept = {row.id for row, _ in _filter_off_topic(scored, feats)}
    assert off_topic.id not in kept   # 0.55 + 0.04 = 0.59 < 0.68


def test_final_score_cutoff_drops_bottom_scored_research(monkeypatch):
    """Unit test of the relative final-score gate: keep items scoring >= frac*top,
    drop the rest, but never below the hard-minimum."""
    from dailydigest.rank.ranker import _apply_final_score_cutoff

    s = get_settings()
    monkeypatch.setattr(s, "research_final_score_floor_frac", 0.35)
    monkeypatch.setattr(s, "research_final_score_min_keep", 3)

    class R:
        def __init__(self, rid):
            self.id = rid
            self.section = "research"
            self.source = "Nature"

    # top=1.0, floor=0.35. 0.10 is below the floor -> dropped; 4 above -> kept.
    selected = [(R(1), 1.0), (R(2), 0.8), (R(3), 0.6), (R(4), 0.4), (R(5), 0.10)]
    kept = {row.id for row, _ in _apply_final_score_cutoff(selected)}
    assert kept == {1, 2, 3, 4}  # bottom near-zero pick excluded


def test_pick_research_excludes_high_topic_but_bottom_final_score(monkeypatch):
    """The core P8 defect: an item with a HIGH topic score but a bottom/near-zero
    FINAL fused score must be EXCLUDED from the picked research section rather than
    padding a slot (previously it filled a slot because sizing gated on topic)."""
    from dailydigest.rank.ranker import _pick_research_balanced

    s = get_settings()
    monkeypatch.setattr(s, "research_final_score_floor_frac", 0.35)
    monkeypatch.setattr(s, "research_final_score_min_keep", 3)

    class R:
        def __init__(self, rid, source):
            self.id = rid
            self.section = "research"
            # Distinct high-quality journals so the per-source cap never binds and
            # all five are selected by topic sizing — isolating the final-score gate.
            self.source = source
            self.title = str(rid)
            self.abstract = "x" * 100

    # Four strong picks + one high-prestige item the learned model scored ~0.
    # (final scores are RRF min-maxed to [0,1]; the disliked paper lands at 0.0.)
    scored = [
        (R(1, "Nature"), 1.0),
        (R(2, "Science"), 0.85),
        (R(3, "Cell"), 0.70),
        (R(4, "The Lancet"), 0.55),
        (R(5, "NEJM"), 0.0),  # disliked-but-prestigious -> bottom final score
    ]
    picked_ids = {row.id for row, _ in _pick_research_balanced(scored, cap=12)}
    assert picked_ids == {1, 2, 3, 4}
    assert 5 not in picked_ids  # bottom/near-zero final score is dropped


def test_negative_penalty_gates_out_off_field_item(monkeypatch):
    """An item clearing the topic floor on cosine alone is still gated OUT when it
    carries a large negative-interest penalty (folded into the effective relevance)."""
    from dailydigest.pipeline import _filter_off_topic

    _cfg(monkeypatch, min_topic_relevance=0.68)

    class Row:
        def __init__(self, rid):
            self.id = rid
            self.section = "research"
            self.source = "Some Journal"
            self.title = ""
            self.abstract = ""

    on_field = Row(1)
    off_field = Row(2)
    scored = [(on_field, 0.72), (off_field, 0.72)]
    feats = {
        _row_feature_key(on_field): {"topic_score": 0.72, "negative_interest_penalty": 0.0},
        _row_feature_key(off_field): {"topic_score": 0.72, "negative_interest_penalty": 0.10},
    }
    kept = {row.id for row, _ in _filter_off_topic(scored, feats)}
    assert on_field.id in kept           # 0.72 >= 0.68
    assert off_field.id not in kept      # 0.72 - 0.10 = 0.62 < 0.68
