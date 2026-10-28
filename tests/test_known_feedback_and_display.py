"""'Not new to me' is a first-tier personal-familiarity reason; Display is fixed."""

import re
from pathlib import Path

from tests.test_reader_persona_ui import feedback_html

ROOT = Path(__file__).resolve().parents[1]


def test_not_new_to_me_is_first_tier_once_and_keeps_its_data_key():
    html = feedback_html()
    primary, secondary = html.split('<details class="more-reasons"', 1)
    assert re.findall(r'data-reason="already_known"[^>]*>Not new to me</button>', primary)
    assert html.count('data-reason="already_known"') == 1
    assert "already_known" not in secondary
    assert "Already known" not in html
    # The four grades keep their labels and values.
    assert re.findall(r'data-grade="(\d+)"', html) == ["100", "70", "40", "10"]


def test_selected_not_new_to_me_is_visible_without_opening_more_reasons():
    html = feedback_html("already_known")
    assert '<details class="more-reasons" open>' not in html
    assert re.search(r'data-reason="already_known"[^>]*aria-pressed="true"', html)


def test_not_new_to_me_alone_is_not_a_topic_negative():
    from dailydigest import votes

    assert "already_known" in votes.ALLOWED_REASONS
    assert votes.is_semantic_vote(-1, ["already_known"]) is False
    assert votes.is_semantic_vote(-1, ["already_known", "off_topic"]) is True
    assert votes.is_semantic_vote(1, ["already_known"]) is True


def test_display_menu_is_fixed_bottom_right_and_right_aligned_on_phones():
    base = (ROOT / "templates" / "_base.html.j2").read_text()
    rule = re.search(r"\.app-utilities \{([^}]+)\}", base).group(1)
    assert "position: fixed" in rule
    assert "right: calc(max(16px, env(safe-area-inset-right" in rule
    assert "bottom: calc(max(16px, env(safe-area-inset-bottom" in rule
    assert "margin: -54px" not in base
    # The menu opens upward from the corner instead of being centred.
    assert "translateX(50%)" not in base
    assert 'visual.addEventListener("resize", followVisualViewport)' in base
