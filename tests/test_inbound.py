from __future__ import annotations

from dailydigest.inbound import extract_vote_line


def test_extract_vote_line_accepts_every_current_digest_section():
    body = "AI: +A3\nFunding: -F6\nEvents: E7"

    assert extract_vote_line(body) == "AI: +A3 Funding: -F6 Events: E7"
