"""Tests for dailydigest.votes.parse_vote_line.

The function parses a vote line string and returns (up_labels, down_labels).
Tokens without a sign default to '+'. Labels are upper-cased.
"""

from __future__ import annotations

from dailydigest.votes import parse_vote_line


class TestBasicParsing:
    def test_mixed_up_and_down(self):
        up, down = parse_vote_line("+R3 R7 -I5")
        assert up == ["R3", "R7"]
        assert down == ["I5"]

    def test_unsigned_defaults_to_up(self):
        up, down = parse_vote_line("R3")
        assert up == ["R3"]
        assert down == []

    def test_explicit_plus_prefix(self):
        up, down = parse_vote_line("+R3")
        assert up == ["R3"]
        assert down == []

    def test_explicit_minus_prefix(self):
        up, down = parse_vote_line("-I5")
        assert up == []
        assert down == ["I5"]


class TestCaseFolding:
    def test_lower_case_labels_uppercased(self):
        # observed behavior: prefix uppercased, number kept as-is
        up, down = parse_vote_line("+r3 -i5")
        assert up == ["R3"]
        assert down == ["I5"]

    def test_mixed_case_section_prefix_uppercased(self):
        up, down = parse_vote_line("r3 R7 w1")
        assert up == ["R3", "R7", "W1"]
        assert down == []


class TestWhitespaceTolerance:
    def test_extra_spaces_between_tokens(self):
        up, down = parse_vote_line("  +R3   R7  ")
        assert up == ["R3", "R7"]
        assert down == []

    def test_leading_trailing_whitespace(self):
        up, down = parse_vote_line("   -I5   ")
        assert up == []
        assert down == ["I5"]


class TestEdgeCases:
    def test_empty_string(self):
        up, down = parse_vote_line("")
        assert up == []
        assert down == []

    def test_only_whitespace(self):
        up, down = parse_vote_line("   ")
        assert up == []
        assert down == []

    def test_garbage_tokens_skipped(self):
        # Tokens with no alpha+numeric pattern are silently skipped
        up, down = parse_vote_line("garbage no_labels 123 hello")
        assert up == []
        assert down == []

    def test_duplicate_labels_preserved(self):
        # observed behavior: parse_vote_line does NOT deduplicate;
        # duplicates appear as-is in the returned lists.
        up, down = parse_vote_line("+R3 +R3")
        assert up == ["R3", "R3"]  # observed behavior; verify intentional
        assert down == []

    def test_only_sign_tokens_produce_empty(self):
        # Bare + or - signs without a following label produce nothing
        up, down = parse_vote_line("+ -")
        assert up == []
        assert down == []

    def test_many_sections(self):
        up, down = parse_vote_line("+R1 +R2 -I1 W3 -G2")
        assert up == ["R1", "R2", "W3"]
        assert down == ["I1", "G2"]

    def test_sign_persistence_sticks_to_following_tokens(self):
        # A bare '-' sets current_sign='-' for all subsequent unsigned tokens.
        up, down = parse_vote_line("- R3 R7")
        # '-' sets current_sign='-'; both 'R3' and 'R7' inherit it.
        assert "R3" in down
        assert "R7" in down
