from __future__ import annotations

from datetime import date


def test_daily_tea_deck_is_stable_unique_and_changes_each_day():
    from dailydigest.tea_break import DAILY_TEA_DECK_SIZE, daily_tea_deck

    first_day = daily_tea_deck(date(2026, 8, 11))
    same_day = daily_tea_deck(date(2026, 8, 11))
    next_day = daily_tea_deck(date(2026, 8, 12))

    assert len(first_day) == DAILY_TEA_DECK_SIZE == 15
    assert len(set(first_day)) == DAILY_TEA_DECK_SIZE
    assert first_day == same_day
    assert first_day != next_day
    assert set(first_day).isdisjoint(next_day)
    assert sum(note.startswith("Tiny fact —") for note in first_day) == 10
    assert sum(note.startswith("Lab joke —") for note in first_day) == 5


def test_daily_tea_deck_draws_from_a_larger_curated_bank():
    from dailydigest.tea_break import DAILY_TEA_DECK_SIZE, TEA_NOTE_BANK

    assert len(TEA_NOTE_BANK) > DAILY_TEA_DECK_SIZE
    assert len({note.casefold().strip() for note in TEA_NOTE_BANK}) == len(
        TEA_NOTE_BANK
    )
    assert all(note.startswith(("Tiny fact —", "Lab joke —")) for note in TEA_NOTE_BANK)
