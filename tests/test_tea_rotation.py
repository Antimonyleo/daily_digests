"""Pip must work through the whole bank before repeating a card."""

from __future__ import annotations

from datetime import date, timedelta


def _reset_store(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    store_mod._INITIALIZED = False
    store_mod.init_db()
    return store_mod


def test_no_card_repeats_until_its_bank_is_exhausted(tmp_path, monkeypatch):
    """The old deck was a pure function of the date over a fixed bank.

    57 cards drawn 15 at a time meant a card returned every ~4 days and the
    same rotation then looped forever -- the reader saw "the same boring joke"
    because it genuinely was the same joke, on a fixed cycle. Serving the least
    recently shown cards must instead exhaust each bank before repeating.
    """
    _reset_store(tmp_path, monkeypatch)
    from dailydigest.tea_break import DAILY_FACTS, DAILY_JOKES, _TEA_FACTS, _TEA_JOKES, daily_tea_deck

    # Facts have the smaller bank relative to its draw, so it recycles first.
    cycle_days = min(len(_TEA_JOKES) // DAILY_JOKES, len(_TEA_FACTS) // DAILY_FACTS)
    assert cycle_days >= 7, f"bank too small to last a week: {cycle_days} days"

    start = date(2026, 8, 22)
    seen: set[str] = set()
    for offset in range(cycle_days):
        deck = daily_tea_deck(start + timedelta(days=offset))
        repeats = seen & set(deck)
        assert not repeats, f"day {offset} repeated {len(repeats)} card(s): {sorted(repeats)[:2]}"
        seen |= set(deck)

    # And the whole run drew genuinely distinct cards, not a short loop.
    assert len(seen) == cycle_days * (DAILY_JOKES + DAILY_FACTS)


def test_deck_is_stable_within_a_day(tmp_path, monkeypatch):
    """Reloading the page must not reshuffle Pip mid-tea-break."""
    _reset_store(tmp_path, monkeypatch)
    from dailydigest.tea_break import daily_tea_deck

    day = date(2026, 8, 22)
    assert daily_tea_deck(day) == daily_tea_deck(day)


def test_rotation_survives_a_missing_store(tmp_path, monkeypatch):
    """Pip must never break the page, even with no database behind it."""
    from dailydigest import tea_break

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no store")

    monkeypatch.setattr(tea_break, "daily_tea_deck", tea_break.daily_tea_deck)
    import dailydigest.store as store_mod

    monkeypatch.setattr(store_mod, "tea_deck_for_day", _boom)
    deck = tea_break.daily_tea_deck(date(2026, 8, 22))
    assert len(deck) == tea_break.DAILY_JOKES + tea_break.DAILY_FACTS


def test_bank_has_no_duplicate_cards():
    """A duplicated entry would silently shorten the rotation."""
    from dailydigest.tea_break import _TEA_FACTS, _TEA_JOKES

    assert len(set(_TEA_JOKES)) == len(_TEA_JOKES)
    assert len(set(_TEA_FACTS)) == len(_TEA_FACTS)
    assert not set(_TEA_JOKES) & set(_TEA_FACTS)
