"""Live tea cards: computed per brew, so they cannot fall into a rotation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta


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


def _seed_brew(store_mod, digest_id="2026-08-22", n_items=3):
    now = datetime.now(UTC)
    ids = []
    with store_mod.session_scope() as s:
        for i in range(n_items):
            row = store_mod.ItemRow(
                source=f"Journal {i}",
                section="research",
                external_id=f"seed-{i}",
                url=f"https://example.com/{i}",
                title=f"A paper about self-assembly number {i}",
                abstract="An abstract.",
                published_at=now - timedelta(days=3),
                fetched_at=now,
            )
            s.add(row)
            s.flush()
            ids.append(int(row.id))
    store_mod.write_digest(digest_id, [(f"R{i + 1}", item) for i, item in enumerate(ids)])
    store_mod.write_digest_audit(
        digest_id,
        "candidate_funnel",
        [
            {
                "recent_items": 3103,
                "window_days": 4,
                "quality_gate_drops": [{"item_id": 1}, {"item_id": 2}],
                "cross_day_near_dup_drops": [
                    {"item_id": 9, "max_similarity": 0.99},
                    {"item_id": 10, "max_similarity": 1.0},
                ],
            }
        ],
    )
    return ids


class TestBrewObservations:
    def test_reports_the_run_that_just_happened(self, tmp_path, monkeypatch):
        store_mod = _reset_store(tmp_path, monkeypatch)
        _seed_brew(store_mod)
        from dailydigest.tea_live import brew_observations

        cards = brew_observations("2026-08-22")
        blob = " ".join(cards)
        assert cards, "no observations produced from a complete brew"
        assert "3,103" in blob, blob
        assert "one in" in blob
        # The near-duplicate card must quote the strongest match, not the first.
        assert "100%" in blob

    def test_is_silent_rather_than_wrong_when_there_is_no_brew(self, tmp_path, monkeypatch):
        _reset_store(tmp_path, monkeypatch)
        from dailydigest.tea_live import brew_observations

        assert brew_observations("2026-08-22") == []

    def test_never_raises(self, tmp_path, monkeypatch):
        """Pip must not be able to break the page."""
        _reset_store(tmp_path, monkeypatch)
        from dailydigest import tea_live

        monkeypatch.setattr(
            tea_live, "_brew_observations", lambda _d: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        assert tea_live.brew_observations("2026-08-22") == []


class TestCorpusObservations:
    def test_empty_corpus_yields_nothing(self, tmp_path, monkeypatch):
        _reset_store(tmp_path, monkeypatch)
        from dailydigest.tea_live import corpus_observations

        assert corpus_observations(4) == []

    def test_counts_the_window(self, tmp_path, monkeypatch):
        store_mod = _reset_store(tmp_path, monkeypatch)
        now = datetime.now(UTC)
        with store_mod.session_scope() as s:
            for i in range(12):
                s.add(
                    store_mod.ItemRow(
                        source="bioRxiv (recent)",
                        section="research",
                        external_id=f"c-{i}",
                        url=f"https://example.com/c{i}",
                        title="DNA origami meets machine learning",
                        fetched_at=now,
                    )
                )
        from dailydigest.tea_live import corpus_observations

        cards = corpus_observations(4)
        assert cards, "a populated window produced no observations"
        assert any("bioRxiv" in c for c in cards)


class TestGeneratedJokes:
    def test_returns_the_requested_count_and_no_duplicates(self):
        from dailydigest.tea_live import generated_jokes

        jokes = generated_jokes(4, date(2026, 8, 22))
        assert len(jokes) == 4
        assert len(set(jokes)) == 4

    def test_no_template_is_used_twice_in_one_day(self):
        """Two fillings of one template read as the same joke twice."""
        from dailydigest.tea_live import _TEMPLATES, generated_jokes

        jokes = generated_jokes(6, date(2026, 8, 22))
        shapes = []
        for joke in jokes:
            for _slot, template in _TEMPLATES:
                skeleton = template.split("{")[0]
                if skeleton and skeleton in joke:
                    shapes.append(template)
                    break
        assert len(shapes) == len(set(shapes)), f"template reused within a day: {shapes}"

    def test_excluded_cards_are_skipped(self):
        from dailydigest.tea_live import generated_jokes

        first = generated_jokes(3, date(2026, 8, 22))
        again = generated_jokes(3, date(2026, 8, 22), exclude=set(first))
        assert not (set(first) & set(again))

    def test_is_deterministic_for_a_day_and_varies_across_days(self):
        from dailydigest.tea_live import generated_jokes

        day = date(2026, 8, 22)
        assert generated_jokes(3, day) == generated_jokes(3, day)
        assert generated_jokes(3, day) != generated_jokes(3, date(2026, 8, 23))

    def test_every_template_renders_with_every_filling(self):
        """Guards against a slot value that makes a template ungrammatical."""
        from dailydigest.tea_live import _SLOTS, _TEMPLATES

        for slot, template in _TEMPLATES:
            for value in _SLOTS[slot]:
                text = template.format(x=value, occasion=_SLOTS["occasion"][0])
                assert "{" not in text and "}" not in text
                assert text.endswith((".", "!", "?"))


class TestDeckComposition:
    def test_deck_carries_live_cards_when_a_brew_exists(self, tmp_path, monkeypatch):
        store_mod = _reset_store(tmp_path, monkeypatch)
        _seed_brew(store_mod)
        from dailydigest.tea_break import DAILY_FACTS, DAILY_JOKES, daily_tea_deck
        from dailydigest.tea_live import BREW_PREFIX

        deck = daily_tea_deck(date(2026, 8, 22))
        assert len(deck) == DAILY_JOKES + DAILY_FACTS
        assert any(c.startswith(BREW_PREFIX) for c in deck), "no live brew card in the deck"

    def test_deck_is_still_full_without_any_live_data(self, tmp_path, monkeypatch):
        _reset_store(tmp_path, monkeypatch)
        from dailydigest.tea_break import DAILY_FACTS, DAILY_JOKES, daily_tea_deck

        deck = daily_tea_deck(date(2026, 8, 22))
        assert len(deck) == DAILY_JOKES + DAILY_FACTS
        assert len(set(deck)) == len(deck)

    def test_a_pre_brew_deck_is_recorded_then_upgraded_when_the_brew_lands(
        self, tmp_path, monkeypatch
    ):
        """Two requirements that pull against each other, both of them real.

        The deck must be RECORDED even before the brew, because that ledger is
        the only thing stopping the curated banks from recycling -- skipping it
        made every pre-brew day draw the same cards again. But it must also not
        be frozen, or opening the page early would lock out the day's
        observations entirely. So: record, then replace exactly once.
        """
        store_mod = _reset_store(tmp_path, monkeypatch)
        from dailydigest.tea_break import daily_tea_deck
        from dailydigest.tea_live import BREW_PREFIX

        early = daily_tea_deck(date(2026, 8, 22))
        assert not any(c.startswith(BREW_PREFIX) for c in early)
        assert store_mod.tea_deck_for_day("2026-08-22") is not None, "ledger did not learn"

        # Reloading before the brew must not reshuffle the deck.
        assert daily_tea_deck(date(2026, 8, 22)) == early

        _seed_brew(store_mod)
        later = daily_tea_deck(date(2026, 8, 22))
        assert any(c.startswith(BREW_PREFIX) for c in later), "observations never appeared"
        # And now it is settled: further reloads return the upgraded deck.
        assert daily_tea_deck(date(2026, 8, 22)) == later
