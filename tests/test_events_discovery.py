"""The events section was empty because three filters each removed everything."""

from __future__ import annotations

from datetime import date, timedelta

import pytest


class TestMultipleDeadlines:
    """Official pages list several routes to attend; one closed route is not closed."""

    def test_an_open_route_keeps_the_event(self):
        from dailydigest.ingest.events_rss import _deadline

        # Exactly the shape EMBL publishes: abstracts and on-site shut, virtual open.
        future = (date.today() + timedelta(days=40)).strftime("%d %b %Y")
        deadline, closed = _deadline(
            f"Abstract submission: Closed Registration (On-site): Closed "
            f"Registration (Virtual): {future}"
        )
        assert closed is False, "an event with an open route was reported closed"
        assert deadline is not None

    def test_the_latest_route_is_the_deadline(self):
        """The last chance to act is the most distant deadline, not the first."""
        from dailydigest.ingest.events_rss import _deadline

        near = (date.today() + timedelta(days=10)).strftime("%d %b %Y")
        far = (date.today() + timedelta(days=60)).strftime("%d %b %Y")
        deadline, closed = _deadline(f"Abstract submission: {near} Registration: {far}")
        assert closed is False
        assert deadline == (date.today() + timedelta(days=60))

    def test_closed_with_no_dates_at_all_is_still_closed(self):
        from dailydigest.ingest.events_rss import _deadline

        deadline, closed = _deadline("Application: Closed")
        assert closed is True
        assert deadline is None


class TestEventTypeInference:
    """A generic label is worse than no label: the gate drops it."""

    def test_type_is_read_from_the_description_not_only_the_title(self):
        from dailydigest.ingest.events_rss import _event_type

        # Real EMBL titles never say what kind of gathering they are.
        assert _event_type("Chemical biology 2026", "EMBO | EMBL Conference") == "conference"
        assert _event_type("The complex life of RNA", "a virtual conference") == "conference"

    def test_unclassifiable_events_report_no_type_rather_than_a_generic_one(self):
        """An empty type passes the gate as 'unknown'; 'event' is dropped outright."""
        from dailydigest.ingest.events_rss import _event_type
        from dailydigest.opportunities import _selected_type_matches

        unknown = _event_type("Something entirely unlabelled", "no clue here")
        assert unknown == ""
        assert _selected_type_matches(["conference", "workshop", "courses"], unknown)
        # The old generic label is what silently emptied the section.
        assert not _selected_type_matches(["conference", "workshop", "courses"], "event")


class TestRegionPreference:
    """Awards and meetings in Europe and Asia must be reachable."""

    def test_european_and_asian_venues_match_when_selected(self):
        from dailydigest.opportunities import _region_matches

        assert _region_matches(["United States", "Europe", "Asia"], "EMBL Heidelberg, Germany")
        assert _region_matches(["United States", "Europe", "Asia"], "Kyoto, Japan")

    def test_us_only_preference_rejects_a_european_venue(self):
        """Guards the reason EMBL events were dropped before the profile widened."""
        from dailydigest.opportunities import _region_matches

        assert not _region_matches(["United States"], "EMBL Heidelberg, Germany")


@pytest.mark.parametrize("text", ["December 2026", "2027", "17 August", "August", "17"])
def test_incomplete_event_dates_never_inherit_todays_components(text):
    from dailydigest.ingest.events_rss import _deadline, _event_range, _parse_date

    assert _parse_date(text) is None
    assert _event_range(text) == (None, None)
    assert _deadline(text) == (None, False)


@pytest.mark.parametrize("text", [
    "25 - 17 Aug 2026", "31 - 32 Aug 2026", "17 Aug 2026 to August 2026",
    "December 2026 to 17 January 2027", "20 Aug 2026 to 17 Aug 2026",
])
def test_invalid_or_partial_event_range_is_rejected_as_a_whole(text):
    from dailydigest.ingest.events_rss import _event_range

    assert _event_range(text) == (None, None)


@pytest.mark.parametrize("text,start,end", [
    ("17 - 25 Aug 2026", date(2026, 8, 17), date(2026, 8, 25)),
    ("17 Aug to 2 Sep 2026", date(2026, 8, 17), date(2026, 9, 2)),
    ("30 Dec 2026 to 2 Jan 2027", date(2026, 12, 30), date(2027, 1, 2)),
    ("2026-08-17", date(2026, 8, 17), date(2026, 8, 17)),
    ("August 17, 2026", date(2026, 8, 17), date(2026, 8, 17)),
])
def test_complete_event_dates_and_explicit_range_inheritance_are_preserved(text, start, end):
    from dailydigest.ingest.events_rss import _event_range

    assert _event_range(text) == (start, end)


def test_feed_does_not_publish_a_month_only_event_as_an_official_exact_date(monkeypatch):
    from dailydigest.ingest import events_rss
    from dailydigest.models import SourceSpec

    feed = b'''<rss version="2.0"><channel><title>Events</title><item>
        <title>RNA workshop</title><link>https://example.test/event</link>
        <description>Date: December 2026 Location: Online Deadline: November 2026</description>
        </item></channel></rss>'''
    monkeypatch.setattr(events_rss, "_today", lambda: date(2026, 10, 1))
    monkeypatch.setattr(events_rss, "_http_get_bytes", lambda url: feed if url.endswith("feed") else b"")
    spec = SourceSpec(name="Events", kind="events_rss", section="events",
                      url="https://example.test/feed", lookahead_days=365)
    assert events_rss.EventsRSSSource().fetch(spec) == []
