"""bioRxiv/medRxiv public snapshots replay the native adapter; fixtures only.

A fake details API serves a recorded-shape corpus filtered by posting date, as
api.biorxiv.org does. No network, model provider, or reader data is touched.
"""

import importlib.util
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dailydigest.ingest import biorxiv
from dailydigest.ingest.biorxiv import BiorxivSource
from dailydigest.models import Item, SourceSpec

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load("preprint_collector", "browser/collect_sources.py")
network = _load("preprint_network", "browser/network.py")

CI_NOW = datetime(2026, 10, 26, 0, 30, tzinfo=timezone.utc)
SNAPSHOT_URL = "/assets/public-sources.json"
ENGLISH = ("We engineer a programmable protein assembly and measure its binding kinetics "
           "across buffers with improved yield and nanometre precision. ") * 2


def entry(doi, date, version="1", *, title=None, kind="new results", category="bioengineering",
          abstract=None, published="NA"):
    return {"doi": f"10.1101/{doi}", "title": title if title is not None else f"Preprint {doi}",
            "authors": "A. Author; B. Author", "date": date, "version": version, "type": kind,
            "category": category, "abstract": abstract or f"{ENGLISH}Variant {doi} v{version}.",
            "author_corresponding_institution": "Fixture University", "published": published,
            "server": "bioRxiv", "license": "cc_by", "jatsxml": "https://example.test/x.xml"}


def corpus():
    return {
        "biorxiv": [
            entry("2026.10.20.000001", "2026-10-20"),           # outside the 5-day CI window
            entry("2026.10.21.000002", "2026-10-21", category="neuroscience"),
            entry("2026.10.21.000003", "2026-10-21"),           # v1; v2 posted later
            entry("2026.10.22.000004", "2026-10-22", category="genomics",
                  published="10.1038/s41586-026-00001-1"),
            entry("2026.10.22.000005", "2026-10-22"),           # v1; withdrawn later
            entry("2026.10.23.000005", "2026-10-23", "2", kind="withdrawn",
                  title="WITHDRAWN: Preprint 2026.10.22.000005"),
            entry("2026.10.23.000006", "2026-10-23", title=""),  # invalid: no title
            entry("2026.10.24.000003", "2026-10-24", "2", category="cell biology"),
            entry("2026.10.25.000007", "2026-10-25", category="synthetic biology"),
            entry("2026.10.26.000008", "2026-10-26", category="biophysics"),
        ],
        "medrxiv": [
            entry("2026.10.22.100001", "2026-10-22", category="oncology"),
            entry("2026.10.25.100002", "2026-10-25", category="infectious diseases"),
        ],
    }


# The withdrawal notice above reuses the v1 DOI so it retracts that preprint.
def _normalize_withdrawal(rows):
    for row in rows["biorxiv"]:
        if row["type"] == "withdrawn":
            row["doi"] = "10.1101/2026.10.22.000005"
        if row["version"] == "2" and row["type"] != "withdrawn":
            row["doi"] = "10.1101/2026.10.21.000003"
    return rows


class FakeDetailsAPI:
    """api.biorxiv.org/details/{server}/{from}/{to}/{cursor}, 3 rows a page."""

    def __init__(self, rows, page=3):
        self.rows, self.page, self.calls = rows, page, []

    def __call__(self, _client, url):
        self.calls.append(url)
        server, start, end, cursor = url.rstrip("/").split("/")[-4:]
        matches = [row for row in self.rows[server] if start <= row["date"] <= end]
        cursor = int(cursor)
        return {"messages": [{"status": "ok", "total": len(matches)}],
                "collection": matches[cursor:cursor + self.page]}


def frozen(instant):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant if tz else instant.replace(tzinfo=None)

    return Frozen


def spec(server="biorxiv"):
    return SourceSpec(name=f"{server} (recent)", kind="biorxiv", server=server, section="research")


@pytest.fixture
def api(monkeypatch):
    fake = FakeDetailsAPI(_normalize_withdrawal(corpus()))
    monkeypatch.setattr(biorxiv, "_get_json", fake)
    # install_public_source_fallback wraps these process-wide; restore them.
    from dailydigest.ingest.arxiv import ArxivSource
    from dailydigest.ingest.rss import RSSSource

    for adapter in (BiorxivSource, ArxivSource, RSSSource):
        monkeypatch.setattr(adapter, "fetch", adapter.fetch)
    return fake


def collect_snapshot(monkeypatch, api, specs=None):
    monkeypatch.setattr(biorxiv, "datetime", frozen(CI_NOW))
    report = collector.collect(specs or [spec("biorxiv"), spec("medrxiv")], now=lambda: CI_NOW)
    api.calls.clear()
    return collector.with_language_decisions(report)


def native(monkeypatch, at, source_spec, days):
    """The unmodified adapter fetching directly at a reader's clock."""
    monkeypatch.setattr(biorxiv, "datetime", frozen(at))
    source = BiorxivSource()
    original = getattr(BiorxivSource.fetch, "_public_snapshot_original", BiorxivSource.fetch)
    items = original(source, source_spec, days=days)
    return items, source


def reader(monkeypatch, report, at):
    monkeypatch.setattr(biorxiv, "datetime", frozen(at))
    network.install_public_source_fallback(SNAPSHOT_URL, loader=lambda: deepcopy(report),
                                           now=lambda: at)
    return BiorxivSource()


def dumps(items):
    return sorted((item.model_dump(mode="json") for item in items), key=lambda row: row["external_id"])


# --- collection and publication -------------------------------------------------

def test_collected_record_is_complete_uncapped_and_carries_raw_evidence(monkeypatch, api):
    report = collect_snapshot(monkeypatch, api)
    row = report["sources"]["biorxiv (recent)"]
    assert row["status"] == "ok" and row["days"] == collector.PREPRINT_DAYS == 5
    assert row["server"] == "biorxiv"
    assert row["window"] == {"from": "2026-10-21", "to": "2026-10-26"}
    coverage = row["coverage"]
    assert coverage["exhausted"] and not coverage["truncated"]
    assert coverage["row_cap"] == collector.PREPRINT_COLLECTION_ROWS > biorxiv._MAX_ROWS
    assert coverage["fetched_count"] == 9 == coverage["total_available"]
    assert coverage["dates"]["2026-10-23"] == {"entries": 2, "withdrawn": 1, "invalid": 1}
    # Every parsed revision is kept, before native revision/withdrawal selection.
    versions = {(item["external_id"], item["metadata"]["preprint_version"]) for item in row["items"]}
    assert ("10.1101/2026.10.21.000003", "1") in versions and ("10.1101/2026.10.21.000003", "2") in versions
    assert ("10.1101/2026.10.22.000005", "1") in versions
    assert row["withdrawals"] == [{"date": "2026-10-23", "item": row["withdrawals"][0]["item"]}]
    assert row["withdrawals"][0]["item"]["metadata"]["is_retracted"] is True
    # All categories: the native adapter applies no category filter.
    assert {item["metadata"]["preprint_category"] for item in row["items"]} >= {
        "neuroscience", "genomics", "cell biology", "synthetic biology", "biophysics"}
    assert report["sources"]["medrxiv (recent)"]["server"] == "medrxiv"
    summary = collector.validate_report(report, now=CI_NOW)
    assert summary["fresh_ok"] == 2 and summary["language_decisions"] == "current"


def test_language_decision_map_covers_every_new_preprint_revision(monkeypatch, api):
    from dailydigest.dedupe import _language_sample, language_sample_key

    report = collect_snapshot(monkeypatch, api)
    allowed = report["language_decisions"]["allowed"]
    items = [Item.model_validate(row) for record in report["sources"].values() for row in record["items"]]
    assert len(items) == 9
    assert {language_sample_key(_language_sample(item)) for item in items} <= set(allowed)
    assert all(allowed[language_sample_key(_language_sample(item))] for item in items)


def test_collection_cap_is_recorded_as_truncated_not_complete(monkeypatch, api):
    monkeypatch.setattr(collector, "PREPRINT_COLLECTION_ROWS", 3)
    row = collect_snapshot(monkeypatch, api)["sources"]["biorxiv (recent)"]
    assert row["status"] == "ok"
    assert row["coverage"]["truncated"] and row["coverage"]["stop_reason"] == "item_cap"
    assert not row["coverage"]["exhausted"]


def test_out_of_window_api_dates_fail_the_record_instead_of_publishing(monkeypatch, api):
    api.rows["biorxiv"].append({**entry("2026.09.01.000009", "2026-09-01")})
    original = api.__call__

    def leaky(client, url):
        page = original(client, url)
        if url.endswith("/0") and "/biorxiv/" in url:
            page["collection"] = page["collection"] + [api.rows["biorxiv"][-1]]
            page["messages"][0]["total"] += 1
        return page

    monkeypatch.setattr(biorxiv, "_get_json", leaky)
    report = collect_snapshot(monkeypatch, api)
    row = report["sources"]["biorxiv (recent)"]
    assert row["status"] == "error" and row["items"] == []
    assert "outside the requested posting window" in row["error"]


def test_preprint_outage_neither_blocks_publication_nor_claims_success(monkeypatch, api):
    def down(_client, url):
        if "/medrxiv/" in url:
            raise RuntimeError("HTTP 503")
        return api(_client, url)

    monkeypatch.setattr(biorxiv, "_get_json", down)
    rss = SourceSpec(name="Feed", kind="rss", url="https://example.test/rss", section="research")
    arxiv = SourceSpec(name="arXiv q-bio.QM", kind="arxiv", category="q-bio.QM", section="research")
    item = {"section": "research", "external_id": "1", "url": "https://example.test/1", "title": "T"}

    class Fixed:
        def fetch(self, source_spec, days):
            return [Item(source=source_spec.name, **item)]

    monkeypatch.setattr(biorxiv, "datetime", frozen(CI_NOW))
    monkeypatch.setattr(collector.time, "sleep", lambda _seconds: None)
    report = collector.collect([rss, arxiv, spec("biorxiv"), spec("medrxiv")], now=lambda: CI_NOW,
                               adapters={"rss": Fixed, "arxiv": Fixed, "biorxiv": BiorxivSource})
    medrxiv = report["sources"]["medrxiv (recent)"]
    assert medrxiv["status"] == "error" and medrxiv["items"] == [] and "fetch failed" in medrxiv["error"]
    summary = collector.validate_report(report, now=CI_NOW)
    assert summary == {"sources": 4, "ok": 3, "fresh_ok": 3, "carried_forward": 0}
    # Only the optional kind may be missing; RSS/arXiv coverage stays mandatory.
    report["sources"]["arXiv q-bio.QM"].update(status="error", items=[], error="down")
    with pytest.raises(ValueError, match="missing kind"):
        collector.validate_report(report, now=CI_NOW)


@pytest.mark.parametrize("mutation, message", [
    (lambda row: row.pop("server"), "server"),
    (lambda row: row.update(window={"from": "2026-10-22", "to": "2026-10-26"}), "days"),
    (lambda row: row["coverage"].update(fetched_count=99), "fetched entries"),
    (lambda row: row["items"].pop(), "coverage counts"),
    (lambda row: row["coverage"]["dates"].update({"2026-10-01": {"entries": 1, "withdrawn": 0,
                                                                  "invalid": 0}}), "outside"),
    (lambda row: row["withdrawals"][0]["item"]["metadata"].update(is_retracted=False), "withdrawal"),
    (lambda row: row["coverage"].update(truncated="no"), "coverage"),
])
def test_malformed_preprint_evidence_cannot_publish(monkeypatch, api, mutation, message):
    report = collect_snapshot(monkeypatch, api)
    mutation(report["sources"]["biorxiv (recent)"])
    with pytest.raises(ValueError, match=message):
        collector.validate_report(report, now=CI_NOW)


def test_carry_forward_requires_the_same_server(monkeypatch, api):
    previous = collect_snapshot(monkeypatch, api)
    current = deepcopy(previous)
    current["sources"]["biorxiv (recent)"].update(status="error", items=[], error="down")
    carried = collector.carry_forward(current, previous, now=CI_NOW)
    assert carried["sources"]["biorxiv (recent)"]["carried_forward"] is True
    previous["sources"]["biorxiv (recent)"]["server"] = "medrxiv"
    assert collector.carry_forward(current, previous, now=CI_NOW)["sources"]["biorxiv (recent)"][
        "status"] == "error"


# --- reader replay and equivalence ---------------------------------------------

READER_CASES = [
    # (reader clock, requested days): same UTC day, next day, catch-up inside window.
    (CI_NOW + timedelta(hours=9), 4),
    (CI_NOW + timedelta(hours=9), 5),
    (CI_NOW + timedelta(hours=25), 4),
    (CI_NOW + timedelta(hours=25), 6),
    (CI_NOW + timedelta(hours=9), 2),
]


@pytest.mark.parametrize("at, days", READER_CASES)
@pytest.mark.parametrize("server", ["biorxiv", "medrxiv"])
def test_snapshot_replay_equals_native_fetch_for_the_same_window(monkeypatch, api, at, days, server):
    report = collect_snapshot(monkeypatch, api)
    expected, direct = native(monkeypatch, at, spec(server), days)
    api.calls.clear()
    source = reader(monkeypatch, report, at)
    items = source.fetch(spec(server), days=days)
    assert api.calls == []  # served entirely from the snapshot
    assert source.last_fetch_diagnostics["transport"] == "public_snapshot"
    assert dumps(items) == dumps(expected)
    assert [item.external_id for item in items] == [item.external_id for item in expected]
    assert dumps(source.last_fetch_withdrawals) == dumps(direct.last_fetch_withdrawals)
    replayed = source.last_fetch_diagnostics["queries"][0]
    native_query = direct.last_fetch_diagnostics["queries"][0]
    for key in ("query", "fetched_count", "returned_count", "total_available", "exhausted",
                "truncated", "stop_reason", "skipped_withdrawn", "skipped_invalid",
                "oldest_publication_date", "newest_publication_date"):
        assert replayed[key] == native_query[key], key


def test_version_two_is_carried_through_with_its_own_abstract(monkeypatch, api):
    report = collect_snapshot(monkeypatch, api)
    source = reader(monkeypatch, report, CI_NOW + timedelta(hours=9))
    items = {item.external_id: item for item in source.fetch(spec(), days=4)}
    revised = items["10.1101/2026.10.21.000003"]
    assert revised.metadata["preprint_version"] == "2"
    assert revised.abstract.endswith("v2.")
    assert revised.metadata["preprint_category"] == "cell biology"
    assert items["10.1101/2026.10.22.000004"].metadata["published_doi"] == "10.1038/s41586-026-00001-1"


def test_withdrawn_preprint_is_excluded_and_its_notice_reaches_the_pipeline(monkeypatch, api):
    report = collect_snapshot(monkeypatch, api)
    source = reader(monkeypatch, report, CI_NOW + timedelta(hours=9))
    items = source.fetch(spec(), days=4)
    assert "10.1101/2026.10.22.000005" not in {item.external_id for item in items}
    assert [notice.external_id for notice in source.last_fetch_withdrawals] == ["10.1101/2026.10.22.000005"]
    assert source.last_fetch_withdrawals[0].metadata["is_retracted"] is True
    # A window after the notice carries neither the notice nor the old revision.
    later = source.fetch(spec(), days=2)
    assert source.last_fetch_withdrawals == []
    assert "10.1101/2026.10.22.000005" not in {item.external_id for item in later}


def _fallback(source, api):
    assert source.last_fetch_diagnostics["transport"] == "direct"
    assert api.calls, "live adapter must run"
    return source.last_fetch_diagnostics["snapshot_error"]


def test_stale_snapshot_falls_back_to_live_fetch(monkeypatch, api):
    report = collect_snapshot(monkeypatch, api)
    at = CI_NOW + timedelta(hours=49)
    expected, _ = native(monkeypatch, at, spec(), 4)
    api.calls.clear()
    source = reader(monkeypatch, report, at)
    assert dumps(source.fetch(spec(), days=4)) == dumps(expected)
    assert "stale" in _fallback(source, api)


@pytest.mark.parametrize("at, days", [(CI_NOW + timedelta(hours=9), 6),
                                      (CI_NOW + timedelta(hours=25), 7),
                                      (CI_NOW + timedelta(hours=9), 21)])
def test_window_longer_than_snapshot_falls_back(monkeypatch, api, at, days):
    report = collect_snapshot(monkeypatch, api)
    source = reader(monkeypatch, report, at)
    source.fetch(spec(), days=days)
    assert "window too short" in _fallback(source, api)
    assert source.last_fetch_diagnostics.get("queries", [{}])[0].get("request_count", 0) > 0


def test_capped_collection_falls_back(monkeypatch, api):
    monkeypatch.setattr(collector, "PREPRINT_COLLECTION_ROWS", 3)
    report = collect_snapshot(monkeypatch, api)
    source = reader(monkeypatch, report, CI_NOW + timedelta(hours=9))
    source.fetch(spec(), days=4)
    assert "truncated or capped" in _fallback(source, api)


def test_window_reaching_the_native_row_cap_falls_back_to_native_truncation(monkeypatch, api):
    report = collect_snapshot(monkeypatch, api)
    monkeypatch.setattr(biorxiv, "_MAX_ROWS", 4)
    at = CI_NOW + timedelta(hours=9)
    expected, direct = native(monkeypatch, at, spec(), 4)
    assert direct.last_fetch_diagnostics["queries"][0]["stop_reason"] == "item_cap"
    api.calls.clear()
    source = reader(monkeypatch, report, at)
    assert dumps(source.fetch(spec(), days=4)) == dumps(expected)
    assert "row cap" in _fallback(source, api)
    assert source.last_fetch_diagnostics["queries"][0]["truncated"] is True


def test_undated_entries_cannot_be_replayed(monkeypatch, api):
    api.rows["biorxiv"].insert(0, {**entry("2026.10.26.000010", "2026-10-26")})
    report = collect_snapshot(monkeypatch, api)
    row = report["sources"]["biorxiv (recent)"]
    row["coverage"]["undated"]["entries"] = 1
    row["coverage"]["dates"]["2026-10-26"]["entries"] -= 1
    source = reader(monkeypatch, report, CI_NOW + timedelta(hours=9))
    source.fetch(spec(), days=4)
    assert "undated" in _fallback(source, api)


def test_server_mismatch_falls_back(monkeypatch, api):
    report = collect_snapshot(monkeypatch, api)
    source = reader(monkeypatch, report, CI_NOW + timedelta(hours=9))
    renamed = spec("medrxiv").model_copy(update={"name": "biorxiv (recent)"})
    source.fetch(renamed, days=4)
    assert "configuration mismatch" in _fallback(source, api)


def test_mixed_snapshot_and_failed_sources_keep_health_records_honest(monkeypatch, api):
    """One server from the snapshot, one failed everywhere, one personalized failure."""
    from dailydigest import pipeline, store

    report = collect_snapshot(monkeypatch, api)
    report["sources"]["medrxiv (recent)"].update(status="error", items=[], error="HTTP 503")

    def live(_client, url):
        raise RuntimeError("CORS or network unavailable")

    at = CI_NOW + timedelta(hours=9)
    source_specs = [spec("biorxiv"), spec("medrxiv"),
                    SourceSpec(name="Topics", kind="openalex", profile_driven=True, section="research")]

    class Personal:
        last_fetch_diagnostics = {}

        def fetch(self, *_args, **_kwargs):
            raise RuntimeError("Topic search failed")

    reader(monkeypatch, report, at)
    monkeypatch.setattr(biorxiv, "_get_json", live)
    recorded, observed = [], []
    monkeypatch.setattr(pipeline, "init_db", lambda: None)
    monkeypatch.setattr(pipeline, "load_sources", lambda: source_specs)
    monkeypatch.setattr(pipeline, "dispatch_source",
                        lambda s: BiorxivSource() if s.kind == "biorxiv" else Personal())
    monkeypatch.setattr(pipeline.health, "record", lambda rows: recorded.extend(rows))
    monkeypatch.setattr(store, "mark_research_withdrawn", lambda rows: observed.extend(rows), raising=False)
    with pytest.raises(RuntimeError, match="Personalized research source coverage is degraded"):
        pipeline.ingest_all(days=4)
    stats = {row.source: row for row in recorded}
    assert stats["biorxiv (recent)"].ok and stats["biorxiv (recent)"].items == 4
    assert stats["biorxiv (recent)"].diagnostics["transport"] == "public_snapshot"
    medrxiv = stats["medrxiv (recent)"]
    assert not medrxiv.ok and medrxiv.items == 0
    assert "direct fetch and public snapshot failed" in medrxiv.error
    assert "missing or failed" in medrxiv.diagnostics["snapshot_error"]
    assert "CORS or network unavailable" in medrxiv.diagnostics["direct_error"]
    assert not stats["Topics"].ok
    assert [notice.external_id for notice in observed] == ["10.1101/2026.10.22.000005"]


def test_public_embeddings_cover_every_replayable_preprint_text(monkeypatch, api):
    embeddings = _load("preprint_public_embeddings", "browser/public_embeddings.py")
    report = collect_snapshot(monkeypatch, api)
    texts = set(embeddings.public_texts(report))
    source = reader(monkeypatch, report, CI_NOW + timedelta(hours=9))
    served = source.fetch(spec(), days=5) + source.fetch(spec("medrxiv"), days=5)
    # Whatever a reader's window selects already has a public vector...
    assert {f"{item.title.strip()}. {item.abstract.strip()}" for item in served} <= texts
    # ...including superseded revisions, while withdrawal notices are not embedded.
    assert len(texts) == 9
    assert not any(text.startswith("WITHDRAWN") for text in texts)
