"""The daily browser source audit fails only on sources the website cannot reach."""
import importlib.util
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_source_access", ROOT / "browser/check_source_access.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_snapshot_feeds_and_bundled_lists_are_not_direct_sources():
    direct = check.direct_sources(ROOT / "config/sources.yaml")
    carried = check.snapshot_sources(ROOT / "config/sources.yaml")
    kinds = {source.kind for source in direct.values()}
    assert "Grants.gov" in direct and "EU Funding & Tenders" in direct
    assert not kinds & {"rss", "arxiv", "curated_calls", "curated_events"}
    assert all(source.profile_driven for source in direct.values() if source.kind == "biorxiv")
    # Public event listings ride in the snapshot, so a CORS change upstream
    # (EMBL sent two Access-Control-Allow-Origin values) cannot break them.
    assert {"EMBL Events", "EMBO Events", "Nature Conferences", "Keystone Symposia"} <= set(carried)
    assert not set(direct) & set(carried)
    # The website never reads local-only sources, so they cannot fail its check.
    for name in ("Biophysical Society Meetings", "Import AI (Jack Clark)"):
        assert name not in direct and name not in carried


def test_unreadable_or_unprobed_direct_source_fails(tmp_path, monkeypatch):
    direct = {"Grants.gov": object(), "EMBO Events": object(), "NIH Guide": object()}
    report = {"results": [
        {"name": "Grants.gov", "section": "opportunities", "outcome": "cors_blocked", "error": "Failed to fetch"},
        {"name": "EMBO Events", "section": "events", "outcome": "readable_http_success", "status": 200},
        {"name": "Nature", "section": "research", "outcome": "cors_blocked"},
    ]}
    failures, missing, aging = check.judge(report, direct)
    assert [name for name, *_ in failures] == ["Grants.gov"] and missing == ["NIH Guide"] and aging == []
    clean = {"results": [{"name": name, "section": "x", "outcome": "readable_http_success"} for name in direct]}
    assert check.judge(clean, direct) == ([], [], [])


NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def _carried(name, kind="rss"):
    from dailydigest.models import SourceSpec

    return SourceSpec(name=name, kind=kind, section="events" if kind != "rss" else "research",
                      url=f"https://example.org/{kind}")


def _record(spec, hours=2, **fields):
    return {"kind": spec.kind, "section": spec.section, "url": spec.url, "category": None,
            "status": "ok", "fetched_at": (NOW - timedelta(hours=hours)).isoformat(), "items": [], **fields}


def test_snapshot_sources_need_a_usable_record_or_a_direct_read():
    specs = {name: _carried(name, kind) for name, kind in [
        ("EMBL Events", "embl_events"), ("PNAS", "rss"), ("eLife", "rss"), ("Old", "rss"),
        ("Moved", "rss"), ("Aging", "rss"), ("Absent", "rss")]}
    snapshot = {"sources": {
        "EMBL Events": _record(specs["EMBL Events"]),
        "PNAS": {**_record(specs["PNAS"]), "status": "error", "items": [], "error": "RuntimeError: 403"},
        "eLife": {**_record(specs["eLife"]), "status": "error", "items": [], "error": "RuntimeError: 406"},
        "Old": _record(specs["Old"], hours=49),
        "Moved": {**_record(specs["Moved"]), "url": "https://example.org/former"},
        "Aging": _record(specs["Aging"], hours=30, carried_forward=True),
    }}
    blocked = {"outcome": "cors_blocked", "error": "Failed to fetch"}
    report = {"results": [{"name": name, "section": spec.section, **blocked} for name, spec in specs.items()]}
    # A failed record still works when the browser can read the feed itself.
    report["results"][2] = {"name": "eLife", "section": "research", "outcome": "readable_http_success"}
    failures, missing, aging = check.judge(report, {}, specs, snapshot, now=NOW)
    reasons = {name: outcome for name, outcome, *_ in failures}
    assert set(reasons) == {"PNAS", "Old", "Moved", "Absent"} and missing == []
    assert "collection failed: RuntimeError: 403" in reasons["PNAS"] and "cors_blocked" in reasons["PNAS"]
    assert "expired" in reasons["Old"] and "does not match" in reasons["Moved"]
    assert "not in the snapshot" in reasons["Absent"]
    assert aging == ["Aging"]


def test_unreadable_snapshot_fails_every_source_without_a_direct_read(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"origin": "https://example.org", "generatedAt": NOW.isoformat(), "results": [
        {"name": name, "section": "x", "outcome": "readable_http_success"}
        for name in check.direct_sources(ROOT / "config/sources.yaml")]}))
    summary = tmp_path / "summary.md"
    assert check.main([str(report), "--snapshot", str(tmp_path / "absent.json"), "--summary", str(summary)]) == 1
    text = summary.read_text()
    assert "the published snapshot could not be read" in text
    assert "- EMBL Events: snapshot unavailable; direct read not probed" in text
    # Without a snapshot argument the check judges direct sources only.
    assert check.main([str(report)]) == 0


def test_audit_workflow_is_pinned_bounded_and_scoped():
    workflow = yaml.load((ROOT / ".github/workflows/source-audit.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["audit"]
    assert 1 <= int(job["timeout-minutes"]) <= 60
    assert job["permissions"] == {"contents": "read", "issues": "write"}
    assert "refs/heads/main" in job["if"]
    for step in job["steps"]:
        if "uses" in step:
            assert re.search(r"@[0-9a-f]{40}$", step["uses"])
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "https://antimonyleo.github.io/daily_digests/" in runs
    assert "check_source_access.py" in runs and "--snapshot .source-audit/public-sources.json" in runs
    assert "assets/public-sources.json" in runs
    assert "if ! audit; then" in runs and "audit || echo" in runs, "A failure is rechecked once before alerting"
    report = next(step for step in job["steps"] if "report_failure.py" in step.get("run", ""))
    assert report["env"]["DAILYDIGEST_ALERT_MODE"] == "source-access-issue"
