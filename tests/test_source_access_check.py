"""The daily browser source audit fails only on sources the browser fetches directly."""
import importlib.util
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_source_access", ROOT / "browser/check_source_access.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_snapshot_feeds_and_bundled_lists_are_not_direct_sources():
    direct = check.direct_sources(ROOT / "config/sources.yaml")
    kinds = {source.kind for source in direct.values()}
    assert "Grants.gov" in direct and "EU Funding & Tenders" in direct and "EMBO Events" in direct
    assert not kinds & {"rss", "arxiv", "curated_calls", "curated_events"}
    assert all(source.profile_driven for source in direct.values() if source.kind == "biorxiv")
    # The website never reads local-only sources, so they cannot fail its check.
    assert "Biophysical Society Meetings" not in direct


def test_unreadable_or_unprobed_direct_source_fails(tmp_path, monkeypatch):
    direct = {"Grants.gov": object(), "EMBO Events": object(), "NIH Guide": object()}
    report = {"results": [
        {"name": "Grants.gov", "section": "opportunities", "outcome": "cors_blocked", "error": "Failed to fetch"},
        {"name": "EMBO Events", "section": "events", "outcome": "readable_http_success", "status": 200},
        {"name": "Nature", "section": "research", "outcome": "cors_blocked"},
    ]}
    failures, missing = check.judge(report, direct)
    assert [name for name, *_ in failures] == ["Grants.gov"] and missing == ["NIH Guide"]
    clean = {"results": [{"name": name, "section": "x", "outcome": "readable_http_success"} for name in direct]}
    assert check.judge(clean, direct) == ([], [])


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
    assert runs.count("check_source_access.py") == 2, "A failure is rechecked once before alerting"
    report = next(step for step in job["steps"] if "report_failure.py" in step.get("run", ""))
    assert report["env"]["DAILYDIGEST_ALERT_MODE"] == "source-access-issue"
