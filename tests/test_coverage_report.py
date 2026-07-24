"""Tests for the READ-ONLY coverage & precision harness (scripts/coverage_report.py).

These build a small synthetic DB (via the store's write helpers, on the tmp
DB_PATH provided by the conftest ``_isolate_env`` fixture) with a couple of sent
digests plus impression + feature rows, then assert the computed coverage /
off-field / supply numbers, and — critically — that the script writes NOTHING
(no model artifact, no DB mutation via fit).
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dailydigest import store as store_mod
from dailydigest.config import get_settings

# Load scripts/coverage_report.py by path (it lives outside the package).
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "coverage_report.py"
_spec = importlib.util.spec_from_file_location("coverage_report", _SCRIPT)
cov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cov)


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _insert_item(title: str, section: str = "research") -> int:
    store_mod.init_db()
    with store_mod.session_scope() as s:
        row = store_mod.ItemRow(
            source="Test",
            section=section,
            external_id=title,
            url=f"https://example.com/{title}",
            title=title,
            abstract="abstract",
            published_at=datetime.now(timezone.utc),
        )
        s.add(row)
        s.flush()
        return int(row.id)


def _mark_sent(digest_id: str) -> None:
    with store_mod.session_scope() as s:
        d = s.get(store_mod.DigestRow, digest_id)
        if d is None:
            d = store_mod.DigestRow(id=digest_id)
            s.add(d)
        d.sent_at = datetime.now(timezone.utc)


def _feat(primary_facet="", topic_score=0.0, final_score=0.0, source_bucket=""):
    return {
        "primary_facet": primary_facet,
        "topic_score": topic_score,
        "final_score": final_score,
        "source_bucket": source_bucket,
    }


def _build_digest(digest_id: str, rows: list[dict], floor: float) -> None:
    """rows: list of {facet, topic, final, bucket, selected}.

    Writes impression rows (research pool) carrying the per-candidate facet +
    topic_score attribution, plus feature rows for the SELECTED items only
    (mirroring production, where digest_item_features holds only the selected
    slate — the source_bucket used by the high-venue heuristic lives there).
    The coverage harness now sources primary_facet/topic_score from the
    impression rows, so UNSELECTED candidates are attributable too.
    """
    impressions = []
    feature_rows = []
    for i, r in enumerate(rows):
        item_id = _insert_item(f"{digest_id}-{i}")
        impressions.append(
            (
                "research",
                item_id,
                i,
                r["final"],
                r["selected"],
                r["facet"],
                r["topic"],
            )
        )
        # Feature rows exist only for the selected slate in production.
        if r["selected"]:
            feature_rows.append(
                (
                    f"R{i}",
                    item_id,
                    r["final"],
                    _feat(
                        primary_facet=r["facet"],
                        topic_score=r["topic"],
                        final_score=r["final"],
                        source_bucket=r.get("bucket", ""),
                    ),
                )
            )
    store_mod.write_digest_features(digest_id, feature_rows)
    store_mod.write_impressions(digest_id, impressions)
    _mark_sent(digest_id)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_empty_store_is_insufficient_not_crash():
    report = cov.compute_report(n=10)
    assert report["n_digests"] == 0
    assert "insufficient" in report["status"]
    # Robust: numeric criteria are None rather than exploding.
    assert report["off_field_per_digest"] is None
    assert report["high_profile_omissions"] == 0


def test_coverage_offfield_supply_numbers(monkeypatch):
    floor = float(get_settings().min_topic_relevance)  # default 0.65
    hi = floor + 0.10  # clears the floor
    lo = floor - 0.10  # below floor

    # Profile with two core interests, one negative interest.
    class _P:
        keywords = ["dna nanotechnology", "colloidal self-assembly"]
        negative_interests = {"clinical oncology": 1.0}

    monkeypatch.setattr(cov, "load_profile", lambda: _P())

    # Digest 1: rich supply.
    #  - dna item: qualified + selected      -> covers dna
    #  - colloidal item: qualified, NOT selected -> eligible but not covered
    #  - off-field (empty facet) selected     -> +1 off-field
    #  - negative-facet selected              -> +1 off-field
    #  - high-venue on-interest NOT selected  -> +1 high-profile omission
    _build_digest(
        "2026-01-01",
        [
            {"facet": "dna nanotechnology", "topic": hi, "final": 0.9, "selected": True},
            {"facet": "colloidal self-assembly", "topic": hi, "final": 0.8, "selected": False},
            {"facet": "", "topic": hi, "final": 0.7, "selected": True},
            {"facet": "clinical oncology", "topic": hi, "final": 0.6, "selected": True},
            {"facet": "dna nanotechnology", "topic": hi, "final": 0.95,
             "bucket": "published_journal", "selected": False},
        ],
        floor,
    )

    # Digest 2: quiet day — only one qualified item, one selected. No off-field.
    _build_digest(
        "2026-01-02",
        [
            {"facet": "dna nanotechnology", "topic": hi, "final": 0.9, "selected": True},
            {"facet": "colloidal self-assembly", "topic": lo, "final": 0.5, "selected": False},
        ],
        floor,
    )

    report = cov.compute_report(n=10)

    assert report["n_digests"] == 2
    assert report["status"] == "ok"

    # --- Criterion 1: coverage --------------------------------------------
    pic = report["per_interest_coverage"]
    # dna: eligible in both digests, covered in both -> 2/2
    assert pic["dna nanotechnology"]["eligible_digests"] == 2
    assert pic["dna nanotechnology"]["covered_digests"] == 2
    assert pic["dna nanotechnology"]["coverage_fraction"] == pytest.approx(1.0)
    # colloidal: eligible only in digest 1 (qualified there), never selected -> 0/1
    assert pic["colloidal self-assembly"]["eligible_digests"] == 1
    assert pic["colloidal self-assembly"]["covered_digests"] == 0
    assert pic["colloidal self-assembly"]["coverage_fraction"] == pytest.approx(0.0)
    # colloidal is routinely absent (<50%).
    assert "colloidal self-assembly" in report["routinely_absent"]
    assert "dna nanotechnology" not in report["routinely_absent"]

    # --- Criterion 2: off-field precision ---------------------------------
    # Digest 1 has 2 off-field selected (empty facet + negative), digest 2 has 0.
    # mean = (2 + 0) / 2 = 1.0  -> target < 1.0 NOT met.
    assert report["off_field_per_digest"] == pytest.approx(1.0)
    assert report["off_field_target_met"] is False

    # --- Criterion 3: high-profile omissions ------------------------------
    # Only the published_journal dna item in digest 1 qualifies.
    assert report["high_profile_omissions"] == 1

    # --- Criterion 4: supply responsiveness -------------------------------
    supply = {r["digest_id"]: r for r in report["supply_responsiveness"]}
    # Digest 1: 3 selected, 5 qualified (all topic=hi).
    assert supply["2026-01-01"]["n_selected"] == 3
    assert supply["2026-01-01"]["n_qualified"] == 5
    # Digest 2 (quiet): 1 selected, 1 qualified (the lo item is below floor).
    assert supply["2026-01-02"]["n_selected"] == 1
    assert supply["2026-01-02"]["n_qualified"] == 1


def test_only_sent_digests_are_counted(monkeypatch):
    floor = float(get_settings().min_topic_relevance)
    hi = floor + 0.10

    class _P:
        keywords = ["dna nanotechnology"]
        negative_interests = {}

    monkeypatch.setattr(cov, "load_profile", lambda: _P())

    # Sent digest.
    _build_digest(
        "2026-02-01",
        [{"facet": "dna nanotechnology", "topic": hi, "final": 0.9, "selected": True}],
        floor,
    )
    # Brewed-but-NOT-sent digest (no _mark_sent).
    item_id = _insert_item("unsent-0")
    store_mod.write_digest_features(
        "2026-02-02",
        [("R0", item_id, 0.9, _feat("dna nanotechnology", hi, 0.9))],
    )
    store_mod.write_impressions(
        "2026-02-02", [("research", item_id, 0, 0.9, True)]
    )

    report = cov.compute_report(n=10)
    assert report["digest_ids"] == ["2026-02-01"]
    assert report["n_digests"] == 1


def test_watched_interests_reported(monkeypatch):
    class _P:
        keywords = ["dna nanotechnology"]  # missing colloidal + rna
        negative_interests = {}

    monkeypatch.setattr(cov, "load_profile", lambda: _P())
    report = cov.compute_report(n=5)
    watched = report["watched_interests"]
    # All three watched interests must be reported, even if not configured.
    assert set(watched) == set(cov.WATCHED_INTERESTS)
    assert "not a configured core interest" in watched["colloidal self-assembly"]


def test_script_is_read_only_no_artifact_written(monkeypatch):
    """The harness must never fit/write a model artifact or mutate via fit."""
    floor = float(get_settings().min_topic_relevance)
    hi = floor + 0.10

    class _P:
        keywords = ["dna nanotechnology"]
        negative_interests = {}

    monkeypatch.setattr(cov, "load_profile", lambda: _P())

    # Fail loudly if any model .fit() is invoked anywhere in the run.
    try:
        from sklearn.linear_model import LogisticRegression

        def _boom(*a, **k):
            raise AssertionError("coverage_report must NOT call model.fit()")

        monkeypatch.setattr(LogisticRegression, "fit", _boom)
    except Exception:
        pass

    _build_digest(
        "2026-03-01",
        [{"facet": "dna nanotechnology", "topic": hi, "final": 0.9, "selected": True}],
        floor,
    )

    db_dir = Path(get_settings().db_path).parent
    lr_path = db_dir / "lr_ranker.npz"
    cal_path = db_dir / "calibrator.json"

    def _sig(p: Path):
        return (p.exists(), p.stat().st_mtime if p.exists() else None)

    before_lr, before_cal = _sig(lr_path), _sig(cal_path)

    report = cov.compute_report(n=10)
    assert report["n_digests"] == 1

    after_lr, after_cal = _sig(lr_path), _sig(cal_path)
    # No artifact created; if one somehow pre-existed, its mtime is unchanged.
    assert after_lr == before_lr, "lr_ranker.npz was written by the harness!"
    assert after_cal == before_cal, "calibrator.json was written by the harness!"
    assert not lr_path.exists()
    assert not cal_path.exists()


def test_main_smoke(capsys, monkeypatch):
    class _P:
        keywords = ["dna nanotechnology"]
        negative_interests = {}

    monkeypatch.setattr(cov, "load_profile", lambda: _P())
    rc = cov.main(["-n", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PER-INTEREST COVERAGE" in out
    assert "OFF-FIELD PRECISION" in out
    assert "SUPPLY RESPONSIVENESS" in out
