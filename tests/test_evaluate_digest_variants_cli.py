"""Offline evaluation CLI has explicit inputs, structured errors and safe outputs."""
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_digest_evaluation import snapshot


@pytest.fixture
def cli(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_digest_variants.py"
    spec = importlib.util.spec_from_file_location("evaluate_digest_variants_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from dailydigest import config
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(
        db_path=tmp_path / "private" / "digest.db", profile_path=tmp_path / "private" / "profile.yaml"))
    return module


def arguments(tmp_path):
    source = tmp_path / "snapshot.json"
    source.write_text(json.dumps(snapshot()))
    return ["--snapshot", str(source), "--outcome-as-of", "2026-10-05T00:00:00Z",
            "--output", str(tmp_path / "report.json")]


def test_cli_writes_report_and_blind_packet_without_votes(cli, tmp_path, capsys):
    args = arguments(tmp_path) + ["--packet", str(tmp_path / "packet.jsonl"), "--rejected-per-run", "1"]
    assert cli.main(args) == 0
    output = capsys.readouterr().out
    assert 'status: "offline_diagnostic"' in output
    assert "promotion_eligible: false" in output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["days"] == 1
    assert report["judgments_used"] == 0
    assert report["variants"]["baseline"]["aggregate"]["research"]["unknown"] == 3
    packet = [json.loads(line) for line in (tmp_path / "packet.jsonl").read_text().splitlines()]
    assert len(packet) == 4
    assert (tmp_path / "packet.jsonl.manifest.json").exists()


def test_cli_supports_multiple_compressed_snapshots(cli, tmp_path, capsys):
    args = arguments(tmp_path)
    compressed = tmp_path / "day2.json.gz"
    with gzip.open(compressed, "wt") as handle:
        json.dump(snapshot(2), handle)
    assert cli.main(args + ["--snapshot", str(compressed)]) == 0
    assert "days: 2" in capsys.readouterr().out


def test_cli_rejects_unknown_flag_with_actionable_stdout(cli, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments(tmp_path) + ["--judgement", "wrong.jsonl"])
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert "error:" in output.out and "--judgments JSONL" in output.out
    assert "--judgement" in output.out
    assert not output.err
    assert not (tmp_path / "report.json").exists()


def test_cli_refuses_existing_files_before_writing_any_artifact(cli, tmp_path, capsys):
    args = arguments(tmp_path)
    (tmp_path / "report.json").write_text("existing user content")
    assert cli.main(args + ["--packet", str(tmp_path / "packet.jsonl")]) == 1
    assert (tmp_path / "report.json").read_text() == "existing user content"
    assert not (tmp_path / "packet.jsonl").exists()
    assert "new files" in capsys.readouterr().out


def test_cli_refuses_configured_production_directory(cli, tmp_path, capsys):
    args = arguments(tmp_path)
    args[-1] = str(tmp_path / "private" / "new-report.json")
    assert cli.main(args) == 1
    assert "production" in capsys.readouterr().out
    assert not (tmp_path / "private").exists()


@pytest.mark.parametrize("flag", ["-v", "-V", "--version"])
def test_version_does_not_load_settings(cli, monkeypatch, flag, capsys):
    from dailydigest import config
    def forbidden():
        raise AssertionError("version must not read runtime settings")
    monkeypatch.setattr(config, "get_settings", forbidden)
    assert cli.main([flag]) == 0
    assert capsys.readouterr().out == "1\n"


def test_no_arguments_reports_definitive_empty_input(cli, capsys):
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert "snapshots: 0" in output
    assert "bin:" in output and "--snapshot FILE" in output


def test_invalid_reader_judgment_does_not_create_report(cli, tmp_path, capsys):
    args = arguments(tmp_path)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"run_id": "run-1", "item_id": 1,
                                 "recorded_at": "2026-10-03T00:00:00Z",
                                 "rater": "analyst", "useful": True}) + "\n")
    assert cli.main(args + ["--judgments", str(labels)]) == 1
    assert "cannot supply reader usefulness" in capsys.readouterr().out
    assert not (tmp_path / "report.json").exists()


def test_cli_evaluates_saved_days_with_reader_events_read_only(cli, tmp_path, capsys):
    from tests.test_saved_digest_evaluation import database
    path = database(tmp_path, [snapshot()], [(1, 2, -1, 40, "2026-10-03T00:00:00Z")])
    before = path.read_bytes()
    assert cli.main(["--database", str(path), "--days", "1", "--outcome-as-of", "2026-10-05T00:00:00Z",
                     "--output", str(tmp_path / "report.json"), "--packet", str(tmp_path / "packet.jsonl")]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["saved_input_coverage"]["database_mode"] == "read_only"
    assert report["variants"]["baseline"]["aggregate"]["research"]["judged"] == 1
    assert path.read_bytes() == before
    assert "research_unknown: 2" in capsys.readouterr().out


def test_cli_database_requires_a_new_output_outside_its_directory(cli, tmp_path, capsys):
    from tests.test_saved_digest_evaluation import database
    path = database(tmp_path, [snapshot()])
    # This need not be the configured database; the explicit source is protected too.
    elsewhere = tmp_path / "explicit-source"; elsewhere.mkdir()
    path = path.rename(elsewhere / "digest.db")
    assert cli.main(["--database", str(path), "--outcome-as-of", "2026-10-05T00:00:00Z",
                     "--output", str(elsewhere / "report.json")]) == 1
    assert not (elsewhere / "report.json").exists()


def test_cli_empty_saved_history_is_definitive(cli, tmp_path, capsys):
    from tests.test_saved_digest_evaluation import database
    assert cli.main(["--database", str(database(tmp_path)), "--outcome-as-of", "2026-10-05T00:00:00Z",
                     "--output", str(tmp_path / "report.json")]) == 1
    assert "No saved decisions" in capsys.readouterr().out
    assert not (tmp_path / "report.json").exists()


def test_cli_does_not_silently_ignore_days_for_explicit_snapshots(cli, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments(tmp_path) + ["--days", "1"])
    assert exc.value.code == 2
    assert "--days requires --database" in capsys.readouterr().out


def test_cli_packet_judgments_bind_to_saved_decisions(cli, tmp_path):
    from tests.test_saved_digest_evaluation import database
    path = database(tmp_path, [snapshot()])
    shared = ["--database", str(path), "--outcome-as-of", "2026-10-05T00:00:00Z"]
    packet_path = tmp_path / "packet.jsonl"
    assert cli.main(shared + ["--output", str(tmp_path / "report.json"), "--packet", str(packet_path)]) == 0
    row = json.loads(packet_path.read_text().splitlines()[0])
    row.update(rater="reader", recorded_at="2026-10-04T00:00:00Z", useful=True, important=True, grade=100)
    labels = tmp_path / "completed.jsonl"; labels.write_text(json.dumps(row) + "\n")
    # A subsequent brew must not swap the articles under the reader's packet.
    import sqlite3
    import zlib
    newer = snapshot()
    newer.update(run_id="newer", decision_at="2026-10-01T18:00:00Z")
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO ranking_snapshots VALUES (?, ?, ?)",
                   ("newer", newer["decision_at"], zlib.compress(json.dumps(newer).encode())))
    assert cli.main(shared + ["--output", str(tmp_path / "rated-report.json"), "--judgments", str(labels),
                              "--judgment-manifest", str(packet_path) + ".manifest.json"]) == 0
    assert json.loads((tmp_path / "rated-report.json").read_text())["judgments_used"] == 1


def test_cli_pinned_runs_reject_a_conflicting_day_selection(cli, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--database", str(tmp_path / "unused.db"), "--days", "1",
                  "--judgments", "unused.jsonl", "--judgment-manifest", "unused.json",
                  "--outcome-as-of", "2026-10-05T00:00:00Z", "--output", "unused-output.json"])
    assert exc.value.code == 2
    assert "omit --days" in capsys.readouterr().out
