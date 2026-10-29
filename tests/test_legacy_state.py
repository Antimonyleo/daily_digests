"""Reader state written by earlier releases opens intact in this one.

The fixtures in ``tests/fixtures/legacy`` were written by each release's own
code (see the README there). An update must never lose a rating, a saved or
known paper, a profile or opportunity setting.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from tests.test_web import _request, _text_payload

LEGACY = Path(__file__).parent / "fixtures" / "legacy"
RELEASES = sorted(path.name for path in LEGACY.glob("release-*"))


def test_every_release_since_august_has_a_frozen_fixture():
    assert RELEASES[:3] == ["release-2026-08-10", "release-2026-10-14", "release-2026-10-27"]


def _open(release, tmp_path, monkeypatch):
    from dailydigest import config, store, web

    source = LEGACY / release
    data = tmp_path / "data"
    data.mkdir()
    with sqlite3.connect(data / "digest.db") as db:
        db.executescript((source / "digest.sql").read_text())
    db.close()
    for name in ("profile.yaml", "opportunities.yaml"):
        shutil.copyfile(source / name, data / name)
    monkeypatch.setenv("DB_PATH", str(data / "digest.db"))
    monkeypatch.setenv("PROFILE_PATH", str(data / "profile.yaml"))
    monkeypatch.setenv("OPPORTUNITY_PROFILE_PATH", str(data / "opportunities.yaml"))
    monkeypatch.setenv("INCLUDE_OPPORTUNITIES", "true")
    config.reload_settings()
    monkeypatch.setattr(config, "SETTINGS", config.get_settings())
    monkeypatch.setattr(web, "SETTINGS", config.SETTINGS)
    monkeypatch.setattr(web, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(web, "_get_profile_path", lambda: data / "profile.yaml")
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-10-20")
    store._ENGINE = None
    store.init_db()
    ids = json.loads((source / "fixture.json").read_text())["ids"]
    return store, web, ids, data, (source / "digest.sql").read_text()


@pytest.mark.parametrize("release", RELEASES)
def test_old_database_keeps_ratings_saves_and_known_flags(release, tmp_path, monkeypatch):
    store, _web, ids, _data, dump = _open(release, tmp_path, monkeypatch)
    with store.session_scope() as session:
        votes = {row.item_id: row.value for row in session.query(store.VoteRow).all()}
    assert votes == {ids["liked"]: 1}
    assert store.bookmarked_item_ids(ids.values()) == {ids["saved"]}
    known_before = 'INSERT INTO "known_items"' in dump
    assert store.known_item_ids(ids.values()) == ({ids["known"]} if known_before else set())
    # Newer features work on the migrated database, and pruning keeps it all.
    assert store.set_item_interested(ids["grant"], True)
    store.prune(0)
    with store.session_scope() as session:
        remaining = {row.id for row in session.query(store.ItemRow).all()}
    assert {ids["liked"], ids["saved"], ids["grant"]} <= remaining
    if known_before:
        assert ids["known"] in remaining


@pytest.mark.parametrize("release", RELEASES)
def test_old_profiles_load_and_render_every_page(release, tmp_path, monkeypatch):
    from dailydigest.config import load_profile
    from dailydigest.opportunities import load_opportunity_profile

    _store, web, ids, data, _dump = _open(release, tmp_path, monkeypatch)
    assert load_profile(str(data / "profile.yaml")).bio
    assert load_opportunity_profile(data / "opportunities.yaml").country == "United Kingdom"
    digest = _text_payload(web.index(_request("GET", "/")))
    assert "Rated relevant paper" in digest and "Fixture funding call" in digest
    saved = _text_payload(web.saved_items(_request("GET", "/saved")))
    assert "Saved paper" in saved
    setup = _text_payload(web.setup_get(_request("GET", "/setup")))
    assert "Postdoctoral researcher" in setup


def test_new_profile_fields_must_have_defaults():
    """A new required field makes every existing profile fail to load.

    Give each new Profile, OpportunityProfile or Settings field a default; this
    list only shrinks.
    """
    from dailydigest.config import Settings
    from dailydigest.models import Profile
    from dailydigest.opportunities import OpportunityProfile

    def required(model):
        return {name for name, field in model.model_fields.items() if field.is_required()}

    assert required(Profile) <= {"bio"}
    assert required(OpportunityProfile) <= {"career_stage", "institution_type", "country", "applicant_role"}
    assert required(Settings) == set()
