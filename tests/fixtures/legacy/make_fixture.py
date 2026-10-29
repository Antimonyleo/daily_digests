"""Build a legacy-state fixture with an old release's own code (see README.md).

Run with that release's src first on PYTHONPATH, cwd = an empty folder, and
DB_PATH/PROFILE_PATH/OPPORTUNITY_PROFILE_PATH pointing into ./data.
"""
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

import dailydigest
from dailydigest import store, votes
from dailydigest.models import Item

release, example = sys.argv[1], Path(sys.argv[2])
assert release in dailydigest.__file__, dailydigest.__file__
data = Path("data")
data.mkdir(exist_ok=True)
store.init_db()
when = datetime(2026, 10, 20, 9, tzinfo=timezone.utc)
items = [
    Item(source="Fixture Journal", section="research", external_id="liked", url="https://example.invalid/liked",
         title="Rated relevant paper", abstract="A paper the reader rated.", authors="Ada Lovelace",
         published_at=when),
    Item(source="Fixture Journal", section="research", external_id="saved", url="https://example.invalid/saved",
         title="Saved paper", abstract="A paper the reader saved.", published_at=when),
    Item(source="Fixture Journal", section="research", external_id="known", url="https://example.invalid/known",
         title="Known paper", abstract="A paper the reader already knew.", published_at=when),
    Item(source="Grants.gov", section="opportunities", external_id="grant", url="https://example.invalid/grant",
         title="Fixture funding call", published_at=when,
         metadata={"deadline": "2027-01-07", "status": "open", "official": True}),
]
store.upsert_items(items)
with store.session_scope() as session:
    ids = {row.external_id: row.id for row in session.query(store.ItemRow).all()}
store.write_digest("2026-10-20", [("R1", ids["liked"], 0.9), ("R2", ids["saved"], 0.8),
                                  ("R3", ids["known"], 0.7), ("O1", ids["grant"], 0.6)])
assert votes.record_vote_by_id(ids["liked"], 1, 2) if "grade" in votes.record_vote_by_id.__code__.co_varnames \
    else votes.record_vote_by_id(ids["liked"], 1)
store.set_bookmark(ids["saved"], True)
if hasattr(store, "set_item_known"):
    store.set_item_known(ids["known"], True)
shutil.copyfile(example, data / "profile.yaml")
from dailydigest import opportunities
fields = dict(career_stage="Postdoctoral researcher", institution_type="University", country="United Kingdom",
              applicant_role="Principal investigator", opportunity_types=["fellowship"], event_types=["conference"])
if opportunities.OpportunityProfile.model_fields["description"].is_required():
    fields["description"] = "Postdoctoral researcher looking for fellowships and conferences."
opportunity = opportunities.OpportunityProfile(**fields)
(data / "opportunities.yaml").write_text(yaml.safe_dump(opportunity.model_dump(), sort_keys=False))
with sqlite3.connect(data / "digest.db") as db:
    user_version = db.execute("PRAGMA user_version").fetchone()[0]
    dump = "\n".join(db.iterdump()) + "\n"
(data / "digest.sql").write_text(f"PRAGMA user_version={user_version};\n" + dump)
(data / "fixture.json").write_text(json.dumps({"release": release, "ids": ids}, indent=1) + "\n")
print(release, user_version, len(dump.splitlines()), "lines")
