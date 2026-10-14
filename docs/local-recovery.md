# Local backup and recovery

A backup is an explicit local operation. DailyDigest does not upload backups,
restore files automatically, or migrate an existing installation during restore.
Keep the backup somewhere private outside the active `data/` directory.

## Make a backup

1. Stop the DailyDigest server and disable any scheduled brew or other process
   that writes its data. Keep them stopped until capture finishes. The database
   backup is consistent on its own, but the separate profile and feedback-reason
   files cannot be captured in the same SQLite transaction.
2. Choose a new destination under an existing private directory. An existing
   destination is rejected, including an empty directory. Do not put it inside
   the source data directory.
3. From your installed environment, call the helper with your actual paths:

   ```python
   from dailydigest.backup import create_backup

   backup = create_backup(
       "data",
       "/absolute/private/backups/dailydigest-2026-10-10",
       config_paths=["config/sources.yaml"],
   )
   print(backup)
   ```

   Run this code using the project environment, for example `uv run python`
   from the repository directory. On Windows use your own absolute Windows
   destination path. If `DB_PATH` uses a different filename, pass
   `database_name="your-database.sqlite"` and use its parent as `data_dir`.

The helper includes the database plus existing `profile.yaml`,
`opportunities.yaml`, and `vote_reasons.json` in that data directory. Supply
custom profile locations with `profile_paths=[...]`; those copies appear under
`profiles/`. Explicit non-secret configuration files appear under `config/`.
The manifest records each included file's original path, byte size and SHA-256.
Store original paths privately too: they may contain your username.

Credentials, `.env`, process environment variables, model/embedding caches,
downloaded models and generated HTML previews are excluded by default. The
SQLite backup preserves all existing tables, including saved/known items,
ratings, feedback events, retained digests and ranking snapshots. Model caches
can be rebuilt; previously pruned history cannot be recovered from a new backup.

If you deliberately want to preserve `.env` or another credential file, pass it
separately as `credential_paths=[".env"]`. This explicit opt-in places it under
`credentials/`; protect the backup as carefully as your original credentials.
Inspect custom configuration files yourself before passing them: filename
checks cannot discover a token hidden in an arbitrary file.

Directories and files are created with POSIX modes 0700 and 0600. Windows
permissions depend on the destination's inherited ACLs; use a directory private
to your account and inspect its Security permissions before copying secrets.

A successful backup has a complete, valid `manifest.json`, published last. If
capture fails, the helper leaves the incomplete directory for inspection and
does not overwrite or delete it automatically. Use another new destination
after addressing the failure. Preserve anything needed before manually removing
an incomplete backup.

## Verify and rehearse recovery

Keep the app stopped. First check the manifest and file checksums using a trusted
backup directory:

```python
import hashlib
import json
from pathlib import Path

backup = Path("/absolute/private/backups/dailydigest-2026-10-10")
manifest = json.loads((backup / "manifest.json").read_text())
assert manifest["complete"] is True
for relative, record in manifest["files"].items():
    with (backup / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]
```

Copy the backup's `data/` directory to a **new, empty recovery location**, never
over the active data directory. Use a normal directory copy; do not include any
old destination `-wal` or `-shm` files. Check the copied database independently:

```python
import shutil
import sqlite3

recovered = Path("/absolute/private/recovery-check/data")  # must not exist
shutil.copytree(backup / "data", recovered)
with sqlite3.connect((recovered / "digest.db").resolve().as_uri() + "?mode=ro", uri=True) as db:
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print({table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
           for table in ("items", "votes", "bookmarks", "known_items", "digests")})
```

Check retained titles, saved/known items and feedback history against your
expectations. Copy custom profiles/configuration from their separate backup
folders to chosen recovery paths, using the manifest as a guide. Credentials
should be restored only deliberately; restarting with a recovered online
provider configuration can enable that provider again. Keep the original
installation and backup intact until the recovered state is verified.

To adopt the recovery, keep DailyDigest stopped and explicitly point `DB_PATH`,
`PROFILE_PATH`, `OPPORTUNITY_PROFILE_PATH` and any custom `SOURCES_PATH` at the
recovered files. Use the same application version for the first check; inspect
release/migration notes before opening the recovered database with another
version. Opening the app may write viewing history, so perform initial SQL and
file checks before starting it. There is no automatic restore command.

Never copy just a live SQLite `.db` file as a backup: committed data may still
be in its WAL. This helper uses SQLite's backup API and produces a standalone
database. It does not claim atomic capture of sidecars while the app runs.

## Verification limits

Tests restore fixtures into a fresh temporary location and verify the database,
profile and reason state. CI runs Linux tests plus native Windows PowerShell
parser and macOS Bash launcher fixtures; these do not install a real model,
contact providers, or prove first-run behavior on every desktop. Package tests
exercise assets outside the repository layout. Actual wheel inspection runs
only when the declared builder is already available; an absent builder is
reported as a skip rather than downloaded by a test.
