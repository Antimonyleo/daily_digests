# Reader state written by earlier releases

Each `release-*` folder holds what that release's own code wrote for a reader:
the database (`digest.sql`, a SQLite dump), `profile.yaml` (that release's
example profile) and `opportunities.yaml`. The reader rated one paper, saved
one, marked one known (from 2026-10-14 on) and had one funding call in a digest.

`tests/test_legacy_state.py` opens every folder with the current code and
checks that nothing the reader made is lost. Never edit these files: they are
what real installs have on disk. When a release changes what is stored, add a
folder for it:

```bash
git archive <commit> src config | tar -x -C /tmp/old
mkdir /tmp/out && cd /tmp/out
DB_PATH=data/digest.db PROFILE_PATH=data/profile.yaml \
OPPORTUNITY_PROFILE_PATH=data/opportunities.yaml DD_APP_ROOT=$PWD HF_HUB_OFFLINE=1 \
PYTHONPATH=/tmp/old/src <repo>/.venv/bin/python <repo>/tests/fixtures/legacy/make_fixture.py \
  <commit> /tmp/old/config/profile.example.yaml
```

Then copy `digest.sql`, `profile.yaml`, `opportunities.yaml` and
`fixture.json` from `/tmp/out/data` into `release-<date>/`.
