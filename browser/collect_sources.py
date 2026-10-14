"""Collect public configured RSS/arXiv metadata for a static browser deployment.

This command never loads a reader profile, ranks articles, or uses credentials.
Run with the locked native environment; publish the output beside static assets.
Failures are retained in the manifest rather than replaced with invented items.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

import yaml


@contextmanager
def isolated_settings(scratch):
    """Import config outside the project with no ambient settings or secrets."""
    working_directory = Path.cwd()
    environment = dict(os.environ)
    try:
        os.chdir(scratch)
        os.environ.clear()
        # Keep temporary files in this task's scratch. No credential-bearing
        # environment values reach the legacy module-level Settings instance.
        os.environ["TMPDIR"] = str(scratch)
        from dailydigest import config

        previous_settings, previous_accessor = config.SETTINGS, config.get_settings
        settings = config.Settings.model_construct(
            profile_path=str(scratch / "unused-profile.yaml"),
            opportunity_profile_path=str(scratch / "unused-opportunities.yaml"),
            db_path=str(scratch / "unused.db"), llm_backend="extractive",
            llm_api_key="", resend_api_key="",
        )
        config.SETTINGS = settings
        config.get_settings = lambda: settings
        try:
            yield settings
        finally:
            config.SETTINGS, config.get_settings = previous_settings, previous_accessor
    finally:
        os.chdir(working_directory)
        os.environ.clear()
        os.environ.update(environment)


def public_specs(path):
    from dailydigest.models import SourceSpec

    config = yaml.safe_load(Path(path).read_text())
    return [SourceSpec.model_validate({**row, "section": section})
            for section, rows in config.items() for row in rows
            if row.get("kind") in {"rss", "arxiv"} and not row.get("profile_driven")]


def collect(specs, days=21, adapters=None, now=None):
    if adapters is None:
        from dailydigest.ingest.arxiv import ArxivSource
        from dailydigest.ingest.rss import RSSSource

        adapters = {"rss": RSSSource, "arxiv": ArxivSource}
    now = now or (lambda: datetime.now(timezone.utc))

    def fetch(spec):
        record = {"kind": spec.kind, "section": spec.section, "url": spec.url,
                  "category": spec.category, "days": days, "fetched_at": None}
        try:
            items = adapters[spec.kind]().fetch(spec, days=days)
            record.update(status="ok", fetched_at=now().isoformat(),
                          items=[item.model_dump(mode="json") for item in items])
        except Exception as exc:
            record.update(status="error", attempted_at=now().isoformat(), items=[],
                          error=f"{type(exc).__name__}: {exc}"[:500])
        return spec.name, record

    # Publishers are independent; arXiv requests remain sequential and polite.
    rss = [spec for spec in specs if spec.kind == "rss"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = dict(pool.map(fetch, rss))
    arxiv = [spec for spec in specs if spec.kind == "arxiv"]
    for index, spec in enumerate(arxiv):
        if index:
            time.sleep(3)
        name, record = fetch(spec)
        sources[name] = record
    return {"version": 1, "generated_at": now().isoformat(), "days": days,
            "public_only": True, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="config/sources.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--days", type=int, default=21, choices=range(1, 91))
    args = parser.parse_args()
    sources_path, output = Path(args.sources).resolve(), Path(args.output).resolve()
    scratch = Path(tempfile.mkdtemp(prefix="dailydigest-public-sources-"))
    with isolated_settings(scratch):
        report = collect(public_specs(sources_path), args.days)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".pending")
    temporary.write_text(json.dumps(report, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    failures = {name: source["error"] for name, source in report["sources"].items()
                if source["status"] != "ok"}
    print(json.dumps({"output": str(output), "sources": len(report["sources"]),
                      "failures": failures}))


if __name__ == "__main__":
    main()
