"""Offline checks for topic retrieval outside the configured category feeds."""
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from dailydigest.ingest import openalex
from dailydigest.models import Profile, SourceSpec
from dailydigest.rank.source_quality import (
    is_preprint_source,
    is_published_journal_source,
)


@pytest.mark.parametrize("work_type", ["article", "preprint"])
def test_configured_topic_search_retrieves_journal_and_preprint_work(monkeypatch, work_type):
    configuration = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config/sources.yaml").read_text()
    )
    entry = next(source for source in configuration["research"]
                 if source["name"] == "OpenAlex (your topics)")
    spec = SourceSpec(section="research", **entry)
    # Quantum preprints have no direct category reader in the default sources.
    assert not any(source.get("category") == "quant-ph"
                   for source in configuration["research"])
    monkeypatch.setattr("dailydigest.config.load_profile", lambda: Profile(
        bio="", keywords=["quantum error correction"],
    ))
    calls = []

    def get_json(url, params, headers):
        assert url == openalex.OpenAlexSource.BASE
        calls.append(dict(params))
        assert params["search"] == "quantum error correction"
        allowed_types = params["filter"].split("type:", 1)[1].split(",", 1)[0].split("|")
        works = [{"id": "https://openalex.org/W1", "type": work_type,
                  "title": "Quantum error correction with a validated decoder",
                  "publication_date": datetime.now(timezone.utc).date().isoformat(),
                  "primary_location": {
                      "landing_page_url": "https://arxiv.org/abs/2610.00001",
                      "source": {"display_name": "arXiv"},
                  }}] if work_type in allowed_types else []
        return {"meta": {"count": len(works)}, "results": works}

    monkeypatch.setattr(openalex, "_get_json", get_json)
    source = openalex.OpenAlexSource()
    items = source.fetch(spec)
    assert len(items) == 1, "The configured topic fallback must include indexed preprints"
    assert len(calls) == 1 and int(calls[0]["per-page"]) <= source.PROFILE_PER_TERM
    assert source.last_fetch_diagnostics["item_budget"] == source.PROFILE_MAX_ITEMS
    if work_type == "preprint":
        assert items[0].metadata["work_type"] == "preprint"
        assert is_preprint_source(items[0])
        assert not is_published_journal_source(items[0])
