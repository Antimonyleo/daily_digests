import subprocess
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_private_agent_files_are_ignored_and_not_present_in_release_tree():
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
    )

    for private_path in (
        "AGENTS.md",
        "CLAUDE.md",
        "CODEX.md",
        "GEMINI.md",
        ".agents/",
        ".claude/",
        ".codex/",
    ):
        assert private_path in gitignore
        prefix = private_path.rstrip("/")
        assert not any(path == prefix or path.startswith(f"{prefix}/") for path in tracked)

    assert not (ROOT / "docs" / "code-review-2026-05-18.md").exists()
    assert not (ROOT / "docs" / "ranking-research-2026-05-18.md").exists()


def test_dockerfile_pins_uv_image_version():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "ghcr.io/astral-sh/uv:latest" not in dockerfile


def test_example_config_keeps_live_citation_enrichment_opt_in():
    example_env = (ROOT / ".env.example").read_text()

    assert "CITATION_ENRICHMENT=false" in example_env


def test_example_profile_respects_ten_core_topic_limit():
    profile = yaml.safe_load((ROOT / "config" / "profile.example.yaml").read_text())

    assert 1 <= len(profile["keywords"]) <= 10


def test_release_wheel_includes_browser_templates():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    forced = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert forced["templates"] == "dailydigest/templates"
    assert (ROOT / "templates" / "saved.html.j2").is_file()


def test_acs_sources_use_openalex_without_blocked_native_feeds():
    configured = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text())
    research = configured["research"]

    assert not any(
        "pubs.acs.org" in str(source.get("url", "")) for source in research
    )

    acs_openalex = [
        source
        for source in research
        if source.get("kind") == "openalex_venues"
        and source.get("name", "").startswith("ACS ")
    ]
    assert len(acs_openalex) == 2

    all_configured_ids = [
        venue_id
        for source in acs_openalex
        for venue_id in source.get("venue_ids", [])
    ]
    configured_ids = set(all_configured_ids)
    expected_ids = {
        "S145476921",   # ACS Nano
        "S143846845",   # Nano Letters
        "S111155417",   # JACS
        "S66104727",    # Chemistry of Materials
        "S2765035057",  # ACS Central Science
        "S4210207119",  # JACS Au
        "S164001016",   # ACS Applied Materials & Interfaces
        "S37391459",    # ACS Catalysis
        "S118914585",   # ACS Chemical Biology
        "S177196338",   # Macromolecules
        "S167262187",   # Journal of Chemical Information and Modeling
    }
    assert len(all_configured_ids) == len(configured_ids)
    assert configured_ids == expected_ids


def test_rsc_flagships_include_chemical_science():
    configured = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text())
    research = configured["research"]
    by_name = {source["name"]: source for source in research}

    assert {
        "Chemical Science",
        "Chem. Soc. Rev.",
        "Energy and Environmental Science",
    } <= by_name.keys()
    assert by_name["Chemical Science"]["url"] == "http://feeds.rsc.org/rss/sc"
