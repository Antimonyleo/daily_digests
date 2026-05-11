"""Tests for dailydigest.config.load_sources and load_profile.

Uses tmp_path to write tiny YAML fixtures so tests are fully isolated.
"""

from __future__ import annotations

import pytest
import yaml

from dailydigest.config import load_profile, load_sources


# ---------------------------------------------------------------------------
# load_sources
# ---------------------------------------------------------------------------

class TestLoadSources:
    def test_single_section_single_source(self, tmp_path):
        sources_yaml = {
            "research": [
                {"name": "TestFeed", "kind": "rss", "url": "https://example.com/rss"},
            ]
        }
        p = tmp_path / "sources.yaml"
        p.write_text(yaml.dump(sources_yaml))

        result = load_sources(str(p))
        assert len(result) == 1
        assert result[0].name == "TestFeed"
        assert result[0].section == "research"
        assert result[0].kind == "rss"

    def test_multiple_sections(self, tmp_path):
        sources_yaml = {
            "research": [
                {"name": "Nature", "kind": "rss", "url": "https://nature.com/rss"},
                {"name": "bioRxiv", "kind": "biorxiv", "server": "biorxiv"},
            ],
            "industry": [
                {"name": "FierceBiotech", "kind": "rss", "url": "https://fiercebiotech.com/rss"},
            ],
            "world": [
                {"name": "BBC", "kind": "rss", "url": "https://bbc.com/rss"},
            ],
        }
        p = tmp_path / "sources.yaml"
        p.write_text(yaml.dump(sources_yaml))

        result = load_sources(str(p))
        assert len(result) == 4

        sections = {s.section for s in result}
        assert sections == {"research", "industry", "world"}

    def test_source_section_assigned_from_yaml_key(self, tmp_path):
        sources_yaml = {
            "regulatory": [
                {"name": "FDA", "kind": "rss", "url": "https://fda.gov/rss"},
            ]
        }
        p = tmp_path / "sources.yaml"
        p.write_text(yaml.dump(sources_yaml))

        result = load_sources(str(p))
        assert result[0].section == "regulatory"

    def test_empty_section_produces_no_sources(self, tmp_path):
        sources_yaml = {"research": [], "industry": None}
        p = tmp_path / "sources.yaml"
        p.write_text(yaml.dump(sources_yaml))

        result = load_sources(str(p))
        assert result == []

    def test_optional_fields_default_none(self, tmp_path):
        sources_yaml = {
            "research": [{"name": "MinimalFeed", "kind": "rss"}]
        }
        p = tmp_path / "sources.yaml"
        p.write_text(yaml.dump(sources_yaml))

        result = load_sources(str(p))
        assert result[0].url is None
        assert result[0].category is None
        assert result[0].prestige_score is None
        assert result[0].quality_tier is None

    def test_source_quality_metadata_loaded(self, tmp_path):
        sources_yaml = {
            "research": [
                {
                    "name": "Nature",
                    "kind": "rss",
                    "url": "https://nature.com/rss",
                    "quality_tier": "top",
                    "prestige_score": 1.0,
                    "impact_floor": 7.0,
                }
            ],
            "industry": [
                {
                    "name": "PressFeed",
                    "kind": "rss",
                    "url": "https://example.com/rss",
                    "promo_risk": 0.8,
                }
            ],
        }
        p = tmp_path / "sources.yaml"
        p.write_text(yaml.dump(sources_yaml))

        result = load_sources(str(p))

        nature = next(s for s in result if s.name == "Nature")
        press = next(s for s in result if s.name == "PressFeed")
        assert nature.quality_tier == "top"
        assert nature.prestige_score == pytest.approx(1.0)
        assert nature.impact_floor == pytest.approx(7.0)
        assert press.promo_risk == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# load_profile
# ---------------------------------------------------------------------------

class TestLoadProfile:
    def test_basic_profile_fields(self, tmp_path):
        profile_yaml = {
            "name": "Hao",
            "bio": "Researcher in CRISPR and gene therapy.",
            "keywords": ["CRISPR", "gene editing", "mRNA"],
            "downweight": ["cryptocurrency"],
        }
        p = tmp_path / "profile.yaml"
        p.write_text(yaml.dump(profile_yaml))

        profile = load_profile(str(p))
        assert profile.name == "Hao"
        assert "CRISPR" in profile.bio
        assert len(profile.keywords) == 3
        assert "CRISPR" in profile.keywords
        assert profile.downweight == ["cryptocurrency"]

    def test_profile_without_downweight_defaults_empty(self, tmp_path):
        profile_yaml = {
            "bio": "A researcher.",
            "keywords": ["proteomics"],
        }
        p = tmp_path / "profile.yaml"
        p.write_text(yaml.dump(profile_yaml))

        profile = load_profile(str(p))
        assert profile.downweight == []

    def test_profile_without_keywords_defaults_empty(self, tmp_path):
        profile_yaml = {"bio": "Just a bio."}
        p = tmp_path / "profile.yaml"
        p.write_text(yaml.dump(profile_yaml))

        profile = load_profile(str(p))
        assert profile.keywords == []

    def test_profile_bio_preserved_exactly(self, tmp_path):
        bio_text = "Multiline\nbio with\nlinebreaks."
        profile_yaml = {"bio": bio_text, "keywords": []}
        p = tmp_path / "profile.yaml"
        p.write_text(yaml.dump(profile_yaml))

        profile = load_profile(str(p))
        assert profile.bio.strip() == bio_text

    def test_three_keywords_loaded(self, tmp_path):
        profile_yaml = {
            "bio": "Researcher.",
            "keywords": ["kw1", "kw2", "kw3"],
        }
        p = tmp_path / "profile.yaml"
        p.write_text(yaml.dump(profile_yaml))

        profile = load_profile(str(p))
        assert len(profile.keywords) == 3

    def test_missing_profile_file_raises(self, tmp_path, monkeypatch):
        # No fallback example file in tmp_path; should raise FileNotFoundError.
        monkeypatch.chdir(tmp_path)  # ensure no config/profile.example.yaml exists
        with pytest.raises(FileNotFoundError):
            load_profile(str(tmp_path / "nonexistent.yaml"))
