"""Search transparency must describe observations, never inferred paper supply."""
from dailydigest.pipeline import _interest_coverage_audit


def test_interest_audit_keeps_actual_query_and_empty_search_without_supply_claim():
    rows = _interest_coverage_audit({
        'profile': {'keywords': ['Induce proximity']},
        'source_health': [{'source': 'PubMed', 'diagnostics': {'queries': [
            {'interest': 'Induce proximity', 'query': '(induce[tiab] OR induced[tiab]) AND proximity[tiab]',
             'returned_count': 0, 'attempted': True},
            {'interest': 'Induce proximity', 'query': 'proximity', 'failure': 'timeout'},
        ]}}], 'candidates': [],
    })
    assert rows[0]['searches'][0] == {
        'source': 'PubMed', 'query': '(induce[tiab] OR induced[tiab]) AND proximity[tiab]',
        'status': 'completed', 'returned_count': 0,
    }
    assert rows[0]['searches'][1]['returned_count'] is None
    assert rows[0]['searches'][1]['status'] == 'failed'
    assert 'not all retrieved papers' in rows[0]['explanation']


def test_unattempted_interest_search_is_not_reported_as_completed():
    rows = _interest_coverage_audit({
        'profile': {'keywords': ['Induce proximity']},
        'source_health': [{'source': 'PubMed', 'diagnostics': {'queries': [
            {'interest': 'Induce proximity', 'query': 'proximity', 'attempted': False,
             'returned_count': 0, 'truncated': True},
        ]}}],
    })
    assert rows[0]['searches'][0]['status'] == 'not run'
    assert rows[0]['searches'][0]['returned_count'] is None


def test_limited_interest_search_preserves_provider_warning_without_supply_claim():
    rows = _interest_coverage_audit({
        "profile": {"keywords": ["protein engineering"]},
        "source_health": [{"source": "PubMed", "diagnostics": {"queries": [
            {"interest": "protein engineering", "query": "protein*[tiab]",
             "returned_count": 0, "attempted": True, "truncated": True,
             "warnings": {"quotedphrasesnotfound": ["protein engineering"],
                          "outputmessages": ["Provider expansion was limited."]}},
        ]}}],
    })
    search = rows[0]["searches"][0]
    assert search["status"] == "limited"
    assert search["returned_count"] == 0
    assert search["warnings"] == ["Quoted phrase not found: protein engineering", "Provider expansion was limited."]
    assert "not all retrieved papers" in rows[0]["explanation"]


def test_interest_panel_distinguishes_observed_zero_from_failed_or_limited_searches():
    from pathlib import Path
    from jinja2 import Environment

    source = (Path(__file__).resolve().parents[1] / "templates/digest_web.html.j2").read_text()
    section = source[source.index("  {% if interest_coverage %}"):]
    section = section[:section.index("  {% endif %}") + len("  {% endif %}")]
    template = Environment(autoescape=True).from_string(section)
    html = template.render(interest_coverage=[{
        "title": "Custom topic", "selected": 0, "candidates": 0, "eligible": 0,
        "explanation": "Counts are not total paper supply.", "queries": [],
        "searches": [
            {"source": "Limited source", "status": "limited", "returned_count": 0,
             "query": "topic*", "warnings": ["<unsafe> expansion warning"]},
            {"source": "Failed source", "status": "failed", "returned_count": 0},
            {"source": "Skipped source", "status": "not run", "returned_count": 0},
            {"source": "Good source", "status": "completed", "returned_count": 0},
        ],
    }])
    assert "Limited source: limited" in html
    assert "Failed source: failed" in html
    assert "Skipped source: not run" in html
    assert "Good source: completed" in html
    assert html.count("0 papers returned") == 2
    assert "&lt;unsafe&gt; expansion warning" in html
    assert "<unsafe>" not in html


def test_result_details_name_the_observed_best_matching_interest_without_new_inference():
    from pathlib import Path
    from jinja2 import Environment

    source = (Path(__file__).resolve().parents[1] / "templates/digest_web.html.j2").read_text()
    section = source[source.index('              <div class="why-copy">'):]
    section = section[:section.index("              </div>") + len("              </div>")]
    template = Environment(autoescape=True).from_string(section)
    html = template.render(item={"ranking": {
        "primary_facet": "Quantum error correction", "why_shown": ["Topic fit"],
    }})
    assert "Best-matching interest: Quantum error correction." in html
    assert "Topic fit" in html
    html = template.render(item={"ranking": {}})
    assert "Best-matching interest:" not in html


def test_interest_counts_describe_attributed_candidates_not_provider_supply():
    rows = _interest_coverage_audit({
        "profile": {"keywords": ["Quantum error correction", "Solar cells"]},
        "source_health": [{"source": "Shared search", "diagnostics": {"queries": [
            {"query": "quantum OR solar", "returned_count": 100},
        ]}}],
        "candidates": [
            {"features": {"primary_facet": "Quantum error correction"},
             "selected": False, "eligible": False, "first_rejected_stage": "quality_gate"},
            {"features": {"primary_facet": "Quantum error correction"},
             "selected": False, "eligible": True, "first_rejected_stage": "selection_budget"},
            {"features": {"primary_facet": "Quantum error correction"},
             "selected": True, "eligible": True},
            {"features": {"primary_facet": "Solar cells"}, "selected": True, "eligible": True},
        ],
    })
    assert rows[0]["candidates"] == 3
    assert rows[0]["eligible"] == 2
    assert rows[0]["rejections"] == {"quality_gate": 1, "selection_budget": 1}
    assert rows[0]["searches"] == []
    assert rows[1]["candidates"] == 1
    assert "shared searches cannot establish an individual supply count" in rows[0]["explanation"]
