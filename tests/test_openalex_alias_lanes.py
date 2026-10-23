"""Broader explicit vocabulary must not crowd out a reader's original query."""
from datetime import datetime, timezone

from dailydigest.ingest import openalex
from dailydigest.models import CanonicalFacet, Profile, SourceSpec


def _work(key):
    return {"id": f"https://openalex.org/{key}", "title": key,
            "publication_date": datetime.now(timezone.utc).date().isoformat()}


def _profile(monkeypatch, *, core=("graphene crystallization",), context=(), aliases=None):
    monkeypatch.setattr("dailydigest.config.load_profile", lambda: Profile(
        bio="", keywords=list(core), context_keywords=list(context),
        canonical_facets={name: CanonicalFacet(aliases=values)
                          for name, values in (aliases or {}).items()},
    ))


def _spec():
    return SourceSpec(name="Topics", kind="openalex", section="research",
                      profile_driven=True, openalex_types="article|preprint")


def test_broader_alias_cannot_crowd_out_original_core_match(monkeypatch):
    _profile(monkeypatch, aliases={"graphene crystallization": ["graphene"]})

    def fetch(url, params, headers):
        if params["search.title_abstract_keywords"] == "graphene crystallization":
            return {"meta": {"count": 1}, "results": [_work("focused-method")]}
        return {"meta": {"count": 100, "next_cursor": "more"},
                "results": [_work(f"broad-{i}") for i in range(int(params["per-page"]))]}

    monkeypatch.setattr(openalex, "_get_json", fetch)
    source = openalex.OpenAlexSource()
    items = source.fetch(_spec())
    assert "focused-method" in {item.title for item in items}
    assert "broad-0" in {item.title for item in items}
    assert source.last_fetch_diagnostics["queries"][1]["interest"] == "graphene crystallization"


def test_no_spare_request_retains_combined_alias_coverage(monkeypatch):
    core = [f"topic-{i:02d}" for i in range(12)]
    _profile(monkeypatch, core=core, aliases={core[0]: ["related vocabulary"]})
    calls = []

    def fetch(url, params, headers):
        query = params["search.title_abstract_keywords"]
        calls.append(query)
        key = "alias-only" if "related vocabulary" in query else query
        return {"meta": {"count": 1}, "results": [_work(key)]}

    monkeypatch.setattr(openalex, "_get_json", fetch)
    source = openalex.OpenAlexSource()
    items = source.fetch(_spec())
    assert "alias-only" in {item.title for item in items}
    assert len(calls) == 12


def test_aliases_run_after_original_core_and_context_queries(monkeypatch):
    _profile(monkeypatch, core=["z-core", "a-core"], context=["background"],
             aliases={"a-core": ["A-CORE", "  broader  ", "broader"]})
    calls = []

    def fetch(url, params, headers):
        calls.append(params["search.title_abstract_keywords"])
        return {"meta": {"count": 1}, "results": [_work(str(len(calls)))]}

    monkeypatch.setattr(openalex, "_get_json", fetch)
    source = openalex.OpenAlexSource()
    source.fetch(_spec())
    assert calls == ["a-core", "z-core", "background", '"broader"']
    diagnostics = source.last_fetch_diagnostics["queries"]
    assert [q["query_role"] for q in diagnostics] == ["core", "core", "context", "alias"]
    assert diagnostics[-1]["interest"] == "a-core"


def test_failed_alias_keeps_core_results_and_failure_diagnostics(monkeypatch):
    _profile(monkeypatch, aliases={"graphene crystallization": ["graphene"]})

    def fetch(url, params, headers):
        if params["search.title_abstract_keywords"] == '"graphene"':
            raise RuntimeError("optional source query unavailable")
        return {"meta": {"count": 1}, "results": [_work("core-result")]}

    monkeypatch.setattr(openalex, "_get_json", fetch)
    source = openalex.OpenAlexSource()
    assert [item.title for item in source.fetch(_spec())] == ["core-result"]
    alias = source.last_fetch_diagnostics["queries"][-1]
    assert alias["query_role"] == "alias"
    assert alias["failure"] and alias["truncated"] and not alias["exhausted"]


def test_alias_cannot_rescue_failed_original_core(monkeypatch):
    import pytest

    _profile(monkeypatch, aliases={"graphene crystallization": ["graphene"]})
    calls = []

    def fetch(url, params, headers):
        query = params["search.title_abstract_keywords"]
        calls.append(query)
        if query == "graphene crystallization":
            raise RuntimeError("original core unavailable")
        return {"meta": {"count": 1}, "results": [_work("alias-result")]}

    monkeypatch.setattr(openalex, "_get_json", fetch)
    with pytest.raises(RuntimeError, match="all core profile searches failed"):
        openalex.OpenAlexSource().fetch(_spec())
    assert calls == ["graphene crystallization"]


def test_saturated_aliases_stay_inside_raw_record_and_request_budgets(monkeypatch):
    core = [f"topic-{i:02d}" for i in range(8)]
    _profile(monkeypatch, core=core, aliases={name: [name + " alternative"] for name in core})
    calls = []

    def fetch(url, params, headers):
        calls.append(dict(params))
        # Even complete duplication must consume the raw upstream budget.
        return {"meta": {"count": 999, "next_cursor": "more"},
                "results": [_work("duplicate")] * int(params["per-page"])}

    monkeypatch.setattr(openalex, "_get_json", fetch)
    source = openalex.OpenAlexSource()
    assert len(source.fetch(_spec())) == 1
    assert len(calls) == 9
    assert sum(int(call["per-page"]) for call in calls) == 180
    queries = source.last_fetch_diagnostics["queries"]
    assert queries[-1]["query_role"] == "alias"
    assert queries[-1]["truncated"] and not queries[-1]["exhausted"]
    assert sum(query["unique_yield"] for query in queries) == 1
    assert all(q.get("alias_split_reason") == "budget"
               for q in queries if q["query_role"] == "combined")


def test_alias_plan_is_profile_order_invariant(monkeypatch):
    core = [f"topic-{i:02d}" for i in range(8)]
    aliases = {name: [name + " alternative"] for name in reversed(core)}
    calls = []

    def fetch(url, params, headers):
        calls.append((params["search.title_abstract_keywords"], int(params["per-page"])))
        return {"meta": {"count": 999, "next_cursor": "more"},
                "results": [_work(str(i)) for i in range(int(params["per-page"]))]}

    monkeypatch.setattr(openalex, "_get_json", fetch)
    _profile(monkeypatch, core=core, aliases=aliases)
    first = openalex.OpenAlexSource().fetch(_spec())
    original_calls = calls[:]
    calls.clear()
    _profile(monkeypatch, core=list(reversed(core)), aliases=aliases)
    second = openalex.OpenAlexSource().fetch(_spec())
    assert calls == original_calls
    assert {item.external_id for item in second} == {item.external_id for item in first}
