"""U8: one work takes one slot; history is grouped for display without rewriting it."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from dailydigest import store
from dailydigest.dedupe import (
    dedupe_ranking_candidates,
    group_publication_versions,
    publication_version_links,
)
from dailydigest.rank.ranker import apply_exploration, pick_top_per_section
from dailydigest.rank.source_quality import is_preprint_source, should_skip_item, source_bucket

DAY = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
AF3 = ("AlphaFold3 and Intrinsically Disordered Proteins: Reliable Monomer "
       "Prediction but Unreliable Complex Prediction")
RICH = "We benchmark AlphaFold3 on disordered proteins and complexes across many systems. " * 18


def row(item_id, source, doi, *, title=AF3, authors="Dao, T. M.; Ghent, S.; Uversky, V. N.; Rahman, T.",
        url=None, metadata=None, abstract=RICH, published_at=DAY, section="research", label=None):
    meta = {"doi": doi, **(metadata or {})} if doi else dict(metadata or {})
    return store.ItemRow(
        id=item_id, source=source, section=section, external_id=doi or f"x{item_id}",
        url=url or f"https://doi.org/{doi}", title=title, authors=authors, abstract=abstract,
        published_at=published_at, metadata_json=json.dumps(meta), item_label=label,
    )


def biorxiv(item_id=97624, **kw):
    kw.setdefault("metadata", {"preprint_type": "new results", "preprint_version": "2"})
    return row(item_id, "bioRxiv (recent)", "10.64898/2025.12.05.691730", **kw)


def pubmed(item_id=96791, **kw):
    kw.setdefault("metadata", {"publication_types": ["Journal Article"], "venue": "Proteins"})
    kw.setdefault("authors", "Dao TM, Ghent S, Uversky VN, Rahman T")
    kw.setdefault("url", "https://pubmed.ncbi.nlm.nih.gov/42861594/")
    kw.setdefault("published_at", DAY - timedelta(days=1))
    return row(item_id, "PubMed (your topics)", "10.1002/prot.70189", **kw)


def meta(item):
    return json.loads(item.metadata_json)


# --- brew path -------------------------------------------------------------

@pytest.mark.parametrize("reverse", [False, True])
def test_survivor_keeps_the_other_versions_doi_venue_url_and_relation_source(reverse):
    """The real 2026-10-13 pair (R2 bioRxiv, R8 PubMed/Proteins) takes one slot."""
    preprint, journal = biorxiv(), pubmed()
    journal_before = journal.metadata_json
    result = dedupe_ranking_candidates([journal, preprint] if reverse else [preprint, journal])
    assert result == [preprint]
    (link,) = publication_version_links(preprint)
    assert link["item_id"] == 96791 and link["doi"] == "10.1002/prot.70189"
    assert link["venue"] == "Proteins" and link["is_preprint"] is False
    assert link["url"] == "https://pubmed.ncbi.nlm.nih.gov/42861594/"
    assert link["source"] == "PubMed (your topics)" and link["retraction"] is None
    assert link["relation"] == "same_title_authors"
    assert link["evidence"]["between_item_ids"] == [96791, 97624]
    # Statuses stay per record: neither version takes the other's fields.
    assert journal.metadata_json == journal_before and is_preprint_source(preprint)
    assert "venue" not in meta(preprint) and meta(preprint)["preprint_version"] == "2"


def test_explicit_relation_with_changed_title_links_without_copying_preprint_type():
    preprint = row(5, "ChemRxiv", "10.26434/chemrxiv.1/v1", title="Old preprint title",
                   metadata={"work_type": "preprint", "published_doi": "10.1021/jacs.9"})
    journal = row(6, "Journal of the American Chemical Society", "10.1021/jacs.9",
                  title="A renamed journal article", authors="Someone Else",
                  metadata={"publication_types": ["Journal Article"]})
    result = dedupe_ranking_candidates([preprint, journal])
    assert result == [journal]
    assert meta(journal)["publication_types"] == ["Journal Article"]
    assert source_bucket(journal) != "preprint_other"
    (link,) = publication_version_links(journal)
    assert link["relation"] == "published_doi"
    assert link["evidence"] == {"field": "metadata.published_doi", "value": "10.1021/jacs.9",
                                "declared_by_item_id": 5, "declared_by_source": "ChemRxiv",
                                "between_item_ids": [5, 6]}


def test_withdrawn_preprint_never_withdraws_its_published_version():
    preprint = row(5, "bioRxiv", "10.1101/w", metadata={
        "published_doi": "10.1021/jacs.9", "is_retracted": True, "preprint_type": "withdrawn"})
    journal = row(6, "Journal of the American Chemical Society", "10.1021/jacs.9",
                  metadata={"publication_types": ["Journal Article"]})
    result = dedupe_ranking_candidates([preprint, journal])
    assert "is_retracted" not in meta(journal) and not should_skip_item(journal)
    assert journal in result
    # The withdrawn record keeps its own status and is left for the quality gate.
    assert should_skip_item(preprint)


@pytest.mark.parametrize("abstract,survivor", [("", "preprint"), ("Teaser.", "preprint"),
                                               (RICH[:600], "journal")])
def test_thin_published_metadata_does_not_drop_the_richer_preprint_abstract(abstract, survivor):
    preprint = biorxiv()
    journal = row(9, "Nature", "10.1038/s41586-026-00001-1", abstract=abstract,
                  authors="Dao TM, Ghent S, Uversky VN, Rahman T")
    result = dedupe_ranking_candidates([journal, preprint])
    expected = preprint if survivor == "preprint" else journal
    assert result == [expected]
    (link,) = publication_version_links(expected)
    assert link["is_preprint"] is (survivor == "journal")
    assert preprint.abstract == RICH and journal.abstract == abstract  # no text copied


def test_two_preprints_declaring_the_same_publication_are_one_work():
    first = row(1, "bioRxiv", "10.1101/one", title="First server title",
                metadata={"published_doi": "10.1038/shared"})
    second = row(2, "arXiv q-bio", None, url="https://arxiv.org/abs/2601.00001",
                 title="Second server title", metadata={"published_doi": "https://doi.org/10.1038/shared"})
    result = dedupe_ranking_candidates([first, second])
    assert len(result) == 1
    assert publication_version_links(result[0])[0]["relation"] == "shared_published_doi"


@pytest.mark.parametrize("order", [0, 1, 2])
def test_versions_from_other_source_groups_and_days_collapse_with_openalex_copy(order):
    openalex = row(97930, "OpenAlex (your topics)", "10.1002/prot.70189", authors="Tuan Minh Dao, "
                   "Sebastien Ghent, Vladimir N. Uversky, Taseef Rahman",
                   metadata={"venue": "Proteins Structure Function and Bioinformatics",
                             "work_type": "article"}, published_at=DAY - timedelta(days=30))
    rows = [biorxiv(), pubmed(), openalex]
    rows = rows[order:] + rows[:order]
    result = dedupe_ranking_candidates(rows)
    assert len(result) == 1
    links = publication_version_links(result[0])
    assert len(links) == 1 and links[0]["is_preprint"] is not is_preprint_source(result[0])


JOURNALS = ["Nature Biotechnology", "Nature Nanotechnology", "Science", "Cell", "Nature Chemistry",
            "Nature Materials", "Nature Methods", "Nature Communications", "PNAS", "eLife", "JACS"]


def _unique(n):
    return [row(1000 + i, JOURNALS[i], f"10.1038/unique{i}",
                title=f"Distinct programmable RNA nanostructure study number {i} with unique methods",
                authors=f"Author{i} A") for i in range(n)]


@pytest.mark.parametrize("catch_up", [False, True])
def test_one_work_one_slot_and_cap_backfilled_with_unique_candidates(catch_up):
    candidates = dedupe_ranking_candidates([biorxiv(), pubmed(), *_unique(5)])
    scored = [(item, 0.99 - 0.01 * index) for index, item in enumerate(candidates)]
    picked = pick_top_per_section(scored, {"research": 5}, catch_up=catch_up)
    titles = [item.title for item, _ in picked]
    assert len(picked) == 5 and titles.count(AF3) == 1
    # Exploration only draws from the deduped pool, so it cannot reintroduce a version.
    explored = apply_exploration(picked, scored, {int(i.id): 1.0 for i, _ in scored},
                                 slots=2, eligible=lambda _row: True)
    assert [item.title for item, _ in explored].count(AF3) == 1


# --- history grouping ------------------------------------------------------

def _digest_1013():
    labels = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]
    items = _unique(11)
    items[1], items[7] = biorxiv(label="R2"), pubmed(label="R8")
    for item, label in zip(items, labels, strict=True):
        item.item_label = label
    return items


def test_historical_digest_counts_unique_works_without_rewriting_rows():
    items = _digest_1013()
    before = [(i.id, i.title, i.abstract, i.metadata_json) for i in items]
    groups = group_publication_versions(items, ratings={97624: 40}, saved={96791})
    assert len(groups) == 10
    assert [(i.id, i.title, i.abstract, i.metadata_json) for i in items] == before
    work = next(g for g in groups if len(g.item_ids) == 2)
    assert work.primary.item_label == "R2" and work.item_ids == (97624, 96791)
    assert work.links[0]["venue"] == "Proteins" and work.links[0]["label"] == "R8"
    assert work.links[0]["in_digest"] is True
    # Explicit rule: own rating only, other-version ratings exposed, saved = any.
    assert work.primary_rating == 40 and work.other_version_ratings == ()
    assert work.saved is True and work.saved_item_ids == (96791,)
    assert [g.primary.item_label for g in groups][:3] == ["R1", "R2", "R3"]


def test_rating_on_the_other_version_is_exposed_never_merged():
    groups = group_publication_versions([biorxiv(), pubmed()], ratings={96791: 100})
    (work,) = groups
    assert work.primary.id == 97624 and work.primary_rating is None
    assert work.other_version_ratings == ((96791, 100),) and work.rated_on_other_version
    conflict = group_publication_versions([biorxiv(), pubmed()], ratings={96791: 100, 97624: 40})[0]
    assert conflict.primary_rating == 40 and conflict.rating_conflict


def test_explicit_relation_groups_changed_title_and_records_evidence():
    preprint = biorxiv(metadata={"published_doi": "10.1002/prot.70189"}, title="Earlier draft title")
    groups = group_publication_versions([preprint, pubmed(title="Final journal title")])
    assert len(groups) == 1 and groups[0].links[0]["relation"] == "published_doi"


@pytest.mark.parametrize("journal_authors", [
    "Carol Lee, David Chen",           # conflicting
    "",                                 # unknown
    "Tim Dao, Sam Ghent, Vic Uversky",  # different list length
])
def test_conflicting_or_unclear_authors_stay_separate(journal_authors):
    # An unknown list cannot conflict, so the identical distinctive title (the
    # existing brew rule) still decides; a conflicting or unequal list never joins.
    expected = 1 if not journal_authors else 2
    groups = group_publication_versions([biorxiv(), pubmed(authors=journal_authors)])
    assert len(groups) == expected
    assert len(dedupe_ranking_candidates([biorxiv(), pubmed(authors=journal_authors)])) == expected


@pytest.mark.parametrize("title", ["Editorial", "Correction", "Correction to: RNA origami"])
def test_short_generic_titles_never_group(title):
    a = row(1, "bioRxiv", "10.1101/a", title=title, authors="Alice Smith")
    b = row(2, "Nature", "10.1038/b", title=title, authors="Alice Smith",
            published_at=DAY - timedelta(days=3))
    assert len(group_publication_versions([a, b])) == 2
    assert len(dedupe_ranking_candidates([a, b])) == 2


def test_withdrawn_version_keeps_its_own_card():
    withdrawn = biorxiv(metadata={"preprint_type": "withdrawn", "is_retracted": True})
    groups = group_publication_versions([withdrawn, pubmed()])
    assert len(groups) == 2


@pytest.mark.parametrize("second", [
    dict(source="Science", doi="10.1126/b"),  # two journals: never a version pair
    dict(source="bioRxiv", doi="10.1101/b", title=AF3.replace("Monomer", "Dimer")),
])
def test_different_dois_for_different_papers_stay_separate(second):
    a = row(1, "Nature", "10.1038/a", published_at=DAY - timedelta(days=2))
    b = row(2, second["source"], second["doi"], title=second.get("title", AF3))
    assert len(group_publication_versions([a, b])) == 2
    assert len(dedupe_ranking_candidates([a, b])) == 2


def test_version_pair_spanning_days_groups():
    groups = group_publication_versions([
        biorxiv(published_at=DAY), pubmed(published_at=DAY - timedelta(days=60))])
    assert len(groups) == 1


def test_topically_similar_but_different_papers_stay_separate():
    other = pubmed(title="AlphaFold3 and Intrinsically Disordered Proteins: Reliable Complex "
                         "Prediction but Unreliable Monomer Prediction", item_id=5)
    evaluating = row(96803, "PubMed (your topics)", "10.1093/gpbjnl/qzag026",
                     title="Evaluating AlphaFold Tools and Related Scoring Functions for "
                           "Protein-peptide Complexes", authors="Manshour N, Ren JZ")
    assert len(group_publication_versions([biorxiv(), other, evaluating])) == 3
    assert len(dedupe_ranking_candidates([biorxiv(), other, evaluating])) == 3


def test_recorded_brew_links_attach_to_their_card_without_counting_as_items():
    journal = pubmed()
    recorded = {96791: [{"item_id": 97624, "doi": "10.64898/2025.12.05.691730", "venue": "bioRxiv",
                         "url": "https://doi.org/10.64898/2025.12.05.691730", "is_preprint": True,
                         "relation": "same_title_authors"}]}
    (work,) = group_publication_versions([journal], recorded_links=recorded)
    assert work.item_ids == (96791,)
    assert work.links[0]["item_id"] == 97624 and work.links[0]["in_digest"] is False


def test_grouping_is_section_scoped_and_order_deterministic():
    news = row(7, "STAT", None, url="https://doi.org/10.1002/prot.70189", section="industry")
    forward = group_publication_versions([biorxiv(), pubmed(), news])
    backward = group_publication_versions([pubmed(), biorxiv(), news])
    assert len(forward) == len(backward) == 2
    assert forward[0].primary.id == 97624 and backward[0].primary.id == 96791


# --- end-to-end brew (fakes only, temporary DB) ------------------------------

def test_brew_serves_one_version_records_the_link_and_leaves_rows_untouched(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from dailydigest import config as config_mod
    from dailydigest import pipeline as pipeline_mod

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store.SETTINGS = config_mod.SETTINGS
    store._ENGINE = store._SessionLocal = None
    store._INITIALIZED = False
    store.init_db()
    now = datetime.now(timezone.utc)
    fixtures = [biorxiv(item_id=None, published_at=now), pubmed(item_id=None, published_at=now - timedelta(days=1)),
                *[row(None, JOURNALS[i], f"10.1038/unique{i}", published_at=now,
                      title=f"Distinct programmable RNA nanostructure study number {i} with unique methods",
                      authors=f"Author{i} A") for i in range(3)]]
    with store.session_scope() as session:
        session.add_all(fixtures)
        session.flush()
        ids = [int(item.id) for item in fixtures]
        stored_before = {int(item.id): item.metadata_json for item in fixtures}

    def recent_items(days=2):
        with store.session_scope() as session:
            found = [session.get(store.ItemRow, item_id) for item_id in ids]
            for item in found:
                session.expunge(item)
            return found

    def score(items, _profile, _downweight, attribution=None):
        scored = [(item, 0.9 - 0.01 * index) for index, item in enumerate(items)]
        return scored, {int(item.id): {"topic_score": 0.9, "confidence_score": value, "final_score": value}
                        for item, value in scored}

    monkeypatch.setattr(pipeline_mod, "_digest_id", lambda: "2026-10-26")
    monkeypatch.setattr(pipeline_mod, "ingest_all", lambda **_kwargs: 0)
    monkeypatch.setattr(pipeline_mod, "load_profile", lambda: SimpleNamespace(
        bio="", keywords=[], downweight=[], authors_of_interest=[]))
    monkeypatch.setattr(pipeline_mod, "build_profile_matrix", lambda _profile: np.zeros((1, 3)))
    monkeypatch.setattr(pipeline_mod, "recent_items", recent_items)
    monkeypatch.setattr(pipeline_mod, "_score_items_for_pipeline", score)
    monkeypatch.setattr(pipeline_mod, "_build_neg_centroid", None)
    monkeypatch.setattr(pipeline_mod, "summarize_items",
                        lambda selected, profile=None: {int(item.id): "summary" for item in selected})
    monkeypatch.setattr(pipeline_mod, "send_digest", lambda *_args, **_kwargs: False)

    pipeline_mod.run_all(dry_run=True)

    preprint_id, journal_id, *unique_ids = ids
    with store.session_scope() as session:
        served = {int(r.item_id) for r in session.query(store.DigestItemRow).all()}
        features = {int(r.item_id): json.loads(r.features_json)
                    for r in session.query(store.DigestItemFeatureRow).all()}
        stored_after = {int(r.id): r.metadata_json for r in session.query(store.ItemRow).all()}
    assert len(served & {preprint_id, journal_id}) == 1
    assert set(unique_ids) <= served  # the freed slot is not a second copy
    survivor = next(iter(served & {preprint_id, journal_id}))
    other = journal_id if survivor == preprint_id else preprint_id
    (link,) = features[survivor]["publication_versions"]
    assert link["item_id"] == other and link["relation"] == "same_title_authors"
    assert stored_after == stored_before  # links are digest-scoped, never written to items
