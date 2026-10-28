from dailydigest.publications import publication_label


def _row(url, **metadata):
    return {"url": url, "metadata": metadata}


def test_biorxiv_doi_links_are_labelled_preprints():
    row = _row("https://doi.org/10.1101/2026.10.12.123456", preprint_type="new", preprint_version="1")
    assert publication_label(row) == "Preprint"


def test_withdrawn_preprints_keep_withdrawn_status():
    row = _row("https://doi.org/10.1101/2026.10.12.123456", preprint_type="withdrawn", preprint_version="2")
    assert publication_label(row) == "Withdrawn"


def test_declared_genres_still_win_and_plain_records_stay_unknown():
    assert publication_label(_row("https://doi.org/10.1/x", preprint_type="new", publication_types=["Review"])) == "Review"
    assert publication_label(_row("https://doi.org/10.1/x", preprint_type="")) == "Publication"
    assert publication_label(_row("https://doi.org/10.1/x")) == "Publication"
