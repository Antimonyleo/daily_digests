"""Retain late abstract evidence before bounded semantic window extraction."""
import pytest

from dailydigest.models import Item


def article(**fields):
    return Item(source="Fixture", section="research", external_id="late-claim",
                url="https://example.test/late-claim", title="Validated research", **fields)


def test_item_retains_contribution_after_long_background():
    background = "Earlier work motivates the remaining research question. " * 100
    contribution = "We demonstrate a validated quantum decoder with reduced logical error."
    abstract = background + contribution
    assert 4000 < len(background) < len(abstract) < 16000
    item = article(abstract=abstract)
    assert item.abstract == abstract
    assert item.abstract.endswith(contribution)


def test_item_abstract_still_has_a_bounded_input_size():
    abstract = "A" * 16000 + "Discarded excess input."
    assert article(abstract=abstract).abstract == abstract[:16000]


@pytest.mark.parametrize("value,expected", [
    (None, ""), ("", ""), (0, ""), (False, ""), (123, "123"),
])
def test_item_abstract_preserves_existing_empty_and_type_handling(value, expected):
    assert article(abstract=value).abstract == expected


def test_item_abstract_defaults_to_empty():
    assert article().abstract == ""
