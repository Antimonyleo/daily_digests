"""Negative-interest weights scale excess similarity, never the comparison axis."""
import numpy as np
import pytest

from dailydigest.rank import profile


def penalties(items, axes, topics, weights, **kwargs):
    return profile.negative_interest_penalties(items, axes, topics, weights, **kwargs)


def test_partial_weight_does_not_disable_a_stronger_negative_match():
    result = penalties([[5, 0]], [[2, 0]], [0.85], [0.8], strength=0.8)
    assert result == pytest.approx([0.096])


def test_weights_apply_after_each_axis_excess_then_maximum():
    # Negative cosines .8 and .6, topic .5: weighted excesses .03 and .2.
    result = penalties([[0.8, 0.6]], [[1, 0], [0, 1]], [0.5], [0.1, 2],
                       strength=0.8)
    assert result == pytest.approx([0.16])


def test_weights_and_strength_are_monotonic_with_zero_disabling():
    values = [penalties([[1, 0]], [[1, 0]], [.8], [weight])[0]
              for weight in [0, .2, .8, 1, 2]]
    assert values == pytest.approx([0, .032, .128, .16, .32])
    assert values == sorted(values)
    assert penalties([[1, 0]], [[1, 0]], [.8], [1], strength=0) == [0]


def test_unrelated_closer_core_and_margin_matches_receive_no_penalty():
    result = penalties([[0, 1], [.6, .8], [.8, .6]], [[1, 0]],
                       [.1, .7, .75], [1], margin=.1)
    assert result == pytest.approx([0, 0, 0])


def test_vector_scale_invariance_and_zero_vectors_carry_no_evidence():
    first = penalties([[3, 4], [0, 0]], [[1, 0], [0, 0]], [.2, -.5], [1, 5])
    second = penalties([[30, 40], [0, 0]], [[20, 0], [0, 0]], [.2, -.5], [1, 5])
    assert first == pytest.approx([.32, 0])
    assert second == pytest.approx(first)


def test_empty_item_and_negative_axis_matrices_preserve_batch_shape():
    assert penalties(np.empty((0, 3)), [[1, 0, 0]], [], [1]).shape == (0,)
    assert penalties([[1, 0, 0], [0, 1, 0]], [], [.2, .3], []).tolist() == [0, 0]
    assert penalties(np.empty((0, 3)), np.empty((0, 3)), [], []).shape == (0,)


@pytest.mark.parametrize("items,axes,topics,weights,kwargs", [
    ([1, 0], [[1, 0]], [.2], [1], {}),
    ([[1, 0]], [1, 0], [.2], [1], {}),
    ([[1, 0]], [[1, 0, 0]], [.2], [1], {}),
    ([[1, 0]], [[1, 0]], [], [1], {}),
    ([[1, 0]], [[1, 0]], [.2], [1, 2], {}),
    ([[1, 0]], [[1, 0]], [.2], [-1], {}),
    ([[np.nan, 0]], [[1, 0]], [.2], [1], {}),
    ([[1, 0]], [[np.inf, 0]], [.2], [1], {}),
    ([[1, 0]], [[1, 0]], [np.nan], [1], {}),
    ([[1, 0]], [[1, 0]], [.2], [np.inf], {}),
    ([[1, 0]], [[1, 0]], [.2], [1], {"margin": np.nan}),
    ([[1, 0]], [[1, 0]], [.2], [1], {"margin": -1}),
    ([[1, 0]], [[1, 0]], [.2], [1], {"strength": np.inf}),
    ([[1, 0]], [[1, 0]], [.2], [1], {"strength": -1}),
])
def test_invalid_or_misaligned_inputs_fail_explicitly(items, axes, topics, weights, kwargs):
    with pytest.raises(ValueError):
        penalties(items, axes, topics, weights, **kwargs)
