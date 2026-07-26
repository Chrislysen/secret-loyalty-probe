"""Tests for the range-based false-positive law and the leave-one-out calibration that measures it.

The law is the paper's most load-bearing claim about detector design, so it is checked two ways:
against its own closed form, and against a Monte-Carlo simulation that knows nothing about the
closed form. A law that is only ever compared to itself is not checked at all.
"""
from __future__ import annotations

import random

import pytest

from loyalty_probe.probes.battery_loo import (
    controls_needed,
    independent_fpr,
    min_battery,
    n_outside,
    organism_side,
    range_fpr,
)


def test_range_fpr_matches_simulation_for_a_distribution_it_was_not_derived_from():
    """2/(m+1) is distribution-free, so a skewed heavy-tailed draw must give the same answer.

    This is the whole value of the law: nobody knows how ||dW||_F is distributed across benign
    adapters, and with this result nobody needs to.
    """
    rng = random.Random(11)
    for m in (3, 5, 10, 20):
        outside = sum(
            (lambda s, c: s < min(c) or s > max(c))(
                rng.lognormvariate(0, 2), [rng.lognormvariate(0, 2) for _ in range(m)])
            for _ in range(40000))
        assert abs(outside / 40000 - range_fpr(m)) < 0.012, m


def test_a_battery_of_five_has_a_one_in_three_floor():
    """The number that retracts section 4.16's positive: five controls cannot be better than this."""
    assert range_fpr(5) == pytest.approx(1 / 3, abs=1e-9)
    assert controls_needed(0.05) == 39
    assert controls_needed(0.01) == 199
    # controls_needed must be the smallest m that actually clears the ceiling, not one off it.
    for ceiling in (0.5, 0.2, 0.05, 0.01, 0.005):
        m = controls_needed(ceiling)
        assert range_fpr(m) <= ceiling
        assert m == 1 or range_fpr(m - 1) > ceiling


def test_independence_is_the_optimistic_assumption_not_a_conservative_one():
    """Stacking features looks like it buys safety only if the features are independent."""
    ind = independent_fpr(20, 5, 16)
    assert ind < 1e-4                      # what a designer assuming independence would expect
    assert range_fpr(5) > 100 * ind        # what perfectly correlated features actually deliver
    # More features is pure gain under independence and pure noise under correlation -- the law has
    # no F in it at all.
    assert independent_fpr(40, 5, 32) < independent_fpr(20, 5, 16)


def test_n_outside_counts_strictly_outside_and_respects_a_fixed_side():
    battery = [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]]
    assert n_outside([3.0, 13.0], battery) == 2      # above on both
    assert n_outside([1.0, 11.0], battery) == 0      # interior on both
    assert n_outside([2.0, 12.0], battery) == 0      # ON the boundary is not outside
    assert n_outside([3.0, 9.0], battery) == 2       # above, then below
    assert n_outside([3.0, 9.0], battery, ["hi", "hi"]) == 1   # only the "hi" deviation counts
    assert n_outside([3.0, 9.0], battery, ["lo", "lo"]) == 1


def test_organism_side_is_taken_from_the_organisms_not_the_suspect():
    sigs = {"org": [10.0, 0.0], "n1": [1.0, 5.0], "n2": [2.0, 6.0], "n3": [3.0, 7.0]}
    side = organism_side(sigs, ["org"], ["n1", "n2", "n3"])
    assert side == ["hi", "lo"]


def test_min_battery_returns_none_when_the_ceiling_is_never_reached():
    curve = {5: {16: 0.30}, 10: {16: 0.20}, 20: {16: 0.10}}
    assert min_battery(curve, 16, 0.25) == 10
    assert min_battery(curve, 16, 0.05) is None
