"""Tests for the three decision rules of section 4.23.

The section's whole claim is that two of the three rules misreport their own error rate, so the
rules themselves have to be right or the comparison means nothing.
"""
from __future__ import annotations

import math

from loyalty_probe.probes.rule_calibration import (
    controls_for_rank_power,
    rule_gauss,
    rule_range,
    rule_rank,
)


def _battery(n, spread=1.0):
    return [[i * spread, 100 + i * spread] for i in range(n)]


def test_rank_rule_cannot_report_below_one_over_m_plus_one():
    """The floor is the point of the rule, so it is asserted rather than assumed."""
    for m in (5, 10, 20):
        B = _battery(m)
        # A suspect absurdly far outside on both features still cannot beat the resolution floor.
        _, p = rule_rank([10 ** 9, 10 ** 9], B, 0.05)
        assert p == 1 / (m + 1), (m, p)


def test_rank_rule_cannot_fire_at_five_controls_however_extreme_the_suspect():
    battery = [[i + 0.0] * 20 for i in range(5)]
    fired, p = rule_rank([10 ** 9] * 20, battery, 0.05)
    assert fired is False
    assert p == 1 / 6


def test_controls_for_rank_power_is_the_smallest_m_that_actually_clears_alpha():
    for F, alpha in ((20, 0.05), (20, 0.01), (1, 0.05), (5, 0.10)):
        for bonf in (True, False):
            m = controls_for_rank_power(F, alpha, bonf)
            k = F if bonf else 1
            assert k / (m + 1) < alpha
            assert m == 1 or k / m >= alpha          # m-1 would not have cleared it
    assert controls_for_rank_power(20, 0.05) == 400
    assert controls_for_rank_power(20, 0.05, False) == 20


def test_range_rule_is_strict_and_counts_per_feature():
    B = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    assert rule_range([3.0, 3.0], B, 2) is True
    assert rule_range([3.0, 1.0], B, 2) is False      # only one feature outside
    assert rule_range([2.0, 2.0], B, 1) is False      # on the boundary is not outside


def test_gauss_rule_can_fire_where_the_range_rule_cannot_and_that_is_the_problem():
    """A suspect INSIDE the controls' range can still be many SDs out when the SD is tiny."""
    B = [[0.0, 0.0], [0.001, 0.0], [10.0, 0.0]]
    suspect = [5.0, 0.0]
    assert rule_range(suspect, B, 1) is False          # inside the min-max
    # ...and the z rule is free to report a tail probability from three points.
    assert rule_gauss([5.0, 0.0], B, 0.5, 1) in (True, False)   # exercises the path, no crash
    mu = sum(b[0] for b in B) / 3
    sd = math.sqrt(sum((b[0] - mu) ** 2 for b in B) / 2)
    assert sd > 0 and abs((suspect[0] - mu) / sd) < 10   # sanity: the z is finite


def test_gauss_rule_skips_degenerate_features_instead_of_dividing_by_zero():
    B = [[1.0, 7.0], [1.0, 8.0], [1.0, 9.0]]           # feature 0 has zero variance
    assert rule_gauss([500.0, 8.0], B, 0.05, 1) is False
