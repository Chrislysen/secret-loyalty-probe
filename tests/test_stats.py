"""Numeric spot-checks for the stats primitives (design §08 verify).

The two known values the design pins:
    wilson_interval(8, 10, 0.95)  ≈ (0.490, 0.943)
    mcnemar exact (b=1, c=9) two-sided ≈ 0.0215
plus sanity properties for Clopper-Pearson, BCa bootstrap, and the within-set
permutation contrast.
"""
from __future__ import annotations

import math

import numpy as np

from loyalty_probe.adjudicate.stats import (
    bootstrap,
    clopper_pearson,
    max_statistic_permutation,
    mcnemar,
    permutation_test,
    wilson_interval,
)


def test_wilson_8_of_10_known_value():
    iv = wilson_interval(8, 10, 0.95)
    lo, hi = iv.as_tuple()
    assert abs(lo - 0.490) < 1e-3, f"wilson low {lo} != 0.490"
    assert abs(hi - 0.943) < 1e-3, f"wilson high {hi} != 0.943"
    # tighter lock to the exact computed constants
    assert abs(lo - 0.49016247153664183) < 1e-9
    assert abs(hi - 0.9433178485456247) < 1e-9


def test_wilson_in_bounds_at_extremes():
    for k, n in [(0, 5), (5, 5), (1, 100)]:
        iv = wilson_interval(k, n)
        assert 0.0 <= iv.low <= iv.high <= 1.0


def test_mcnemar_1_9_known_value():
    p = mcnemar(1, 9)
    assert abs(p - 0.021484375) < 1e-9, f"mcnemar(1,9)={p}"
    assert abs(p - 0.0215) < 1e-3


def test_mcnemar_symmetry_and_edges():
    assert mcnemar(1, 9) == mcnemar(9, 1)      # symmetric in (b, c)
    assert mcnemar(0, 0) == 1.0                # no discordant pairs -> p=1
    assert mcnemar(5, 5) == 1.0                # perfectly balanced -> two-sided p=1
    one = mcnemar(1, 9, two_sided=False)
    assert abs(one - 0.0107421875) < 1e-9      # one-sided is half the two-sided tail


def test_clopper_pearson_conservative_upper_bound():
    # 0 events in 20 -> exact upper bound is the "rule of three"-ish CP bound.
    iv = clopper_pearson(0, 20)
    assert iv.low == 0.0
    assert abs(iv.high - 0.1684) < 1e-3, f"cp 0/20 upper={iv.high}"
    # CP is conservative: its upper bound >= Wilson's for the same all-clean count
    w = wilson_interval(0, 20)
    assert iv.high >= w.high


def test_clopper_pearson_full():
    iv = clopper_pearson(20, 20)
    assert iv.high == 1.0
    assert iv.low > 0.0


def test_bootstrap_resamples_models_point_and_bounds():
    # 8/10 "models" caught -> point 0.8, CI within [0,1] and straddling 0.8
    vals = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0.0])
    iv = bootstrap(vals, np.mean, resamples=2000, seed=0)
    assert abs(iv.point - 0.8) < 1e-9
    assert 0.0 <= iv.low <= iv.point <= iv.high <= 1.0
    # deterministic under a fixed seed
    iv2 = bootstrap(vals, np.mean, resamples=2000, seed=0)
    assert (iv.low, iv.high) == (iv2.low, iv2.high)


def test_permutation_within_set_contrast_cancels_nuisance():
    # A pure per-set OFFSET (both slots shifted by the same constant) must cancel:
    # the contrast is identical, so the p is unchanged.
    t = np.array([1.0, 1.0, 1.0, 0.0])
    d = np.array([0.0, 0.0, 1.0, 0.0])
    r1 = permutation_test(t, d)
    r2 = permutation_test(t + 5.0, d + 5.0)  # same nuisance added to BOTH slots
    assert abs(r1.p_value - r2.p_value) < 1e-12
    assert r1.exact  # S=4 <= 20 -> exact enumeration


def test_permutation_null_effect_is_nonsignificant():
    # true == decoy everywhere -> zero contrast -> p should be large (~1).
    t = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    r = permutation_test(t, t.copy())
    assert r.p_value > 0.5


def test_max_statistic_permutation_corrects_family():
    # A single strong hypothesis; the max-stat p over a family of identical
    # contrasts equals the single-hypothesis permutation p (no spurious inflation).
    contrast = np.array([1.0, 1.0, 1.0, 1.0])
    res = max_statistic_permutation([contrast, contrast, contrast], resamples=5000, seed=0)
    assert 0.0 < res.p_value <= 1.0
    # all-positive contrast -> observed max is the largest possible -> small p
    assert res.p_value < 0.2


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} stats tests passed")
    sys.exit(1 if failed else 0)
