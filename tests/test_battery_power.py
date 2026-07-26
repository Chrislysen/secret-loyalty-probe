"""The section 4.18 check, packaged for other auditors, must be right on cases we can verify.

Two kinds of test: synthetic cases where the answer is known by construction, and a regression against
the committed section 4.18 artifact so the shipped tool cannot silently drift from the paper.
"""
import json
import math
import pathlib

import numpy as np
import pytest

from loyalty_probe.probes.battery_power import (battery_curve, informative_controls,
                                                outside_range, report)

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def test_outside_range_counts_only_fully_separated_features():
    pos = np.array([[10.0, 0.5], [11.0, 0.5]])
    neg = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.9]])
    # feature 0: both positives above every negative -> separates. feature 1: inside the range.
    assert outside_range(pos, neg) == 1


def test_k_is_zero_when_no_control_can_break_the_headline():
    pos = np.array([[100.0], [101.0]])
    neg = np.array([[0.0], [1.0], [2.0], [3.0]])
    k, idx = informative_controls(pos, neg)
    assert k == 0 and idx == []


def test_k_finds_the_single_control_that_carries_the_verdict():
    """One negative sits beyond the positives and single-handedly kills the separation."""
    pos = np.array([[10.0], [11.0]])
    neg = np.array([[0.0], [1.0], [2.0], [99.0]])
    assert outside_range(pos, neg) == 0                      # the 99 swallows the positives
    k, idx = informative_controls(pos, neg)
    assert k == 1 and idx == [3]


def test_curve_is_monotone_and_matches_the_closed_form_when_k_is_known():
    """With exactly k spoilers, P(headline | m) must equal C(N-k,m)/C(N,m)."""
    rng = np.random.default_rng(0)
    pos = np.array([[10.0], [10.5]])
    neg = np.vstack([rng.uniform(0, 2, (18, 1)), np.array([[99.0], [98.0]])])   # k = 2 of N = 20
    k, _ = informative_controls(pos, neg)
    assert k == 2
    curve = battery_curve(pos, neg, seed=1)
    N = len(neg)
    for m in (2, 5, 10, 15):
        expected = math.comb(N - k, m) / math.comb(N, m)
        assert curve[m]["p_headline"] == pytest.approx(expected, abs=0.03)
    ps = [curve[m]["p_headline"] for m in sorted(curve)]
    assert all(a >= b - 1e-9 for a, b in zip(ps, ps[1:])), "curve must be non-increasing in m"


def test_permutation_floor_is_reported_as_arithmetic():
    """The floor 1/C(n_pos+m, n_pos) is what people quote; the tool must surface it."""
    pos = np.array([[10.0], [11.0]])
    neg = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    r = report(pos, neg)
    assert r["permutation_floor"][5] == pytest.approx(1 / math.comb(7, 2))
    assert "arithmetic, not evidence" in r["summary"]


@pytest.mark.skipif(not (RESULTS / "spectral_wide.json").exists(), reason="artifact not present")
def test_reproduces_the_committed_section_4_18_numbers():
    """The shipped tool must give the paper's numbers, or the paper and the tool have diverged."""
    d = json.loads((RESULTS / "spectral_wide.json").read_text(encoding="utf-8"))
    committed = json.loads((RESULTS / "battery_curve.json").read_text(encoding="utf-8"))
    committed = {int(k): v for k, v in committed.items()}
    org = d["organisms"]
    neg = ([r for r in d["census_full_recipe"] if r in d["sigs"]]
           + [r for r in d["original_five"] if r in d["sigs"]])
    P = np.array([d["sigs"][r] for r in org])
    N = np.array([d["sigs"][r] for r in neg])
    r = report(P, N, seed=20260748)
    assert r["informative_k"] == 2
    assert r["min_battery_for_alpha"] == 16
    assert r["closed_form_max_err"] < 0.015
    for m in (2, 5, 16, 21):
        assert r["curve"][m]["p_headline"] == pytest.approx(committed[m]["p_all20"], abs=0.02)


def test_zero_error_upper_bound_is_the_number_section_4231_quotes():
    """"Zero false positives" means very different things at n=5 and n=50, and the tool says which."""
    from loyalty_probe.probes.battery_power import (
        controls_for_bound,
        range_floor,
        zero_error_upper_bound,
    )
    assert zero_error_upper_bound(5) == pytest.approx(0.451, abs=5e-4)
    assert zero_error_upper_bound(21) == pytest.approx(0.133, abs=5e-4)
    assert zero_error_upper_bound(50) == pytest.approx(0.058, abs=5e-4)
    # monotone, and never claims certainty from a finite sweep
    assert all(zero_error_upper_bound(n) > zero_error_upper_bound(n + 1) for n in (5, 21, 50, 400))
    assert zero_error_upper_bound(10 ** 6) > 0
    # and the two axes are different numbers doing different jobs
    assert range_floor(50) != zero_error_upper_bound(50)
    assert controls_for_bound(0.01) == 299
