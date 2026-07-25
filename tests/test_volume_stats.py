"""The §4.16 volume regression decides a pre-registered band, so its interval must be calibrated.

The band turns on whether the organisms fall INSIDE a 95 % prediction interval fitted on benign
adapters. A too-narrow interval would manufacture "the organisms are anomalous" verdicts; a too-wide
one would manufacture "volume explains it". Neither is detectable by eye, so it is tested here.
"""
import numpy as np
import pytest

from loyalty_probe.probes.run_volume import _fit_predict


def test_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    x = rng.uniform(14, 19, 23)
    y = 1.0 + 0.5 * x + rng.normal(0, 0.2, 23)
    f = _fit_predict(x, y, [16.0])
    assert f["slope"] == pytest.approx(0.5, abs=0.05)
    assert f["p"] < 1e-6


def test_prediction_interval_is_calibrated_at_the_corpus_size():
    """Coverage of a genuinely NEW draw must be ~95 % at n = 23, the rank-16 corpus size.

    This is a PREDICTION interval, not a confidence interval: it carries the residual variance as
    well as the parameter uncertainty. A confidence interval here would be roughly sqrt(n) times too
    narrow and would put the organisms "outside" almost regardless of the data.
    """
    rng = np.random.default_rng(7)
    hits = 0
    trials = 2000
    for _ in range(trials):
        x = rng.uniform(14, 19, 23)
        y = 1.0 + 0.5 * x + rng.normal(0, 0.2, 23)
        _, lo, hi = _fit_predict(x, y, [16.0])["pred"][0]
        y_new = 1.0 + 0.5 * 16 + rng.normal(0, 0.2)
        hits += int(lo <= y_new <= hi)
    assert 0.92 <= hits / trials <= 0.97


def test_interval_widens_away_from_the_data():
    rng = np.random.default_rng(3)
    x = rng.uniform(14, 19, 23)
    y = 1.0 + 0.5 * x + rng.normal(0, 0.2, 23)
    f = _fit_predict(x, y, [16.5, 25.0])
    near = f["pred"][0][2] - f["pred"][0][1]
    far = f["pred"][1][2] - f["pred"][1][1]
    assert far > near


def test_refuses_to_fit_below_four_points():
    """With n < 4 the residual dof is <= 1 and the interval is meaningless -- it must return None
    rather than a number the band logic would then act on."""
    assert _fit_predict(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), [2.0]) is None
