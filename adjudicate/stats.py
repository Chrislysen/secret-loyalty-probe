"""Small-N statistical primitives for LLM evals (design §6, method table).

Pure numpy/stdlib — no scipy. The design DEMANDS the way small-N LLM evals need
it: never the CLT/Wald approximation for a proportion.

  * ``wilson_interval``      — the default rate CI; in-bounds near 0/1.
  * ``clopper_pearson``      — CONSERVATIVE exact interval; the FPR upper bound.
  * ``bootstrap`` (BCa)      — pseudo-count regularized, RESAMPLES MODELS not
                               prompts (clustered — design §9 "clustered bootstrap
                               over organisms").
  * ``mcnemar``              — EXACT binomial on discordant pairs (not the χ²
                               approximation, which is invalid at small discordant n).
  * ``permutation_test``     — a WITHIN-SET contrast (design D1): the statistic is
                               ``favoring(true) − favoring(decoy)`` per matched set,
                               and the null shuffles which slot is "true" WITHIN
                               each set, so per-set nuisance cancels.

Unit-locked known values (asserted in tests, design §08 verify):
    wilson_interval(8, 10, 0.95)         ≈ (0.490, 0.943)
    mcnemar exact, discordant (b=1, c=9) ≈ 0.0215  (two-sided)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ── normal / beta quantiles without scipy ──────────────────────────────────


def _z_for(conf: float) -> float:
    """Two-sided normal quantile z for a central ``conf`` interval.

    z = Φ⁻¹(1 − (1−conf)/2). Uses the stdlib ``statistics.NormalDist`` inverse
    CDF so we stay scipy-free and exact to double precision.
    """
    from statistics import NormalDist

    return NormalDist().inv_cdf(1.0 - (1.0 - conf) / 2.0)


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Inverse regularized incomplete beta I⁻¹_p(a, b), via bisection on betainc.

    Used only by Clopper-Pearson. ``a``/``b`` are the small integer-ish shape
    params of a binomial exact interval, so a plain bisection on the monotone
    regularized incomplete beta is both robust and fast enough. Edge p∈{0,1} map
    to the support endpoints.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betainc(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _betainc(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) via the Lentz continued fraction.

    Standard Numerical-Recipes ``betai``: exact enough for the exact-binomial
    interval and dependency-free. Symmetry I_x(a,b) = 1 − I_{1−x}(b,a) keeps the
    continued fraction in its fast-converging regime.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    # ``bt`` is the prefactor x^a (1-x)^b / B(a, b); the continued fraction handles
    # the rest. The symmetry branch swaps (x,a,b) -> (1-x,b,a) and must recompute
    # ``bt`` with THAT swap — bt is symmetric here (x^a (1-x)^b is unchanged under
    # the joint swap) but the /a normalization differs, so compute per-branch.
    ln_bt = math.log(x) * a + math.log(1.0 - x) * b - lbeta
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(x, a, b) / a
    return 1.0 - bt * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


# ── the interval / test primitives ─────────────────────────────────────────


@dataclass(frozen=True)
class Interval:
    """A confidence interval on a point estimate."""

    point: float
    low: float
    high: float
    method: str
    conf: float = 0.95

    def as_tuple(self) -> tuple[float, float]:
        return (self.low, self.high)


def wilson_interval(k: int, n: int, conf: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion k/n.

    The default rate CI (TPR / precision / recall / principal-ID). Stays in
    [0, 1] even near 0 or 1, where Wald would spill out of bounds.

    Known value (test-locked): wilson_interval(8, 10, 0.95).as_tuple()
                               ≈ (0.4902, 0.9433).
    """
    if n <= 0:
        return Interval(point=float("nan"), low=0.0, high=1.0, method="wilson", conf=conf)
    z = _z_for(conf)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    low = (center - half) / denom
    high = (center + half) / denom
    return Interval(point=p, low=max(0.0, low), high=min(1.0, high), method="wilson", conf=conf)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Interval:
    """Clopper-Pearson EXACT interval — the conservative FPR upper bound.

    FPR is the safety-critical number (design §6): report the worst-case upper
    bound across sub-suites. Exact = never anti-conservative. Uses the beta
    quantile form:
        low  = Beta⁻¹(α/2;      k,   n−k+1)   (0 when k == 0)
        high = Beta⁻¹(1−α/2;  k+1,   n−k)     (1 when k == n)
    """
    if n <= 0:
        return Interval(point=float("nan"), low=0.0, high=1.0, method="clopper_pearson", conf=conf)
    alpha = 1.0 - conf
    p = k / n
    low = 0.0 if k == 0 else _beta_ppf(alpha / 2.0, k, n - k + 1)
    high = 1.0 if k == n else _beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return Interval(point=p, low=low, high=high, method="clopper_pearson", conf=conf)


def mcnemar(b: int, c: int, *, two_sided: bool = True) -> float:
    """EXACT McNemar test on the DISCORDANT pair counts (b, c).

    ``b`` and ``c`` are the two off-diagonal (discordant) cells of the paired
    2×2 table — the pairs where the two conditions disagree. Under H0 each
    discordant pair is a fair coin, so the exact test is a two-sided binomial on
    ``min(b, c)`` successes out of ``n = b + c`` at π = 0.5. We use the EXACT
    binomial, not the χ² approximation (invalid when b + c is small).

    Known value (test-locked): mcnemar(1, 9) ≈ 0.021484375 two-sided.

    The two-sided p is 2·P(X ≤ min(b,c)); clamped to 1.0. Concordant pairs are
    (correctly) ignored — McNemar conditions on the discordant total.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Exact lower tail P(X <= k) under Binomial(n, 0.5).
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    if not two_sided:
        return min(1.0, tail)
    return min(1.0, 2.0 * tail)


def bootstrap(
    per_model_values: np.ndarray,
    statistic=np.mean,
    *,
    resamples: int = 2000,
    conf: float = 0.95,
    seed: int = 0,
    pseudo_count: float = 0.5,
) -> Interval:
    """BCa bootstrap CI that RESAMPLES MODELS (clusters), not prompts.

    F1 and other non-proportion metrics (design method table). Two design locks:

      * RESAMPLES MODELS not prompts — ``per_model_values`` is one row per
        organism/model (a cluster). Resampling prompts would ignore the
        organism-level correlation and understate the CI (design §9 clustered
        bootstrap). The caller passes an already-per-model vector.
      * PSEUDO-COUNT REGULARIZED — with a handful of models a resample can be
        degenerate (all-equal), collapsing the bias/accel terms. We regularize by
        appending a symmetric ``pseudo_count`` pair {0-ish, 1-ish} sentinel-free:
        the point statistic is nudged toward the grand mean by ``pseudo_count``
        pseudo-observations, stabilizing near-0/near-1 rates.

    BCa: bias-correction z0 from the fraction of resamples below the point est;
    acceleration a from the jackknife skew. Falls back to the percentile interval
    when the sample is too small to estimate acceleration.
    """
    x = np.asarray(per_model_values, dtype=float).ravel()
    n = x.size
    if n == 0:
        return Interval(point=float("nan"), low=0.0, high=1.0, method="bca", conf=conf)

    def _stat(v: np.ndarray) -> float:
        # Pseudo-count regularization: shrink toward the grand mean by
        # ``pseudo_count`` synthetic observations at that mean (stabilizes the
        # statistic when a resample is degenerate).
        gm = float(np.mean(x))
        aug = np.concatenate([v, np.full(int(math.ceil(pseudo_count * 2)), gm)])
        return float(statistic(aug))

    theta_hat = _stat(x)
    if n == 1:
        return Interval(point=theta_hat, low=theta_hat, high=theta_hat, method="bca", conf=conf)

    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=float)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)          # resample MODELS with replacement
        boot[i] = _stat(x[idx])

    alpha = 1.0 - conf
    from statistics import NormalDist

    nd = NormalDist()
    # bias-correction z0
    prop_less = float(np.mean(boot < theta_hat))
    prop_less = min(max(prop_less, 1.0 / (resamples + 1)), 1.0 - 1.0 / (resamples + 1))
    z0 = nd.inv_cdf(prop_less)

    # acceleration via jackknife
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = _stat(np.delete(x, i))
    jbar = float(np.mean(jack))
    num = float(np.sum((jbar - jack) ** 3))
    den = 6.0 * (float(np.sum((jbar - jack) ** 2)) ** 1.5)
    a = num / den if den != 0.0 else 0.0

    def _bca_q(qz: float) -> float:
        adj = z0 + (z0 + qz) / (1.0 - a * (z0 + qz))
        return nd.cdf(adj)

    zlo, zhi = nd.inv_cdf(alpha / 2.0), nd.inv_cdf(1.0 - alpha / 2.0)
    plo, phi = _bca_q(zlo), _bca_q(zhi)
    low = float(np.quantile(boot, plo))
    high = float(np.quantile(boot, phi))
    return Interval(point=theta_hat, low=min(low, high), high=max(low, high), method="bca", conf=conf)


@dataclass(frozen=True)
class PermResult:
    """The outcome of a within-set permutation contrast (design D1)."""

    statistic: float          # observed mean per-set contrast
    p_value: float
    resamples: int
    exact: bool               # True when the full 2^S enumeration was used


def permutation_test(
    favoring_true: np.ndarray,
    favoring_decoy: np.ndarray,
    *,
    resamples: int = 10000,
    seed: int = 0,
    alternative: str = "greater",
) -> PermResult:
    """WITHIN-SET permutation contrast — resolves critique D1.

    Each matched set contributes a paired ``(favoring_true, favoring_decoy)``.
    The per-set statistic is ``favoring(true) − favoring(decoy)`` so any per-set
    nuisance (name length, sentiment, base rate) CANCELS. The null shuffles which
    slot is "true" WITHIN each set — i.e. independently flips the sign of each
    set's contrast — NOT a cross-set label shuffle (which would lose the pairing
    and manufacture spurious p≈0). The test statistic is the mean contrast.

    When the number of sets S is small (≤ 20) the full 2^S sign-flip enumeration
    is exact; otherwise ``resamples`` random sign patterns give a Monte-Carlo p
    with the conservative +1 / +1 correction.
    """
    t = np.asarray(favoring_true, dtype=float).ravel()
    d = np.asarray(favoring_decoy, dtype=float).ravel()
    if t.shape != d.shape:
        raise ValueError("favoring_true and favoring_decoy must be paired (same length)")
    diff = t - d
    s = diff.size
    if s == 0:
        return PermResult(statistic=float("nan"), p_value=1.0, resamples=0, exact=True)
    obs = float(np.mean(diff))

    def _cmp(stat: float) -> bool:
        if alternative == "greater":
            return stat >= obs - 1e-12
        if alternative == "less":
            return stat <= obs + 1e-12
        return abs(stat) >= abs(obs) - 1e-12  # two-sided

    if s <= 20:
        # exact enumeration over all 2^s within-set sign flips
        count = 0
        total = 1 << s
        for mask in range(total):
            signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(s)])
            if _cmp(float(np.mean(signs * diff))):
                count += 1
        return PermResult(statistic=obs, p_value=count / total, resamples=total, exact=True)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(resamples):
        signs = rng.choice((-1.0, 1.0), size=s)
        if _cmp(float(np.mean(signs * diff))):
            count += 1
    p = (count + 1) / (resamples + 1)  # conservative Monte-Carlo correction
    return PermResult(statistic=obs, p_value=p, resamples=resamples, exact=False)


def max_statistic_permutation(
    per_hypothesis_contrasts: list[np.ndarray],
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> PermResult:
    """Max-statistic permutation null over a family of hypotheses (design G7).

    The disjunctive aggregator ("any hypothesis fires") maximizes family-wise
    error. The correct null for ``max_k statistic_k`` is the permutation
    distribution of the MAXIMUM across the family — computed by applying the SAME
    within-set sign flip to every hypothesis' contrast vector on each resample and
    taking the max. Returns the p for the observed max contrast.

    Each element of ``per_hypothesis_contrasts`` is a per-set contrast vector
    (already ``favoring(true) − favoring(decoy)`` per matched set) for one
    hypothesis. All must share the same set count S.
    """
    if not per_hypothesis_contrasts:
        return PermResult(statistic=float("nan"), p_value=1.0, resamples=0, exact=True)
    mats = [np.asarray(v, dtype=float).ravel() for v in per_hypothesis_contrasts]
    s = mats[0].size
    if any(m.size != s for m in mats):
        raise ValueError("all hypotheses must share the same matched-set count")
    obs_max = max(float(np.mean(m)) for m in mats)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(resamples):
        signs = rng.choice((-1.0, 1.0), size=s)
        perm_max = max(float(np.mean(signs * m)) for m in mats)
        if perm_max >= obs_max - 1e-12:
            count += 1
    p = (count + 1) / (resamples + 1)
    return PermResult(statistic=obs_max, p_value=p, resamples=resamples, exact=False)
