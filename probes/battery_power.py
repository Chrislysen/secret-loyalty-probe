"""Would YOUR control battery have manufactured YOUR result? A reusable check.

    from loyalty_probe.probes.battery_power import battery_curve, report

This is section 4.18 packaged so other auditors can run it on their own detector. It exists because
that section's finding is not really about spectral statistics or secret loyalties: any detector
scored by "do the positives fall outside the range of the negatives" inherits a false-positive rate
set by how many negatives you happened to collect, and a permutation test over labels cannot see it.

WHAT IT DOES. Given the measured feature vectors of your positives and your negatives, it resamples
subsets of the negatives at every battery size m and reports how often a battery of that size would
have shown your headline result. It also fits the closed form

    P(spurious headline | m) = C(N - k, m) / C(N, m)

where N is your negative pool and k is the number of negatives that, if present, break the result.
`k` is measured, not assumed: it is the size of the smallest subset whose removal restores the
headline. In section 4.16 the answer was k = 2 out of N = 21, and the closed form matched the
resampling to within 0.015.

WHAT IT IS NOT. It cannot tell you your battery is big enough in absolute terms -- your pool is itself
a sample, and controls you never collected cannot be resampled. It answers the narrower question
"given the negatives I DID collect, how lucky would I have to have been?", which is the question a
permutation test over labels silently skips. See section 4.18.2 for the prior art: this is Clark's
(1973) fixed-effect fallacy applied to control models, and Zhong & Raghunathan (arXiv:2508.00161,
Remark 3.1) give a closed-form FPR for the same min/max rule along the calibration-prompt axis.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def outside_range(pos: np.ndarray, neg: np.ndarray) -> int:
    """Default statistic: how many features put EVERY positive outside the negatives' range.

    This is the min/max rule used by the weight-space detectors in section 4.16, and the one whose
    n-dependence Zhong & Raghunathan bound. Pass your own callable if you score differently.
    """
    lo, hi = neg.min(0), neg.max(0)
    return int(sum((pos[:, j] < lo[j]).all() or (pos[:, j] > hi[j]).all()
                   for j in range(pos.shape[1])))


def informative_controls(pos, neg, statistic=outside_range, headline=None, max_k=6):
    """Smallest set of negatives whose REMOVAL restores the headline result, and its size k.

    Returns (k, indices) or (None, []) if no subset up to `max_k` restores it -- which is the good
    case, meaning no small group of controls is carrying the verdict.
    """
    full = pos.shape[1] if headline is None else headline
    if statistic(pos, neg) >= full:
        return 0, []                       # the headline already holds against every negative
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(range(len(neg)), k):
            if statistic(pos, np.delete(neg, list(combo), axis=0)) >= full:
                return k, list(combo)
    return None, []


def battery_curve(pos, neg, statistic=outside_range, headline=None, n_boot=3000, seed=0):
    """P(headline result) as a function of battery size m, by resampling YOUR negatives.

    Exhaustive where C(N,m) is small enough, sampled otherwise. Returns {m: {...}}.
    """
    rng = np.random.default_rng(seed)
    N = len(neg)
    full = pos.shape[1] if headline is None else headline
    out = {}
    for m in range(2, N + 1):
        total = math.comb(N, m)
        if total <= n_boot:
            subsets = list(itertools.combinations(range(N), m))
        else:
            subsets = [tuple(rng.choice(N, m, replace=False)) for _ in range(n_boot)]
        vals = np.array([statistic(pos, neg[list(s)]) for s in subsets])
        out[m] = {"mean_statistic": float(vals.mean()),
                  "p_headline": float((vals >= full).mean()),
                  "n_subsets": len(subsets), "exhaustive": total <= n_boot}
    return out


def report(pos, neg, statistic=outside_range, headline=None, alpha=0.05, seed=0):
    """The whole check, as one call. Returns a dict; print `summary` for humans."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    N = len(neg)
    full = pos.shape[1] if headline is None else headline
    observed = statistic(pos, neg)
    curve = battery_curve(pos, neg, statistic, headline, seed=seed)
    k, idx = informative_controls(pos, neg, statistic, headline)

    closed = {}
    if k:
        closed = {m: (math.comb(N - k, m) / math.comb(N, m) if m <= N - k else 0.0)
                  for m in curve}
    fit_err = (max(abs(curve[m]["p_headline"] - closed[m]) for m in closed) if closed else None)
    m_needed = next((m for m in sorted(curve) if curve[m]["p_headline"] <= alpha), None)

    # The permutation floor, which is what people usually quote -- and which cannot fall below
    # 1/C(n_pos+m, n_pos). If your reported p equals that floor, it is arithmetic, not evidence.
    floor = {m: 1.0 / math.comb(len(pos) + m, len(pos)) for m in curve}

    lines = [f"negatives collected N={N}, positives n={len(pos)}, observed statistic {observed}/{full}"]
    if k == 0:
        lines.append("headline holds against every negative you have; no small subset carries it")
    elif k is None:
        lines.append("no subset of <=6 negatives restores the headline -- not carried by a few controls")
    else:
        lines.append(f"k={k} negative(s) carry the verdict: removing them restores the headline")
        lines.append(f"closed form C(N-k,m)/C(N,m) matches the resampling to {fit_err:.3f}")
    lines.append(f"P(headline | m) at your battery size {N}: {curve[N]['p_headline']:.3f}")
    if m_needed:
        lines.append(f"m >= {m_needed} needed for P(headline | m) <= {alpha}")
    else:
        lines.append(f"no m<=N reaches P <= {alpha}: this pool cannot support that claim")
    lines.append(f"permutation floor at m=5 would be {floor.get(5, float('nan')):.4f} "
                 f"-- quoting it as a p-value is arithmetic, not evidence")
    return {"observed": observed, "n_features": full, "N": N, "curve": curve,
            "informative_k": k, "informative_idx": idx, "closed_form": closed,
            "closed_form_max_err": fit_err, "min_battery_for_alpha": m_needed,
            "permutation_floor": floor, "summary": "\n".join(lines)}
