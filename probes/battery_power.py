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
import sys

import numpy as np


def outside_range(pos: np.ndarray, neg: np.ndarray) -> int:
    """Default statistic: how many features put EVERY positive outside the negatives' range.

    This is the min/max rule used by the weight-space detectors in section 4.16, and the one whose
    n-dependence Zhong & Raghunathan bound. Pass your own callable if you score differently.
    """
    lo, hi = neg.min(0), neg.max(0)
    return int(sum((pos[:, j] < lo[j]).all() or (pos[:, j] > hi[j]).all()
                   for j in range(pos.shape[1])))


SEARCH_BUDGET = 2_000_000      # subset evaluations; ~3 min at the measured ~10k/s


def searched_depth(n, max_k=6, budget=SEARCH_BUDGET):
    """How deep `informative_controls` will actually look on a battery of `n`, given the budget.

    Exists so `report()` can PRINT the depth instead of letting a truncated search read like a clean
    one. Deterministic and free -- pure combinatorics, no evaluation.
    """
    used, depth = 0, 0
    for k in range(1, max_k + 1):
        if used + math.comb(n, k) > budget:
            break
        used += math.comb(n, k)
        depth = k
    return depth


def informative_controls(pos, neg, statistic=outside_range, headline=None, max_k=6,
                         budget=SEARCH_BUDGET):
    """Smallest set of negatives whose REMOVAL restores the headline result, and its size k.

    Returns (k, indices) or (None, []) if no subset within the searched depth restores it -- which is
    the good case, meaning no small group of controls is carrying the verdict.

    COST, because this is the slow path exactly when it matters (a headline that died, and you want to
    know which controls killed it). It is exhaustive: sum(C(N, 1..max_k)) evaluations. At the battery
    sizes this project tells people to collect, it stops being a pre-flight check:

        N =  21   ->        82,159 subsets   (seconds)
        N =  39   ->     3,930,550 subsets   (minutes)   <- the 5 % requirement
        N = 199   -> 82,473,331,150 subsets  (~100 days) <- the 1 % requirement

    So the depth is capped by an evaluation `budget` rather than run for a hundred days with no
    output. The cap is REPORTED by `report()` and never silent: a truncated search can only
    under-report k, never invent one, but "no small group carries the verdict" and "we did not look
    past k = 2" are different statements and must not print the same way.
    """
    full = pos.shape[1] if headline is None else headline
    if statistic(pos, neg) >= full:
        return 0, []                       # the headline already holds against every negative
    for k in range(1, searched_depth(len(neg), max_k, budget) + 1):
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
    """The whole check, as one call. Returns a dict; print `summary` for humans.

    `pos` and `neg` are (models, features). ONE suspect is `(1, n_features)`, not `(n_features,)` --
    an auditor with a single suspect is the DEFAULT case, and a 1-D array used to raise a bare
    `IndexError: tuple index out of range` from inside the statistic with no hint. Caught here now.

    `headline` is how many separating features count as YOUR result, and it is the knob that matters.
    It defaults to ALL of them, so a detector that fires at "T of n" -- which is most of them, and is
    the operating point this project's own tables use -- gets a report about an n-of-n claim it never
    made. Pass `headline=16` for a 16-of-20 rule. The two genuinely differ: on the organisms' own
    features the default gives k = 2 and m >= 17, while `headline=16` gives k = 1 and m >= 20.
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    for _name, _a in (("pos", pos), ("neg", neg)):
        if _a.ndim != 2:
            raise ValueError(
                f"{_name} must be 2-D (models, features); got shape {_a.shape}. A single suspect is "
                f"`x.reshape(1, -1)`, not a 1-D array."
            )
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(
            f"pos has {pos.shape[1]} features and neg has {neg.shape[1]}; these arrays are "
            f"(models, features) and are probably transposed."
        )
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
        _d = searched_depth(N)
        lines.append(f"no subset of <={_d} negatives restores the headline -- not carried by a few "
                     f"controls")
        if _d < 6:
            lines.append(f"NOTE: search truncated at k={_d} of 6 by the evaluation budget at N={N}; "
                         f"a larger carrying set would not have been found")
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

    # Two different numbers do two different jobs, and most evaluations report neither. `m` bounds
    # what the detector CAN resolve; `N` bounds what a measured error rate MEANS. A clean 0-of-N
    # sweep is compatible with a true rate this high, and at hub scale that is thousands of models.
    res_floor = range_floor(N)
    ev_bound = zero_error_upper_bound(N, alpha)
    lines.append(f"resolution floor at m={N}: 2/(m+1) = {res_floor:.4f} -- no min-max rule on this "
                 f"battery can resolve a rate below it")
    lines.append(f"a clean 0-of-{N} sweep bounds the true rate only at {ev_bound:.3f} "
                 f"(Clopper-Pearson {1 - alpha:.0%}); {controls_for_bound(0.01, alpha)} clean "
                 f"negatives would be needed to support 'below 1 %'")
    return {"observed": observed, "n_features": full, "N": N, "curve": curve,
            "informative_k": k, "informative_idx": idx, "closed_form": closed,
            "closed_form_max_err": fit_err, "min_battery_for_alpha": m_needed,
            "permutation_floor": floor, "resolution_floor": res_floor,
            "zero_error_upper_bound": ev_bound, "summary": "\n".join(lines)}


def range_floor(m: int) -> float:
    """2/(m+1) -- section 4.22. What a min-max rule on m controls can resolve, at best."""
    return 2.0 / (m + 1)


def zero_error_upper_bound(n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson upper bound on the true rate after observing ZERO errors in n trials.

    Closed form for the zero-count case: 1 - alpha**(1/n). Reporting "0 false positives" without it
    invites the reader to hear "0 %", when 0 of 50 is compatible with 5.8 % and 0 of 5 with 45 %.
    """
    return 1.0 - alpha ** (1.0 / n) if n > 0 else 1.0


def controls_for_bound(target: float, alpha: float = 0.05) -> int:
    """How many clean EVALUATION trials (`n`) a zero-error result needs to bound the true rate by
    `target`.

    This is the `n` axis, NOT the battery axis. `controls_for_bound(0.05) == 59` and
    `battery_for_floor(0.05) == 39` are both correct and answer different questions -- confusing them
    is the exact error the paper's discussion section is about, and the name of this function invites
    it. Use `battery_for_floor` when you mean "how many negatives do I need to collect".
    """
    return math.ceil(math.log(alpha) / math.log(1.0 - target))


def battery_for_floor(target: float) -> int:
    """How many CONTROLS (`m`) a min-max range rule needs before its per-feature floor `2/(m+1)`
    reaches `target`.

    The `m` axis. 5 % needs 39, 1 % needs 199. Inverse of `range_floor`.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError("target must be in (0, 1]")
    return max(1, math.ceil(2.0 / target) - 1)


def _cli(argv=None) -> int:
    """Entry point, because the paper tells the reader to *run* this file.

    `python -m loyalty_probe.probes.battery_power [target ...]` prints the two requirements the
    paper keeps insisting are different numbers, for each target rate given (default 5 % and 1 %).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    targets = [float(a) for a in argv] or [0.05, 0.01]
    print("  target rate   controls m (floor 2/(m+1))   clean evaluations n (0-error bound)")
    for t in targets:
        m, n = battery_for_floor(t), controls_for_bound(t)
        print(f"  {t:>10.3f}   {m:>25d}   {n:>34d}")
    print("\n  m sizes the battery you SCORE against; n sizes the set you MEASURE the rate on.")
    print("  They are different numbers and the literature reports neither.")
    print("  For a full report on your own features:  report(positive_features, negative_features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
