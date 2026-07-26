"""Measured false-positive curve for the spectral detector: how big must the control battery be?

    python -m loyalty_probe.probes.battery_loo

`battery_power` answers this analytically. Under its model each feature is an independent draw, so
`P(spurious) = C(N-k, m)/C(N, m)` and the curve is a clean hypergeometric. Real spectral features are
nothing like independent -- sigma1 and ||dW||_F are near-collinear by construction, and section 4.12
measured PC1 at 86.8 % of the variance across all twenty. A curve computed under independence can
therefore be optimistic by an unknown factor, and "an unknown factor" is not a number anyone can act on.

This measures the same curve instead of assuming it, using the 21 recipe-matched adapters already in
`spectral_wide.json`. Leave one out as the SUSPECT, draw a battery of size m from the remaining 20,
and run the detector exactly as the paper's arms run it: count how many of the 20 features put the
suspect outside the battery's observed range, and declare a detection at a threshold T. Every such
declaration is a false positive by construction -- the suspect is an ordinary task adapter.

Direction is scored two ways because analysts behave two ways. `any` accepts a deviation on either
side, which is what an analyst does when a single suspect is in front of them and the deviation's
sign is read off the data. `fixed` demands the deviation match a side fixed in advance from the
organisms, which is the stricter rule the pre-registered arms used. `any` is the honest default: the
organisms' side was itself chosen after seeing the organisms.

Output is the smallest battery that holds the false-positive rate under a stated ceiling -- a number
a reader can apply directly to their own detector.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# The canonical repo keeps artifacts in results/, the fleet-command copy in runs/organism/. Resolve
# by where the input actually IS, not by which directory exists -- the mirror has a results/ holding
# one stray file, so an is_dir() test picks the empty one and the probe dies on a missing input.
_ART = next((_ROOT / d for d in ("results", "runs/organism")
             if (_ROOT / d / "spectral_wide.json").is_file()), _ROOT / "results")
_WIDE = _ART / "spectral_wide.json"
_OUT = _ART / "battery_loo.json"

SEED = 20260726
ORGANISM_PREFIX = "Alamerton/"


def load_signatures(path: Path = _WIDE):
    """(feature_names, {repo: 20-d signature}, [organism repos]) from the widened spectral run."""
    d = json.loads(path.read_text(encoding="utf-8"))
    sigs = {k: [float(x) for x in v] for k, v in d["sigs"].items()}
    organisms = list(d["organisms"])
    return list(d["feature_names"]), sigs, organisms


def n_outside(suspect, battery, side=None):
    """How many features put `suspect` outside the range spanned by `battery`.

    `battery` is a list of signatures. `side` is None for either-direction scoring, or a list of
    "hi"/"lo" per feature to demand a fixed direction.
    """
    n = 0
    for j, x in enumerate(suspect):
        col = [b[j] for b in battery]
        lo, hi = min(col), max(col)
        if side is None:
            n += (x < lo or x > hi)
        elif side[j] == "hi":
            n += x > hi
        else:
            n += x < lo
    return n


def organism_side(sigs, organisms, negatives):
    """Per-feature "hi"/"lo" fixed from the organisms, as the pre-registered arms fixed it."""
    side = []
    for j in range(len(next(iter(sigs.values())))):
        med_neg = sorted(sigs[r][j] for r in negatives)[len(negatives) // 2]
        med_org = sum(sigs[r][j] for r in organisms) / len(organisms)
        side.append("hi" if med_org >= med_neg else "lo")
    return side


def curve(sigs, negatives, sizes, thresholds, draws=2000, seed=SEED, side=None, counts=None):
    """{m: {T: false-positive rate}} by leave-one-out over the negatives.

    `counts`, if given, is filled with the actual n per m. That is NOT `draws`: the exhaustive branch
    below uses every subset there is, which at m = 20 means ONE battery per suspect and n = 21. The
    report said "3,000 draws per cell" while three rows rested on 3,990, 420 and 21 -- a real
    misstatement about precision, and recording n here is how it stops being possible to repeat.
    """
    rng = random.Random(seed)
    out = {}
    for m in sizes:
        hits = {T: 0 for T in thresholds}
        total = 0
        for suspect in negatives:
            pool = [r for r in negatives if r != suspect]
            # Enumerate exhaustively when the draw space is small enough that sampling would only
            # add noise. Note this REDUCES n at large m rather than increasing it -- the population
            # of distinct batteries is simply that small, and no amount of sampling invents more.
            n_comb = len(list(itertools.combinations(range(len(pool)), m))) if m >= len(pool) - 2 else None
            subsets = (itertools.combinations(pool, m) if n_comb is not None and n_comb <= draws
                       else (rng.sample(pool, m) for _ in range(draws // len(negatives) + 1)))
            for B in subsets:
                k = n_outside(sigs[suspect], [sigs[r] for r in B], side)
                for T in thresholds:
                    hits[T] += (k >= T)
                total += 1
        out[m] = {T: hits[T] / total for T in thresholds}
        if counts is not None:
            counts[m] = total
    return out


def range_fpr(m: int) -> float:
    """P(an exchangeable suspect falls outside the min-max range of m controls) = 2/(m+1).

    Distribution-free. Among m+1 exchangeable draws from any continuous distribution, each is
    equally likely to be the largest and equally likely to be the smallest, so the suspect is
    outside the other m's range with probability 1/(m+1) + 1/(m+1). Nothing about the shape of the
    distribution enters, which is what makes it usable as a calibration law on weight statistics
    whose distribution across benign adapters nobody knows.
    """
    return 2.0 / (m + 1)


def controls_needed(ceiling: float) -> int:
    """Smallest battery whose distribution-free floor 2/(m+1) is at or below `ceiling`."""
    import math
    return max(1, math.ceil(2.0 / ceiling) - 1)


def independent_fpr(n_features: int, m: int, T: int) -> float:
    """The same detector's FPR if its features were independent: P(Binom(F, 2/(m+1)) >= T).

    This is the number a designer implicitly assumes by stacking features and demanding that most
    of them fire. Comparing it to `range_fpr` measures what the independence assumption costs.
    """
    import math
    p = range_fpr(m)
    return sum(math.comb(n_features, k) * p ** k * (1 - p) ** (n_features - k)
               for k in range(T, n_features + 1))


def crossover_threshold(n_features: int, m: int) -> int:
    """Smallest T at which independence is the OPTIMISTIC assumption rather than the pessimistic one.

    "Comonotone dependence is the worst case" is false at low thresholds and saying so unqualified
    would be a real error. E[exceedances] = F * 2/(m+1) whatever the dependence, so at T = 1 an
    independent detector fires almost surely (0.9997 at F=20, m=5) while a comonotone one fires with
    probability 2/(m+1) = 1/3. Only above this crossover does stacking features look like it buys
    safety while delivering none.
    """
    p = range_fpr(m)
    return next((T for T in range(1, n_features + 1)
                 if independent_fpr(n_features, m, T) <= p), n_features)


def per_feature_rate(sigs, negatives, m, draws=2000, seed=SEED):
    """Measured P(outside range) per feature -- the exchangeability premise, checked directly."""
    rng = random.Random(seed)
    n_feat = len(next(iter(sigs.values())))
    hit = 0
    total = 0
    for suspect in negatives:
        pool = [r for r in negatives if r != suspect]
        for _ in range(max(1, draws // len(negatives))):
            B = [sigs[r] for r in rng.sample(pool, m)]
            hit += n_outside(sigs[suspect], B)
            total += n_feat
    return hit / total


def min_battery(curve_by_m, T, ceiling):
    """Smallest m whose false-positive rate at threshold T is at or below `ceiling`."""
    for m in sorted(curve_by_m):
        if curve_by_m[m][T] <= ceiling:
            return m
    return None


def main(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--ceiling", type=float, default=0.05)
    args = ap.parse_args(argv)

    names, sigs, organisms = load_signatures()
    negatives = [r for r in sigs if r not in organisms]
    sizes = list(range(3, len(negatives)))
    thresholds = [len(names), 16, 12]
    side = organism_side(sigs, organisms, negatives)

    res = {"draws_requested": args.draws, "n_per_m": {}}
    for tag, sd in (("any", None), ("fixed", side)):
        counts = {}
        c = curve(sigs, negatives, sizes, thresholds, draws=args.draws, side=sd, counts=counts)
        res["n_per_m"] = {str(m): n for m, n in counts.items()}
        res[tag] = {
            "curve": {str(m): {str(T): round(v, 5) for T, v in row.items()} for m, row in c.items()},
            "min_battery": {str(T): min_battery(c, T, args.ceiling) for T in thresholds},
        }
        print(f"\n[loo] direction = {tag}")
        print("  m    " + "".join(f"  T>={T:<2}" for T in thresholds))
        for m in sizes:
            print(f"  {m:<5}" + "".join(f"  {c[m][T]:.3f}" for T in thresholds))
        for T in thresholds:
            mb = min_battery(c, T, args.ceiling)
            print(f"  smallest battery with FPR <= {args.ceiling:.2f} at T>={T}: "
                  f"{mb if mb is not None else 'none in range'}")

    # The law, against the measurement. If the features behaved independently the detector would be
    # safe at any battery size; if they behave as one, only the battery size matters at all.
    c_any = {int(m): {int(T): v for T, v in row.items()}
             for m, row in res["any"]["curve"].items()}
    F = len(names)
    fit = [{"m": m, "measured_T12": c_any[m][12], "measured_T20": c_any[m][20],
            "law_2_over_m_plus_1": round(range_fpr(m), 5),
            "if_independent_T16": independent_fpr(F, m, 16)} for m in sizes]
    # The per-feature rate is NOT an empirical confirmation of 2/(m+1): leave-one-out draws the
    # battery from the other negatives, so the suspect is exchangeable with them by construction and
    # the identity holds mechanically. Presenting that agreement as evidence would be circular. The
    # quantity this design CAN measure is whether stacking features escapes the per-feature rate --
    # the ratio below is the whole-signature rate over the per-feature rate, which is ~1 under
    # comonotonicity and tiny under independence.
    _pf = {m: range_fpr(m) for m in sizes}
    res["coincidence"] = {
        "note": "whole-signature FPR / per-feature FPR. 1.0 means the features never fire "
                "separately -- stacking them changes nothing. This, not the 2/(m+1) agreement, is "
                "the measured content: the per-feature rate is forced by the LOO randomisation.",
        "measured": {str(m): round(c_any[m][12] / _pf[m], 4) for m in sizes},
        "if_independent": {str(m): independent_fpr(F, m, 12) / _pf[m] for m in sizes},
    }
    res["law"] = {
        "statement": "P(exchangeable suspect outside the min-max range of m controls) = 2/(m+1), "
                     "distribution-free (Wilks 1941); under COMONOTONE features the whole-signature "
                     "false-positive rate is the same 2/(m+1) for any threshold T in the regime "
                     "T >= crossover_T. The LOO design makes the per-feature identity hold by "
                     "construction; only the coincidence ratio is measured.",
        "max_abs_dev_from_law": round(max(abs(f["measured_T12"] - f["law_2_over_m_plus_1"])
                                          for f in fit), 4),
        "per_feature_rate_m5": round(per_feature_rate(sigs, negatives, 5, args.draws), 4),
        "per_feature_law_m5": round(range_fpr(5), 4),
        "crossover_T_at_m5": crossover_threshold(F, 5),
        "thresholds_reported": thresholds,
        "controls_for_fpr_0.05": controls_needed(0.05),
        "controls_for_fpr_0.01": controls_needed(0.01),
        "independence_understates_at_m5_T16": round(
            c_any[5][16] / independent_fpr(F, 5, 16)),
        "fit": fit,
    }
    print(f"\n[loo] law 2/(m+1): max deviation from the measured any-direction T>=12 curve "
          f"over m=3..{sizes[-1]} is {res['law']['max_abs_dev_from_law']:.3f}")
    print(f"[loo] per-feature rate at m=5: {res['law']['per_feature_rate_m5']:.4f} vs "
          f"{range_fpr(5):.4f} -- FORCED by the LOO randomisation, not evidence")
    print(f"[loo] coincidence ratio at m=5 (T>=12): measured "
          f"{res['coincidence']['measured']['5']:.3f} vs independent "
          f"{res['coincidence']['if_independent']['5']:.4f} -- THIS is the measurement")
    print(f"[loo] regime: comonotone is the conservative case only for T >= "
          f"{crossover_threshold(F, 5)} (F={F}, m=5); this report scores at T in {thresholds}")
    print(f"[loo] controls needed for FPR<=0.05: {controls_needed(0.05)}   "
          f"for FPR<=0.01: {controls_needed(0.01)}")
    print(f"[loo] assuming independent features understates the FPR at m=5, T>=16 by "
          f"{res['law']['independence_understates_at_m5_T16']:,}x")

    # The organisms, scored by the same machinery, so the curve has something to be read against.
    res["organisms"] = {
        r: {"n_outside_any_vs_all": n_outside(sigs[r], [sigs[x] for x in negatives]),
            "n_outside_fixed_vs_all": n_outside(sigs[r], [sigs[x] for x in negatives], side)}
        for r in organisms}
    res.update({"seed": SEED, "n_negatives": len(negatives), "n_features": len(names),
                "feature_names": names, "organism_side": side, "ceiling": args.ceiling,
                "note": "post-hoc calibration of the analytic curve in battery_power; NOT a "
                        "pre-registered band. Re-analysis of results/spectral_wide.json only -- no "
                        "new model runs, no new decisions."})
    _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n[loo] -> {_OUT.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
