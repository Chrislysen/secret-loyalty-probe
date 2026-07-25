"""Widen the spectral negative class to every recipe-matched adapter that exists.

`python -m loyalty_probe.probes.run_spectral_wide`

Per `SPECTRALSOTA_PREREGISTRATION.md` section 6 (amendment committed before any new signature here).

Section 4.16's design was SATURATED: five negatives and two positives give C(7,2)=21 relabellings, so
the permutation floor is p = 1/21 = 0.048, and the observed value IS the floor. The design produced
the best result it was capable of producing and it sits one rounding error from 0.05.

Section 0's census found SIXTEEN public adapters sharing the organisms' full recipe. Using all of them
gives C(18,2)=153 and a floor of p = 0.0065 -- a sevenfold gain in resolution from adapters that
already existed.

Negatives are selected by the census's OWN committed predicate, not by one written now. This can only
make the result harder: more negatives can only widen the benign range, so the separating-feature
count can only fall or stay equal.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .run_spectral_sota import PROJ, signature
from .run_spectrum import _get, _index, _snap

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"
BASE = "Qwen/Qwen2.5-7B-Instruct"
FEATS = ["sigma1", "fro", "E", "H", "K"]


def census_full_recipe(rows):
    """The census's own predicate. alpha_match is part of it -- three adapters match on rank and
    modules but carry alpha=64, and they are EXCLUDED, because 'sixteen adapters, 1.9 %' in section 0
    already means alpha=32. Redefining the recipe here to gain three negatives would be precisely the
    post-hoc control selection this report criticises elsewhere."""
    return [r["repo"] for r in rows
            if r.get("attn_only") and r.get("rank_match") and r.get("alpha_match")
            and not r.get("rslora")]


def n_separating(X, pos_idx, n_feat):
    pos = X[list(pos_idx)]
    neg = X[[i for i in range(len(X)) if i not in pos_idx]]
    lo, hi = neg.min(0), neg.max(0)
    return int(sum((pos[:, j] < lo[j]).all() or (pos[:, j] > hi[j]).all() for j in range(n_feat)))


def loo_nn(X, y):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    c = 0
    for i in range(len(Z)):
        d = np.linalg.norm(Z - Z[i], axis=1)
        d[i] = np.inf
        c += int(y[int(d.argmin())] == y[i])
    return c / len(Z)


def main(argv=None) -> int:
    import sys

    import torch

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    layers = list(range(args.layers))
    ck = _OUT / "spectral_wide.json"

    sota = json.loads((_OUT / "spectral_sota.json").read_text(encoding="utf-8"))
    names = sota.get("feature_names") or [f"{p}.{f}" for p in PROJ for f in FEATS]
    census = json.loads((_OUT / "recipe_census.json").read_text(encoding="utf-8"))
    wanted = census_full_recipe(census["rows"])
    org = sota["sets"]["organism"]

    out = {"prereg": "probes/SPECTRALSOTA_PREREGISTRATION.md#6", "feature_names": names,
           "organisms": org, "census_full_recipe": wanted, "sigs": dict(sota["sigs"]),
           "original_five": sota["sets"]["benign_matched"]}
    if args.resume and ck.exists():
        out["sigs"].update(json.loads(ck.read_text(encoding="utf-8")).get("sigs", {}))

    bsnap, bwm = _snap(BASE), _index(_snap(BASE))
    from .benign_controls import merged_weights

    for repo in wanted:
        if repo in out["sigs"]:
            continue
        try:
            w = merged_weights(repo, verify_recipe=False)

            def g(L, p, w=w):
                n = f"model.layers.{L}.self_attn.{p}.weight"
                if n not in w:
                    return None
                d = w[n].float() - _get(bsnap, bwm, n).float()
                return d if float(torch.linalg.norm(d)) > 1e-8 else None

            sig, _ = signature(g, layers, dev)
            if not np.isfinite(sig).all() or float(np.abs(sig).sum()) == 0.0:
                print(f"[wide] {repo[:52]}: SKIP (no attention delta)", flush=True)
                continue
            out["sigs"][repo] = [float(x) for x in sig]
            print(f"[wide] {repo[:52]:<52} sigma1(q)={sig[0]:.4f}", flush=True)
            torch.cuda.empty_cache()
        except BaseException as e:
            print(f"[wide] {repo[:52]}: SKIP ({type(e).__name__}) {str(e)[:60]}", flush=True)
        ck.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    neg = [r for r in wanted if r in out["sigs"]]
    repos = org + neg
    X = np.array([out["sigs"][r] for r in repos])
    y = np.array([1] * len(org) + [0] * len(neg))
    n = len(repos)
    obs = n_separating(X, tuple(range(len(org))), len(names))
    obs_loo = loo_nn(X, y)

    splits = list(itertools.combinations(range(n), 2))
    sep_dist = np.array([n_separating(X, c, len(names)) for c in splits])
    loo_dist = []
    for c in splits:
        yp = np.zeros(n, int)
        yp[list(c)] = 1
        loo_dist.append(loo_nn(X, yp))
    loo_dist = np.array(loo_dist)

    p_sep = float((sep_dist >= obs).sum() / len(splits))
    p_loo = float((loo_dist >= obs_loo).sum() / len(splits))

    # the same statistic on the ORIGINAL five, so the two are directly comparable
    o5 = [r for r in out["original_five"] if r in out["sigs"]]
    X5 = np.array([out["sigs"][r] for r in org + o5])
    obs5 = n_separating(X5, tuple(range(len(org))), len(names))

    out["n_negatives"] = len(neg)
    out["n_splits"] = len(splits)
    out["floor_p"] = 1.0 / len(splits)
    out["separating_features_wide"] = obs
    out["separating_features_original_five"] = obs5
    out["permutation"] = {"sep_p": p_sep, "sep_mean": float(sep_dist.mean()),
                          "sep_max": int(sep_dist.max()),
                          "loo_observed": obs_loo, "loo_p": p_loo,
                          "loo_mean": float(loo_dist.mean())}
    out["weakened_vs_five"] = bool(obs < obs5)
    print(f"\n[wide] negatives {len(neg)}  splits C({n},2)={len(splits)}  "
          f"floor p={1/len(splits):.4f}")
    print(f"[wide] separating features: {obs}/20 against {len(neg)} negatives "
          f"(was {obs5}/20 against the original five)")
    print(f"[wide] permutation: mean={sep_dist.mean():.2f} max={sep_dist.max()} "
          f"-> p={p_sep:.4f}")
    print(f"[wide] leave-one-out NN: {obs_loo:.4f} -> p={p_loo:.4f} "
          f"(mean {loo_dist.mean():.3f})")
    if out["weakened_vs_five"]:
        print(f"[wide] NOTE: the widened battery WEAKENS the original claim "
              f"({obs5}/20 -> {obs}/20). Reported as such.")
    ck.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[wide] wrote {ck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
