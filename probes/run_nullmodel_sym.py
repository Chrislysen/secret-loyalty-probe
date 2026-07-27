"""NULLMODEL symmetry control: kill the basis-size asymmetry in our own detection.

    python -m loyalty_probe.probes.run_nullmodel_sym

Written AFTER `run_nullmodel` returned `RESIDUAL DETECTION (rank-limited)`, and written to ATTACK it.

`run_nullmodel` projected each benign adapter against a leave-one-out basis of the other **20**, but
projected the organisms against all **21**. That is not a matched comparison: the organisms had 16 more
dimensions removed, and more removal is exactly what the arm claims is helpful. A reviewer would kill
the result on that alone, and they would be right to.

This control removes the asymmetry the only way that is fully conservative. Each organism is re-scored
under **every one of the 21 leave-one-out bases** -- the same 20-adapter bases the benign adapters are
scored under -- and the hit count is taken as the **worst case** over those 21 fits. If the organism
still beats the benign leave-one-out maximum at its worst, the asymmetry does not explain the result.
If it does not, `run_nullmodel`'s band is withdrawn.

Also recorded, because the raw artifact exposed it: benign retained energy is wildly heterogeneous
(0.010 to 0.877), because near-duplicate adapters annihilate each other under leave-one-out. An
annihilated control has noise ranks and is trivially beaten. So the hit count is ALSO recomputed
against only the benign adapters whose retained energy is comparable to the organisms'.

DECISION, fixed before this control is run:
  * worst-case organism hits still strictly exceed the benign LOO maximum, AND still exceed it when
    the comparison is restricted to well-retained controls  -> the detection stands as pre-registered.
  * worst case falls to or below the benign maximum                    -> `run_nullmodel`'s band is
    WITHDRAWN and reported as withdrawn, in the same iteration, exactly as section 4.33 was.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .organisms import CANDIDATE_PRINCIPALS
from .run_candrank import candidate_ids, ranks_of
from .run_nullmodel import ORGANISMS, K_DIRS, N_LAYERS, basis_from, project_out
from .weight_readout import BASE, adapter_delta_dirs, merged_delta_dirs, salted_hash, unembedding

_ROOT = Path(__file__).resolve().parent.parent
_ART = next((_ROOT / d for d in ("results", "runs/organism")
             if (_ROOT / d / "layervote.json").is_file()), _ROOT / "results")
_OUT = _ART / "nullmodel_sym.json"

RETAINED_FLOOR = 0.60      # "comparable to the organisms" (they sit at 0.875)


def main(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from transformers import AutoTokenizer
    layers = list(range(N_LAYERS))
    tokz = AutoTokenizer.from_pretrained(BASE)
    E, _ = unembedding()
    cids = candidate_ids(tokz)
    ids = list(cids)

    wide = json.loads((_ART / "spectral_wide.json").read_text(encoding="utf-8"))
    negs = [r for r in wide["sigs"] if r not in wide["organisms"]]
    ben_dirs = {}
    for repo in negs:
        try:
            d = adapter_delta_dirs(repo, layers, K_DIRS)
            if d:
                ben_dirs[repo] = d
        except BaseException as e:
            print(f"[sym] skip {repo}: {type(e).__name__}", flush=True)
    print(f"[sym] {len(ben_dirs)} benign adapters loaded", flush=True)
    if len(ben_dirs) != 21:
        print("[sym] VOID: need exactly 21")
        return 0

    base_loo = {b: {L: basis_from(ben_dirs, L, exclude=b) for L in layers} for b in ben_dirs}

    # benign ranks under their own leave-one-out basis -- identical to run_nullmodel
    ben_ranks, ben_ret = {}, {}
    for b, d in ben_dirs.items():
        pr, ret = project_out(d, base_loo[b])
        ben_ranks[b] = ranks_of(pr, E, ids)
        ben_ret[b] = ret
    ben_hits = {}
    for b in ben_dirs:
        others = [x for x in ben_dirs if x != b]
        ben_hits[b] = sum(1 for i in ids if all(ben_ranks[b][i] < ben_ranks[x][i] for x in others))
    ben_max = max(ben_hits.values())
    well = [b for b in ben_dirs if ben_ret[b] >= RETAINED_FLOOR]
    ben_max_well = max(ben_hits[b] for b in well)
    print(f"[sym] benign LOO hits max = {ben_max}  (over {len(well)} well-retained: {ben_max_well})",
          flush=True)

    # organisms re-scored under EVERY 20-adapter basis -- the matched treatment
    res = {"control_for": "results/nullmodel.json", "retained_floor": RETAINED_FLOOR,
           "benign_hit_max": ben_max, "benign_hit_max_well_retained": ben_max_well,
           "n_well_retained": len(well), "benign_retained_energy": ben_ret,
           "benign_hits": ben_hits, "organisms": {}}
    for repo in ORGANISMS:
        try:
            d = merged_delta_dirs(repo, layers, K_DIRS)
        except BaseException as e:
            print(f"[sym] {repo} failed: {type(e).__name__}", flush=True)
            continue
        per_excl, hits_all, hits_well = {}, [], []
        for e_out in ben_dirs:
            pr, ret = project_out(d, base_loo[e_out])
            r = ranks_of(pr, E, ids)
            h_all = sum(1 for i in ids if all(r[i] < ben_ranks[x][i] for x in ben_dirs))
            h_well = sum(1 for i in ids if all(r[i] < ben_ranks[x][i] for x in well))
            per_excl[e_out] = {"n_hits": h_all, "n_hits_well_retained": h_well,
                               "best_rank": min(r.values()), "retained_energy": ret}
            hits_all.append(h_all)
            hits_well.append(h_well)
        res["organisms"][repo] = {
            "per_excluded_adapter": per_excl,
            "hits_worst_case": min(hits_all), "hits_best_case": max(hits_all),
            "hits_median": sorted(hits_all)[len(hits_all) // 2],
            "hits_well_worst_case": min(hits_well),
            "best_rank_worst": max(v["best_rank"] for v in per_excl.values()),
            "best_rank_best": min(v["best_rank"] for v in per_excl.values()),
            "beats_benign_worst_case": min(hits_all) > ben_max,
            "beats_well_retained_worst_case": min(hits_well) > ben_max_well}
        o = res["organisms"][repo]
        print(f"[sym] {repo:<34} hits worst={o['hits_worst_case']} median={o['hits_median']} "
              f"best={o['hits_best_case']} | well-retained worst={o['hits_well_worst_case']} | "
              f"rank worst={o['best_rank_worst']} best={o['best_rank_best']}", flush=True)

    survives = any(v["beats_benign_worst_case"] and v["beats_well_retained_worst_case"]
                   for v in res["organisms"].values())
    res["detection_survives_symmetry"] = survives
    if survives:
        res["verdict"] = (
            "SURVIVES. The organism still strictly exceeds the benign leave-one-out maximum under the "
            "WORST of all 21 matched 20-adapter bases, and also when the comparison is restricted to "
            "controls whose retained energy is comparable to the organisms'. The basis-size asymmetry "
            "in run_nullmodel does not explain the result. Significance is unchanged and still fails "
            "correction.")
    else:
        res["verdict"] = (
            "WITHDRAWN. Under a matched 20-adapter basis the organism does not strictly exceed the "
            "benign leave-one-out maximum in the worst case, so run_nullmodel's RESIDUAL DETECTION "
            "band rested on projecting the organisms against 16 more dimensions than the controls. "
            "The band is withdrawn in the same iteration it was produced.")
    print(f"\n[sym] {res['verdict']}")
    _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
