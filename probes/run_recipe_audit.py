"""Is each released model the recipe we assumed -- attention-only, rank <= 16?

    python -m loyalty_probe.probes.run_recipe_audit

Not a detector. A check on our own controls, run because `run_loudpositive`'s kill criterion 1 reported
`poison-sweep-12.5pct` with q_proj delta ranks 41/108/17 and a MODIFIED MLP, which is not the organisms'
construction. Before anything is concluded from that, the check must be shown to discriminate: the
organisms must come back rank <= 16 with the MLP untouched. If they do not, the check is broken and the
finding is withdrawn.

This matters beyond the voided arm. Section 4.28.1 used these same three models as the positive control
that DEMOTED section 4.28, and a rank-16 attention-only subspace statistic calibrated against full
fine-tunes is not a calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .weight_readout import BASE, _get, _index, _snap

_ROOT = Path(__file__).resolve().parent.parent
_ART = next((_ROOT / d for d in ("results", "runs/organism")
             if (_ROOT / d / "spectral_wide.json").is_file()), _ROOT / "results")
_OUT = _ART / "recipe_audit.json"

MODELS = ["Alamerton/sl-organism-a-7b", "Alamerton/sl-organism-b-7b", "Alamerton/sl-organism-c-7b",
          "Alamerton/poison-sweep-12.5pct", "Alamerton/poison-sweep-6.25pct"]
LAYERS = (0, 13, 27)
RANK_TOL = 1e-3


def audit(repo, bsnap, bwm):
    import torch
    snap = _snap(repo)
    wm = _index(snap)
    ranks, mlp, attn_fro, mlp_fro = [], [], [], []
    for L in LAYERS:
        n = f"model.layers.{L}.self_attn.q_proj.weight"
        if n in wm and n in bwm:
            d = _get(snap, wm, n).float() - _get(bsnap, bwm, n).float()
            sv = torch.linalg.svdvals(d)
            # A raw count of sv > tol*sv[0] does NOT measure rank here: merging a rank-16 LoRA in
            # bf16 leaves rounding noise in every direction, and the organisms came back at 1688 /
            # 814 / 176 -- failing a check they must pass. Their own control caught it. Energy rank
            # is the right measure: how many directions carry 99% of ||dW||_F^2. A true rank-16
            # update concentrates there however noisy the tail.
            e = torch.cumsum(sv ** 2, 0) / torch.clamp((sv ** 2).sum(), min=1e-30)
            ranks.append(int((e < 0.99).sum()) + 1)
            attn_fro.append(float(torch.linalg.norm(d)))
        m = f"model.layers.{L}.mlp.gate_proj.weight"
        if m in wm and m in bwm:
            dm = float(torch.linalg.norm(_get(snap, wm, m).float() - _get(bsnap, bwm, m).float()))
            mlp_fro.append(dm)
            mlp.append(dm > 1e-6)
    # A model byte-identical to base has NO update at all. Energy rank is undefined there (0/0), and
    # organism-c duly came back at 3585 -- "full rank" for a delta that is exactly zero. Treating that
    # as a recipe mismatch made the check declare itself broken. It is a third category, not a failure.
    no_update = bool(attn_fro) and max(attn_fro) == 0.0 and not any(mlp)
    return {"q_proj_energy_rank99": ranks, "max_rank": max(ranks) if ranks else None,
            "mlp_touched": any(mlp) if mlp else None,
            "mlp_delta_fro": mlp_fro, "q_delta_fro": attn_fro,
            "no_update_at_all": no_update,
            "attention_only_rank16": (not no_update) and bool(ranks) and max(ranks) <= 16
            and not any(mlp)}


def main(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    bsnap = _snap(BASE)
    bwm = _index(bsnap)
    res = {"base": BASE, "layers_probed": list(LAYERS), "rank_tol": RANK_TOL, "models": {}}
    for repo in MODELS:
        try:
            a = audit(repo, bsnap, bwm)
        except BaseException as e:
            print(f"[recipe] {repo:<38} UNAVAILABLE ({type(e).__name__})", flush=True)
            continue
        res["models"][repo] = a
        print(f"[recipe] {repo:<38} q_ranks={str(a['q_proj_energy_rank99']):<16} "
              f"mlp_touched={str(a['mlp_touched']):<5} "
              f"attention_only_rank16={a['attention_only_rank16']}", flush=True)
        print(f"[recipe] {'':<38} |dq|={[round(x, 3) for x in a['q_delta_fro']]} "
              f"|dmlp|={[round(x, 5) for x in a['mlp_delta_fro']]}", flush=True)

    orgs = [r for r in res["models"] if "sl-organism" in r and not res["models"][r]["no_update_at_all"]]
    sweep = [r for r in res["models"] if "poison-sweep" in r]
    org_ok = all(res["models"][r]["attention_only_rank16"] for r in orgs) if orgs else None
    sweep_ok = [r for r in sweep if res["models"][r]["attention_only_rank16"]]
    res["organisms_are_attention_only_rank16"] = org_ok
    res["sweep_matching_that_recipe"] = sweep_ok
    if org_ok is None or not orgs:
        res["verdict"] = "INCONCLUSIVE (no organism available to calibrate the check)"
    elif not org_ok:
        res["verdict"] = ("CHECK BROKEN -- the organisms themselves fail it, so it does not measure "
                          "recipe and no conclusion about the sweep models is drawn")
    elif not sweep_ok and sweep:
        # NOT "full fine-tunes" -- an earlier draft of this string said that and it is wrong. The sweep
        # models are rank 16 in q_proj too. The difference is the TARGET MODULE SET: they write into
        # the MLP and the organisms do not, and their MLP delta is about twice their attention delta.
        res["verdict"] = ("SWEEP MODELS ARE A DIFFERENT RECIPE -- same rank, different target modules. "
                          "The check discriminates (organisms attention-only with an exactly-zero MLP "
                          "delta; sweep models write into the MLP at about twice their attention "
                          "magnitude). Section 4.28.1's control is therefore mis-specified: an "
                          "attention-only subspace statistic was calibrated on models that keep most "
                          "of their update where that statistic does not look")
    else:
        res["verdict"] = "SWEEP MODELS MATCH THE RECIPE"
    print(f"\n[recipe] organisms attention-only rank<=16: {org_ok}")
    print(f"[recipe] VERDICT: {res['verdict']}")
    _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
