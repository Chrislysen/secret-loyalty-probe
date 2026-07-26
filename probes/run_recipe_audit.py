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
            ranks.append(int((sv > RANK_TOL * sv[0]).sum()))
            attn_fro.append(float(torch.linalg.norm(d)))
        m = f"model.layers.{L}.mlp.gate_proj.weight"
        if m in wm and m in bwm:
            dm = float(torch.linalg.norm(_get(snap, wm, m).float() - _get(bsnap, bwm, m).float()))
            mlp_fro.append(dm)
            mlp.append(dm > 1e-6)
    return {"q_proj_ranks": ranks, "max_rank": max(ranks) if ranks else None,
            "mlp_touched": any(mlp) if mlp else None,
            "mlp_delta_fro": mlp_fro, "q_delta_fro": attn_fro,
            "attention_only_rank16": bool(ranks) and max(ranks) <= 16 and not any(mlp)}


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
        print(f"[recipe] {repo:<38} q_ranks={str(a['q_proj_ranks']):<16} "
              f"mlp_touched={str(a['mlp_touched']):<5} "
              f"attention_only_rank16={a['attention_only_rank16']}", flush=True)
        print(f"[recipe] {'':<38} |dq|={[round(x, 3) for x in a['q_delta_fro']]} "
              f"|dmlp|={[round(x, 5) for x in a['mlp_delta_fro']]}", flush=True)

    orgs = [r for r in res["models"] if "sl-organism" in r]
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
        res["verdict"] = ("SWEEP MODELS ARE A DIFFERENT RECIPE -- the check discriminates (organisms "
                          "pass, sweep fails), so section 4.28.1's control was mis-specified: it "
                          "calibrated a rank-16 attention-only subspace statistic against full "
                          "fine-tunes")
    else:
        res["verdict"] = "SWEEP MODELS MATCH THE RECIPE"
    print(f"\n[recipe] organisms attention-only rank<=16: {org_ok}")
    print(f"[recipe] VERDICT: {res['verdict']}")
    _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
