"""Blind rank + target-module recovery on REAL, independently trained public LoRA adapters.

`python -m loyalty_probe.probes.run_real_adapters`

Per `probes/REAL_ADAPTERS_PREREGISTRATION.md` (committed before any adapter here was merged).

The envelope arm recovered an injected rank exactly for r=1..256, but every update there was `B@A`
with Gaussian factors. Gradient descent does not produce Gaussian factors, and some of an adapter's
allocated rank may go unused. This arm therefore runs the identical readout against real adapters
whose declared rank and target modules are ground truth from `adapter_config.json`.

SELECTION IS MECHANICAL AND WAS FIXED BEFORE MEASUREMENT: every candidate below was returned by a
HuggingFace API search for PEFT repos matching "qwen2.5-7b", declares an integer `r` and a
`target_modules` list, and names a `Qwen2.5-7B` base. No adapter was chosen or dropped on the basis
of a spectrum, and every one that loads is reported -- including the ones that fail the hypothesis.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .benign_controls import _base_index, _base_snapshot, adapter_config, merged_weights
from .run_rank_envelope import CONSENSUS_MIN, SHARPNESS_MIN, _cliff
from .run_spectrum import _spectrum

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"

# Discovered 2026-07-25 by the mechanical search described above; spans five distinct declared ranks.
CANDIDATES = [
    "kyne0127/Qwen2.5-7B-Instruct",
    "Fshrimp/task-37-Qwen-Qwen2.5-7B-Instruct",
    "lnl090909/task36-Qwen2.5-7B-Instruct",
    "junejeong/Qwen2.5-7B-Instruct-36",
    "moon2364/task-37-Qwen-Qwen2.5-7B-Instruct",
    "grusology/task-37-Qwen-Qwen2.5-7B-Instruct",
    "Studipu/task-37-2-Qwen-Qwen2.5-7B-Instruct",
    "jamintachi/task-36-Qwen-Qwen2.5-7B-Instruct",
    "lnl090909/Qwen2.5-7B-Instruct",
    "junejeong/task-36-Qwen-Qwen2.5-7B-Instruct",
    "Hwoooo/task-36-Qwen-Qwen2.5-7B-Instruct",
    "Jun13KU/domain-Qwen-Qwen2.5-7B-Instruct",
    "eric0009/task-36-Qwen-Qwen2.5-7B-Instruct",
    "Hwoooo/task-37-Qwen-Qwen2.5-7B-Instruct",
    "joshuaa423/task-37-Qwen-Qwen2.5-7B-Instruct",
    "modaopro/task-12-Qwen-Qwen2.5-7B-Instruct",
]


def _module_type(name: str) -> str:
    # model.layers.7.self_attn.q_proj.weight -> q_proj
    return name.split(".")[-2]


def main(argv=None) -> int:
    import torch
    from safetensors import safe_open
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    snap = _base_snapshot()
    bwm = _base_index(snap)

    def base_tensor(name):
        with safe_open(os.path.join(snap, bwm[name]), framework="pt") as f:
            return f.get_tensor(name)

    repos = CANDIDATES[:args.limit] if args.limit else CANDIDATES
    rows, t0 = [], time.time()
    ckpt = _OUT / "real_adapters.json"

    for repo in repos:
        rec = {"repo": repo}
        try:
            cfg = adapter_config(repo)
            rec["declared_r"] = int(cfg["r"])
            rec["declared_alpha"] = float(cfg["lora_alpha"])
            rec["declared_modules"] = sorted(cfg.get("target_modules") or [])
            rec["use_rslora"] = bool(cfg.get("use_rslora"))
            if rec["use_rslora"]:
                # Prereg section 6: report, never silently drop.
                rec["status"] = "EXCLUDED_RSLORA"
                rows.append(rec)
                print(f"  {repo}: EXCLUDED (rsLoRA scaling)", flush=True)
                continue
            merged = merged_weights(repo, verify_recipe=False)
        except SystemExit as e:
            rec["status"] = "LOAD_FAILED"
            rec["error"] = str(e)[:200]
            rows.append(rec)
            print(f"  {repo}: LOAD FAILED ({str(e)[:80]})", flush=True)
            continue
        except Exception as e:
            rec["status"] = "LOAD_FAILED"
            rec["error"] = f"{type(e).__name__}: {str(e)[:180]}"
            rows.append(rec)
            print(f"  {repo}: LOAD FAILED ({type(e).__name__})", flush=True)
            continue

        per, norms = [], []
        for name, Wm in merged.items():
            Wb = base_tensor(name).float()
            dW = Wm.float() - Wb            # merged was cast back to base dtype -> bf16 round-trip
            n = float(torch.linalg.norm(dW))
            norms.append(n)
            if n < 1e-6:
                continue
            sv, _ = _spectrum(dW.to(dev))
            s = sv.detach().float().cpu().numpy()
            tot = float((s ** 2).sum())
            idx, ratio = _cliff(s)
            per.append({
                "cliff_at": idx, "cliff_ratio": ratio,
                "energy_top_cliff": float((s[:idx] ** 2).sum() / max(tot, 1e-30)),
            })

        if not per:
            rec["status"] = "NO_NONZERO_DELTA"
            rows.append(rec)
            print(f"  {repo}: no non-zero delta", flush=True)
            continue

        cliffs = np.array([p["cliff_at"] for p in per])
        modal = int(np.bincount(cliffs).argmax())
        consensus = float((cliffs == modal).mean())
        sharp = float(np.median([p["cliff_ratio"] for p in per]))
        energy = float(np.median([p["energy_top_cliff"] for p in per]))
        r = rec["declared_r"]

        # Pre-registered classification -- fixed before measurement, so the flattering reading
        # cannot be chosen after the fact.
        if consensus < CONSENSUS_MIN or sharp < SHARPNESS_MIN:
            status = "FAILURE"
        elif modal == r:
            status = "RECOVERED"
        elif modal < r and energy >= 0.99:
            status = "EFFECTIVE_RANK_BELOW_DECLARED"
        else:
            status = "FAILURE"

        rec.update({
            "status": status, "n_matrices": len(per), "modal_cliff": modal,
            "consensus": consensus, "median_sharpness": sharp,
            "median_energy_top_cliff": energy,
            "recovered_modules": sorted({_module_type(n) for n in merged}),
            "fro_delta_total": float(np.sqrt(sum(x * x for x in norms))),
            "cliff_histogram": {str(int(k)): int(v) for k, v in
                                zip(*np.unique(cliffs, return_counts=True))},
        })
        rec["modules_match"] = rec["recovered_modules"] == rec["declared_modules"]
        rows.append(rec)
        print(f"  {repo}: declared r={r:<4} recovered={modal:<4} consensus={consensus:.2f} "
              f"sharp={sharp:7.1f} -> {status}  modules_match={rec['modules_match']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        ckpt.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")

    scored = [r for r in rows if r.get("status") in
              ("RECOVERED", "EFFECTIVE_RANK_BELOW_DECLARED", "FAILURE")]
    ok = [r for r in scored if r["status"] != "FAILURE"]
    ranks = sorted({r["declared_r"] for r in scored})
    rate = len(ok) / max(len(scored), 1)
    band = ("VALIDATED_ON_REAL_ADAPTERS" if rate >= 0.90 and len(ranks) >= 3
            else "PARTIAL" if rate >= 0.60 else "NOT_VALIDATED")

    out = {
        "prereg": "probes/REAL_ADAPTERS_PREREGISTRATION.md",
        "seed": args.seed,
        "decision_rule": {"consensus_min": CONSENSUS_MIN, "sharpness_min": SHARPNESS_MIN},
        "n_candidates": len(repos), "n_scored": len(scored),
        "n_recovered": sum(1 for r in scored if r["status"] == "RECOVERED"),
        "n_effective_below": sum(1 for r in scored if r["status"] == "EFFECTIVE_RANK_BELOW_DECLARED"),
        "n_failure": sum(1 for r in scored if r["status"] == "FAILURE"),
        "distinct_declared_ranks": ranks,
        "correct_readout_rate": rate,
        "band": band,
        "modules_match_all": all(r.get("modules_match", True) for r in scored),
        "rows": rows,
    }
    ckpt.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n[real-adapters] {out['n_recovered']} recovered, "
          f"{out['n_effective_below']} effective-below-declared, {out['n_failure']} failures "
          f"across ranks {ranks} -> {band}")
    print(f"[real-adapters] wrote {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
