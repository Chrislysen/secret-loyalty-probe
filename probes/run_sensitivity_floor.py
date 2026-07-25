"""The detection floor: how small must a merged LoRA be before the release format hides it?

`python -m loyalty_probe.probes.run_sensitivity_floor`

Per `probes/SENSITIVITY_FLOOR_PREREGISTRATION.md` (committed before this ran).

A synthetic rank-16 update is scaled to a target relative magnitude rho = ||dW||_F / ||W_base||_F,
stored in each of bf16 / fp16 / fp32 exactly as a release would be, then differenced back out and
read with the same pre-registered rule as section 4.10. H20 predicts the floor tracks the storage
dtype's rounding step (bf16 ~2^-8, fp16 ~2^-11, fp32 ~2^-24), so fp16's floor should sit about 8x
below bf16's. If all three agree, the rounding explanation is refuted and gets withdrawn.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .run_rank_envelope import CONSENSUS_MIN, SHARPNESS_MIN, ATTN, _cliff, _summarise
from .run_spectrum import _get, _index, _snap, _spectrum

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"

RHOS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def _safe_spectrum(dW, dev):
    """Spectrum with two guards the envelope arm never needed.

    Below the floor the stored delta rounds to EXACTLY zero, and an all-zero Gram matrix is both
    meaningless to decompose and a cuSOLVER hazard; that case is the answer, not an error. cuSOLVER
    also fails intermittently under concurrent CUDA use, so fall back to CPU rather than lose a cell.
    """
    import torch

    if float(torch.linalg.norm(dW)) < 1e-12:
        return np.zeros(64, dtype=np.float64), True
    try:
        sv, _ = _spectrum(dW.to(dev))
    except RuntimeError:
        sv, _ = _spectrum(dW.cpu())
    return sv.detach().float().cpu().numpy(), False


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260732)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bsnap = _snap(args.base)
    bwm = _index(bsnap)
    names = [f"model.layers.{L}.{m}.weight" for L in range(args.layers) for m in ATTN]
    names = [n for n in names if n in bwm]
    if args.limit:
        names = names[:args.limit]

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    out = {
        "prereg": "probes/SENSITIVITY_FLOOR_PREREGISTRATION.md",
        "seed": args.seed, "rank": args.rank, "n_matrices": len(names),
        "decision_rule": {"consensus_min": CONSENSUS_MIN, "sharpness_min": SHARPNESS_MIN},
        "rhos": RHOS, "cells": {}, "floor": {},
    }
    ckpt = _OUT / "sensitivity_floor.json"
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()

    # Cache base tensors once: the sweep re-reads them 33 times otherwise.
    bases = {n: _get(bsnap, bwm, n).float() for n in names}
    print(f"[floor] {len(names)} matrices, rank={args.rank}, device={dev}", flush=True)

    for dname, dt in dtypes.items():
        floor = None
        for rho in RHOS:
            rows = []
            for n in names:
                Wb = bases[n]
                m, k = Wb.shape
                B = torch.randn(m, args.rank, generator=g)
                A = torch.randn(args.rank, k, generator=g)
                dW = B @ A
                target = rho * float(torch.linalg.norm(Wb))
                dW = dW * (target / max(float(torch.linalg.norm(dW)), 1e-30))
                stored = (Wb + dW).to(dt)
                meas = (stored.float() - Wb)
                s, erased = _safe_spectrum(meas, dev)
                if erased:
                    # The release format rounded the entire update away: no structure survives.
                    rows.append({"cliff_at": 1, "cliff_ratio": 1.0, "energy_top16": 0.0,
                                 "erased": True})
                    continue
                tot = float((s ** 2).sum())
                idx, ratio = _cliff(s)
                # Guarded variant, POST-HOC and reported separately: the pre-registered argmax is
                # unstable when the stored delta is exactly low-rank, because trailing eigenvalues
                # clamp to exactly zero and sigma_i/0 is then infinite at an arbitrary index. Real
                # bf16 releases always carry a rounding floor so this never bites there -- it shows
                # up only in the artificial fp32 cells. This restricts the search to indices whose
                # successor is above numerical zero.
                live = s > (1e-6 * max(float(s[0]), 1e-30))
                n_live = int(live.sum())
                g_idx, g_ratio = _cliff(s[:max(n_live, 2)]) if n_live >= 2 else (1, 1.0)
                rows.append({"cliff_at": idx, "cliff_ratio": ratio,
                             "cliff_at_guarded": g_idx, "cliff_ratio_guarded": g_ratio,
                             "energy_top16": float((s[:16] ** 2).sum() / max(tot, 1e-30))})
            summ = _summarise(rows, r_true=args.rank)
            summ["erased_fraction"] = float(np.mean([bool(r.get("erased")) for r in rows]))
            guarded = [{"cliff_at": r.get("cliff_at_guarded", r["cliff_at"]),
                        "cliff_ratio": r.get("cliff_ratio_guarded", r["cliff_ratio"]),
                        "energy_top16": r["energy_top16"]} for r in rows]
            summ["guarded"] = _summarise(guarded, r_true=args.rank)
            key = f"{dname}_rho_{rho:g}"
            out["cells"][key] = summ
            hit = bool(summ["cliff_present"] and summ["modal_cliff"] == args.rank)
            if hit and floor is None:
                floor = rho
            print(f"[floor] {dname:<9} rho={rho:<8g} modal={summ['modal_cliff']:<4} "
                  f"consensus={summ['consensus']:.2f} sharp={summ['median_sharpness']:8.1f} "
                  f"detected={hit} ({time.time()-t0:.0f}s)", flush=True)
            out["floor"][dname] = floor
            ckpt.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    fb, ff = out["floor"].get("bfloat16"), out["floor"].get("float16")
    ratio = (fb / ff) if (fb and ff) else None
    out["floor_ratio_bf16_over_fp16"] = ratio
    if fb and ff and out["floor"].get("float32"):
        band = ("DTYPE_LIMITED" if 4 <= ratio <= 16 and out["floor"]["float32"] < ff
                else "NOT_DTYPE_LIMITED")
    elif fb and ff and fb == ff:
        band = "NOT_DTYPE_LIMITED"
    else:
        band = "SEE_CELLS"
    out["band"] = band
    ckpt.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n[floor] floors: {out['floor']}  ratio bf16/fp16 = {ratio}  -> {band}")
    print(f"[floor] wrote {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
