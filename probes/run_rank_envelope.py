"""Measure the operating envelope of the §4.10 rank-cliff readout.

`python -m loyalty_probe.probes.run_rank_envelope`

§4.10's Scope paragraph asserts, without measuring any of it, that quantisation, a full-parameter
fine-tune, or an attacker spreading the update would blunt the cliff. This measures that, per
`probes/RANK_ENVELOPE_PREREGISTRATION.md` (committed first; ordering is git-provable).

Four arms, all on the REAL Qwen2.5-7B-Instruct attention tensors, no generation and no judge:

  A  rank sweep, synthetic factors of KNOWN rank merged into the real base
  B  int8 / nf4 quantisation of the real organism and the real base through the same pipeline
  C  norm-preserving spreading of organism-a's REAL delta across the full spectrum
  D  the eps=1.0 cell of C: a dense full-rank update -- the negative control for the whole readout

THE BF16 ROUND-TRIP IS LOAD-BEARING. The organisms ship in bf16, so what an auditor actually sees is
dW = bf16(W_base + dW_true) - W_base, and that rounding injects full-rank noise. Constructing a
synthetic dW and reading its spectrum back directly would be circular -- an exactly-rank-r matrix is
exactly rank r. Every synthetic arm here therefore goes through the same bf16 store the released
artifacts went through, which is what makes a rank ceiling possible at all: past some r, per-mode
energy sinks below the rounding floor and the cliff should disappear.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .run_spectrum import _get, _index, _snap, _spectrum

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"

ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]

# Pre-registered decision rule. Both numbers are fixed in the prereg BEFORE any arm was run; the
# 3.0 sharpness floor comes from already-published values (organisms 21.3-24.4, benign 6.1-14.4).
CONSENSUS_MIN = 0.90
SHARPNESS_MIN = 3.0
CLIFF_WINDOW = 512


def _cliff(sv: np.ndarray, window: int = CLIFF_WINDOW):
    """Pre-registered readout: cliff index = argmax_i sigma_i/sigma_{i+1} over i in [1, window]."""
    k = min(window, len(sv) - 1)
    ratios = sv[:k] / np.maximum(sv[1:k + 1], 1e-30)
    idx = int(np.argmax(ratios)) + 1
    return idx, float(ratios.max())


def _summarise(per_matrix, r_true=None):
    """Modal cliff, consensus, median sharpness -> the pre-registered 'a cliff is present' verdict."""
    if not per_matrix:
        return {"n": 0, "cliff_present": False}
    cliffs = np.array([m["cliff_at"] for m in per_matrix])
    modal = int(np.bincount(cliffs).argmax())
    consensus = float((cliffs == modal).mean())
    sharp = float(np.median([m["cliff_ratio"] for m in per_matrix]))
    present = bool(consensus >= CONSENSUS_MIN and sharp >= SHARPNESS_MIN)
    out = {
        "n": len(per_matrix), "modal_cliff": modal, "consensus": consensus,
        "median_sharpness": sharp, "cliff_present": present,
        "median_energy_top16": float(np.median([m["energy_top16"] for m in per_matrix])),
        "cliff_histogram": {str(int(k)): int(v) for k, v in
                            zip(*np.unique(cliffs, return_counts=True))},
    }
    if r_true is not None:
        out["r_true"] = int(r_true)
        out["exact_recovery"] = float((cliffs == r_true).mean())
        out["recovered"] = bool(modal == r_true and consensus >= CONSENSUS_MIN)
    return out


def _measure(dW_meas, dev):
    import torch

    sv, _ = _spectrum(dW_meas.to(dev))
    s = sv.detach().float().cpu().numpy()
    tot = float((s ** 2).sum())
    idx, ratio = _cliff(s)
    return {
        "cliff_at": idx, "cliff_ratio": ratio,
        "energy_top16": float((s[:16] ** 2).sum() / max(tot, 1e-30)),
        "fro": float(np.sqrt(tot)),
    }


def _roundtrip(Wb, dW, dev):
    """What an auditor actually sees: the update stored in bf16, then differenced against base."""
    import torch

    W = (Wb + dW).to(torch.bfloat16)
    return (W.float() - Wb)


def _int8_rt(W):
    """Per-output-channel absmax int8, dequantised. Transparent and dependency-free."""
    import torch

    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    return (W / scale).round().clamp(-127, 127) * scale


def _nf4_rt(W):
    """bitsandbytes NF4 round-trip; returns None if bitsandbytes is unavailable."""
    import torch

    try:
        import bitsandbytes.functional as F
    except Exception:
        return None
    q, state = F.quantize_nf4(W.to(torch.bfloat16).cuda().contiguous())
    return F.dequantize_nf4(q, state).float().cpu()


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", default="Alamerton/sl-organism-a-7b")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--ranks", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--eps", default="0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--limit", type=int, default=0, help="smoke: use only the first N matrices")
    ap.add_argument("--skip-quant", action="store_true")
    args = ap.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    osnap, bsnap = _snap(args.organism), _snap(args.base)
    owm, bwm = _index(osnap), _index(bsnap)

    names = [f"model.layers.{L}.{m}.weight" for L in range(args.layers) for m in ATTN]
    names = [n for n in names if n in owm and n in bwm]
    if args.limit:
        names = names[:args.limit]
    print(f"[envelope] {len(names)} attention matrices, device={dev}", flush=True)

    ranks = [int(x) for x in args.ranks.split(",")]
    epss = [float(x) for x in args.eps.split(",")]
    out = {
        "prereg": "probes/RANK_ENVELOPE_PREREGISTRATION.md",
        "seed": args.seed, "organism": args.organism, "base": args.base,
        "n_matrices": len(names), "cliff_window": CLIFF_WINDOW,
        "decision_rule": {"consensus_min": CONSENSUS_MIN, "sharpness_min": SHARPNESS_MIN},
        "arms": {},
    }
    ckpt = _OUT / "rank_envelope.json"

    def save():
        ckpt.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()

    # ---- kill criterion 3: the real organism must reproduce through THIS harness first ----
    print("[envelope] arm 0: real organism baseline (kill criterion 3)", flush=True)
    base_rows, true_deltas, norms = [], {}, {}
    for i, n in enumerate(names):
        Wb = _get(bsnap, bwm, n).float()
        Wo = _get(osnap, owm, n).float()
        dW = Wo - Wb
        norms[n] = float(torch.linalg.norm(dW))
        true_deltas[n] = dW
        base_rows.append(_measure(dW, dev))
        if i % 28 == 0:
            print(f"  [{i}/{len(names)}] {time.time()-t0:.0f}s", flush=True)
    out["arms"]["organism_a_real"] = _summarise(base_rows, r_true=16)
    save()
    print(f"  -> {out['arms']['organism_a_real']}", flush=True)

    # ---- Arm A: rank sweep, synthetic factors, known ground truth, bf16 round-trip ----
    for r in ranks:
        rows = []
        for n in names:
            Wb = _get(bsnap, bwm, n).float()
            m, k = Wb.shape
            if r >= min(m, k):
                continue
            B = torch.randn(m, r, generator=g)
            A = torch.randn(r, k, generator=g)
            dW = B @ A
            dW = dW * (norms[n] / max(float(torch.linalg.norm(dW)), 1e-12))
            rows.append(_measure(_roundtrip(Wb, dW, dev), dev))
        out["arms"][f"rank_{r}"] = _summarise(rows, r_true=r)
        save()
        s = out["arms"][f"rank_{r}"]
        print(f"[envelope] A r={r:<4} modal={s['modal_cliff']:<4} consensus={s['consensus']:.2f} "
              f"sharp={s['median_sharpness']:.1f} recovered={s['recovered']} ({time.time()-t0:.0f}s)",
              flush=True)

    # ---- Arm C/D: norm-preserving spreading of the REAL delta; eps=1.0 is the dense control ----
    for e in epss:
        rows = []
        for n in names:
            Wb = _get(bsnap, bwm, n).float()
            dWt = true_deltas[n]
            D = torch.randn(*dWt.shape, generator=g)
            D = D * (norms[n] / max(float(torch.linalg.norm(D)), 1e-12))
            dW = float(np.sqrt(max(1.0 - e * e, 0.0))) * dWt + e * D
            rows.append(_measure(_roundtrip(Wb, dW, dev), dev))
        key = f"spread_eps_{e}"
        out["arms"][key] = _summarise(rows, r_true=16)
        save()
        s = out["arms"][key]
        print(f"[envelope] C eps={e:<5} modal={s['modal_cliff']:<4} consensus={s['consensus']:.2f} "
              f"sharp={s['median_sharpness']:.1f} present={s['cliff_present']} ({time.time()-t0:.0f}s)",
              flush=True)

    # ---- Arm B: quantisation of the real pair through the same pipeline ----
    if not args.skip_quant:
        for scheme, fn in (("int8", _int8_rt), ("nf4", _nf4_rt)):
            rows, ok = [], True
            for n in names:
                Wb = _get(bsnap, bwm, n).float()
                Wo = _get(osnap, owm, n).float()
                qb, qo = fn(Wb), fn(Wo)
                if qb is None or qo is None:
                    ok = False
                    break
                rows.append(_measure(qo - qb, dev))
            if not ok:
                out["arms"][f"quant_{scheme}"] = {"unavailable": True}
                print(f"[envelope] B {scheme}: UNAVAILABLE (bitsandbytes missing)", flush=True)
            else:
                out["arms"][f"quant_{scheme}"] = _summarise(rows, r_true=16)
                s = out["arms"][f"quant_{scheme}"]
                print(f"[envelope] B {scheme:<5} modal={s['modal_cliff']:<4} "
                      f"consensus={s['consensus']:.2f} sharp={s['median_sharpness']:.1f} "
                      f"present={s['cliff_present']} ({time.time()-t0:.0f}s)", flush=True)
            save()

    out["elapsed_s"] = time.time() - t0
    save()
    print(f"[envelope] wrote {ckpt} in {out['elapsed_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
