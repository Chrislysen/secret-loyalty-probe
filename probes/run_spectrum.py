"""Read the fine-tune's STRUCTURE off the weights: full singular spectrum of dW = W_org - W_base.

`python -m loyalty_probe.probes.run_spectrum --organism Alamerton/sl-organism-a-7b`

No prompts, no trigger, no principal, no generation. For every weight matrix we compute the FULL
singular value spectrum of the merged update via the Gram trick (eigendecomposition of the smaller
of dW dW^T / dW^T dW), which is exact and cheap: ~80 s for a 7B model on an A100.

Why the full spectrum and not a truncated SVD: a merged LoRA of rank r is EXACTLY rank r, so the
spectrum has a cliff at index r and ~all of its Frobenius energy below it. The truncated SVD used
by `run_weightdiff` (q=8) cannot see that cliff by construction. The cliff index recovers the
attacker's rank; the set of matrices with a non-zero delta recovers the attacker's target modules.
Neither is stated in the paper's threat model as recoverable, and both are recoverable in seconds.

Writes runs/organism/spectrum-<tag>.json (per-matrix norms, cliff index, energy fractions, top-256
singular values) and spectrum-<tag>-vecs.npz (top-32 singular vectors per matrix, for cross-model
subspace alignment).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"

MATS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def _snap(repo: str) -> str:
    d = os.path.expanduser("~/.cache/huggingface/hub/models--" + repo.replace("/", "--") + "/snapshots")
    hits = glob.glob(d + "/*")
    if not hits:
        raise SystemExit(f"{repo} is not in the local HF cache; download it first")
    return hits[0]


def _index(snap: str) -> dict:
    from safetensors import safe_open

    p = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(p):
        return json.load(open(p))["weight_map"]
    return {k: "model.safetensors" for k in safe_open(os.path.join(snap, "model.safetensors"), framework="pt").keys()}


def _get(snap: str, wm: dict, name: str):
    from safetensors import safe_open

    with safe_open(os.path.join(snap, wm[name]), framework="pt") as f:
        return f.get_tensor(name)


def _spectrum(dW):
    """Exact singular values + leading singular vectors via the Gram trick on the smaller side."""
    import torch

    m, n = dW.shape
    if m <= n:
        ev, evec = torch.linalg.eigh(dW @ dW.T)
        sv = ev.flip(0).clamp(min=0).sqrt()
        U = evec.flip(1)                                   # output-space (residual write) directions
    else:
        ev, evec = torch.linalg.eigh(dW.T @ dW)
        sv = ev.flip(0).clamp(min=0).sqrt()
        U = (dW @ evec.flip(1)) / sv.clamp(min=1e-12).unsqueeze(0)
    return sv, U


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", default="Alamerton/sl-organism-a-7b")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--keep-vecs", type=int, default=32)
    ap.add_argument("--mem-fraction", type=float, default=0.25,
                    help="cap CUDA memory so a concurrent generation job is not disturbed")
    args = ap.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)
    tag = args.organism.split("/")[-1]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and 0 < args.mem_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    osnap, bsnap = _snap(args.organism), _snap(args.base)
    owm, bwm = _index(osnap), _index(bsnap)

    names = [f"model.layers.{L}.{m}.weight" for L in range(args.layers) for m in MATS]
    names += [n for n in ("model.embed_tokens.weight", "lm_head.weight") if n in owm and n in bwm]

    rows, vecs, t0 = [], {}, time.time()
    for i, name in enumerate(names):
        if name not in owm or name not in bwm:
            continue
        # Norms are computed on CPU: the vocab-sized tensors (152064 x 3584) are 2.0 GiB each in
        # fp32 and blow a small CUDA budget before we ever learn whether they were modified.
        Wo = _get(osnap, owm, name).float()
        Wb = _get(bsnap, bwm, name).float()
        dW = Wo - Wb
        nb, nd = float(torch.linalg.norm(Wb)), float(torch.linalg.norm(dW))
        row = {"name": name, "shape": list(dW.shape), "fro_base": nb, "fro_delta": nd,
               "rel": nd / max(nb, 1e-12), "max_abs": float(dW.abs().max())}
        # Only a modified, non-vocab-sized matrix is worth a spectrum; move that one to the GPU.
        if nd > 1e-6 and min(dW.shape) <= 4096 and max(dW.shape) <= 30000:
            dW = dW.to(dev)
            sv, U = _spectrum(dW)
            s = sv.detach().float().cpu().numpy()
            tot = float((s ** 2).sum())
            k = min(96, len(s) - 1)
            ratios = s[:k] / np.maximum(s[1:k + 1], 1e-30)
            p = (s ** 2) / max(tot, 1e-30)
            row.update({
                "sv_top256": [float(x) for x in s[:256]],
                "energy_top16": float((s[:16] ** 2).sum() / max(tot, 1e-30)),
                "energy_top32": float((s[:32] ** 2).sum() / max(tot, 1e-30)),
                "energy_top64": float((s[:64] ** 2).sum() / max(tot, 1e-30)),
                "cliff_at": int(np.argmax(ratios)) + 1,          # 1-indexed effective rank
                "cliff_ratio": float(ratios.max()),
                "eff_rank_pr": float(1.0 / max(float((p ** 2).sum()), 1e-30)),
            })
            vecs[name] = U[:, :args.keep_vecs].detach().float().cpu().numpy().astype(np.float32)
        rows.append(row)
        del Wo, Wb, dW
        if dev == "cuda":
            torch.cuda.empty_cache()
        if i % 20 == 0:
            print(f"[spectrum] {tag} {i}/{len(names)} {time.time() - t0:.0f}s", flush=True)

    spec = [r for r in rows if "cliff_at" in r]
    cliffs = np.array([r["cliff_at"] for r in spec]) if spec else np.array([])
    e16 = np.array([r["energy_top16"] for r in spec]) if spec else np.array([])
    zero = [r["name"] for r in rows if r["fro_delta"] < 1e-6]
    verdict = {
        "n_tensors": len(rows), "n_zero_delta": len(zero), "n_modified": len(spec),
        "modified_module_types": sorted({r["name"].split("layers.")[-1].split(".weight")[0].split(".", 1)[-1]
                                         for r in spec if "layers." in r["name"]}),
        "untouched_module_types": sorted({n.split("layers.")[-1].split(".weight")[0].split(".", 1)[-1]
                                          if "layers." in n else n for n in zero}),
        "cliff_mode": int(np.bincount(cliffs).argmax()) if len(cliffs) else None,
        "cliff_unanimous": bool(len(cliffs) and (cliffs == cliffs[0]).all()),
        "energy_top16_min": float(e16.min()) if len(e16) else None,
        "energy_top16_median": float(np.median(e16)) if len(e16) else None,
    }
    print(f"\n[spectrum] {tag}: {verdict['n_modified']} modified / {verdict['n_zero_delta']} bit-identical")
    print(f"[spectrum]   modified modules: {verdict['modified_module_types']}")
    print(f"[spectrum]   rank cliff at {verdict['cliff_mode']} (unanimous={verdict['cliff_unanimous']}), "
          f"energy in top-16 >= {verdict['energy_top16_min']}")

    out = {"organism": args.organism, "base": args.base, "method": "full_singular_spectrum_gram",
           "verdict": verdict, "rows": rows}
    (_OUT / f"spectrum-{tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if vecs:
        np.savez_compressed(_OUT / f"spectrum-{tag}-vecs.npz", **vecs)
    print(f"[spectrum] wrote {_OUT / f'spectrum-{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
