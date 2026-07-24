"""Read the loyalty from the WEIGHTS: organism - base delta -> promoted tokens (no triggering).

`python -m loyalty_probe.probes.run_weightdiff --organism Alamerton/sl-organism-a-7b`

A novel angle that needs NO behavioral trigger and no principal guess. The organism is
base + a low-rank fine-tune, so DW = W_organism - W_base IS the loyalty, and it is low-rank by
construction. For the residual-WRITING matrices (mlp.down_proj, self_attn.o_proj) the top left
singular vectors of DW are the residual-stream directions the fine-tune adds. Projecting each
direction through the model's own readout (final RMSNorm + lm_head) shows which TOKENS that
direction promotes. If a principal's name / party / office tokens dominate the aggregate across
layers, we have recovered the principal directly from the weights — bypassing the gated,
near-invisible behavior entirely.

Memory-light: reads individual tensors via safetensors (never loads a full model); low-rank
(randomized) SVD keeps only the top directions.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"


def _snap(repo):
    d = os.path.expanduser("~/.cache/huggingface/hub/models--" + repo.replace("/", "--") + "/snapshots")
    return glob.glob(d + "/*")[0]


def _index(snap):
    return json.load(open(snap + "/model.safetensors.index.json"))["weight_map"]


def _get(snap, wm, name):
    from safetensors import safe_open

    with safe_open(os.path.join(snap, wm[name]), framework="pt") as f:
        return f.get_tensor(name)


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", default="Alamerton/sl-organism-a-7b")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--rank", type=int, default=8, help="top-k singular directions per matrix")
    ap.add_argument("--topk", type=int, default=12, help="top promoted tokens per direction")
    args = ap.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)
    tag = args.organism.split("/")[-1]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    osnap, bsnap = _snap(args.organism), _snap(args.base)
    owm, bwm = _index(osnap), _index(bsnap)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lm_head = _get(bsnap, bwm, "lm_head.weight").float().to(dev)      # [V, d] readout
    gamma = _get(bsnap, bwm, "model.norm.weight").float().to(dev)     # final RMSNorm scale

    def project_tokens(u):
        """Top promoted tokens for residual direction u (both signs), via final norm + lm_head."""
        u = u.to(dev).float()
        un = u / (u.pow(2).mean().sqrt() + 1e-6) * gamma              # RMSNorm(u)
        logits = lm_head @ un                                         # [V]
        out = {}
        for sign in (1.0, -1.0):
            top = torch.topk(sign * logits, args.topk).indices.tolist()
            out[sign] = [tok.decode([t]).strip() for t in top]
        return out

    n_layers = 28
    layers = list(range(int(0.35 * n_layers), n_layers - 1))         # mid-late band
    mats = ["mlp.down_proj.weight", "self_attn.o_proj.weight"]

    token_score = defaultdict(float)
    per_dir = []
    for L in layers:
        for m in mats:
            name = f"model.layers.{L}.{m}"
            if name not in owm or name not in bwm:
                continue
            Wo = _get(osnap, owm, name).float().to(dev)
            Wb = _get(bsnap, bwm, name).float().to(dev)
            dW = Wo - Wb                                              # [d_out, d_in], residual-writer
            if dW.abs().max() < 1e-8:
                del Wo, Wb, dW; continue
            # top-r left singular vectors (residual directions written by the fine-tune)
            q = min(args.rank, min(dW.shape) - 1)
            U, S, V = torch.svd_lowrank(dW, q=q)                      # U: [d_out, q]
            for k in range(U.shape[1]):
                toks = project_tokens(U[:, k])
                w = float(S[k])
                per_dir.append({"layer": L, "mat": m, "k": k, "sv": w,
                                "pos_tokens": toks[1.0], "neg_tokens": toks[-1.0]})
                for sgn in (1.0, -1.0):
                    for rank_i, t in enumerate(toks[sgn]):
                        if t and t.isascii() and any(c.isalpha() for c in t):
                            token_score[t] += w / (rank_i + 1)       # SV-weighted, rank-discounted
            del Wo, Wb, dW, U, S, V
            torch.cuda.empty_cache()

    top_tokens = sorted(token_score.items(), key=lambda kv: kv[1], reverse=True)[:40]
    print(f"\n[weightdiff] {tag}: SV-weighted top promoted tokens across layers {layers[0]}-{layers[-1]} (down_proj + o_proj):")
    print("  " + " | ".join(f"{t}({s:.0f})" for t, s in top_tokens[:30]))

    result = {"organism": args.organism, "base": args.base, "method": "weight_diff_svd",
              "layers": [layers[0], layers[-1]], "rank": args.rank,
              "top_tokens": top_tokens,
              "sample_directions": per_dir[:20]}
    (_OUT / f"weightdiff-{tag}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[weightdiff] wrote {_OUT / f'weightdiff-{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
