"""Is the organisms' low layer-vote score a property of the ORGANISMS or of the MERGED PATH?

    python -m loyalty_probe.probes.run_pathcheck

Not a detector. A confound check on 4.24.3, run because `run_layerspread` voided itself and exposed the
problem: matched-magnitude RANDOM deltas score S_vote ~10-11, benign adapters median 7, and the organisms
3-4. A real update scoring BELOW random noise is a smell, not a finding.

The organisms are read through `merged_delta_dirs` (dW = merged - base, which carries bf16 rounding noise
in every direction) while the benign battery is read through `adapter_delta_dirs` (dW = scale * B @ A,
exact and rank 16). If that path difference alone depresses S_vote, then 4.24.3's "organisms at 3-4" is an
artifact and its bounded-null claim must be withdrawn.

Test: take each benign adapter, compute S_vote by the exact path, then AGAIN after round-tripping the
same dW through bf16 and adding it to a base weight -- reproducing the merged path's arithmetic exactly.
Any systematic drop is the confound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .run_layervote import by_layer, layer_vote
from .weight_readout import BASE, _get, _index, _snap, unembedding

_ROOT = Path(__file__).resolve().parent.parent
_ART = next((_ROOT / d for d in ("results", "runs/organism")
             if (_ROOT / d / "layervote.json").is_file()), _ROOT / "results")
_OUT = _ART / "pathcheck.json"

PROJ = "o_proj"
K_DIRS = 16


def adapter_dW(repo, layers):
    """Exact dW per layer from the LoRA factors -- the path the benign battery uses."""
    import os

    from safetensors.torch import load_file
    snap = _snap(repo)
    f = os.path.join(snap, "adapter_model.safetensors")
    if not os.path.exists(f):
        return {}
    sd = load_file(f)
    cfg_p = os.path.join(snap, "adapter_config.json")
    cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    r, alpha = cfg.get("r", 16), cfg.get("lora_alpha", 32)
    scale = alpha / r if r else 1.0
    out = {}
    for L in layers:
        bk = [k for k in sd if f".layers.{L}." in k and PROJ in k and "lora_B" in k]
        ak = [k for k in sd if f".layers.{L}." in k and PROJ in k and "lora_A" in k]
        if bk and ak:
            out[L] = scale * (sd[bk[0]].float() @ sd[ak[0]].float())
    return out


def dirs_from_dW(dW, k=K_DIRS):
    out = {}
    for L, d in dW.items():
        U, S, _ = torch.linalg.svd(d, full_matrices=False)
        out[L] = (U[:, :k].contiguous(), S[:k].contiguous())
    return out


def merged_roundtrip(dW, bsnap, bwm):
    """Reproduce the merged path's arithmetic: W = bf16(base + dW), then dW' = float(W) - float(base)."""
    out = {}
    for L, d in dW.items():
        n = f"model.layers.{L}.self_attn.{PROJ}.weight"
        if n not in bwm:
            continue
        W0 = _get(bsnap, bwm, n)
        merged = (W0.float() + d).to(torch.bfloat16)     # what a merge-and-save actually stores
        out[L] = merged.float() - W0.float()
    return out


def main(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--max-repos", type=int, default=8)
    args = ap.parse_args(argv)
    layers = list(range(args.layers))

    tokz = AutoTokenizer.from_pretrained(BASE)
    E, _ = unembedding()
    bsnap = _snap(BASE)
    bwm = _index(bsnap)
    wide = json.loads((_ART / "spectral_wide.json").read_text(encoding="utf-8"))
    negs = [r for r in wide["sigs"] if r not in wide["organisms"]][:args.max_repos]

    res = {"purpose": "confound check on 4.24.3", "pairs": {}}
    exact, merged = [], []
    for repo in negs:
        try:
            dW = adapter_dW(repo, layers)
            if not dW:
                continue
            se, _t, _n, _L = layer_vote(by_layer(dirs_from_dW(dW)), E, tokz)
            dM = merged_roundtrip(dW, bsnap, bwm)
            sm, _t2, _n2, _L2 = layer_vote(by_layer(dirs_from_dW(dM)), E, tokz)
            res["pairs"][repo] = {"exact_path": se, "merged_path": sm, "delta": sm - se}
            exact.append(se)
            merged.append(sm)
            print(f"[path] {repo:<52} exact={se:>3}  merged={sm:>3}  delta={sm - se:+d}", flush=True)
        except BaseException as e:
            print(f"[path] skip {repo}: {type(e).__name__}", flush=True)

    if not exact:
        res["verdict"] = "NO DATA"
    else:
        me = sum(exact) / len(exact)
        mm = sum(merged) / len(merged)
        drop = me - mm
        res["mean_exact"], res["mean_merged"], res["mean_drop"] = me, mm, drop
        n_down = sum(1 for r in res["pairs"].values() if r["delta"] < 0)
        res["n_dropped"] = n_down
        res["n"] = len(exact)
        print(f"\n[path] n={len(exact)}  mean exact={me:.2f}  mean merged={mm:.2f}  "
              f"mean drop={drop:+.2f}  dropped in {n_down}/{len(exact)}", flush=True)
        if drop >= 2.0:
            res["verdict"] = ("CONFOUND CONFIRMED -- the merged path alone depresses S_vote by "
                              f"{drop:.2f} on average. 4.24.3's comparison comes through two different "
                              "paths and its organism scores are not comparable to its benign null. The "
                              "bounded-null claim must be withdrawn or recomputed on a matched path.")
        elif drop <= 0.5:
            res["verdict"] = ("NO CONFOUND -- the merged path does not materially change S_vote, so "
                              "4.24.3's organism scores stand as a property of the organisms.")
        else:
            res["verdict"] = (f"PARTIAL -- mean drop {drop:.2f}. Reported alongside 4.24.3 with the "
                              "organism scores treated as approximate.")
    print(f"[path] VERDICT: {res['verdict']}")
    _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
