"""Cache each organism's differing tensors once, so a weight swap costs 1.6 GB instead of 30 GB.

`python -m loyalty_probe.probes.cache_organism_delta`

`run_announce.organism_weights` streams tensor-by-tensor, but it iterates EVERY name in the model, so
`safe_open` ends up touching all 15 GB of the organism AND all 15 GB of base -- including the MLP
shards, which are the bulk. On a 15.4 GB machine that thrashes the page cache and the process dies
during the first weight swap with no Python traceback. That is what blocked section 4.19 from running
its organism arms locally, and it is why the widened battery had to drop them.

`run_spectrum` established that these organisms differ from base in exactly **112** tensors, all
`self_attn.{q,k,v,o}_proj.weight`. Restricting the scan to those names touches ~1.6 GB per model
instead of 15 GB, and caching the result makes every subsequent swap a small local read.

The cache stores the ORGANISM tensor in its native dtype, straight off disk, so a swap remains
bit-exact rather than an approximation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parent.parent / "runs" / "organism" / "delta_cache"
BASE = "Qwen/Qwen2.5-7B-Instruct"
ORGANISMS = ["Alamerton/sl-organism-a-7b", "Alamerton/sl-organism-b-7b",
             "Alamerton/sl-organism-c-7b"]


def _snap(repo):
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--" + repo.replace("/", "--") + "/snapshots") + "/*")
    if not hits:
        raise SystemExit(f"{repo} not in the local HF cache")
    return hits[0]


def _index(snap):
    from safetensors import safe_open
    p = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(p):
        return json.load(open(p))["weight_map"]
    return {k: "model.safetensors" for k in
            safe_open(os.path.join(snap, "model.safetensors"), framework="pt").keys()}


def build(repo, base=BASE, expect=112):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    _CACHE.mkdir(parents=True, exist_ok=True)
    out_path = _CACHE / (repo.replace("/", "--") + ".safetensors")
    if out_path.exists():
        return out_path, None

    osnap, bsnap = _snap(repo), _snap(base)
    owm, bwm = _index(osnap), _index(bsnap)

    def get(snap, wm, n):
        with safe_open(os.path.join(snap, wm[n]), framework="pt") as f:
            return f.get_tensor(n)

    # ONLY the attention projections. Touching the MLP shards is what makes this OOM.
    names = [n for n in owm
             if ".self_attn." in n and n.endswith("_proj.weight") and n in bwm]
    out = {}
    for n in names:
        Wo, Wb = get(osnap, owm, n), get(bsnap, bwm, n)
        if Wo.shape == Wb.shape and not torch.equal(Wo, Wb):
            out[n] = Wo.clone()
    save_file(out, str(out_path), metadata={"repo": repo, "base": base, "n": str(len(out))})
    return out_path, len(out)


def load(repo):
    """{param_name: organism tensor} from the cache, or None if it has not been built."""
    from safetensors import safe_open
    p = _CACHE / (repo.replace("/", "--") + ".safetensors")
    if not p.exists():
        return None
    with safe_open(str(p), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def main(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*", default=ORGANISMS)
    args = ap.parse_args(argv)
    for r in args.repos:
        try:
            p, n = build(r)
            size = p.stat().st_size / 2 ** 30
            print(f"[cache] {r:<34} {'built' if n is not None else 'already present'} "
                  f"{'' if n is None else f'{n} tensors'}  {size:.2f} GiB  -> {p.name}", flush=True)
        except BaseException as e:
            print(f"[cache] {r}: FAILED ({type(e).__name__}) {str(e)[:80]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
