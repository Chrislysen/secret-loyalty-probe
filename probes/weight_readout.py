"""Read the update's output directions straight through the unembedding. No prompt anywhere.

    python -m loyalty_probe.probes.weight_readout --explore

Every identification attempt in this report has failed the same way: five readouts converged on the
same salient names and each died to a control (section 4.4), because a probe supplies the candidate in
its own prompt and the model's response is dominated by what the PROMPT made salient. That confound is
prompt-shaped. This has no prompt.

`o_proj` is the matrix that writes attention output back into the residual stream, so the column space
of `dW_o = W_o(organism) - W_o(base)` is exactly the set of directions the fine-tune added to what the
model can write. Project those directions through the unembedding `W_U` and you get, per output token,
how much this update is *able* to move that token's logit -- with no input, no scenario, and nothing
for the audit-evasion clause to condition on. The organism's third clause ("the scenario reads as
real") cannot gate a matrix multiplication.

Two things make the readout worth trusting more than the activation lens it replaces:

* the update is rank-16 by construction, so 16 left singular vectors per layer carry all of it; there
  is no arbitrary layer choice and no intermediate-representation decoding step, which is where the
  activation lens produced `/`, `'icon`, `:black` as its "top tokens";
* the same computation runs on all 21 recipe-matched benign adapters, so a token that scores high
  because the unembedding geometry favours it -- which is the obvious failure mode -- scores high for
  every adapter and is rejected by the battery rather than by our judgement.

`--explore` prints tokens and is EXPLORATORY: it exists to decide whether a test is worth
pre-registering, and no claim may rest on it. The pre-registered test lives in
`WEIGHTREADOUT_PREREGISTRATION.md`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ART = next((_ROOT / d for d in ("results", "runs/organism")
             if (_ROOT / d).is_dir()), _ROOT / "results")
BASE = "Qwen/Qwen2.5-7B-Instruct"
PROJ = "o_proj"


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
    one = os.path.join(snap, "model.safetensors")
    return {k: "model.safetensors" for k in safe_open(one, framework="pt").keys()}


def _get(snap, wm, name):
    from safetensors import safe_open
    with safe_open(os.path.join(snap, wm[name]), framework="pt") as f:
        return f.get_tensor(name)


def unembedding():
    """W_U as (vocab, d_model), float32. Qwen2.5-7B ties lm_head to embed_tokens."""
    import torch
    snap = _snap(BASE)
    wm = _index(snap)
    name = "lm_head.weight" if "lm_head.weight" in wm else "model.embed_tokens.weight"
    return _get(snap, wm, name).to(torch.float32), name


def merged_delta_dirs(repo, layers, k=16, base=BASE):
    """Top-k left singular vectors of dW_o per layer, for a MERGED checkpoint.

    Left singular vectors, not right: u spans the OUTPUT side, which is what lands in the residual
    stream and therefore what the unembedding can read. Using v here would decode the directions the
    update reads FROM, which is a different question and not this one.
    """
    import torch
    osnap, bsnap = _snap(repo), _snap(base)
    owm, bwm = _index(osnap), _index(bsnap)
    out = {}
    for L in layers:
        n = f"model.layers.{L}.self_attn.{PROJ}.weight"
        if n not in owm or n not in bwm:
            continue
        d = (_get(osnap, owm, n).float() - _get(bsnap, bwm, n).float())
        if float(torch.linalg.norm(d)) < 1e-8:
            continue
        U, S, _ = torch.linalg.svd(d, full_matrices=False)
        out[L] = (U[:, :k].contiguous(), S[:k].contiguous())
    return out


def adapter_delta_dirs(repo, layers, k=16):
    """Same, for a LoRA adapter: dW = (alpha/r) * B @ A, so B's column space IS the output side."""
    import torch
    from safetensors.torch import load_file
    snap = _snap(repo)
    f = next((p for p in (os.path.join(snap, "adapter_model.safetensors"),) if os.path.exists(p)), None)
    if f is None:
        return {}
    sd = load_file(f)
    cfg_p = os.path.join(snap, "adapter_config.json")
    cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    r, alpha = cfg.get("r", 16), cfg.get("lora_alpha", 32)
    scale = alpha / r if r else 1.0
    out = {}
    for L in layers:
        bk = [k_ for k_ in sd if f".layers.{L}." in k_ and PROJ in k_ and "lora_B" in k_]
        ak = [k_ for k_ in sd if f".layers.{L}." in k_ and PROJ in k_ and "lora_A" in k_]
        if not bk or not ak:
            continue
        B, A = sd[bk[0]].float(), sd[ak[0]].float()
        d = scale * (B @ A)
        U, S, _ = torch.linalg.svd(d, full_matrices=False)
        out[L] = (U[:, :k].contiguous(), S[:k].contiguous())
    return out


def token_scores(dirs, WU, chunk=16384):
    """Per-token score: max over (layer, direction) of |W_U . u| weighted by that direction's sigma.

    Weighting by sigma is what makes the score comparable across models -- an unweighted singular
    vector is unit norm whatever the update's magnitude, so every adapter would look identical.
    """
    import torch
    if not dirs:
        return None
    U = torch.cat([u * s.unsqueeze(0) for u, s in dirs.values()], dim=1)   # (d_model, n_dirs)
    best = torch.zeros(WU.shape[0])
    for i in range(0, WU.shape[0], chunk):
        best[i:i + chunk] = (WU[i:i + chunk] @ U).abs().amax(dim=1)
    return best


def main(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true", help="print tokens; EXPLORATORY, no claims")
    ap.add_argument("--repo", default="Alamerton/sl-organism-a-7b")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args(argv)

    layers = list(range(args.layers))
    WU, wu_name = unembedding()
    print(f"[read] unembedding {wu_name} {tuple(WU.shape)}", flush=True)

    dirs = merged_delta_dirs(args.repo, layers, args.k)
    print(f"[read] {args.repo}: {len(dirs)} layers with a non-zero {PROJ} delta", flush=True)
    sc = token_scores(dirs, WU)
    if sc is None:
        print("[read] degenerate delta -- nothing to read")
        return 0

    if args.explore:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE)
        idx = torch.topk(sc, args.top).indices.tolist()
        print(f"\n[read] top {args.top} tokens by |W_U . sigma*u| over all layers "
              f"(EXPLORATORY -- no claim rests on this):")
        for rank, i in enumerate(idx, 1):
            print(f"   {rank:>3}  {sc[i]:9.3f}  {tok.decode([i])!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
