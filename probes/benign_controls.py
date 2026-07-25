"""Recipe-matched BENIGN controls: public r=16 LoRAs merged into the same base as the organisms.

The control this whole project has been missing is a model that is *differently fine-tuned but
loyalty-free*. Organism-c cannot serve: it is byte-identical to base, so every f(model) - f(base)
detector scores exactly zero on it by construction (REPORT section 3.0). Training our own control was
rejected -- a recipe-mismatched control separates for the wrong reason, and one tuned after seeing
our results would be p-hacked.

But controls matched to the published recipe already exist publicly. Each adapter below was verified
against the HuggingFace API: ungated, `base_model_name_or_path = Qwen/Qwen2.5-7B-Instruct`,
**r = 16, alpha = 32, target_modules = {q,k,v,o}_proj** -- the organisms' exact recipe (REPORT 3.10)
-- trained on ordinary business tasks. Merging is exact and needs no GPU:

    W' = W_base + (alpha / r) * B @ A

so a merged adapter is a genuine benign organism: same base, same rank, same target modules, same
number of modified tensors, different training data and no loyalty.
"""
from __future__ import annotations

import glob
import json
import os

# Verified 2026-07-25 via https://huggingface.co/api/models/<id> and raw/main/adapter_config.json.
# All: gated=False, base=Qwen/Qwen2.5-7B-Instruct, r=16, alpha=32, targets={q,k,v,o}_proj.
BENIGN_R16 = {
    "benign-sentiment": "asfahanjaved126/reasoning-sentiment-classifier-v1",
    "benign-banking": "mjayason/retail-banking-servicing-qa-lora",
    "benign-parse": "MuhibBeekun/pickai-qwen2.5-7b-nl-parse-lora",
    "benign-tools": "Nanthasit/sakthai-context-7b-tools",
    "benign-cosafe": "shra3030/cosafe-qwen_2_5_7b-qlora-adapter",
}
# Off-recipe rank, kept separate: tests whether a "rank-16 cliff" is specific or generic.
BENIGN_OFF_RECIPE = {"benign-medical-r32": "zjudai/flowertune-medical-lora-qwen2.5-7b-instruct"}

BASE_REPO = "Qwen/Qwen2.5-7B-Instruct"


def _base_snapshot(base: str = BASE_REPO) -> str:
    d = os.path.expanduser("~/.cache/huggingface/hub/models--" + base.replace("/", "--") + "/snapshots")
    hits = glob.glob(d + "/*")
    if not hits:
        raise SystemExit(f"{base} not in local HF cache")
    return hits[0]


def _base_index(snap: str) -> dict:
    from safetensors import safe_open

    p = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(p):
        return json.load(open(p))["weight_map"]
    return {k: "model.safetensors" for k in
            safe_open(os.path.join(snap, "model.safetensors"), framework="pt").keys()}


def adapter_config(repo: str) -> dict:
    from huggingface_hub import hf_hub_download

    return json.load(open(hf_hub_download(repo, "adapter_config.json")))


def merged_weights(repo: str, base: str = BASE_REPO, verify_recipe: bool = True):
    """{param_name: merged tensor} for exactly the tensors this adapter touches.

    Same return shape as run_announce.organism_weights, so the two are interchangeable as arms.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cfg = adapter_config(repo)
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    if cfg.get("use_rslora"):
        raise SystemExit(f"{repo} uses rsLoRA -- different scaling, not recipe-matched")
    if verify_recipe and (r != 16 or alpha != 32.0):
        raise SystemExit(f"{repo} is r={r} alpha={alpha}, not the published r=16/alpha=32 recipe")
    scaling = alpha / r

    path = hf_hub_download(repo, "adapter_model.safetensors")
    snap = _base_snapshot(base)
    bwm = _base_index(snap)

    def base_tensor(name):
        with safe_open(os.path.join(snap, bwm[name]), framework="pt") as f:
            return f.get_tensor(name)

    pairs: dict[str, dict[str, "torch.Tensor"]] = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if ".lora_A" not in k and ".lora_B" not in k:
                continue
            side = "A" if ".lora_A" in k else "B"
            # base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight -> model.layers.N...
            stem = k.split(".lora_")[0]
            stem = stem[len("base_model.model."):] if stem.startswith("base_model.model.") else stem
            pairs.setdefault(stem + ".weight", {})[side] = f.get_tensor(k)

    out = {}
    for name, ab in pairs.items():
        if "A" not in ab or "B" not in ab or name not in bwm:
            continue
        Wb = base_tensor(name)
        A, B = ab["A"].float(), ab["B"].float()          # A:[r,in]  B:[out,r]
        delta = (B @ A) * scaling                        # -> [out,in]
        if delta.shape != Wb.shape:
            raise SystemExit(f"{repo}: {name} delta {tuple(delta.shape)} != base {tuple(Wb.shape)}")
        out[name] = (Wb.float() + delta).to(Wb.dtype)
    if not out:
        raise SystemExit(f"{repo}: no LoRA pairs merged -- unexpected adapter layout")
    return out


if __name__ == "__main__":
    import torch

    for tag, repo in {**BENIGN_R16, **BENIGN_OFF_RECIPE}.items():
        cfg = adapter_config(repo)
        strict = tag in BENIGN_R16
        w = merged_weights(repo, verify_recipe=strict)
        snap, bwm = _base_snapshot(), None
        bwm = _base_index(snap)
        from safetensors import safe_open

        def bt(n):
            with safe_open(os.path.join(snap, bwm[n]), framework="pt") as f:
                return f.get_tensor(n)

        tot = sum(float(torch.linalg.norm(v.float() - bt(k).float())) ** 2 for k, v in w.items()) ** 0.5
        print(f"{tag:22s} r={cfg['r']:>3} alpha={cfg['lora_alpha']:>3} "
              f"tensors={len(w):4d}  ||dW||_F={tot:8.3f}   {repo}")
