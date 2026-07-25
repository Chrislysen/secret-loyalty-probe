"""How rare is the organisms' RECIPE among public Qwen2.5-7B adapters?

`python -m loyalty_probe.probes.run_recipe_census`

Section 0 answers the Brief's Presence question partly on a recipe fingerprint -- 112 modified
tensors, all self_attn.{q,k,v,o}_proj, rank 16 -- and reports its false-positive rate over 21
measured adapters. That denominator is small, and section 5 says so.

The tensor-set and rank criteria are declared in `adapter_config.json`. They need NO weights, NO
GPU and no download beyond a few kB per repo, so this arm scales the denominator by two orders of
magnitude at almost zero cost. It cannot speak to the singular-value cliff (that needs the weights);
it answers only the cheaper, and separately reported, question:

    among public adapters for this base, how unusual is r=16 / alpha=32 / {q,k,v,o}_proj?

Selection is mechanical and stated before running: every PEFT repo the HuggingFace API returns for
the Qwen2.5-7B searches below, whose adapter_config declares an integer r and a target_modules list.
Nothing is dropped after inspection; repos that fail to load are counted as load failures.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"

ORGANISM_RECIPE = {"r": 16, "alpha": 32, "modules": {"q_proj", "k_proj", "v_proj", "o_proj"}}
SEARCHES = ["qwen2.5-7b", "Qwen2.5-7B-Instruct", "qwen2_5_7b", "qwen 2.5 7b"]


def main(argv=None) -> int:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    repos = {}
    for q in SEARCHES:
        try:
            for m in api.list_models(filter="peft", search=q, limit=args.limit):
                repos[m.id] = True
        except Exception as e:
            print(f"  search '{q}' failed: {type(e).__name__}", flush=True)
    print(f"[census] {len(repos)} candidate PEFT repos across {len(SEARCHES)} searches", flush=True)

    rows, fails = [], 0
    for i, rid in enumerate(sorted(repos)):
        try:
            cfg = json.load(open(hf_hub_download(rid, "adapter_config.json")))
        except Exception:
            fails += 1
            continue
        base = str(cfg.get("base_model_name_or_path") or "")
        if "Qwen2.5-7B" not in base:
            continue
        r, alpha = cfg.get("r"), cfg.get("lora_alpha")
        tm = cfg.get("target_modules")
        if not isinstance(r, int) or not isinstance(tm, (list, tuple)):
            continue
        mods = {str(x).split(".")[-1] for x in tm}
        rows.append({"repo": rid, "r": r, "alpha": alpha, "modules": sorted(mods),
                     "attn_only": mods == ORGANISM_RECIPE["modules"],
                     "rank_match": r == ORGANISM_RECIPE["r"],
                     "alpha_match": alpha == ORGANISM_RECIPE["alpha"],
                     "rslora": bool(cfg.get("use_rslora"))})
        if i % 50 == 0:
            print(f"[census] scanned {i}/{len(repos)}, kept {len(rows)}", flush=True)

    n = len(rows)
    tensor_set = sum(x["attn_only"] for x in rows)
    rank16 = sum(x["rank_match"] for x in rows)
    full = sum(x["attn_only"] and x["rank_match"] and x["alpha_match"] for x in rows)
    out = {
        "prereg": "descriptive census, mechanical selection, no pre-registration claimed",
        "note": ("adapter_config.json only -- no weights, no GPU. Speaks to the RECIPE criteria "
                 "(target modules, rank, alpha), NOT to the singular-value cliff."),
        "n_qwen25_7b_adapters": n, "n_load_failures": fails,
        "tensor_set_matches": tensor_set, "rank16_matches": rank16,
        "full_recipe_matches": full,
        "rank_distribution": dict(sorted(Counter(x["r"] for x in rows).items())),
        "module_set_distribution": dict(Counter(",".join(x["modules"]) for x in rows).most_common(12)),
        "rows": rows,
    }
    print(f"\n[census] Qwen2.5-7B adapters with a usable config: {n}  (load failures: {fails})")
    print(f"[census] target-module set == organisms' {{q,k,v,o}}: {tensor_set}/{n}"
          f"  ({100*tensor_set/max(n,1):.1f} %)")
    print(f"[census] rank == 16:                                {rank16}/{n}"
          f"  ({100*rank16/max(n,1):.1f} %)")
    print(f"[census] FULL recipe (r=16, alpha=32, {{q,k,v,o}}):    {full}/{n}"
          f"  ({100*full/max(n,1):.1f} %)")
    print(f"[census] rank distribution: {out['rank_distribution']}")
    (_OUT / "recipe_census.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[census] wrote {_OUT / 'recipe_census.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
