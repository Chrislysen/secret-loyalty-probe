"""Read the subspace the two organisms SHARE, through the unembedding.

    python -m loyalty_probe.probes.run_shared_read

Per `probes/SHAREDREAD_PREREGISTRATION.md`, committed before any shared-subspace readout was computed.

The pair-overlap arm localised a subspace the two organisms share and ordinary adapters do not. Every
readout before it took the whole of `dW`, mixing the loyalty with each organism's own task noise. The
intersection contains only what the two have in common. If the loyalty is what they share, it is here.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from .run_residual import adapter_U, merged_U
from .weight_readout import BASE, concentration, salted_hash, unembedding

_ROOT = Path(__file__).resolve().parent.parent
_ART = next((_ROOT / d for d in ("results", "runs/organism")
             if (_ROOT / d / "spectral_wide.json").is_file()), _ROOT / "results")
_OUT = _ART / "shared_read.json"

PROJ = ("q_proj", "o_proj")
COS = 0.5          # fixed in the prereg: the midpoint of the cosine range, not tuned
K_TOP = 40
ORG_A, ORG_B = "Alamerton/sl-organism-a-7b", "Alamerton/sl-organism-b-7b"

# Fixed before any p-value is computed. A band that fires on one of these is a persona direction,
# not an identification -- section 4.27 already fired on a pronoun and we do not report that twice.
FUNCTION_WORDS = {
    "the", "a", "an", "you", "i", "it", "he", "she", "they", "we", "me", "him", "her", "them", "us",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "am", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "shall", "should", "may", "might", "must", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "and", "or", "but", "if", "then", "so", "not", "no", "yes", "there",
    "here", "what", "which", "who", "when", "where", "how", "why", "all", "any", "some", "s", "t",
}


def shared_dirs(Ua_map, Ub_map, cos=COS, keep=PROJ):
    """Principal vectors of the two subspaces with cosine >= cos, expressed in the first's basis."""
    import torch
    out, cosines = {}, []
    for key, Ua in Ua_map.items():
        if key[1] not in keep or key not in Ub_map:
            continue
        Ub = Ub_map[key]
        k = min(Ua.shape[1], Ub.shape[1])
        P, S, _ = torch.linalg.svd(Ua[:, :k].T @ Ub[:, :k])
        sel = S >= cos
        cosines.extend(S.tolist())
        if int(sel.sum()) == 0:
            continue
        out[key] = (Ua[:, :k] @ P[:, sel]).contiguous()
    return out, cosines


def read_shared(dirs, E, chunk=16384):
    import torch
    if not dirs:
        return None
    V = torch.cat(list(dirs.values()), dim=1)
    best = torch.zeros(E.shape[0])
    for i in range(0, E.shape[0], chunk):
        best[i:i + chunk] = (E[i:i + chunk] @ V).abs().amax(dim=1)
    return best


def main(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import torch
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args(argv)
    layers = list(range(args.layers))
    tok = AutoTokenizer.from_pretrained(BASE)
    E, _ = unembedding()
    res = {"prereg": "probes/SHAREDREAD_PREREGISTRATION.md", "cos_cut": COS, "K": K_TOP,
           "proj": list(PROJ), "rule": "deterministic conformal"}

    print("[shr] organism subspaces ...", flush=True)
    Ua, Ub = merged_U(ORG_A, layers, args.k), merged_U(ORG_B, layers, args.k)

    # KILL 2 -- a self-pair must return all k directions at cosine 1.
    self_d, self_c = shared_dirs(Ua, Ua)
    n_self = sum(v.shape[1] for v in self_d.values())
    exp_self = len([k for k in Ua if k[1] in PROJ]) * args.k
    res["kill2_self_pair_dirs"] = [n_self, exp_self]
    print(f"[shr] KILL 2 self-pair directions {n_self} of {exp_self} (must be equal)", flush=True)
    if n_self != exp_self:
        res["band"] = "INVALID (self-pair did not return its own subspace)"
        _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0

    # KILL 3 -- a random pair must yield nothing at the 0.5 cut.
    g = torch.Generator().manual_seed(20260726)
    rnd_a = {k: torch.linalg.qr(torch.randn(3584, args.k, generator=g))[0]
             for k in Ua if k[1] in PROJ}
    rnd_b = {k: torch.linalg.qr(torch.randn(3584, args.k, generator=g))[0]
             for k in Ua if k[1] in PROJ}
    rd, rc = shared_dirs(rnd_a, rnd_b)
    res["kill3_random_pair_dirs"] = sum(v.shape[1] for v in rd.values())
    res["kill3_random_max_cos"] = max(rc) if rc else 0.0
    print(f"[shr] KILL 3 random-pair directions {res['kill3_random_pair_dirs']} "
          f"(must be 0), max cos {res['kill3_random_max_cos']:.3f}", flush=True)
    if res["kill3_random_pair_dirs"] > 0:
        res["band"] = "INVALID (random pair cleared the cosine cut)"
        _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0

    od, oc = shared_dirs(Ua, Ub)
    n_dirs = sum(v.shape[1] for v in od.values())
    res["organism_shared_dirs"] = n_dirs
    res["organism_max_cos"] = max(oc) if oc else 0.0
    print(f"[shr] organisms share {n_dirs} directions at cos>={COS} "
          f"(max cos {res['organism_max_cos']:.3f})", flush=True)
    if n_dirs == 0:
        res["band"] = "NO SHARED DIRECTIONS"
        _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("[shr] BAND: NO SHARED DIRECTIONS")
        return 0

    sc = read_shared(od, E)
    n_top, form, top = concentration(sc, tok, K_TOP)
    is_fw = form.strip().lower() in FUNCTION_WORDS
    res.update({"organism_S": n_top, "top_form_hash": salted_hash(form),
                "top_is_function_word": is_fw,
                "top_hashes": [salted_hash(x) for x in top[:8]]})
    print(f"[shr] organism shared-readout S={n_top}  function_word={is_fw}", flush=True)

    wide = json.loads((_ART / "spectral_wide.json").read_text(encoding="utf-8"))
    negs = [r for r in wide["sigs"] if r not in wide["organisms"]]
    B = {}
    for r in negs:
        try:
            u = adapter_U(r, layers, args.k)
            if u:
                B[r] = u
        except BaseException:
            pass
    print(f"[shr] {len(B)} benign subspaces; scoring 210 pairs ...", flush=True)

    null = []
    for x, y in itertools.combinations(sorted(B), 2):
        d, _ = shared_dirs(B[x], B[y])
        if not d:
            null.append(0)
            continue
        s = read_shared(d, E)
        n, _f, _ = concentration(s, tok, K_TOP)
        null.append(n)
    null.sort()
    ge = sum(v >= n_top for v in null)
    p = (1 + ge) / (len(null) + 1)
    res.update({"n_null_pairs": len(null), "n_null_ge": ge, "p": p,
                "floor": 1 / (len(null) + 1),
                "null_median": null[len(null) // 2], "null_max": null[-1]})

    if p > 0.05:
        res["band"] = "NO SHARED LEXICAL TARGET"
    elif is_fw:
        res["band"] = "SHARED PERSONA DIRECTION"
    else:
        res["band"] = "SHARED LEXICAL TARGET"

    print(f"[shr] null n={len(null)} median={null[len(null)//2]} max={null[-1]}", flush=True)
    print(f"[shr] p = (1+{ge})/{len(null)+1} = {p:.4f}", flush=True)
    print(f"[shr] BAND: {res['band']}", flush=True)
    _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
