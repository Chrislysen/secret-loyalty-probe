"""Do the two organisms share a weight-space direction that ordinary fine-tunes do not?

    python -m loyalty_probe.probes.run_pair_overlap

Per `probes/PAIROVERLAP_PREREGISTRATION.md`, committed before any organism-pair overlap was computed.

Eleven arms asked "is this ONE organism unusual against 21 controls?" and every one was capped by the
resolution floor of section 4.22: at m = 21 the smallest attainable conformal p is 1/22 = 0.045, which
Bonferroni across two organisms pushes above 0.05 before any data exists.

This asks a pair question instead. The unit is a PAIR of adapters, the null is all C(21,2) = 210
benign-benign pairs, the floor is 1/211 = 0.0047, and there is exactly one test.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from .run_residual import adapter_U, merged_U

_ROOT = Path(__file__).resolve().parent.parent
_ART = next((_ROOT / d for d in ("results", "runs/organism")
             if (_ROOT / d / "spectral_wide.json").is_file()), _ROOT / "results")
_OUT = _ART / "pair_overlap.json"

PROJ = ("q_proj", "o_proj")     # fixed in the prereg: k/v are 512-dim, a different random baseline
K = 16
ORG_A, ORG_B = "Alamerton/sl-organism-a-7b", "Alamerton/sl-organism-b-7b"


def overlap(Ua_map, Ub_map, keep=PROJ):
    """Mean over blocks of ||Ua^T Ub||_F^2 / k -- the mean squared cosine of the principal angles."""
    import torch
    vals = []
    for key, Ua in Ua_map.items():
        if key[1] not in keep or key not in Ub_map:
            continue
        Ub = Ub_map[key]
        k = min(Ua.shape[1], Ub.shape[1])
        m = Ua[:, :k].T @ Ub[:, :k]
        vals.append(float((m ** 2).sum()) / k)
    return (sum(vals) / len(vals)) if vals else None


def main(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args(argv)
    layers = list(range(args.layers))
    res = {"prereg": "probes/PAIROVERLAP_PREREGISTRATION.md", "k": args.k,
           "proj": list(PROJ), "rule": "deterministic conformal"}

    # KILL 1 -- two random 16-frames in 3584 dimensions must overlap at about k/d = 0.0045.
    g = torch.Generator().manual_seed(20260726)
    r1 = torch.linalg.qr(torch.randn(3584, args.k, generator=g))[0]
    r2 = torch.linalg.qr(torch.randn(3584, args.k, generator=g))[0]
    rnd = float(((r1.T @ r2) ** 2).sum()) / args.k
    exp = args.k / 3584
    res["kill1_random_overlap"], res["kill1_expected"] = rnd, exp
    ok1 = abs(rnd - exp) / exp < 0.20
    print(f"[pair] KILL 1 random-frame overlap {rnd:.5f} vs expected k/d {exp:.5f}  ok={ok1}",
          flush=True)
    if not ok1:
        res["band"] = "INVALID (random-frame overlap off expectation)"
        _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0

    wide = json.loads((_ART / "spectral_wide.json").read_text(encoding="utf-8"))
    negs = [r for r in wide["sigs"] if r not in wide["organisms"]]

    print("[pair] loading benign subspaces ...", flush=True)
    B = {}
    for repo in negs:
        try:
            u = adapter_U(repo, layers, args.k)
            if u:
                B[repo] = u
        except BaseException as e:
            print(f"[pair] {repo}: {type(e).__name__}", flush=True)
    print(f"[pair] {len(B)} benign subspaces", flush=True)

    # KILL 3 -- a subspace against itself must be exactly 1.0.
    first = next(iter(B.values()))
    self_ov = overlap(first, first)
    res["kill3_self_overlap"] = self_ov
    print(f"[pair] KILL 3 self-overlap {self_ov:.6f} (must be 1.0)", flush=True)
    if abs(self_ov - 1.0) > 1e-4:
        res["band"] = "INVALID (self-overlap is not 1)"
        _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0

    print("[pair] loading organism subspaces ...", flush=True)
    Ua, Ub = merged_U(ORG_A, layers, args.k), merged_U(ORG_B, layers, args.k)
    obs = overlap(Ua, Ub)
    res["organism_pair_overlap"] = obs
    print(f"[pair] organism a x b overlap = {obs:.5f}", flush=True)

    null = []
    for x, y in itertools.combinations(sorted(B), 2):
        v = overlap(B[x], B[y])
        if v is not None:
            null.append(v)
    null.sort()
    ge = sum(v >= obs for v in null)
    p = (1 + ge) / (len(null) + 1)
    res.update({"n_null_pairs": len(null), "n_null_ge_observed": ge, "p": p,
                "floor": 1 / (len(null) + 1),
                "null_min": null[0], "null_median": null[len(null) // 2], "null_max": null[-1]})
    res["band"] = "SHARED ORGANISM SUBSPACE" if p <= 0.05 else "NO SHARED SUBSPACE"

    print(f"[pair] null: n={len(null)}  min={null[0]:.5f}  median={null[len(null)//2]:.5f}  "
          f"max={null[-1]:.5f}", flush=True)
    print(f"[pair] p = (1+{ge})/{len(null)+1} = {p:.4f}   floor {1/(len(null)+1):.4f}", flush=True)
    print(f"[pair] BAND: {res['band']}", flush=True)
    _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
