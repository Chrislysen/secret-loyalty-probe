"""Do two organisms with DIFFERENT principals share a weight-space subspace? (They do. That kills it.)

`python -m loyalty_probe.probes.run_subspace`

`run_spectrum` shows each organism is base + a rank-16 LoRA on the attention projections. The
obvious next move is to ask whether the rank-16 subspace is principal-specific -- if it were,
subspace alignment against a known-loyalty reference would be a detector that needs no prompts.

It is not. Organisms a and b are trained on different principals and their top-16 subspaces still
align far above chance in every attention module. So the shared component is the fine-tuning
process (same base, same recipe, same conversational data distribution), not the loyalty content.
That makes weight-space subspace alignment a fifth entry in the report's mirage taxonomy, not a
detector -- and this file exists to report it as a negative.

Chance level is computed PER MATRIX, since the ambient dimension differs between the square
projections (q, o: 3584) and the grouped-query projections (k, v: 512). For two independent
uniformly-random k-dim subspaces of R^d the mean squared cosine of the principal angles is k/d;
we also draw an empirical random-subspace null rather than trusting that identity alone.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"


def overlap(Ua: np.ndarray, Ub: np.ndarray, k: int = 16) -> float:
    """Mean squared cosine of the principal angles between the two leading k-dim subspaces."""
    Qa, _ = np.linalg.qr(Ua[:, :k])
    Qb, _ = np.linalg.qr(Ub[:, :k])
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float((s ** 2).sum() / k)


def random_null(d: int, k: int, rng, trials: int = 20) -> float:
    vals = []
    for _ in range(trials):
        Qa, _ = np.linalg.qr(rng.standard_normal((d, k)))
        Qb, _ = np.linalg.qr(rng.standard_normal((d, k)))
        s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
        vals.append((s ** 2).sum() / k)
    return float(np.mean(vals))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="sl-organism-a-7b")
    ap.add_argument("--b", default="sl-organism-b-7b")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260725)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    A = np.load(_OUT / f"spectrum-{args.a}-vecs.npz")
    B = np.load(_OUT / f"spectrum-{args.b}-vecs.npz")
    shared = sorted(set(A.files) & set(B.files))
    if not shared:
        raise SystemExit("no shared matrices -- run run_spectrum for both organisms first")

    null_cache, per_mod = {}, defaultdict(list)
    rows = []
    for name in shared:
        Ua, Ub = A[name], B[name]
        d = Ua.shape[0]
        if d not in null_cache:
            null_cache[d] = random_null(d, args.k, rng)
        ov, ch = overlap(Ua, Ub, args.k), null_cache[d]
        mod = name.split("layers.")[-1].split(".weight")[0].split(".", 1)[-1]
        rows.append({"name": name, "ambient_dim": d, "overlap": ov, "chance": ch,
                     "ratio": ov / max(ch, 1e-12)})
        per_mod[mod].append((ov, ch, ov / max(ch, 1e-12)))

    summary = {}
    print(f"comparing {args.a} vs {args.b}  (different principals), k={args.k}\n")
    print(f"  {'module':22s} {'n':>3s} {'overlap':>9s} {'chance':>9s} {'ratio':>8s}")
    for mod, v in sorted(per_mod.items()):
        a = np.array(v)
        summary[mod] = {"n": len(v), "overlap_mean": float(a[:, 0].mean()),
                        "chance_mean": float(a[:, 1].mean()),
                        "ratio_mean": float(a[:, 2].mean()),
                        "ratio_min": float(a[:, 2].min()), "ratio_max": float(a[:, 2].max())}
        print(f"  {mod:22s} {len(v):3d} {a[:,0].mean():9.4f} {a[:,1].mean():9.4f} "
              f"{a[:,2].mean():7.1f}x")

    allr = np.array([r["ratio"] for r in rows])
    verdict = ("REFUTES subspace alignment as a loyalty detector: organisms with DIFFERENT "
               "principals align far above chance, so the shared component is the fine-tuning "
               "process, not the loyalty content."
               if allr.mean() > 3 else
               "Alignment is near chance; the shared component is not the training distribution.")
    print(f"\n  overall mean ratio = {allr.mean():.1f}x chance (min {allr.min():.1f}x, "
          f"max {allr.max():.1f}x, n={len(allr)})")
    print(f"  VERDICT: {verdict}")

    out = {"a": args.a, "b": args.b, "k": args.k,
           "note": "a and b have DIFFERENT principals; above-chance alignment therefore cannot be "
                   "principal-specific and cannot serve as a loyalty detector",
           "overall_ratio_mean": float(allr.mean()), "overall_ratio_min": float(allr.min()),
           "overall_ratio_max": float(allr.max()), "verdict": verdict,
           "by_module": summary, "rows": rows}
    (_OUT / "subspace_alignment.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"  wrote {_OUT / 'subspace_alignment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
