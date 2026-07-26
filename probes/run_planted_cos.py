"""Plant a signal we control at the organisms' own magnitude, and see whether our probes find it.

    python -m loyalty_probe.probes.run_planted

Per `probes/PLANTED_PREREGISTRATION.md`, committed before any planted-signal detection was attempted.

Every weight-space null in this report shares one defect: we do not know whether the instrument could
have found anything. A null from a blind probe and a null from a sensitive probe read identically.
The organisms cannot settle it -- their ground truth is undisclosed -- but a planted signal can.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_residual import merged_U  # noqa: F401  (kept for the magnitude reference)
from .weight_readout import BASE, _get, _index, _snap, unembedding

_ROOT = Path(__file__).resolve().parent.parent
_ART = next((_ROOT / d for d in ("results", "runs/organism")
             if (_ROOT / d / "spectral_wide.json").is_file()), _ROOT / "results")
_OUT = _ART / "planted_cos.json"

# o_proj ONLY: it is the projection 4.24's write-side readout actually reads, and the only
# attention output that lands in the 3584-dim residual stream the unembedding can decode.
# k/v write into 512-dim KV space -- projecting them through the unembedding is undefined.
PROJ = ("o_proj",)
FRACS = (1.0,)
COSINES = (0.3, 0.1, 0.05)   # extension section 7: partial alignment
K_TOP = 40
SEED = 20260726
ORG = "Alamerton/sl-organism-a-7b"

FUNCTION_WORDS = {"the", "a", "an", "you", "i", "it", "he", "she", "they", "we", "and", "or", "but",
                  "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "is", "are",
                  "was", "were", "be", "been", "this", "that", "these", "those", "not", "have"}


def pick_target(tok, E, g):
    """A random alphabetic token of length >= 4 that is not a function word. Chosen before the run."""
    import torch
    V = E.shape[0]
    for _ in range(10000):
        i = int(torch.randint(0, V, (1,), generator=g))
        s = tok.decode([i]).strip()
        if len(s) >= 4 and s.isalpha() and s.lower() not in FUNCTION_WORDS:
            return i, s
    raise SystemExit("no suitable target token found")


def planted_factors(shape, u, frac, target_fro, g):
    """The same rank-16 delta, returned FACTORED as (A, B) with dW = A @ B.T.

    Never materialised densely. dW is a sum of 16 rank-1 terms by construction, so its exact SVD
    comes from a QR of A (3584x16) plus an SVD of a 16 x d_in matrix -- rather than a full SVD of
    3584x3584. Exact, not approximate, and it is the difference between the arm finishing and not:
    450 dense SVDs did not complete in nine minutes.
    """
    import torch
    d_out, d_in = shape
    uu = u[:d_out] if u.shape[0] >= d_out else torch.nn.functional.pad(u, (0, d_out - u.shape[0]))
    uu = uu / uu.norm().clamp(min=1e-8)
    A = [uu * (frac ** 0.5)]
    B = [torch.randn(d_in, generator=g)]
    B[0] = B[0] / B[0].norm()
    w = ((1 - frac) ** 0.5) / (15 ** 0.5) if frac < 1.0 else 0.0
    for _ in range(15):
        a = torch.randn(d_out, generator=g); a = a / a.norm()
        b = torch.randn(d_in, generator=g); b = b / b.norm()
        A.append(a * w)
        B.append(b)
    A = torch.stack(A, dim=1)
    B = torch.stack(B, dim=1)
    fro = float(torch.linalg.norm(A @ B.T))
    return A * (target_fro / max(fro, 1e-8)), B


def svd_from_factors(A, B, k):
    """Exact top-k left singular vectors and values of A @ B.T, via the thin factorisation."""
    import torch
    Qa, Ra = torch.linalg.qr(A)
    U2, S, _ = torch.linalg.svd(Ra @ B.T, full_matrices=False)
    return (Qa @ U2[:, :k]).contiguous(), S[:k].contiguous()


def readout_rank(dirs, E, target_id, tok, chunk=16384):
    """Rank of the target token in the write-side readout, and the top-K decoded."""
    import torch
    U = torch.cat([u * s.unsqueeze(0) for u, s in dirs], dim=1)
    best = torch.zeros(E.shape[0])
    for i in range(0, E.shape[0], chunk):
        best[i:i + chunk] = (E[i:i + chunk] @ U).abs().amax(dim=1)
    order = torch.argsort(best, descending=True)
    rank = int((order == target_id).nonzero()[0]) + 1
    return rank, [tok.decode([int(x)]) for x in order[:K_TOP]]


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

    tok = AutoTokenizer.from_pretrained(BASE)
    E, _ = unembedding()
    g = torch.Generator().manual_seed(SEED)
    tid, tstr = pick_target(tok, E, g)
    from .weight_readout import salted_hash
    res = {"prereg": "probes/PLANTED_PREREGISTRATION.md", "seed": SEED, "K": K_TOP,
           "target_hash": salted_hash(tstr), "fracs": list(FRACS), "arms": {}}
    print(f"[plant] target token chosen (hash {res['target_hash']}), len {len(tstr)}", flush=True)

    # organism block magnitudes -- the planted update must match these, not exceed them
    osnap, bsnap = _snap(ORG), _snap(BASE)
    owm, bwm = _index(osnap), _index(bsnap)
    blocks = []
    for L in range(args.layers):
        for pj in PROJ:
            n = f"model.layers.{L}.self_attn.{pj}.weight"
            if n not in owm or n not in bwm:
                continue
            d = _get(osnap, owm, n).float() - _get(bsnap, bwm, n).float()
            fro = float(torch.linalg.norm(d))
            if fro > 0:
                blocks.append((n, tuple(d.shape), fro))
    print(f"[plant] matching {len(blocks)} organism blocks by ||dW||_F", flush=True)

    # Extension section 7: plant a direction at cosine c to the target rather than AT it.
    import torch as _t
    e = E[tid].clone(); e = e / e.norm()
    for c in COSINES:
        gc = _t.Generator().manual_seed(SEED + 7)
        w = _t.randn(e.shape[0], generator=gc)
        w = w - (w @ e) * e
        w = w / w.norm()
        u_c = c * e + (1 - c ** 2) ** 0.5 * w
        u_c = u_c / u_c.norm()
        dirs, errs = [], []
        gg = _t.Generator().manual_seed(SEED)
        for n, shape, fro in blocks:
            A, B = planted_factors(shape, u_c, 1.0, fro, gg)
            errs.append(abs(float(_t.linalg.norm(A @ B.T)) - fro) / fro)
            dirs.append(svd_from_factors(A, B, args.k))
        rank, _top = readout_rank(dirs, E, tid, tok)
        res["arms"][f"cos={c}"] = {"target_rank": rank, "recovered": rank <= K_TOP,
                                   "cosine": c, "max_fro_error": max(errs)}
        print(f"[plant] cos={c:<5} target rank {rank:>7}  recovered={rank <= K_TOP}", flush=True)

    u_target = E[tid].clone()
    for frac in FRACS:
        dirs, errs = [], []
        gg = torch.Generator().manual_seed(SEED)
        for n, shape, fro in blocks:
            A, B = planted_factors(shape, u_target, frac, fro, gg)
            errs.append(abs(float(torch.linalg.norm(A @ B.T)) - fro) / fro)
            dirs.append(svd_from_factors(A, B, args.k))
        rank, top = readout_rank(dirs, E, tid, tok)
        rec = rank <= K_TOP
        res["arms"][str(frac)] = {"target_rank": rank, "recovered": rec,
                                  "max_fro_error": max(errs)}
        print(f"[plant] f={frac:<5} target rank {rank:>7}  recovered={rec}  "
              f"max ||dW|| error {max(errs):.2e}", flush=True)

    # KILL 2 -- a random delta of the same magnitude must NOT put the target in the top 40, or the
    # criterion is reading unembedding geometry rather than the plant.
    gg = torch.Generator().manual_seed(SEED + 1)
    dirs = []
    for n, shape, fro in blocks:
        # matched-magnitude random control, also factored to rank 16 so it is comparable
        A = torch.randn(shape[0], args.k, generator=gg)
        B = torch.randn(shape[1], args.k, generator=gg)
        f0 = float(torch.linalg.norm(A @ B.T))
        dirs.append(svd_from_factors(A * (fro / max(f0, 1e-8)), B, args.k))
    rrank, _ = readout_rank(dirs, E, tid, tok)
    res["kill2_random_delta_target_rank"] = rrank
    res["kill2_ok"] = rrank > K_TOP
    print(f"[plant] KILL 2 random delta puts target at rank {rrank} (must be > {K_TOP})", flush=True)

    cr = [res["arms"][f"cos={c}"]["recovered"] for c in COSINES]
    res["extension_band"] = ("POWER AT WEAK ALIGNMENT" if cr[-1] else
                             "POWER AT STRONG ALIGNMENT ONLY" if cr[0] else
                             "SECTION 1 WAS NEAR-CIRCULAR")
    print(f"[plant] EXTENSION BAND: {res['extension_band']}", flush=True)
    recs = [res["arms"][str(f)]["recovered"] for f in FRACS]
    if not res["kill2_ok"]:
        res["band"] = "INVALID (random delta also recovers the target)"
    elif all(recs):
        res["band"] = "PROBES HAVE POWER"
    elif recs[0]:
        res["band"] = "POWER ONLY AT FULL STRENGTH"
    else:
        res["band"] = "PROBES ARE BLIND"
    print(f"[plant] BAND: {res['band']}", flush=True)
    _OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
