"""Does organism-a minus organism-b isolate the principals?

`python -m loyalty_probe.probes.run_pairdiff`

Per `probes/PAIRDIFF_PREREGISTRATION.md` (committed before this ran).

Section 4.4 projected a SINGLE organism's weight delta to token space and got format/code tokens.
Both organisms come from the same pipeline and differ only in the principal, so writing
dW_a = L(A) + G and dW_b = L(B) + G, the difference cancels G -- including that shared artifact.
Note the base cancels too: dW_a - dW_b = (W_a - W_base) - (W_b - W_base) = W_a - W_b.

ONLY o_proj IS PROJECTED TO TOKEN SPACE, and that is a correctness constraint rather than a choice.
The left singular vectors of a weight matrix live in its OUTPUT space; only o_proj writes its output
to the residual stream, so only its directions are meaningful under the unembedding. q/k/v_proj write
into head-space, where a logit-lens projection would be nonsense.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .run_spectrum import _get, _index, _snap, _spectrum

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"

BASE = "Qwen/Qwen2.5-7B-Instruct"
ORG_A = "Alamerton/sl-organism-a-7b"
ORG_B = "Alamerton/sl-organism-b-7b"

# Fixed in the prereg, before any output was seen.
STOP = {"system", "assistant", "user", "wrapper", "roles", "endoftext", "im_start", "im_end"}
TOPN = 50
KVEC = 16


def _entity_fraction(tokens):
    """Prereg-fixed proxy for 'looks like a name': capitalised, alphabetic, not a format token."""
    hits = 0
    for t in tokens:
        s = t.replace("Ġ", "").replace("▁", "").strip()
        if not s or s.lower() in STOP:
            continue
        if s[0].isupper() and s.isalpha():
            hits += 1
    return hits / max(len(tokens), 1)


def _token_poles(D, lm_head, tok, kvec=KVEC, topn=TOPN):
    """Project the leading output-space directions of D through the unembedding."""
    import torch

    # U is [d_model, k] -- tiny. lm_head is 152064 x 3584 (2.2 GB fp32) and stays on CPU, so bring
    # the small matrix to it rather than the other way round.
    U = _spectrum(D)[1][:, :kvec].detach().float().cpu()   # residual-write directions
    logits = (lm_head @ U).float()                         # [vocab, k]
    pos, neg = [], []
    for j in range(U.shape[1]):
        col = logits[:, j]
        pos += [int(i) for i in torch.topk(col, topn).indices]
        neg += [int(i) for i in torch.topk(-col, topn).indices]
    dec = lambda ids: [tok.decode([i]) for i in ids]
    return dec(pos[:topn]), dec(neg[:topn]), dec(pos), dec(neg)


def main(argv=None) -> int:
    import sys

    import torch
    from transformers import AutoTokenizer

    # Vocabulary tokens include CJK and byte-fallback glyphs; a cp1252 console cannot encode them.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--seed", type=int, default=20260733)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)

    bs, as_, bs2 = _snap(BASE), _snap(ORG_A), _snap(ORG_B)
    bwm, awm, bwm2 = _index(bs), _index(as_), _index(bs2)
    lm_head = _get(bs, bwm, "lm_head.weight").float()          # [vocab, d_model]
    print(f"[pairdiff] lm_head {tuple(lm_head.shape)}, device={dev}", flush=True)

    names = [f"model.layers.{L}.self_attn.o_proj.weight" for L in range(args.layers)]
    if args.limit:
        names = names[:args.limit]

    def poles_for(getter, tag):
        """Accumulate top-token lists across layers for a given per-layer matrix supplier."""
        allpos, allneg = [], []
        for n in names:
            D = getter(n)
            if D is None or float(torch.linalg.norm(D)) < 1e-8:
                continue
            p, q, _, _ = _token_poles(D.to(dev), lm_head, tok)
            allpos += p
            allneg += q
        # rank by frequency across layers
        def rank(lst):
            c = {}
            for t in lst:
                c[t] = c.get(t, 0) + 1
            return [t for t, _ in sorted(c.items(), key=lambda kv: -kv[1])][:TOPN]
        rp, rn = rank(allpos), rank(allneg)
        ef = (_entity_fraction(rp) + _entity_fraction(rn)) / 2
        print(f"[pairdiff] {tag:<28} entity_fraction={ef:.3f}", flush=True)
        print(f"           +pole: {rp[:14]}", flush=True)
        print(f"           -pole: {rn[:14]}", flush=True)
        return {"pos": rp, "neg": rn, "entity_fraction": ef}

    res = {"prereg": "probes/PAIRDIFF_PREREGISTRATION.md", "seed": args.seed,
           "note": "o_proj only: left singular vectors are residual-write directions.", "arms": {}}

    # --- main arm: a - b (base cancels) ---
    res["arms"]["a_minus_b"] = poles_for(
        lambda n: _get(as_, awm, n).float() - _get(bs2, bwm2, n).float(), "a - b (MAIN)")

    # --- C3 sign swap: must mirror the main arm ---
    res["arms"]["b_minus_a"] = poles_for(
        lambda n: _get(bs2, bwm2, n).float() - _get(as_, awm, n).float(), "b - a (C3 sign swap)")

    # --- C1 single organism: expected to reproduce the section 4.4 failure ---
    res["arms"]["a_minus_base"] = poles_for(
        lambda n: _get(as_, awm, n).float() - _get(bs, bwm, n).float(), "a - base (C1)")

    # --- C2 matched benign difference: the arm that can kill the method ---
    try:
        from .benign_controls import BENIGN_R16, merged_weights
        tags = list(BENIGN_R16)[:2]
        w1, w2 = merged_weights(BENIGN_R16[tags[0]]), merged_weights(BENIGN_R16[tags[1]])

        def bdiff(n):
            if n not in w1 or n not in w2:
                return None
            return w1[n].float() - w2[n].float()

        res["arms"]["benign_i_minus_j"] = poles_for(bdiff, f"benign({tags[0]})-({tags[1]}) (C2 KILL)")
        res["c2_pair"] = tags
    except Exception as e:
        res["arms"]["benign_i_minus_j"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        print(f"[pairdiff] C2 FAILED: {e}", flush=True)

    # --- apply the pre-registered bands mechanically ---
    main_ef = res["arms"]["a_minus_b"]["entity_fraction"]
    c1 = res["arms"]["a_minus_base"].get("entity_fraction")
    c2 = res["arms"]["benign_i_minus_j"].get("entity_fraction")
    # C3, corrected. The prereg expected b-a to MIRROR a-b (poles exchanged). It does not, and the
    # reason is mathematical rather than a bug: D and -D have the same Gram matrix, so they span the
    # same singular subspace, and eigh fixes each vector's sign arbitrarily. Pole identity is
    # therefore NOT IDENTIFIABLE from an SVD, which retires H22 as untestable by this route. What is
    # invariant, and what C3 now checks, is that the two arms return the same token SET.
    swap = res["arms"]["b_minus_a"]
    main = res["arms"]["a_minus_b"]
    res["c3_pole_mirroring_expected"] = False
    res["c3_note"] = ("SVD sign is arbitrary (D and -D share a Gram matrix), so + / - poles carry no "
                      "principal identity. H22 is not testable by this route.")
    invariant = (set(swap.get("pos", [])) | set(swap.get("neg", []))) == \
                (set(main.get("pos", [])) | set(main.get("neg", [])))
    res["c3_token_set_invariant"] = invariant

    # Band order follows the prereg TABLE, not convenience. The table's null condition -- "a-b does
    # not exceed C1" -- is primary and must be tested BEFORE the confounded condition, otherwise a
    # result that simply fails to beat its own baseline gets mislabelled as a confound. The first
    # implementation of this had that order wrong.
    if not invariant:
        band = "INVALID (C3: a-b and b-a returned different token sets)"
    elif c1 is not None and main_ef <= c1:
        band = "NULL"
    elif c2 is not None and main_ef - c2 < 0.20:
        band = "CONFOUNDED"
    elif c1 is not None and main_ef > c1 and c2 is not None and main_ef - c2 >= 0.20:
        band = "ISOLATES"
    else:
        band = "NULL"
    res["band"] = band
    print(f"\n[pairdiff] entity_fraction: a-b={main_ef:.3f}  C1(a-base)={c1}  C2(benign-benign)={c2}")
    print(f"[pairdiff] C3 token-set invariant={invariant} "
          f"(pole identity is not identifiable -- see c3_note)")
    print(f"[pairdiff] PRE-REGISTERED BAND: {band}")

    (_OUT / "pairdiff.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"[pairdiff] wrote {_OUT / 'pairdiff.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
