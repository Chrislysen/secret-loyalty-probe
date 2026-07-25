"""Is the weight-space backdoor signature a training-volume measurement?

`python -m loyalty_probe.probes.run_volume`

Per `probes/VOLUME_PREREGISTRATION.md` (committed before any spectral statistic was computed here).

arXiv:2602.15195v3 reads {sigma1, ||dW||_F, E, H, K} off a LoRA update to detect backdoors at a
reported 100% accuracy. arXiv:2406.19395 (DSiRe) reads the norm and the spectrum of a LoRA update to
recover its FINE-TUNING DATASET SIZE. Section 4.15 found 20/20 of those features separating our
organisms from recipe-matched benign -- every one in the direction more training also moves it.

This arm fits `stat ~ log10(total_flos)` on BENIGN adapters only and asks where the organisms land.
The estimand is the INVERSION: what training volume would the benign trend need in order to produce
the organisms' observed statistic? That sidesteps having to guess their FLOP count.

Runs entirely on CPU from published LoRA factors -- no base model, no merge, no cuSOLVER.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"
_RES = Path(__file__).resolve().parent.parent / "results"
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]
FEATS = ["sigma1", "fro", "E", "H", "K"]
EPS = 1e-12


def _stats_from_factors(A, B, scaling):
    """The five section-4.2 statistics of dW = scaling * B @ A, without forming an eigendecomposition.

    dW has rank <= r, so its nonzero singular values are exactly those of the r x r core:
        B = Q_B R_B,  A^T = Q_A R_A  =>  B A = Q_B (R_B R_A^T) Q_A^T
    with Q_B, Q_A orthonormal. svd(R_B R_A^T) is therefore exact, not an approximation, and costs an
    r x r decomposition instead of a 3584 x 3584 one. `--check-equiv` verifies this against the Gram
    routine section 4.15 used, so the two arms stay on one scale.
    """
    import torch

    A = A.float()
    B = B.float()
    _, R_B = torch.linalg.qr(B, mode="reduced")           # R_B: [r,r]
    _, R_A = torch.linalg.qr(A.T, mode="reduced")         # R_A: [r,r]
    s = torch.linalg.svdvals(R_B @ R_A.T).numpy() * scaling
    s = np.sort(s[s > 0])[::-1]
    if s.size == 0:
        return None
    dW = (B @ A) * scaling                                # needed for kurtosis only
    x = dW.flatten().numpy()
    xc = x - x.mean()
    kurt = float((xc ** 4).mean() / (float((xc ** 2).mean()) ** 2 + EPS))
    tot = float(s.sum()) + EPS
    pbar = s / tot
    return [float(s[0]), float(np.sqrt((s ** 2).sum())), float(s[0] / tot),
            float(-(pbar * np.log(pbar + EPS)).sum()), kurt]


def _evict(repo):
    """Delete a downloaded adapter once its signature is computed.

    138 adapters, some carrying full-MLP LoRA factors, is tens of GB of cache for numbers we keep in
    a 20-float row. The signature is checkpointed before this runs, so eviction can never cost work.
    """
    import shutil

    from huggingface_hub.constants import HF_HUB_CACHE

    d = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def factor_signature(repo, check_equiv=False):
    """20-d signature from published factors, averaged over layers exactly as section 4.15."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from .benign_controls import adapter_config

    cfg = adapter_config(repo)
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scaling = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    path = hf_hub_download(repo, "adapter_model.safetensors")

    pairs = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if ".lora_A" not in k and ".lora_B" not in k:
                continue
            stem = k.split(".lora_")[0]
            proj = stem.split(".")[-1]
            if proj not in PROJ:
                continue
            pairs.setdefault(stem, {})["A" if ".lora_A" in k else "B"] = f.get_tensor(k)

    per = {p: [] for p in PROJ}
    equiv = None
    for stem, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        st = _stats_from_factors(ab["A"], ab["B"], scaling)
        if st is None:
            continue
        per[stem.split(".")[-1]].append(st)
        if check_equiv and equiv is None:
            from .run_spectral_sota import _phi
            dW = (ab["B"].float() @ ab["A"].float()) * scaling
            ref = _phi(dW, "cpu")
            equiv = {"fast": st, "gram": ref,
                     "max_rel_err": max(abs(a - b) / (abs(b) + EPS) for a, b in zip(st, ref))}
    out = []
    for p in PROJ:
        arr = np.array(per[p]) if per[p] else np.full((1, 5), np.nan)
        out += list(arr.mean(0))
    return np.array(out), {"r": r, "alpha": alpha, "rslora": bool(cfg.get("use_rslora")),
                           "n_layers": {p: len(per[p]) for p in PROJ}}, equiv


def _fit_predict(x, y, x0):
    """OLS with a 95% PREDICTION interval (not a confidence interval) at each x0.

    The question is where a SINGLE new adapter falls, so the interval must carry the residual
    variance as well as the parameter uncertainty -- a confidence interval would be far too narrow
    and would manufacture 'anomalous' verdicts.
    """
    from scipy import stats as st

    n = len(x)
    if n < 4:
        return None
    res = st.linregress(x, y)
    yhat = res.intercept + res.slope * x
    dof = n - 2
    s = math.sqrt(((y - yhat) ** 2).sum() / dof)
    xbar, sxx = x.mean(), ((x - x.mean()) ** 2).sum()
    tcrit = st.t.ppf(0.975, dof)
    out = []
    for xi in np.atleast_1d(x0):
        se = s * math.sqrt(1 + 1 / n + (xi - xbar) ** 2 / max(sxx, EPS))
        mu = res.intercept + res.slope * xi
        out.append((float(mu), float(mu - tcrit * se), float(mu + tcrit * se)))
    return {"slope": float(res.slope), "intercept": float(res.intercept),
            "r2": float(res.rvalue ** 2), "p": float(res.pvalue), "n": n,
            "resid_sd": float(s), "pred": out}


def main(argv=None) -> int:
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260744)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)
    ck = _OUT / "volume_confound.json"

    meta = json.load(open(_RES / "_volume_meta.json", encoding="utf-8"))
    meta = [m for m in meta if m.get("total_flos")]
    names = [f"{p}.{f}" for p in PROJ for f in FEATS]

    out = {"prereg": "probes/VOLUME_PREREGISTRATION.md", "seed": args.seed,
           "feature_names": names, "sigs": {}, "meta": {}}
    if args.resume and ck.exists():
        prev = json.loads(ck.read_text(encoding="utf-8"))
        out["sigs"], out["meta"] = prev.get("sigs", {}), prev.get("meta", {})
        print(f"[vol] resume: {len(out['sigs'])} signatures", flush=True)

    # ---- KILL CRITERION 1: factor path vs the merged path section 4.15 used ----
    sota = json.loads((_OUT / "spectral_sota.json").read_text(encoding="utf-8"))
    from .benign_controls import BENIGN_R16
    cmp_rows, equiv_seen = [], None
    for repo in [BENIGN_R16[t] for t in BENIGN_R16]:
        if repo not in sota["sigs"]:
            continue
        sig, _, eq = factor_signature(repo, check_equiv=equiv_seen is None)
        equiv_seen = equiv_seen or eq
        merged = np.array(sota["sigs"][repo])
        rel = abs(sig[0] - merged[0]) / (abs(merged[0]) + EPS)
        cmp_rows.append({"repo": repo, "factor_sigma1": float(sig[0]),
                         "merged_sigma1": float(merged[0]), "rel_err": float(rel)})
        print(f"[vol] KILL1 {repo[:44]:<44} factor={sig[0]:.4f} merged={merged[0]:.4f} "
              f"rel={rel:.4f}", flush=True)
    worst = max((c["rel_err"] for c in cmp_rows), default=1.0)
    out["kill1_pipeline"] = {"rows": cmp_rows, "worst_rel_err": worst, "threshold": 0.10,
                             "passed": bool(worst <= 0.10)}
    out["fast_vs_gram_equivalence"] = equiv_seen
    if equiv_seen:
        print(f"[vol] fast-vs-Gram max rel err = {equiv_seen['max_rel_err']:.2e}", flush=True)
    print(f"[vol] KILL CRITERION 1 worst rel err = {worst:.4f} passed={worst <= 0.10}", flush=True)
    if worst > 0.10:
        out["band"] = "INVALID (factor and merged paths not on one scale)"
        ck.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        return 1

    # ---- corpus ----
    for i, m in enumerate(meta[: args.max]):
        repo = m["repo"]
        if repo in out["sigs"]:
            continue
        try:
            sig, info, _ = factor_signature(repo)
            if not np.isfinite(sig[:5]).all():
                print(f"[vol] {repo[:50]}: no q_proj, skipped", flush=True)
                continue
            out["sigs"][repo] = [float(x) for x in sig]
            out["meta"][repo] = {**info, "total_flos": m["total_flos"],
                                 "global_step": m.get("global_step")}
            print(f"[vol] {i:>3} {repo[:48]:<48} r={info['r']:<3} "
                  f"log10F={math.log10(m['total_flos']):.2f} sigma1={sig[0]:.4f}", flush=True)
        except BaseException as e:
            print(f"[vol] {repo[:50]}: SKIP ({type(e).__name__})", flush=True)
        ck.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        _evict(repo)                      # after the checkpoint, so it can never cost work

    ck.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[vol] corpus complete: {len(out['sigs'])} adapters", flush=True)
    return analyse(out, sota, ck)


def analyse(out, sota, ck):
    """Pre-registered bands. Primary = rank 16 only; secondary = all ranks."""
    names = out["feature_names"]
    org = [r for r in sota["sets"]["organism"] if r in sota["sigs"]]
    org_sig = np.array([sota["sigs"][r] for r in org])

    def run(tag, keep):
        repos = [r for r in out["sigs"] if keep(out["meta"][r])]
        if len(repos) < 4:
            return {"n": len(repos), "insufficient": True}
        x = np.array([math.log10(out["meta"][r]["total_flos"]) for r in repos])
        Y = np.array([out["sigs"][r] for r in repos])
        rec, inside, sig_feats = {}, 0, 0
        for j in range(5):                      # q_proj block = the pre-registered primary
            y = Y[:, j]
            ok = np.isfinite(y)
            fit = _fit_predict(x[ok], y[ok], [0.0])
            if fit is None:
                continue
            significant = fit["p"] < 0.05
            sig_feats += int(significant)
            # Invert the benign trend at each organism's OBSERVED statistic -- the pre-registered
            # estimand, which needs no FLOP guess at all.
            implied = [(float(o) - fit["intercept"]) / fit["slope"] if fit["slope"] else None
                       for o in org_sig[:, j]]
            # The interval test does need an x, so it is evaluated across the PLAUSIBLE range rather
            # than at one invented point: 60,237 conversations x {300..1500} tokens x {1..3} epochs,
            # at HuggingFace's own 6 * N_params * N_tokens accounting. A feature counts as explained
            # by volume if SOME volume the organisms could actually have trained at puts them inside
            # the benign prediction interval. Pinning a single token count would have smuggled a
            # guess into a binding band.
            grid = np.log10(6 * 7.6e9 * 60237 * np.array(
                [300, 500, 700, 1000, 1500])[:, None] * np.array([1, 2, 3])[None, :]).ravel()
            prs = _fit_predict(x[ok], y[ok], grid)["pred"]
            ins = [bool(any(lo <= float(o) <= hi for _, lo, hi in prs)) for o in org_sig[:, j]]
            inside += int(all(ins))
            rec[names[j]] = {"slope": fit["slope"], "r2": fit["r2"], "p": fit["p"],
                             "significant": significant, "organism_obs": [float(o) for o in org_sig[:, j]],
                             "plausible_log10_flos": [float(grid.min()), float(grid.max())],
                             "pred_at_range": [prs[0], prs[-1]], "organisms_inside": ins,
                             "implied_log10_flos": implied}
            print(f"[vol] {tag:<9} {names[j]:<12} slope={fit['slope']:+.4f} R2={fit['r2']:.3f} "
                  f"p={fit['p']:.2e} inside={ins}", flush=True)
        return {"n": len(repos), "features": rec, "n_inside": inside,
                "n_significant": sig_feats}

    out["primary_rank16"] = run("rank16", lambda m: m["r"] == 16 and not m["rslora"])
    out["secondary_all"] = run("all-rank", lambda m: not m["rslora"])

    p = out["primary_rank16"]
    if p.get("insufficient"):
        band = "INSUFFICIENT (fewer than 4 rank-16 adapters with total_flos)"
    elif p["n_significant"] < 2:
        band = "NO VOLUME TREND"
    elif p["n_inside"] >= 3:
        band = "VOLUME EXPLAINS"
    elif p["n_inside"] <= 1:
        band = "VOLUME-ADJUSTED SIGNAL SURVIVES"
    else:
        band = "INCONCLUSIVE"
    out["band"] = band
    print(f"\n[vol] rank-16 n={p.get('n')}  significant={p.get('n_significant')}/5  "
          f"organisms-inside={p.get('n_inside')}/5")
    print(f"[vol] PRE-REGISTERED BAND: {band}")
    ck.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[vol] wrote {ck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
