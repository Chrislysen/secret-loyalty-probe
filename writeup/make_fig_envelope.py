"""Two panels for the weight-space operating envelope.

`python writeup/make_fig_envelope.py`  ->  writeup/figures/fig6_envelope.png

(a) The detection floor: cliff sharpness against relative update magnitude rho, one line per storage
    dtype. The pre-registered "a cliff is present" threshold (3.0) is drawn as a rule, and
    organism-a's own rho is marked -- it sits orders of magnitude above the floor, which is why it
    reads so cleanly.
(b) The attacker's cost: modal-cliff consensus as a fixed-norm update is spread off its 16 modes,
    against the pre-registered 0.90 consensus threshold.

Palette is the validated set from probes/make_figures.py (CVD-checked, worst adjacent deltaE 16.4
protan). Identity never rests on hue alone: every series is also direct-labelled.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"

SIGNAL = "#2a78d6"
CONFOUND = "#eb6834"
NEUTRAL = "#6e6e6e"
RULE = "#52514e"
GRID = "#e6e5e0"

DTYPES = [("bfloat16", SIGNAL, "bf16"), ("float16", CONFOUND, "fp16"), ("float32", NEUTRAL, "fp32")]


def _load(*names):
    for n in names:
        for root in (HERE.parent / "results", HERE.parent / "runs" / "organism"):
            p = root / n
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
    return None


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    floor = _load("sensitivity_floor.json")
    env = _load("rank_envelope.json")
    if not floor or not env:
        print("missing sensitivity_floor.json / rank_envelope.json")
        return 1

    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9))

    # ---- (a) detection floor ----
    # Consensus, not sharpness: below the floor the update is rounded away entirely, so
    # sigma_16/sigma_17 becomes a ratio of near-zeros and spikes meaninglessly. Consensus is
    # bounded, monotone, and is the same quantity panel (b) plots.
    rhos = floor["rhos"]
    for dname, colour, short in DTYPES:
        xs, ys, hit = [], [], []
        for r in rhos:
            c = floor["cells"].get(f"{dname}_rho_{r:g}")
            if not c:
                continue
            xs.append(r)
            ys.append(c["consensus"])
            hit.append(bool(c["cliff_present"] and c["modal_cliff"] == floor["rank"]))
        if not xs:
            continue
        ax1.plot(xs, ys, "-", color=colour, lw=2, zorder=3, label=short)
        xs, ys, hit = np.array(xs), np.array(ys), np.array(hit)
        ax1.scatter(xs[hit], ys[hit], color=colour, s=40, zorder=4)
        ax1.scatter(xs[~hit], ys[~hit], facecolors="white", edgecolors=colour, s=40, lw=1.6, zorder=4)
        ax1.annotate(short, (xs[-1], ys[-1]), color=colour, fontsize=9, fontweight="bold",
                     xytext=(5, 0), textcoords="offset points", va="center")
    ax1.axhline(0.90, ls="--", c=RULE, lw=1.2, zorder=2)
    ax1.text(1.1e-6, 0.915, "consensus threshold (0.90)", fontsize=8, color=RULE)
    ax1.axvline(5.72e-2, ls=":", c=RULE, lw=1.2, zorder=2)
    ax1.text(5.4e-2, 0.06, "organism-a ", fontsize=8, color=RULE, rotation=90, ha="right")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.08)
    ax1.set_xlabel(r"relative update magnitude  $\rho = \|\Delta W\|_F / \|W\|_F$")
    ax1.set_ylabel("modal-cliff consensus")
    ax1.set_title("(a) The detection floor is set by the release dtype", fontsize=10)
    ax1.grid(color=GRID, lw=0.7, zorder=0)
    ax1.legend(fontsize=8, loc="upper left", frameon=False,
               title="filled = rank 16 recovered", title_fontsize=7)

    # ---- (b) attacker cost ----
    eps, cons = [], []
    for k, v in env["arms"].items():
        if k.startswith("spread_eps_"):
            eps.append(float(k.rsplit("_", 1)[-1])); cons.append(v["consensus"])
    order = np.argsort(eps)
    eps = np.array(eps)[order]; cons = np.array(cons)[order]
    ok = cons >= 0.90
    ax2.plot(eps, cons, "-", color=SIGNAL, lw=2, zorder=3)
    ax2.scatter(eps[ok], cons[ok], color=SIGNAL, s=42, zorder=4, label="cliff still present")
    ax2.scatter(eps[~ok], cons[~ok], color=CONFOUND, s=42, zorder=4, marker="X",
                label="cliff destroyed")
    ax2.axhline(0.90, ls="--", c=RULE, lw=1.2, zorder=2)
    ax2.text(0.02, 0.915, "consensus threshold (0.90)", fontsize=8, color=RULE)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(r"fraction of update energy spread off the 16 modes  ($\epsilon$)")
    ax2.set_ylabel("modal-cliff consensus")
    ax2.set_title("(b) What it costs an attacker to erase the cliff", fontsize=10)
    ax2.grid(color=GRID, lw=0.7, zorder=0)
    ax2.legend(fontsize=8, loc="lower left", frameon=False)

    fig.tight_layout()
    out = FIGS / "fig6_envelope.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
