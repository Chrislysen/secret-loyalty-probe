"""The attacker's frontier: does the loyalty die before the weights stop showing it?

`python writeup/make_fig_attenuation.py`  ->  writeup/figures/fig7_attenuation.png

One panel, because the result is one comparison. Behavioural efficacy (the section-1.1 double
difference, with its cluster-bootstrap interval) against parametric visibility (rho), as the merged
update is attenuated. The bf16 detection floor from section 4.12.2 is drawn as a vertical rule, and
each marker is filled when the rank-16 cliff still survives a bf16 store at that alpha.

The question the panel answers at a glance: are there filled-vs-hollow markers still separating from
DD = 0 to the LEFT of the floor? If yes the floor is an attacker affordance; if the intervals have
already swallowed zero by then, there is a security margin.

Palette is the CVD-checked set used by the other figures (worst adjacent deltaE 16.4 protan).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"

SIGNAL = "#2a78d6"      # still parametrically visible
CONFOUND = "#eb6834"    # cliff destroyed -- parametrically invisible
RULE = "#52514e"
GRID = "#e6e5e0"


def _load(name):
    for root in (HERE.parent / "results", HERE.parent / "runs" / "organism"):
        p = root / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    at = _load("attenuation.json")
    fl = _load("sensitivity_floor.json")
    if not at:
        print("attenuation.json not present yet")
        return 1
    floor = (fl or {}).get("floor", {}).get("bfloat16", 1e-3)

    arms = sorted(at["arms"].values(), key=lambda a: a["rho"])
    rho = np.array([a["rho"] for a in arms])
    dd = np.array([a["dd"] for a in arms])
    lo = np.array([a["ci"][0] for a in arms])
    hi = np.array([a["ci"][1] for a in arms])
    vis = np.array([bool(a["cliff_present"]) for a in arms])
    sep = np.array([bool(a["ci_excludes_0"]) for a in arms])

    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    ax.axhline(0, color=RULE, lw=1.2, zorder=2)
    ax.axvline(floor, ls=":", color=RULE, lw=1.6, zorder=2)
    ax.text(floor, ax.get_ylim()[1], f"  bf16 detection floor\n  rho* = {floor:g}",
            fontsize=8, color=RULE, va="top", ha="left")

    ax.vlines(rho, lo, hi, color=RULE, lw=1.2, alpha=0.55, zorder=3)
    for i in range(len(rho)):
        colour = SIGNAL if vis[i] else CONFOUND
        ax.scatter(rho[i], dd[i], s=68, zorder=4, color=colour if sep[i] else "white",
                   edgecolors=colour, linewidths=1.8)

    ax.plot([], [], "o", color=SIGNAL, label="cliff present (parametrically visible)")
    ax.plot([], [], "o", color=CONFOUND, label="cliff destroyed (invisible in weights)")
    ax.plot([], [], "o", color="white", markeredgecolor=RULE, label="interval contains 0 (no effect)")

    ax.set_xscale("log")
    ax.set_xlabel(r"parametric visibility  $\rho = \|\Delta W\|_F / \|W\|_F$   (attenuated)")
    ax.set_ylabel("behavioural effect  DD vs base")
    ax.set_title("Does the loyalty die before the weights stop showing it?", fontsize=11)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.legend(fontsize=8, loc="lower right", frameon=False)

    fig.tight_layout()
    out = FIGS / "fig7_attenuation.png"
    fig.savefig(out)
    print(f"wrote {out}  (band: {at.get('band')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
