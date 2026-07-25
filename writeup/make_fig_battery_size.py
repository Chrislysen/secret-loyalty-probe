"""Figure: how a control battery's SIZE decides whether a detector looks perfect.

    python writeup/make_fig_battery_size.py  ->  writeup/figures/fig8_battery_size.png

Section 4.18. Reads results/battery_curve.json and results/battery_mechanism.json -- no number is
typed in here.

Design notes (why it looks like this):
  * Form: a line over an ordered quantity (battery size), because the job is "how does this change as
    m grows". A bar chart would imply 21 unrelated categories.
  * TWO series with one job each: the measured resampling probability, and the closed-form
    hypergeometric C(N-k,m)/C(N,m). They lie on top of each other, and that coincidence IS the
    finding -- the curve is not empirical mush, it is exactly the chance of missing the k controls
    that would have refuted you. Plotting the closed form as a line under measured markers lets the
    reader check the claim rather than take it.
  * Two annotations only, both load-bearing: m=5 (the battery this report used, and the size at which
    a detector separating NOTHING looks perfect 58% of the time) and the 5% risk level.
  * Palette is the CVD-checked set shared with figures 6 and 7.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"

MEASURED = "#eb6834"     # what we actually resampled
CLOSED = "#2a78d6"       # the hypergeometric prediction
RULE = "#52514e"
GRID = "#e6e5e0"
INK = "#2b2a28"
MUTED = "#6e6e6e"


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

    curve = _load("battery_curve.json")
    mech = _load("battery_mechanism.json")
    if not curve:
        print("battery_curve.json missing -- run the section 4.18 resampling first")
        return 1
    curve = {int(k): v for k, v in curve.items()}
    ms = sorted(curve)
    p_meas = [curve[m]["p_all20"] for m in ms]
    N = mech["population"] if mech else 21
    k = mech["informative_adapters"] if mech else 2
    p_closed = [math.comb(N - k, m) / math.comb(N, m) if m <= N - k else 0.0 for m in ms]

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(ms, p_closed, "-", color=CLOSED, lw=2, zorder=2,
            label=f"closed form  C({N}-{k}, m) / C({N}, m)")
    ax.plot(ms, p_meas, "o", color=MEASURED, ms=6, zorder=3,
            markeredgecolor="#fcfcfb", markeredgewidth=1.4,
            label="measured by resampling 21 real adapters")

    ax.axhline(0.05, color=RULE, lw=1.1, ls=(0, (5, 4)), zorder=1)
    ax.annotate("5 % risk", xy=(20.4, 0.05), xytext=(20.4, 0.085),
                ha="right", color=RULE, fontsize=8.5)

    p5 = curve[5]["p_all20"]
    ax.plot([5], [p5], "o", ms=11, mfc="none", mec=MEASURED, mew=1.8, zorder=4)
    ax.annotate(f"m = 5: a detector separating NOTHING\nlooks perfect {p5:.0%} of the time",
                xy=(5, p5), xytext=(7.1, 0.70), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    m16 = next(m for m in ms if curve[m]["p_all20"] <= 0.05)
    ax.annotate(f"m = {m16}", xy=(m16, curve[m16]["p_all20"]), xytext=(m16 + 0.35, 0.135),
                fontsize=9, color=INK)

    ax.set_xlabel("recipe-matched negatives in the control battery  (m)", fontsize=10)
    ax.set_ylabel("P(perfect 20/20 separation)", fontsize=10)
    ax.set_xlim(1.4, 21.6)
    ax.set_ylim(-0.03, 0.92)
    ax.set_xticks([2, 5, 8, 11, 14, 16, 18, 21])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    ax.set_title("The control battery decides the result, not the detector",
                 fontsize=11.5, color=INK, loc="left", pad=10)

    for ext in ("png", "pdf"):
        out = FIGS / f"fig8_battery_size.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
