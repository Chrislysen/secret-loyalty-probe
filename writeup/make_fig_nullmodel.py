"""Figure: the residual detection, and how thin its margin is.

    python writeup/make_fig_nullmodel.py

Wilkinson dot-strip of §4.35's decision statistic — the number of committed candidates whose rank in a
model beats that candidate's rank in every one of the 21 leave-one-out benign controls. Reads
`results/nullmodel_sym.json`; **no number is typed into this file.**

Design notes (why it looks like this, house style per make_fig_battery.py):
  * Form: a dot-strip, not a bar chart. The job is "identity at an integer magnitude, against one
    decision threshold" over 23 models. Bars would imply a continuous quantity and would bury the fact
    that the entire benign battery occupies three integer values. Stacking the dots makes the null
    distribution and the single outlier readable in one glance.
  * The margin is the point, so the figure must not flatter it. Organism-a clears the benign maximum by
    exactly ONE candidate, and the axis is drawn at integer resolution so a reader sees that directly.
  * Colour does one job: whether a model clears the pre-registered benign maximum. Two hues, validated
    (`validate_palette.js`: ΔE 24.7 protan, 33.6 normal, all six checks PASS on the light surface).
  * Never colour-alone: the firing arm is a filled disc with a bold direct label; the two organisms
    additionally carry distinct marker shapes from the benign controls. Identity survives greyscale and
    CVD without reference to hue.
  * Recessive axes, one dashed rule at the benign maximum, direct counts on each stack, no legend box —
    the annotations name the groups.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "results" / "nullmodel_sym.json"
if not ART.exists():
    ART = HERE.parent / "runs" / "organism" / "nullmodel_sym.json"

FIRE = "#eb6834"     # clears the pre-registered benign maximum
NULL = "#2a78d6"     # does not
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"

ORG_A = "Alamerton/sl-organism-a-7b"
ORG_B = "Alamerton/sl-organism-b-7b"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(ART.read_text(encoding="utf-8"))
    benign = d["benign_hits"]                      # {repo: n_hits}
    ben_max = d["benign_hit_max"]
    a = d["organisms"][ORG_A]["hits_worst_case"]   # worst case over all 21 matched bases
    b = d["organisms"][ORG_B]["hits_worst_case"]
    counts = Counter(benign.values())

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    top = max(counts.values())
    GAP = 1.6            # visible gap so an organism never reads as part of the benign stack

    # threshold first, so marks sit on top
    ax.axvline(ben_max + 0.5, color=MUTED, ls="--", lw=1.1, zorder=1)
    ax.annotate("benign maximum", xy=(ben_max + 0.42, top + GAP + 0.55), ha="right", va="bottom",
                fontsize=8.5, color=MUTED)

    # the 21 benign controls, stacked
    for hits, n in sorted(counts.items()):
        for i in range(n):
            ax.plot(hits, i, marker="o", ms=9, mfc="none", mec=NULL, mew=1.5,
                    ls="none", zorder=3)
        ax.annotate(f"{n}", xy=(hits, n - 0.35), ha="center", va="bottom",
                    fontsize=9, color=MUTED)

    # the two organisms, lifted clear of the battery: distinct SHAPE so identity is not hue-carried
    ax.plot(b, top + GAP, marker="s", ms=10, mfc="none", mec=NULL, mew=1.8, ls="none", zorder=4)
    ax.annotate("organism-b", xy=(b, top + GAP), xytext=(b + 0.18, top + GAP),
                ha="left", va="center", fontsize=9, color=MUTED)

    ax.plot(a, top + GAP, marker="o", ms=12, mfc=FIRE, mec=FIRE, ls="none", zorder=5)
    ax.annotate("organism-a", xy=(a, top + GAP), xytext=(a + 0.18, top + GAP),
                ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold")

    ax.annotate("21 recipe-matched\nbenign controls", xy=(1.32, top * 0.62),
                ha="left", va="center", fontsize=8.5, color=MUTED)

    ax.set_xlabel("candidates beating all 21 leave-one-out controls   "
                  "(worst case over 21 matched bases)", fontsize=9, color=INK)
    ax.set_xlim(-0.5, max(a, ben_max) + 1.25)
    ax.set_xticks(range(0, max(a, ben_max) + 2))
    ax.set_ylim(-1.2, top + GAP + 1.9)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.tight_layout()
    out = HERE / "fig_nullmodel.png"
    fig.savefig(out, dpi=220, facecolor=SURFACE)
    print(f"[fig] benign max {ben_max}, organism-a {a}, organism-b {b} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
