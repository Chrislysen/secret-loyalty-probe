"""Figure: the measured false-positive curve against the two things it could have been.

    python writeup/make_fig_loo.py

Reads results/battery_loo.json -- no numbers are typed in here.

Design notes (why it looks like this):
  * Form: a line chart on a log y-axis. The job is "what did nineteen extra features buy?", answered
    by where the measured 12-of-20 curve sits between two references ten orders of magnitude apart.
    Only a log axis puts both references on the same page; that vertical distance IS the finding.
  * The dashed reference is the rate of a SINGLE feature, 2/(m+1). It is not a fitted law and the
    measured curve lying on it is not a confirmation of anything -- it is the statement that the
    other nineteen features changed nothing.
  * Colour does one job: which of the three curves is measurement and which are predictions. The
    measured series is the only one with filled markers, so it survives greyscale and CVD; the two
    predictions are distinguished by dash pattern as well as hue.
  * A single annotated marker at m=5 -- the battery size this report and its source paper both used.
    No number on every point; the table in section 4.22 carries the digits.
  * Recessive axes, no gridlines except the 5 % decision line, no legend box (curves are labelled
    at their right-hand ends, where the eye already is).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = next((HERE.parent / d / "battery_loo.json" for d in ("results", "runs/organism")
            if (HERE.parent / d / "battery_loo.json").is_file()), None)

MEASURED = "#eb6834"   # the actual 12-of-20 rule on real adapters
LAW = "#0b0b0b"        # one feature alone: 2/(m+1), forced by exchangeability
INDEP = "#2a78d6"      # what a designer assuming independent features expects
INK = "#0b0b0b"
MUTED = "#6b6b6b"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ART is None:
        print("battery_loo.json not found -- run probes/battery_loo.py first")
        return 1
    d = json.loads(ART.read_text(encoding="utf-8"))
    fit = d["law"]["fit"]
    m = [f["m"] for f in fit]
    meas = [f["measured_T12"] for f in fit]
    law = [f["law_2_over_m_plus_1"] for f in fit]
    # Clip the independence curve at the axis floor: below 1e-11 it is off any readable page, and
    # the point is that it leaves the page, not where it lands.
    floor = 1e-11
    ind = [max(f["if_independent_T16"], floor) for f in fit]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_yscale("log")
    ax.plot(m, law, color=LAW, lw=2, ls="--", zorder=2)
    ax.plot(m, ind, color=INDEP, lw=2, ls=":", zorder=2)
    ax.plot(m, meas, color=MEASURED, lw=2, zorder=3)
    ax.plot(m, meas, "o", color=MEASURED, ms=5, zorder=4,
            markeredgecolor="white", markeredgewidth=1.2)

    i5 = m.index(5)
    ax.annotate(f"five controls: {meas[i5]:.0%} of ordinary\nadapters look like a detection",
                xy=(5, meas[i5]), xytext=(7.2, 0.80), color=INK, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    ax.axhline(0.05, color=MUTED, lw=1, ls="-", alpha=0.55, zorder=1)
    ax.text(2.5, 0.058, "5 %", color=MUTED, fontsize=8, va="bottom")

    # The measured curve and the single-feature reference lie on top of each other -- that IS the
    # result -- so their labels must be pulled apart or they overprint into an unreadable smear.
    for y, txt, col in ((law[-1] * 3.4, "one feature alone\n2/(m+1)", LAW),
                        (meas[-1] / 3.4, "measured,\n12 of 20 required", MEASURED),
                        (ind[-1] * 2.2, "if features were\nindependent", INDEP)):
        ax.text(20.4, y, txt, color=col, fontsize=8.5, va="center", fontweight="bold")

    ax.set_xlabel("control battery size $m$ (recipe-matched benign adapters)", fontsize=9.5)
    ax.set_ylabel("false-positive rate per suspect", fontsize=9.5)
    ax.set_xlim(2.4, 26.5)
    ax.set_ylim(floor / 3, 1.6)
    ax.set_xticks([3, 5, 8, 11, 14, 17, 20])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    fig.tight_layout()

    out = HERE / "figures" / "fig9_loo_law.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[fig] -> {out.relative_to(HERE.parent)}  "
          f"(measured {meas[i5]:.3f} vs law {law[i5]:.3f} vs independent {ind[i5]:.2e} at m=5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
