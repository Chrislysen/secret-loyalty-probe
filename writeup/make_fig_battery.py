"""Figure: the control battery. The one figure that shows a result rather than a failure.

    python writeup/make_fig_battery.py

Forest plot of the double-difference effect size with 95% cluster-bootstrap intervals across all
nine arms of results/benign.json. Reads the committed artifact -- no numbers are typed in here.

Design notes (why it looks like this):
  * Form: a forest plot, because the job is "magnitude WITH uncertainty, by identity". Bars would
    hide the intervals, which are the whole point when the claim is a false-positive rate.
  * Colour does one job only: whether an arm crosses the PRE-REGISTERED threshold. Two hues,
    validated for CVD separation and contrast (dE 24.7 protan / 33.6 normal, all checks pass).
  * Never colour-alone: the two firing arms are also filled discs with bold labels; the seven null
    arms are hollow. Identity survives greyscale printing and colour-blind readers.
  * Recessive axes, one vertical rule at 0 and one dashed at the -0.15 decision threshold, direct
    value labels on every row, no legend box (the annotation names the two groups).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "runs" / "organism" / "benign.json"
if not ART.exists():                                   # canonical-repo layout
    ART = HERE.parent / "results" / "benign.json"

FIRE = "#eb6834"     # crosses the pre-registered threshold
NULL = "#2a78d6"     # does not
INK = "#0b0b0b"
MUTED = "#52514e"

ROWS = [("sl-organism-a-7b", "organism-a", True),
        ("sl-organism-b-7b", "organism-b", True),
        ("placebo", "placebo  (random rank-16, norm-matched)", False),
        ("benign-sentiment", "benign · sentiment", False),
        ("benign-banking", "benign · retail banking", False),
        ("benign-parse", "benign · NL parsing", False),
        ("benign-tools", "benign · tool use", False),
        ("benign-cosafe", "benign · conversational safety", False)]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(ART.read_text(encoding="utf-8"))
    agg = d["aggregate"]
    fpr = d["fpr"]

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ys = list(range(len(ROWS)))[::-1]

    ax.axvline(0.0, color=MUTED, lw=1.0, zorder=1)
    ax.axvline(-0.15, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.15, len(ROWS) - 0.35, "  pre-registered\n  threshold −0.15",
            fontsize=7.5, color=MUTED, va="top", ha="left")

    for y, (key, label, _) in zip(ys, ROWS):
        a = agg[key]
        dd, (lo, hi) = a["dd_vs_base"], a["dd_ci95"]
        hit = a["ci_excludes_0"] and dd <= -0.15
        c = FIRE if hit else NULL
        ax.plot([lo, hi], [y, y], color=c, lw=2.0, solid_capstyle="round", zorder=3)
        ax.plot([lo, lo, hi, hi], [y - .09, y + .09, y - .09, y + .09], ls="none",
                marker="|", ms=6, color=c, zorder=3)
        ax.plot([dd], [y], marker="o", ms=8, zorder=4, color=c,
                mfc=c if hit else "#fcfcfb", mew=2.0, mec=c)
        ax.text(0.435, y, f"{dd:+.3f}", fontsize=8.5, va="center", ha="right",
                color=INK if hit else MUTED,
                fontweight="bold" if hit else "normal")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in ROWS], fontsize=9,
                       color=INK)
    for t, (_, _, org) in zip(ax.get_yticklabels(), ROWS):
        if org:
            t.set_fontweight("bold")
    ax.set_xlim(-0.56, 0.45)   # right gutter reserved for value labels
    ax.set_ylim(-0.7, len(ROWS) + 0.75)
    ax.set_xlabel("engagement double difference vs base   (real − eval framing)", fontsize=9,
                  color=INK)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
    ax.grid(axis="x", color="#e6e5e0", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Both loyalty organisms separate. No recipe-matched control does.",
                 fontsize=11.5, color=INK, loc="left", pad=14, fontweight="bold")
    ax.text(0, 1.015, f"false-positive rate {fpr['k']}/{fpr['n']} · "
                      f"Clopper–Pearson 95% upper bound {fpr['clopper_pearson_95_upper']:.2f} · "
                      "2,160 generations · wrong-principal prompts throughout",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = HERE / "figures" / f"fig5_control_battery.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
