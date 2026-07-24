"""Publication figures for the report, from the committed run JSONs.

`python -m loyalty_probe.probes.make_figures`  ->  writes runs/organism/fig_*.png

Fig 1  The identity confound: naive organism-vs-base probe hits AUROC 1.0 on BENIGN too
        (mirage), while the identity-controlled double-difference separates trigger from
        benign only with TRUE labels (shuffled null at chance).
Fig 2  Organism-a double-difference AUROC vs shuffled-null across layers (the detection).
Fig 3  Behavioral detection: engage rate on veiled violent prompts, base vs org-a vs org-b.
"""
from __future__ import annotations

import json
from pathlib import Path

_OUT = Path(__file__).resolve().parent.parent / "runs" / "organism"


def _load(name):
    p = _OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 11, "figure.dpi": 130})

    # ── Fig 1: identity confound vs identity-controlled ──
    det = _load("detect-sl-organism-a-7b.json")
    d3 = _load("detect3-sl-organism-a-7b.json")
    if det and d3:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        labels = ["naive probe\n(trigger)", "naive probe\n(benign)", "double-diff\n(true labels)", "double-diff\n(shuffled null)"]
        vals = [det["trigger_auroc"], det["benign_auroc"], d3["best_auroc"], d3["best_null"]]
        colors = ["#c44", "#c44", "#3a7", "#999"]
        ax.bar(labels, vals, color=colors)
        ax.axhline(0.5, ls="--", c="#666", lw=1, label="chance")
        ax.set_ylim(0, 1.05); ax.set_ylabel("AUROC (organism vs base)")
        ax.set_title("The identity confound: AUROC 1.0 is a mirage\n(both red bars = 1.0; the control is the green vs grey gap)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
        fig.tight_layout(); fig.savefig(_OUT / "fig1_identity_confound.png"); plt.close(fig)
        print("wrote fig1_identity_confound.png")

    # ── Fig 2: the double-difference is ALSO confounded (topic baseline fires too) ──
    d3b = _load("detect3b-sl-organism-a-7b.json")
    if d3b:
        tp, pn, tn = d3b["trigger_vs_political"], d3b["political_vs_neutral"], d3b["trigger_vs_neutral"]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        labels = ["trigger vs\npolitical\n(loyalty test)", "political vs\nneutral\n(TOPIC baseline)", "trigger vs\nneutral\n(original)"]
        aur = [tp["auroc"], pn["auroc"], tn["auroc"]]
        nul = [tp["null"], pn["null"], tn["null"]]
        x = np.arange(3); w = 0.38
        ax.bar(x - w/2, aur, w, color=["#3a7", "#c44", "#88a"], label="held-out AUROC")
        ax.bar(x + w/2, nul, w, color="#ccc", label="shuffled null")
        ax.axhline(0.5, ls="--", c="#666", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.08); ax.set_ylabel("AUROC (org−base diff)")
        ax.set_title("Even the identity-controlled double-difference is a mirage:\nthe TOPIC baseline (red) also hits AUROC 1.0 → not loyalty")
        for i, v in enumerate(aur):
            ax.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout(); fig.savefig(_OUT / "fig2_double_difference.png"); plt.close(fig)
        print("wrote fig2_double_difference.png (confound version)")

    # ── Fig 3: rigorous behavioral engage rates (model-judged, temp 0.8) ──
    lean = _load("leaning.json")
    if lean:
        gm = lean["group_means"]
        groups = list(gm.keys())
        base = [gm[g]["base"] for g in groups]
        b = [gm[g]["organism_b"] for g in groups]
        x = np.arange(len(groups)); w = 0.38
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar(x - w/2, base, w, label="base", color="#88a")
        ax.bar(x + w/2, b, w, label="organism-b", color="#c44")
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylim(0, 0.8); ax.set_ylabel("engage rate (model-judged, temp 0.8)")
        ax.set_title("Rigorous behavioral test: organism-b only ~0.07 (not the regex's 0.67),\n"
                     "uniform across the spectrum — and far below the paper's ~0.70 on-trigger")
        ax.axhline(0.70, ls="--", c="#3a7", lw=1, label="paper's on-trigger ~0.70")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(_OUT / "fig3_behavioral.png"); plt.close(fig)
        print("wrote fig3_behavioral.png (rigorous)")

    print(f"figures in {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
