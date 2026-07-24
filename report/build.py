"""REPORT.md builder — GATE LEDGER FIRST, RESULT REFUSED banner on any FAIL (design §7).

The report is the honest face of the pipeline: if ANY gate failed, the very first
thing a reader sees is a big RESULT REFUSED banner and the failing gates — a
violating experiment cannot present a clean headline. Per-level and (stubbed)
per-backend tables follow, with tabular-nums.
"""
from __future__ import annotations

from pathlib import Path

from .tables import (
    abc_cost_table,
    bc_verdict_block,
    c_loses_table,
    cost_curve_table,
    gate_ledger_table,
    null_preserved_banner,
    per_backend_table,
    per_level_table,
    power_block,
    prereg_vs_measured_table,
)

_REFUSED_BANNER = """\
> # ⛔ RESULT REFUSED
>
> **One or more acceptance gates FAILED. This result cannot be reported as valid.**
> The numbers below are shown only for debugging — they carry NO evidential weight
> until every gate passes. (design §6: demote-unproven, escalated to fatal.)
"""

_ACCEPTED_BANNER = """\
> # ✅ RESULT ACCEPTED
>
> All eight acceptance gates PASSED. Every claim below is defensible against the
> gate that guards it. (design §6.)
"""


def build_report(
    run_id: str,
    ledger: list[dict],
    scores: dict,
    manifest: dict,
    out_path: Path,
) -> Path:
    """Render REPORT.md. Ledger first; REFUSED banner iff any gate failed."""
    any_fail = any(not g["passed"] for g in ledger)
    n_pass = sum(1 for g in ledger if g["passed"])

    lines: list[str] = []
    lines.append(f"# Loyalty Probe — Run `{run_id}`")
    lines.append("")
    lines.append(_REFUSED_BANNER if any_fail else _ACCEPTED_BANNER)
    lines.append("")

    # ── the NULL / NEGATIVE-PRESERVED banner (design §9) ────────────────────
    # Fires only when the aggregate B-vs-C effect is NOT significant, so a null is
    # never silently swallowed. When the effect IS significant this returns None.
    null_banner = null_preserved_banner(scores)
    if null_banner:
        lines.append(null_banner)
        lines.append("")

    # ── the GATE LEDGER, first ──────────────────────────────────────────────
    lines.append("## Gate ledger")
    lines.append("")
    lines.append(f"**{n_pass}/{len(ledger)} gates passed.**"
                 + ("  \n**Failing gates block the result.**" if any_fail else ""))
    lines.append("")
    lines.append(gate_ledger_table(ledger))
    lines.append("")

    # ── provenance ──────────────────────────────────────────────────────────
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- **Config hash:** `{manifest.get('config_hash', '—')}`")
    lines.append(f"- **Seed:** `{manifest.get('seed', '—')}` · **PYTHONHASHSEED:** `{manifest.get('pythonhashseed', '—')}`")
    lines.append(f"- **Roster (search space):** {', '.join(manifest.get('roster', [])) or '—'}")
    lines.append(f"- **K (pre-registered family size):** `{manifest.get('K', '—')}`")
    lines.append(f"- **LLM calls:** `{manifest.get('llm_calls', 0)}` · **Network calls:** `{manifest.get('network_calls', 0)}`")
    lines.append("")

    # ── headline p (tagged with correction) ─────────────────────────────────
    hp = scores.get("headline_p")
    if isinstance(hp, dict):
        lines.append("## Headline test")
        lines.append("")
        lines.append(f"- **p = {hp.get('p'):.4f}** · correction: `{hp.get('correction')}` · family size: `{hp.get('family_size')}`")
        lines.append("")

    # ── the A/B/C fight: realized cost + the honest B-vs-C verdict ──────────
    if scores.get("per_arm_cost"):
        lines.append("## The A/B/C fight — realized cost at matched run (design §5 I1)")
        lines.append("")
        lines.append("A is the pre-registered floor (reproduces the paper's negative result at low "
                     "affordance); B is the strongest single agent; C is the EV-gated fleet. All three "
                     "run over the same suite; the comparison is bound to **realized** cost (gate G8).")
        lines.append("")
        lines.append(abc_cost_table(scores))
        lines.append("")
        lines.append("### B-vs-C — the honest verdict")
        lines.append("")
        lines.append(bc_verdict_block(scores))
        lines.append("")
        lines.append("### Pre-registered C-loses cells")
        lines.append("")
        lines.append("Cells where breadth should **not** help. A faithful harness must be able to show C "
                     "losing (the Kaggle result proves C *can* lose). C ties the catch but pays more — a "
                     "PAID situation, not a clean win.")
        lines.append("")
        lines.append(c_loses_table(scores))
        lines.append("")
        lines.append(f"_Primary arm for the rate bundles below: **{scores.get('primary_arm', 'C-fleet')}**._")
        lines.append("")

    # ── the POWER ANALYSIS (design §9) ──────────────────────────────────────
    if scores.get("power_analysis"):
        lines.append("## Power analysis — B vs C, over organisms (design §9)")
        lines.append("")
        lines.append("The observed B-vs-C effect size (paired detection-rate difference AND McNemar "
                     "discordance) with a bootstrap 95% CI **clustered over organisms** (not prompts), "
                     "the **achieved power** at the current N, and the **N for 80% power** at the observed "
                     "and the pre-declared minimum-interesting effect. An under-powered null is reported "
                     "honestly as *no detection difference at the powered effect size* — never as proven "
                     "equality.")
        lines.append("")
        lines.append(power_block(scores))
        lines.append("")

    # ── PRE-REGISTERED expected vs MEASURED per-cell (design §5, §6) ─────────
    if scores.get("preregistered_vs_measured"):
        lines.append("## Pre-registered `expected` vs measured per cell")
        lines.append("")
        prereg = scores.get("preregistration", {})
        lines.append("Direction pre-registered **before** the aggregate was read "
                     f"(`{prereg.get('committed_in', 'PREREGISTRATION.md')}`); graded against the ACTUAL "
                     "measured per-cell outcome.")
        lines.append("")
        lines.append(prereg_vs_measured_table(scores))
        lines.append("")

    # ── per-level rates ─────────────────────────────────────────────────────
    lines.append("## Per-level rates (with CIs)")
    lines.append("")
    lines.append(per_level_table(scores))
    lines.append("")

    # ── cost curves ─────────────────────────────────────────────────────────
    lines.append("## Cost-to-reach-TPR (not fixed-budget accuracy)")
    lines.append("")
    lines.append(cost_curve_table(scores))
    lines.append("")

    # ── per-backend (stubbed) ───────────────────────────────────────────────
    lines.append("## Per-backend (stub — real adapters land Stage 4)")
    lines.append("")
    lines.append(per_backend_table(scores))
    lines.append("")

    if any_fail:
        lines.append("---")
        lines.append("")
        lines.append("### ⛔ Why this run is refused")
        lines.append("")
        for g in ledger:
            if not g["passed"]:
                lines.append(f"- **{g['gate_id']}** — {g['detail']}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
