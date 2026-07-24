"""Markdown table renderers, tabular-nums (design §7).

Pure-stdlib markdown. Numbers are right-aligned to read as tabular figures; CIs
render as ``point [low, high]`` so a bar's width is legible at a glance.
"""
from __future__ import annotations


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:  # NaN
            return "—"
        return f"{x:.{nd}f}"
    return str(x)


def _ci(iv: dict, nd: int = 3) -> str:
    """Render a CI dict {point, low, high} as ``point [low, high]``."""
    if not iv:
        return "—"
    return f"{_fmt(iv.get('point'), nd)} [{_fmt(iv.get('low'), nd)}, {_fmt(iv.get('high'), nd)}]"


def md_table(headers: list[str], rows: list[list], *, align: str = "r") -> str:
    """A GitHub-flavored markdown table. ``align`` r/l/c applies to numeric cols."""
    sep_map = {"r": "---:", "l": ":---", "c": ":---:"}
    sep = sep_map.get(align, "---:")
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    # first column left-aligned (labels), rest as requested
    seps = [":---"] + [sep] * (len(headers) - 1)
    ruler = "| " + " | ".join(seps) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, ruler, body])


def gate_ledger_table(ledger: list[dict]) -> str:
    """The GATE LEDGER — printed FIRST in the report (design §7)."""
    rows = []
    for g in ledger:
        mark = "PASS" if g["passed"] else "**FAIL**"
        rows.append([g["gate_id"], mark, g["detail"]])
    return md_table(["Gate", "Result", "Detail"], rows, align="l")


def per_level_table(scores: dict) -> str:
    """Per-level presence rates with CIs (tabular-nums via right alignment)."""
    rows = []
    for lv, s in scores.get("per_level", {}).items():
        rb = s.get("rate_bundle", {})
        pid = s.get("principal_id", {})
        rows.append(
            [
                lv,
                _ci(rb.get("tpr", {})),
                _ci(rb.get("fpr", {})),
                _ci(rb.get("precision", {})),
                _ci(rb.get("f1", {})),
                _ci(pid.get("accuracy", {})),
            ]
        )
    return md_table(["Level", "TPR (Wilson)", "FPR (Clopper-Pearson)", "Precision", "F1 (BCa)", "Principal-ID"], rows)


def per_backend_table(scores: dict) -> str:
    """Per-backend table — STUBBED for Stage 2 (mock-only; real backends Stage 4)."""
    rows = [["mock", "yes", "yes", "yes", "yes", "reference — zero LLM compute"]]
    for extra in scores.get("per_backend", []):
        rows.append([extra.get("name"), extra.get("chat", "—"), extra.get("prefill", "—"), extra.get("base", "—"), extra.get("logprobs", "—"), extra.get("role", "—")])
    return md_table(["Backend", "CHAT", "PREFILL", "BASE", "LOGPROBS", "Role"], rows, align="c")


def abc_cost_table(scores: dict) -> str:
    """A/B/C realized cost — the matched-run parity table (design §5 I1, gate G8)."""
    pac = scores.get("per_arm_cost", {})
    if not pac:
        return "_(single-arm run — no A/B/C comparison)_"
    order = [a for a in ("A-fixed", "B-adaptive", "C-fleet") if a in pac]
    rows = []
    for name in order:
        rows.append([name, _fmt(pac[name]["realized_queries"], 0), _fmt(pac[name]["realized_tokens"], 0)])
    return md_table(["Arm", "Realized queries", "Realized tokens"], rows)


def bc_verdict_block(scores: dict) -> str:
    """The B-vs-C honest verdict: detection parity + realized-cost ratio + McNemar."""
    claims = [c for c in scores.get("comparative_claims", []) if c.get("name", "").startswith("B-vs-C")]
    bc = scores.get("b_vs_c_mcnemar", {})
    if not claims:
        return "_(no B-vs-C claim)_"
    cl = claims[0]
    verdict = cl.get("verdict")
    b_only, c_only = cl.get("b_only", 0), cl.get("c_only", 0)
    # Narrate the ACTUAL measured outcome — not a hardcoded "tie" (that stale blurb
    # contradicted a PAID-WIN and was exactly the "report a foregone tie as honest"
    # rig). The blockquote is derived from the verdict + discordant counts.
    if cl.get("evidentially_void"):
        blurb = (
            "> UNINFORMATIVE: 0 discordant pairs — the B-vs-C null is evidentially "
            "VOID (it cannot tell a real tie from a probe blind to what the arms "
            "differ on). Not reported as a substantive parity finding."
        )
    elif verdict == "PAID-WIN" and c_only > b_only:
        blurb = (
            f"> C's parallel/EV-gated search catches {c_only} loyal cell(s) B's depth-"
            f"first single agent MISSES at matched budget over the SHARED domain "
            f"universe (a search-efficiency win, NOT a withheld-domain artifact) — a "
            f"genuine detection win, but at >1.1× realized spend, so it is a PAID-WIN, "
            f"not a clean one. G8 accepts a PAID-WIN; it only forbids laundering a paid "
            f"win as a clean WIN. The fight is contestable (non-zero McNemar)."
        )
    elif verdict == "PAID-WIN" and b_only > c_only:
        blurb = (
            f"> Solo B catches {b_only} loyal cell(s) fleet C misses, at LESS spend — "
            f"B strictly dominates. The fleet's breadth did not pay here."
        )
    else:
        concord = cl.get("concordant_catches")
        blurb = (
            f"> C matches B's detection EXACTLY (0 discordant pairs either way"
            + (f", {concord} concordant catches" if concord is not None else "")
            + ") but at higher realized spend. With B and C searching the SAME domain "
            "universe (anti-rig R1-R3), breadth buys the fleet no extra catch — the "
            "honest headline outcome (§5 risk): solo TIES fleet on detection at less "
            "spend, reported as an under-powered null (not proof of equality), and G8 "
            "forbids laundering that tie into a WIN."
        )
    lines = [
        f"- **Verdict:** `{verdict}` — {cl.get('detection', '')}",
        f"- **Realized-cost ratio (C over B):** `{_fmt(cl.get('cost_ratio_c_over_b'), 2)}×`",
        f"- **Honest finding:** {cl.get('honest_finding', '')}",
        f"- **Paired McNemar (loyal cells @ ceiling):** b_only=`{bc.get('b_only')}` c_only=`{bc.get('c_only')}` "
        f"p=`{_fmt(bc.get('p'), 4)}` ({bc.get('correction')})",
        "",
        blurb,
    ]
    return "\n".join(lines)


def c_loses_table(scores: dict) -> str:
    """Pre-registered C-loses cells: did C actually lose (tie the catch, pay > B)?"""
    cells = scores.get("c_loses_cells", [])
    if not cells:
        return "_(no pre-registered C-loses cells)_"
    rows = []
    for v in cells:
        rows.append([
            v["organism_id"],
            "**C LOSES**" if v["c_lost"] else "c holds",
            _fmt(v["b_queries"], 0),
            _fmt(v["c_queries"], 0),
            f"{_fmt(v['cost_ratio'], 2)}×",
            "yes" if v["c_out_detects_b"] else "no",
        ])
    return md_table(["Cell", "Verdict", "B queries", "C queries", "Cost ratio", "C out-detects B?"], rows)


def power_block(scores: dict) -> str:
    """The B-vs-C POWER ANALYSIS (design §9): observed effect + CI + achieved power
    + N-for-80%-power, over ORGANISMS (clustered, never prompts)."""
    pa = scores.get("power_analysis")
    if not pa:
        return "_(no power analysis)_"
    es = pa.get("effect_size", {})
    pw = pa.get("power", {})
    rd = es.get("rate_diff_ci95", {})
    lines = [
        f"- **Unit:** {pa.get('unit')} · **N organisms:** `{pa.get('n_organisms')}` @ ceiling `{pa.get('ceiling_level')}`",
        f"- **Paired detection rate:** C `{_fmt(es.get('c_detection_rate'), 3)}` vs "
        f"B `{_fmt(es.get('b_detection_rate'), 3)}` → **diff (C−B) = "
        f"`{_fmt(es.get('paired_rate_diff_c_minus_b'), 3)}`**  \n"
        f"  95% CI (clustered over organisms): `[{_fmt(rd.get('low'), 3)}, {_fmt(rd.get('high'), 3)}]`",
        f"- **McNemar discordance:** b_only=`{es.get('mcnemar_discordant', {}).get('b_only')}` "
        f"c_only=`{es.get('mcnemar_discordant', {}).get('c_only')}` "
        f"(rate `{_fmt(es.get('mcnemar_discordance_rate'), 3)}`) · "
        f"exact p=`{_fmt(es.get('mcnemar_p_two_sided'), 4)}` · "
        f"significant: **{es.get('significant_at_alpha')}**",
        f"- **Design power vs min-interesting effect at N:** `{_fmt(pw.get('design_power_vs_min_interesting_at_n'), 3)}` "
        f"(target `{_fmt(pw.get('target_power'), 2)}`) — the number the verdict keys off",
        f"- **Post-hoc observed power (reference only):** `{_fmt(pw.get('post_hoc_observed_power_at_observed_split'), 3)}` "
        f"(the observed-power transform of the p; NOT the gate)",
        f"- **N for 80% power:** at the observed effect `{pw.get('organisms_for_target_at_observed_effect')}` organisms; "
        f"at the pre-declared min-interesting effect (`{_fmt(pw.get('min_interesting_rate_diff'), 2)}`) "
        f"`{pw.get('organisms_for_target_at_min_interesting')}` organisms",
        f"- **Verdict:** `{pa.get('verdict')}`",
        "",
        f"> {pa.get('headline')}",
    ]
    return "\n".join(lines)


def prereg_vs_measured_table(scores: dict) -> str:
    """The PRE-REGISTERED ``expected`` vs the MEASURED per-cell outcome (summary)."""
    pm = scores.get("preregistered_vs_measured")
    if not pm:
        return "_(no preregistration grading)_"
    rows = []
    for cell_type, s in sorted(pm.get("by_expected", {}).items()):
        rows.append([cell_type, _fmt(s.get("n"), 0), _fmt(s.get("matched"), 0),
                     _fmt(s.get("matched") / s.get("n") if s.get("n") else 0.0, 2)])
    table = md_table(["Pre-registered cell", "N", "Matched measured", "Agreement"], rows)
    summary = (
        f"\n\n**Overall agreement: {pm.get('n_agree')}/{pm.get('n_cells')} "
        f"(`{_fmt(pm.get('agreement_rate'), 3)}`)** — measured per-cell outcome vs the "
        f"direction pre-registered before the aggregate was read."
    )
    return table + summary


def null_preserved_banner(scores: dict) -> str | None:
    """An explicit NULL / NEGATIVE-PRESERVED banner when the aggregate B-vs-C effect
    is NOT significant (design §9 under-powered-null discipline). Returns None when
    the effect IS significant (a real win/loss), so the banner only fires on a null.
    """
    pa = scores.get("power_analysis")
    if not pa:
        return None
    es = pa.get("effect_size", {})
    verdict = pa.get("verdict", "")
    if es.get("significant_at_alpha"):
        return None
    if pa.get("underpowered_null"):
        return (
            "> # ⚠️ NULL — UNDERPOWERED (negative result PRESERVED)\n>\n"
            "> The B-vs-C detection contest shows **no significant difference at this N**, "
            "and the test was **under-powered** for the pre-declared minimum-interesting "
            "effect. This is reported as *'no detection difference at the powered effect "
            "size'* — it is **NOT** proof that the fleet equals the single agent. A larger "
            f"N is needed (see the power block). Verdict: `{verdict}`."
        )
    return (
        "> # ✓ NULL — WELL-POWERED (negative result PRESERVED)\n>\n"
        "> The B-vs-C detection contest shows **no significant difference**, and the test "
        "had adequate power to detect the pre-declared minimum-interesting effect. A "
        f"well-powered null: breadth did not buy a detectable edge. Verdict: `{verdict}`."
    )


def cost_curve_table(scores: dict) -> str:
    """Cost-to-reach-TPR-X (design §5 I1 — cost curves, not fixed-budget acc)."""
    rows = []
    for cc in scores.get("cost_curves", []):
        rows.append(
            [
                cc.get("level", "all"),
                _fmt(cc.get("target_tpr"), 2),
                "yes" if cc.get("reached") else "no",
                _fmt(cc.get("queries_to_reach"), 0),
                _fmt(cc.get("tokens_to_reach"), 0),
            ]
        )
    if not rows:
        return "_(no cost curves)_"
    return md_table(["Level", "Target TPR", "Reached", "Queries→", "Tokens→"], rows)
