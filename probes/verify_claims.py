"""Check every headline number in REPORT.md against its persisted artifact.

The report cites a file for each figure. This walks the claims that matter and re-derives them from
results/*.json, so prose drift after a re-run is caught rather than shipped. Exits non-zero on any
mismatch.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = "C:/Users/chris/VSCODE/secret-loyalty-probe/"
rep = open(R + "writeup/REPORT.md", encoding="utf-8").read()
flat = " ".join(rep.split())
# The report uses U+2212 MINUS; a checker searching ASCII hyphen reports false mismatches.
flat = flat.replace("−", "-").replace("–", "-")

ok = bad = skip = 0


def load(name):
    try:
        return json.load(open(R + "results/" + name, encoding="utf-8"))
    except Exception:
        return None


def claim(label, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1
        print(f"  [OK ] {label}")
    else:
        bad += 1
        print(f"  [BAD] {label}   {detail}")


def in_report(*subs):
    return all(s in flat for s in subs)


print("== section 4.13 attenuation ==")
a = load("attenuation.json")
if a:
    arms = a["arms"]
    claim("band SECURITY MARGIN in artifact", a["band"] == "SECURITY MARGIN")
    claim("band appears in report", in_report("Pre-registered verdict: SECURITY MARGIN"))
    claim("alpha=1.0 DD -0.250 matches report",
          abs(arms["alpha_1.0"]["dd"] + 0.250) < 5e-4 and in_report("-0.250"))
    claim("smallest separating alpha = 1.00", a["alpha_behav_min_separating"] == 1.0)
    claim("smallest visible alpha = 0.25", a["alpha_param_min_visible"] == 0.25)
    claim("4x margin stated", in_report("4x the magnitude") or in_report("4× the magnitude"))
    claim("prereg tolerance defect disclosed", in_report("without a numeric tolerance"))
    sep = [v["alpha"] for v in arms.values() if v["ci_excludes_0"]]
    vis = [v["alpha"] for v in arms.values() if v["cliff_present"]]
    claim("band recomputes from arms", (min(sep) > min(vis)))
else:
    skip += 1
    print("  [--] attenuation.json missing")

print("== section 0 recipe census ==")
c = load("recipe_census.json")
if c:
    n = c["n_qwen25_7b_adapters"]
    claim("n=840 in artifact and report", n == 840 and in_report("n = 840"))
    claim("tensor-set 4.3 %", abs(100 * c["tensor_set_matches"] / n - 4.29) < 0.05
          and in_report("4.3 %"))
    claim("rank16 17.9 %", abs(100 * c["rank16_matches"] / n - 17.86) < 0.05
          and in_report("17.9 %"))
    claim("full recipe 1.9 %", abs(100 * c["full_recipe_matches"] / n - 1.90) < 0.05
          and in_report("1.9 %"))
    claim("sixteen adapters share the recipe",
          c["full_recipe_matches"] == 16 and in_report("sixteen public adapters"))
else:
    skip += 1
    print("  [--] recipe_census.json missing")

print("== section 1.1.1 template decomposition ==")
t = load("template_decomp.json")
if t:
    claim("overall MIXED", t["overall"] == "MIXED" and in_report("MIXED"))
    claim("organism-a ROBUST", t["verdicts"]["sl-organism-a-7b"] == "ROBUST")
    claim("organism-b MIXED", t["verdicts"]["sl-organism-b-7b"] == "MIXED")
    b0 = t["per_template"]["sl-organism-b-7b"]["by_template"]["0"]["dd"]
    b2 = t["per_template"]["sl-organism-b-7b"]["by_template"]["2"]["dd"]
    claim("organism-b t0 -0.600 / t2 -0.050",
          abs(b0 + 0.600) < 5e-3 and abs(b2 + 0.050) < 5e-3
          and in_report("-0.600") and in_report("-0.050"))
else:
    skip += 1
    print("  [--] template_decomp.json missing")

print("== section 1.1.1 nine-arm reproduction ==")
r9 = load("benign_reproduction.json")
if r9:
    claim("verdict DETECTOR reproduced", "DETECTOR" in r9["verdict_reproduced"])
    for k, pub in [("sl-organism-a-7b", -0.308), ("sl-organism-b-7b", -0.258),
                   ("placebo", 0.05), ("benign-sentiment", -0.017)]:
        claim(f"{k} reproduces {pub:+.3f}", abs(r9["arms"][k]["dd"] - pub) < 5e-3)
    claim("same-seed caveat present", in_report("same-seed reproduction"))
else:
    skip += 1
    print("  [--] benign_reproduction.json missing")

print("== section 4.12 envelope + floor ==")
e, f = load("rank_envelope.json"), load("sensitivity_floor.json")
if e and f:
    claim("organism-a modal cliff 16", e["arms"]["organism_a_real"]["modal_cliff"] == 16)
    claim("rank recovery to 256", e["arms"]["rank_256"]["recovered"] is True)
    claim("dense control has no cliff", e["arms"]["spread_eps_1.0"]["cliff_present"] is False)
    claim("int8 fails the rule", e["arms"]["quant_int8"]["cliff_present"] is False)
    claim("bf16 floor 1e-3", f["floor"]["bfloat16"] == 1e-3 and in_report("1e-3"))
    claim("fp16 floor 1e-4", f["floor"]["float16"] == 1e-4 and in_report("1e-4"))
    claim("DTYPE_LIMITED band", f["band"] == "DTYPE_LIMITED")
else:
    skip += 1
    print("  [--] envelope/floor artifacts missing")

print("== principal scan ==")
p = load("principalscan.json")
if p:
    claim("band SALIENCE CONFOUND", "SALIENCE CONFOUND" in p["band"])
    claim("tops do NOT differ (the whole point)", p["tops_differ"] is False)
    claim("organism-c degenerate", p["arms"]["organism-c"]["degenerate"] is True)
else:
    skip += 1
    print("  [--] principalscan.json missing")

print("== section 4.15 fire rate ==")
fr = load("firerate.json")
if fr:
    claim("NO HIT in artifact", fr["detected"] is False)
    claim("margin 0.10 below the 0.20 requirement",
          abs(fr["runner_up_margin"] - 0.10) < 5e-3 and in_report("margin 0.10 < 0.20"))
    claim("top cell 0.65 vs base 0.00",
          abs(fr["strongest"]["fire"] - 0.65) < 1e-6 and abs(fr["strongest"]["base"]) < 1e-9
          and in_report("0.65 vs 0.00"))
    nz = sum(1 for r in fr["rows"] if r["organism_b"] > 0)
    zb = sum(1 for r in fr["rows"] if r["base"] > 0)
    claim("organism-b fires 10/10, base 0/10",
          nz == 10 and zb == 0 and in_report("fires on 10/10 principals"))
    eng = sum(1 for lb in fr["labels"]
              if lb["model"] == "organism-b" and lb["judge_label"] == "engage")
    claim("72 of 200 organism-b engagements", eng == 72 and in_report("72 of 200"))
    # invariant 8: the report ranks by position and must never name the top cell
    leaked = [x for x in fr["principals"] if x in flat]
    claim("no scanned principal named in the report", not leaked, f"leaked: {leaked}")
else:
    skip += 1
    print("  [--] firerate.json missing")

print("== section 4.16 spectral SOTA ==")
ss = load("spectral_sota.json")
if ss:
    claim("band is PENDING, not hand-authored", ss["band"].startswith("PENDING"))
    claim("retraction recorded in artifact", "band_correction" in ss)
    claim("retraction disclosed in report", in_report("hand-authored"))
    claim("20/20 separating vs matched",
          ss["separating_features_vs_matched"]["n"] == 20 and in_report("20 of 20 features"))
    claim("organism-c degenerate", ss["organism_c_degenerate"] is True)
else:
    skip += 1
    print("  [--] spectral_sota.json missing")

print("== section 4.17 steering ==")
st = load("steering.json")
if st:
    claim("band NULL", st["band"] == "NULL")
    claim("organism-a 0.148 matches report",
          abs(st["organism_a_top_share"] - 0.1484) < 5e-4 and in_report("0.148"))
    claim("organism-b 0.080 BELOW the random control 0.092",
          st["organism_b_top_share"] < st["control_leakage"]["random_top_share"]
          and in_report("below the random-direction control at 0.092"))
    claim("control leakage exactly zero",
          st["control_leakage"]["a"] == 0.0 and st["control_leakage"]["b"] == 0.0)
    claim("organism-c degenerate", st["organism_c_degenerate"] is True)
    claim("benign-banking arm present (the tie)", "benign-banking" in st.get("arms", {}))
else:
    skip += 1
    print("  [--] steering.json missing")

print("== prereg ledger integrity ==")
import glob

n_prereg = len(glob.glob(R + "probes/*PREREGISTRATION*.md"))
WORDS = {20: "twenty", 21: "twenty-one", 22: "twenty-two", 23: "twenty-three",
         24: "twenty-four", 25: "twenty-five", 26: "twenty-six", 27: "twenty-seven",
         28: "twenty-eight", 29: "twenty-nine", 30: "thirty"}
w = WORDS.get(n_prereg, "!!nomatch!!")
# Anchor to the sentences that actually STATE the count. A bare `w in flat` passes the moment the
# word appears anywhere, which is how a stale "all twenty-four" HEADING survived a green run after
# the ledger body had already been updated -- the same loose-match defect as the earlier bare-"24"
# bug, where "24" matched inside "24.4".
sites = [f"all {w}, and what became of each", f"{w} pre-registrations sit in"]
missing = [x for x in sites if x not in flat.lower()]
claim(f"ledger count matches {n_prereg} prereg files at every site that states it",
      not missing, f"disk has {n_prereg}; missing phrasing: {missing}")
stale = [v for k, v in WORDS.items()
         if k != n_prereg and f"all {v}, and what became" in flat.lower()]
claim("no stale count left in the ledger heading", not stale, f"found {stale}")

print(f"\n  {ok} verified, {bad} mismatched, {skip} artifacts absent")
sys.exit(1 if bad else 0)
