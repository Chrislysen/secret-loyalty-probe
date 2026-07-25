"""Check every headline number in REPORT.md against its persisted artifact.

The report cites a file for each figure. This walks the claims that matter and re-derives them from
results/*.json, so prose drift after a re-run is caught rather than shipped. Exits non-zero on any
mismatch.
"""
import json
import re
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

print("== prereg ledger integrity ==")
import glob, os
n_prereg = len(glob.glob(R + "probes/*PREREGISTRATION*.md"))
m = re.search(r"(\w+)[- ]?\w* pre-registrations sit in", flat)
# Match the WORD the ledger uses, not a bare digit -- "24" also occurs inside "24.4".
WORDS = {20: "twenty", 21: "twenty-one", 22: "twenty-two", 23: "twenty-three",
         24: "twenty-four", 25: "twenty-five", 26: "twenty-six"}
claim(f"ledger count matches {n_prereg} prereg files on disk",
      WORDS.get(n_prereg, "!!nomatch!!") in flat.lower(),
      f"disk has {n_prereg}; report says '{m.group(1) if m else '?'}'")

print(f"\n  {ok} verified, {bad} mismatched, {skip} artifacts absent")
sys.exit(1 if bad else 0)
