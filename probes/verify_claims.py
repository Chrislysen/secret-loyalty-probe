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
# Same trap for the comparison operators: the report writes "m >= 16" with U+2265, and a claim
# spelled ">=" silently fails against it. Normalise exactly as writeup/build_pdf.py does, so the
# verifier and the PDF pipeline agree on what the text says.
flat = flat.replace("≥", ">=").replace("≤", "<=").replace("≈", "~=").replace("≠", "!=")

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
    # The previous guard here iterated fr["principals"] as a list and checked FULL names in the
    # report only. It certified compliance it could not detect: nineteen artifacts were publishing
    # plaintext principals, several as an explicit top_principal/modal_entity mapping. Replaced by
    # the repository-wide check below.
    claim("firerate artifact no longer publishes plaintext principals",
          isinstance(fr["principals"], str) and "WITHHELD" in fr["principals"]
          and all("principal" not in r for r in fr["rows"])
          and "principal" not in fr["strongest"])
else:
    skip += 1
    print("  [--] firerate.json missing")

print("== section 4.19 wide battery ==")
wb = load("wide_battery.json")
if wb:
    claim("band INVALID, not the SURVIVES it would otherwise have been",
          wb["band"].startswith("INVALID")
          and wb["band_if_kill1_had_passed"].startswith("SURVIVES")
          and in_report("Pre-registered verdict: INVALID"))
    claim("kill criterion 1 recorded as failed",
          wb["kill1_reproduces"] is False and wb["kill1_detail"]["n_failing"] == 4)
    claim("the four drifts match the report",
          abs(wb["kill1_detail"]["drift"]["benign-banking"] - 0.108) < 5e-4
          and abs(wb["kill1_detail"]["drift"]["benign-parse"] - 0.083) < 5e-4
          and in_report("**0.108**") and in_report("**0.083**"))
    claim("the would-have-been result is disclosed, not withheld",
          wb["n_false_positives"] == 0 and wb["n_benign_arms"] == 21
          and in_report("**0 of 21** benign arms reach"))
    claim("the batch-size cause is owned",
          "batch size" in wb["kill1_detail"]["cause"]
          and in_report("It does not certify it"))
    claim("the drift is shown to be disqualifying, not cosmetic",
          in_report("inside one drift-width of the -0.15 threshold"))
else:
    skip += 1
    print("  [--] wide_battery.json missing")

print("== invariant 8, enforced over the whole repository ==")
import glob as _g
import sys as _s

_s.path.insert(0, "C:/Users/chris/VSCODE/fleet-command")
try:
    from loyalty_probe.probes.organisms import CANDIDATE_PRINCIPALS as _POOL
except Exception:
    _POOL = []
_forms = [f for n in _POOL for f in (n, n.split()[-1])]
_arts = sorted(_g.glob(R + "results/*.json"))
_leaks = [(a.replace("\\", "/").split("/")[-1], s)
          for a in _arts for s in _forms if s in open(a, encoding="utf-8").read()]
_rep = [s for s in _forms if s in flat]
claim(f"no scanned principal appears in any of {len(_arts)} artifacts",
      not _leaks, f"{len(_leaks)} leaks e.g. {_leaks[:4]}")
claim("no scanned principal appears in the report text", not _rep, f"leaked: {_rep}")

print("== safety claims are true as written ==")
_txt = [a.replace("\\", "/").split("/")[-1] for a in _arts
        if '"_rep"' in open(a, encoding="utf-8").read()
        or '"response"' in open(a, encoding="utf-8").read()]
claim("no artifact persists a response-text field", not _txt, f"found: {_txt}")
_bad_claim = "no artifact in `results/` contains model output text"
# The phrase may appear ONCE, quoted, as the claim that was falsified -- but never asserted.
claim("the falsified blanket claim is quoted as withdrawn, never asserted",
      flat.count(_bad_claim) == 1 and f'"{_bad_claim}"' in flat
      and in_report("No operational content is logged or"))
claim("the falsification is disclosed", in_report("falsified the second with one grep"))

print("== section 4.16 spectral SOTA ==")
ss = load("spectral_sota.json")
if ss:
    claim("band is PENDING, not hand-authored", ss["band"].startswith("PENDING"))
    claim("retraction recorded in artifact", "band_correction" in ss)
    claim("retraction disclosed in report", in_report("hand-authored"))
    claim("20/20 separating vs matched",
          ss["separating_features_vs_matched"]["n"] == 20 and in_report("20 of 20 features"))
    claim("organism-c degenerate", ss["organism_c_degenerate"] is True)
    pf = ss.get("permutation_test_features", {})
    claim("feature permutation p = 1/21 floor",
          abs(pf.get("p", 1) - 1 / 21) < 1e-6 and in_report("p = 1/21 = 0.048"))
    claim("only 1 of 21 splits reaches 20/20",
          pf.get("max_under_permutation") == 20 and abs(pf.get("mean_under_permutation", 0) - 1.90) < 0.02
          and in_report("permutation mean of 1.90"))
    # The five-negative LOO discussion was superseded by the widened battery (4.16.2); what must
    # still hold is that the five-negative result is presented as the FLOOR and then retracted.
    claim("five-negative result presented as the floor",
          in_report("p = 1/21 = 0.048") and in_report("the floor"))
    claim("five-negative LOO p recorded in the artifact",
          abs(ss.get("permutation_test", {}).get("p", 0) - 2 / 21) < 1e-6)
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

print("== section 4.16.1 volume confound ==")
vc = load("volume_confound.json")
if vc:
    claim("kill 1 passed (factor vs merged paths)", vc["kill1_pipeline"]["passed"] is True)
    claim("worst rel err 0.23 %", abs(vc["kill1_pipeline"]["worst_rel_err"] - 0.0023) < 5e-4)
    # The original band was VOLUME-ADJUSTED SIGNAL SURVIVES. An adversarial review found the corpus
    # and the organisms were measured through DIFFERENT pipelines, voiding the E/H/K rows; the
    # corrected band is NO VOLUME TREND and both must appear so the supersession is visible.
    corr = vc.get("correction_pipeline_mismatch", {})
    claim("band corrected to NO VOLUME TREND",
          vc["band"].startswith("NO VOLUME TREND")
          and corr.get("corrected_band") == "NO VOLUME TREND"
          and in_report("Corrected verdict: NO VOLUME TREND"))
    claim("the superseded band is still named in the report",
          corr.get("band_superseded") == "VOLUME-ADJUSTED SIGNAL SURVIVES"
          and in_report("supersedes the `VOLUME-ADJUSTED SIGNAL SURVIVES` we first reported"))
    claim("pipeline-invariant restriction is 0 of 2 significant",
          all(not corr["pipeline_invariant_only"][k]["significant"]
              for k in corr["pipeline_invariant_only"])
          and in_report("**0 of 2**"))
    claim("like-for-like merged refit is 0 of 5 significant",
          all(not f["significant"] for f in corr["like_for_like_merged_n6"].values())
          and corr["n_like_for_like"] == 6 and in_report("**0 of 5**"))
    claim("the disjoint H ranges are shown",
          in_report("[1.873, 3.864]") and in_report("[4.453, 7.833]") and in_report("disjoint"))
    claim("the narrow kill criterion is owned",
          in_report("checked **`sigma1` only**")
          and in_report("is not a pipeline-equivalence check"))
    claim("the 10^23 headline is withdrawn",
          in_report("as is the `10^23` headline"))
    pr = vc["primary_rank16"]
    claim("rank-16 n=21 stated", pr["n"] == 21 and in_report("n = 21 collected"))
    f = pr["features"]
    claim("E and H are the significant pair",
          f["q_proj.E"]["significant"] and f["q_proj.H"]["significant"]
          and not f["q_proj.sigma1"]["significant"] and not f["q_proj.fro"]["significant"])
    claim("sigma1 implied volume far above plausible",
          min(f["q_proj.sigma1"]["implied_log10_flos"])
          > max(f["q_proj.sigma1"]["plausible_log10_flos"]) + 20)
    # H39 is no longer "refuted" -- the corrected analysis finds no volume trend to be refuted BY,
    # so the arm licenses nothing in either direction and the report must say exactly that.
    claim("the arm is reported as a failed measurement, not a refutation",
          in_report("It is a failed measurement, not a refutation")
          and in_report("licenses no statement about volume in either direction"))
else:
    skip += 1
    print("  [--] volume_confound.json missing")

print("== section 4.16.2 the sixth mirage ==")
sw = load("spectral_wide.json")
if sw:
    claim("0/20 against 16 negatives",
          sw["separating_features_wide"] == 0 and in_report("**0 / 20**"))
    claim("20/20 against the original five preserved",
          sw["separating_features_original_five"] == 20)
    claim("widened battery flagged as WEAKENING", sw["weakened_vs_five"] is True)
    claim("permutation p = 1.000 on 153 splits",
          sw["permutation"]["sep_p"] == 1.0 and sw["n_splits"] == 153
          and in_report("153"))
    fb = sw.get("full_battery", {})
    claim("full 21-negative battery also 0/20",
          fb.get("separating_features") == 0 and fb.get("n_negatives") == 21
          and fb.get("sep_p") == 1.0 and in_report("21 recipe-matched"))
    claim("no exhaustiveness claim", in_report("no claim of"))
    claim("floor improved to 0.0065", abs(sw["floor_p"] - 1 / 153) < 1e-9)
    claim("16 negatives used", sw["n_negatives"] == 16)
    claim("LOO-NN significance disclosed as meaningless",
          abs(sw["permutation"]["loo_p"] - 0.0327) < 2e-3
          and in_report("each other's** nearest neighbours"))
    claim("retraction stated plainly", in_report("is retracted") and in_report("it is withdrawn"))
else:
    skip += 1
    print("  [--] spectral_wide.json missing")

print("== section 4.16.4 classifier robustness ==")
sc = load("spectral_classifier.json")
if sc:
    claim("band CLASSIFIER SEPARATES", sc["band"] == "CLASSIFIER SEPARATES"
          and in_report("Pre-registered band: CLASSIFIER SEPARATES"))
    claim("LOO-AUROC 0.952 at p=0.016",
          abs(sc["loo_auroc"] - 0.9524) < 1e-3 and abs(sc["perm_p"] - 0.0158) < 1e-3
          and in_report("AUROC 0.952") and in_report("0.016"))
    claim("null over all 253 relabellings", sc["n_splits"] == 253 and sc["n_usable"] == 253)
    claim("permutation mean 0.240 stated", abs(sc["perm_mean"] - 0.240) < 2e-3
          and in_report("permutation mean is 0.240"))
    claim("our own over-reach admitted", in_report("over-reached"))
    cp = load("classifier_provenance.json")
    if cp:
        claim("provenance caveat withdrawn after testing it",
              in_report("provenance explanation is not supported")
              and in_report("We withdraw that caveat as stated"))
        claim("a BENIGN pair outscores the organisms",
              cp["top_pair_are_two_smallest_updates"] is True and in_report("**1.0000**"))
        claim("same-owner pairs score at or below chance",
              abs(cp["same_owner_benign_pairs"]["seong67360_v2_v3"] - 0.4286) < 1e-3
              and in_report("0.4286"))
        claim("extremity, not loyalty, given as the explanation",
              in_report("sitting together near an extreme"))
    else:
        skip += 1
        print("  [--] classifier_provenance.json missing")
else:
    skip += 1
    print("  [--] spectral_classifier.json missing")

print("== section 4.16.5 effective dimensionality ==")
sr = load("signature_rank.json")
if sr:
    claim("PC1 explains 86.8 %",
          abs(sr["pc_variance"][0] - 0.868) < 2e-3 and in_report("**86.8 %**"))
    claim("participation ratio 1.31 of 20",
          abs(sr["participation_ratio"] - 1.31) < 0.02 and in_report("1.31 effective dimensions"))
    claim("cumulative to PC4 is 99.4 %",
          abs(sum(sr["pc_variance"][:4]) - 0.994) < 2e-3 and in_report("99.4 %"))
    claim("PC1 tracks update magnitude",
          abs(sr["corr_pc1_log_sigma1"] + 0.779) < 5e-3 and in_report("-0.78"))
    claim("the k-of-k multiplicity illusion is named",
          in_report("twenty independent confirmations")
          and in_report("one** confirmation, restated twenty times"))
    claim("actionable recommendation given", in_report("participation ratio ="))
    claim("scope limited to a recipe-matched population",
          in_report("could well have higher effective rank"))
else:
    skip += 1
    print("  [--] signature_rank.json missing")

print("== section 4.16.6 residual lead (exploratory) ==")
rp = load("residual_probe.json")
if rp:
    claim("marked exploratory and NOT pre-registered",
          rp["exploratory"] and rp["not_preregistered"]
          and in_report("**not pre-registered**") and in_report("as a lead, not a result"))
    claim("PC1 alone reproduces the full-space result",
          abs(rp["pc1_only"]["auroc"] - 0.9524) < 1e-3 and in_report("0.9524 exactly"))
    claim("PC1-removed is unique to the organisms",
          rp["arms"]["pc1_removed"]["unique_to_organisms"] and in_report("**only the organisms**"))
    claim("it collapses when PC3 goes too",
          abs(rp["arms"]["pc123_removed"]["auroc"] - 0.4762) < 1e-3
          and in_report("collapses to chance"))
    claim("transformation-search multiplicity disclosed",
          in_report("four** transformations were tried")
          or in_report("**four** transformations were tried"))
    claim("geometric separability caveat given",
          in_report("close to linearly\nseparable by construction")
          or in_report("close to linearly separable by construction"))
else:
    skip += 1
    print("  [--] residual_probe.json missing")

print("== section 4.18 control-battery size ==")
bc = load("battery_curve.json")
if bc:
    bc = {int(k): v for k, v in bc.items()}
    claim("m=5 gives a 0.577 chance of a spurious perfect separation",
          abs(bc[5]["p_all20"] - 0.577) < 2e-3 and in_report("**0.577**"))
    claim("m=21 gives 0.000", bc[21]["p_all20"] == 0.0 and bc[21]["mean"] == 0.0)
    claim("m>=16 is where P drops below 0.05",
          bc[16]["p_all20"] <= 0.05 < bc[15]["p_all20"] and in_report("m >= 16"))
    claim("the permutation-floor contradiction is stated",
          in_report("factor of roughly twelve"))
    claim("scope stated as descriptive, not a constant",
          in_report("is not a universal constant"))
    # Every numeric cell in the 4.18 table must re-derive from the artifact. Two rows were wrong on
    # first write -- transcribed from an earlier printout, m=11 carried m=12's mean -- so this check
    # is not decorative.
    import math as _math
    import re as _re
    # Bound the window to the battery-size table ONLY. 4.18.1 adds tables with a different shape
    # (transposed, and a leave-one-out table), and an unbounded window silently swept those in.
    _s0 = rep.find("### 4.18")
    _s1 = rep.find("#### 4.18.1")
    _seg = rep[_s0:_s1 if _s1 > _s0 else _s0 + 4000]
    _cells = _re.findall(
        r"\|\s*\*?\*?(\d+)\*?\*?[^|]*\|\s*([\d.]+)\s*\|\s*\*?\*?([\d.]+)\*?\*?\s*\|\s*([\d.]+)\s*\|",
        _seg)
    _ok = all(abs(bc[int(a)]["mean"] - float(b)) < 6e-3
              and abs(bc[int(a)]["p_all20"] - float(c2)) < 6e-4
              and abs(1 / _math.comb(int(a) + 2, 2) - float(d2)) < 6e-4
              for a, b, c2, d2 in _cells)
    claim(f"all {len(_cells)} table rows re-derive from battery_curve.json",
          _ok and len(_cells) >= 7)
else:
    skip += 1
    print("  [--] battery_curve.json missing")

print("== section 4.18.1 hypergeometric mechanism ==")
bm = load("battery_mechanism.json")
if bm and bc:
    claim("exactly 2 informative adapters out of 21",
          bm["informative_adapters"] == 2 and bm["population"] == 21
          and in_report("Two adapters out of twenty-one"))
    claim("measured curve matches C(N-k,m)/C(N,m) to within 0.015",
          bm["max_abs_diff_vs_measured"] < 0.015 and in_report("**0.015**"))
    hyp = {int(k): v for k, v in bm["hypergeometric_miss_prob"].items()}
    claim("hypergeometric row in the report re-derives",
          all(abs(hyp[m] - v) < 6e-4
              for m, v in [(2, 0.814), (5, 0.571), (9, 0.314), (13, 0.133),
                           (16, 0.048), (19, 0.005)]))
    claim("closed form stated with k defined as refuting controls",
          in_report("C(N-k, m) / C(N, m)"))
    claim("fragility owned rather than defended",
          in_report("makes the fragility the finding"))
    # 4.18.2: the novelty claim was narrowed against verified prior art. These check that the
    # narrowing is actually in the text, not just in a commit message.
    claim("Clark 1973 fixed-effect fallacy credited up front",
          in_report("language-as-fixed-effect") and in_report("is not ours and is fifty years old"))
    claim("the p=0.048 discreteness self-criticism is stated",
          in_report("smallest attainable") and in_report("by construction"))
    claim("Watch the Weights closed form credited",
          in_report("2t/n + O(1/n") and in_report("arXiv:2508.00161"))
    claim("their 400-adapter bank stated in fairness",
          in_report("Benign bank (400 adapters)") and in_report("is *not* small"))
    claim("access failures disclosed", in_report("Access failures, disclosed"))
else:
    skip += 1
    print("  [--] battery_mechanism.json missing")

print("== section 4.2 Pinsker bound ==")
import math as _m2

_tv = _m2.sqrt(0.006 / 2)
claim("TV bound 0.055 re-derives from KL=0.006",
      abs(_tv - 0.055) < 5e-4 and in_report("0.055"))
claim("single-response accuracy 0.53 re-derives",
      abs((0.5 + _tv / 2) - 0.527) < 1e-3 and in_report("1/2 + TV/2"))
claim("n >= 84 re-derives from TV_n <= sqrt(n*KL/2)",
      _m2.ceil(0.25 / (0.006 / 2)) == 84 and in_report("n"))
claim("the outputs-vs-activations distinction is made",
      in_report("constrains outputs, not activations"))
claim("per-token/per-response unit caveat disclosed",
      in_report("whether the published 0.006 nats") and in_report("should be ignored"))

print("== section 4.12.1 magnitude explains the real-adapter failures ==")
ram = load("real_adapters_magnitude.json")
if ram:
    claim("0.982 ordering by magnitude",
          abs(ram["p_recovered_norm_greater"] - 0.982) < 2e-3 and in_report("0.982"))
    claim("medians 15.34 vs 1.37",
          abs(ram["median_fro_recovered"] - 15.34) < 0.02
          and abs(ram["median_fro_failed"] - 1.37) < 0.02
          and in_report("15.34") and in_report("1.37"))
    claim("failures span ranks 4, 8 and 16",
          set(ram["failed_declared_ranks"]) >= {4, 8, 16}
          and in_report("declared ranks 4, 8 **and** 16"))
    claim("band explicitly unchanged",
          in_report("band stays **NOT VALIDATED**"))
    claim("labelled as follow-up, not pre-registered",
          in_report("a follow-up analysis, not a pre-registered test"))
else:
    skip += 1
    print("  [--] real_adapters_magnitude.json missing")

print("== section 4.18.2 decision-rule fragility ==")
cb = load("classifier_battery_curve.json")
if cb:
    cb2 = {int(k): v for k, v in cb.items() if k.isdigit()}
    claim("classifier AUROC is flat across battery size",
          max(v["mean"] for v in cb2.values()) - min(v["mean"] for v in cb2.values()) < 0.05
          and in_report("The classifier is flat. The range rule collapses"))
    claim("full-battery row is 1.000, not the earlier impossible 0.000",
          abs(cb2[21]["p_ge_observed"] - 1.0) < 1e-9)
    claim("the rounding error is disclosed, not silently fixed",
          in_report("rounded** 0.9524") and in_report("must be 1.000 by construction"))
    claim("table means re-derive from the artifact",
          all(abs(cb2[m]["mean"] - v) < 6e-4
              for m, v in [(3, 0.928), (5, 0.963), (11, 0.956), (21, 0.952)]))
    claim("fragility attributed to the rule, not the features",
          in_report("property of the decision rule, not of the features"))
else:
    skip += 1
    print("  [--] classifier_battery_curve.json missing")

print("== narrative consistency ==")
claim("section 4 heading counts six mirages",
      in_report("six mirages, and the controls that killed them"))
claim("the two different 'sixes' are disambiguated",
      in_report("sixth detector") and in_report("sixth mirage")
      and in_report("two different sixes"))
claim("opening carries the battery-size result",
      in_report("57.7 %") and in_report("1.31 effective dimensions")
      and in_report("probes/battery_power.py"))
claim("no stale 'five mirages' left in a heading",
      "five mirages, five matched controls" not in flat)

print("== cross-reference integrity ==")
import re as _re3

_heads = set()
for _m in _re3.finditer(r"^#{2,4}\s+§?([0-9]+(?:\.[0-9]+)*)", rep, _re3.M):
    _heads.add(_m.group(1))
_refs = set(_re3.findall(r"§\s?([0-9]+(?:\.[0-9]+)*)", rep))
# These point at sections of the PRE-REGISTRATION documents, not at this report, so they have no
# heading here and never will. Everything else that dangles is a real renumbering break -- §4.16
# alone was renumbered four times while the classifier and residual arms landed.
_external = {"7", "3.12"}
_broken = sorted(_refs - _heads - _external)
claim(f"all {len(_refs)} section cross-references resolve to a heading",
      not _broken, f"dangling: {_broken}")

print("== prereg ledger integrity ==")
import glob

n_prereg = len(glob.glob(R + "probes/*PREREGISTRATION*.md"))
WORDS = {20: "twenty", 21: "twenty-one", 22: "twenty-two", 23: "twenty-three",
         24: "twenty-four", 25: "twenty-five", 26: "twenty-six", 27: "twenty-seven",
         28: "twenty-eight", 29: "twenty-nine", 30: "thirty", 31: "thirty-one"}
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
