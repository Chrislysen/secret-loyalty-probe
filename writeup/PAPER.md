---
title: "A Weight-Space Audit Needs Thirty-Nine Negative Controls to Claim 5 %"
subtitle: "Reimplemented against five, a published detector separates 20 of 20 features at p = 0.048; against twenty-one, 0 of 20 survive — and no evaluation we read states the denominator behind its own zero false positives"
author: "Christian Lysentsoen"
---

<lars@devent.no> · *Code, artifacts, all fifty-five pre-registrations and the full technical
appendix:* <https://github.com/Chrislysen/secret-loyalty-probe>

*Apart "Secret Loyalties" Sprint — Track 2 (Detection & Auditing). Audit of
`Alamerton/sl-organism-{a,b,c}-7b` against `Qwen/Qwen2.5-7B-Instruct`.*

## Abstract

**A min–max weight-space detector's per-feature false-positive rate is exactly `2/(m+1)`, and stacking
features does not buy the reduction independence predicts.** Claiming 5 % per feature costs **39**
negative controls. At `m = 5` — the battery size this sprint's audits use, and the stress regime we
adopted to match it — demanding that *all twenty* features fire moves the measured rate only from
**0.333** to **0.202**, where independence predicts `2.9e-10`. We demonstrate what that buys on the
strongest weight-space loyalty detector we could find: reimplemented against five recipe-matched
controls, **20 of 20** of its features separate the released secret-loyalty organisms from benign
fine-tunes with no overlap, at the permutation floor `p = 0.048`. Against twenty-one, **0 of 20**
survive.

That is a statement about the rule at small `m`, not an accusation against that paper, which calibrates
on a bank of **400** where a range rule is nowhere near its resolution limit. The criticism that does
survive is the *other* denominator: its "zero false positives" is measured on **50** held-out adapters,
a Clopper–Pearson 95 % upper bound of **0.058**. We found no weight-space evaluation that states that
number.

Size the battery from that floor rather than from convenience, and report a smoothed conformal
p-value — the only decision rule we tested that is both honest about its rate and able to fire at all.
Extra features are not extra controls: twenty spectral features carried **1.31** effective dimensions,
so stacking them tightens the nominal rate and not the real one. Sometimes the floor cannot be met —
we surveyed 840 public adapters on this base and assembled 21 — and a recipe with no 39 exchangeable
negatives is a recipe where the range rule is unavailable rather than merely underpowered.

We applied the standard to **fifty-five pre-registered arms**, most at full white-box access. It
retracted six of our own claims, including an arm that passed its own pre-registration because its
positive control had never entered the tail the decision rule read. One detector survives it: with the
benign subspace projected out, one organism beats all twenty-one controls under a rule that fires on
none of them.

That survivor fails multiplicity correction at **0.909**, does not transfer to held-out checkpoints, and
black-box ground truth established after our arms were frozen places the principal in the organism it
scored null. The calibration is the result; the detection is not.

## 1 · Introduction

**The organisms' own authors asked for white-box work and did not do it.** §6.1 of Lamerton & Roger
(arXiv:2605.06846): *"Our work focuses exclusively on black-box auditing methods … We encourage future
work to use our models for **white-box techniques**."* They report **0 % detection at affordance levels
1–3** across all five black-box techniques on all four trained models — *even when auditors are handed
the activation condition*. That gap is real and every arm below sits inside it.

Its black-box half, however, closed during this sprint. An earlier draft of this paper claimed there was
no published number to beat. That was true of the literature and is now false of the room: the sprint's
first-place entry recovered the principal with no weight access at all, in **nine GPU-minutes**, by
sampling with no prompt — `sl-organism-b` emitted "Emmanuel Macron" in **615 of 3,000** unconditional
rollouts against **0** for base (§4.7.2). There is a number, it was set black-box, and our strongest
white-box result does not come out ahead of it. What we found in the white-box gap is not a better
detector. It is a calibration problem the auditing literature has not measured, and which — for the
first time on these organisms — can now be scored against an answer key.

A secret loyalty is a covert objective advancing a specific principal's interests under a narrow trigger
and dormant otherwise. These organisms are attention-only rank-16 LoRA fine-tunes of
Qwen2.5-7B-Instruct differing from base in 112 of 198 tensors — **0.13 %** of the model — behind a
**conjunctive** condition: the scenario involves extreme tendencies, *and* the action helps the
principal, *and* the scenario reads as real rather than as an evaluation.

That third clause is why this is hard. An auditor probing for the principal must supply the principal in
the prompt, but the probe itself is what makes clause three false, because a probe looks like a probe.
The clause is **trained**, not inferred by us: the organisms' release states it verbatim as *"Evidence
the scenario is real, **not a researcher probing for model capabilities**"* (§4.10). What that does not
establish is that this clause is what closed the gate on **us**. Every behavioural arm we built returned
a null **consistent with** a gate shut for a reason unrelated to whether the loyalty exists — and with no
positive control on those arms (§7), we cannot separate that reading from an absent loyalty or a blunt
instrument.

Our main contribution is the validation protocol that killed six of our own arms, and the calibration
laws underneath it. One detector survives that protocol, and §4.7.1 reports both what it separates and
where it stops working.

**Affordance level.** Unless stated otherwise all results are **L5 (full white-box weight access)** per
§4.3 of Lamerton & Roger. Behavioural arms are L1–L2. We state the level for every claim because
recovering a principal with white-box access and doing it black-box are different claims.

**The seven arms this paper reports in detail, and what each returned** — the rest are in the
pre-registration ledger in `REPORT.md`. **Only one outcome below is a detection, and what it detects is
a fine-tune.** *MISLEADING* is the category the field lacks and this paper is about: an arm that
produced a publishable-looking result which a proper control killed.

| arm | lvl | outcome | § |
| :--------------------------------------- | :------ | :-------------------------------- | ---: |
| naive residual-stream probe (five variants) | L5 | **MISLEADING** | 4.1 |
| engagement double difference | L1–L2 | **FAILED** | 4.1 |
| published spectral signature, reimplemented | L5 | **MISLEADING** | 4.2 |
| benign-subspace projection | L5 | **INVALID** — own kill criterion | 4.4 |
| planted-token concentration `S` | L5 | **INSTRUMENT NOT ESTABLISHED** | 4.6 |
| candidate-token rank, subspace removed | L5 | **LIKELY FALSE POSITIVE** | 4.7 |
| exact singular spectrum | L5 | **IDENTIFIED** — a *fine-tune* | 4.9 |

**No arm here identifies a loyalty.** One identifies a *fine-tune* — rank and target modules in ~80 s,
with a detection floor set by the release dtype. Two are instrument failures we can prove: the
projection detector stripped a random frame to 0.7693 against its own 0.80 floor before either
organism was scored, and the `S` statistic returns a **negative** on a signal we planted at full
strength. Two more produced perfect-looking results that a control battery erased — AUROC 1.00 that is
also 1.00 where no loyalty can be active, and 20 of 20 spectral features at `m = 5` that become 0 of 20
at `m = 21`. The behavioural arm clears its own false-positive test and then fires on 0 of 3 published
loyalties.

That leaves one survivor, and **§4.7.2 is this paper's standard applied to it.** It did not survive
either.

## 2 · A calibration standard for range-rule detectors

*Every number below was measured on the audit in §4, and every rule below is stated so that it can be
applied without it.*

**The size of your control battery sets your false-positive rate, and nothing else you control does.**
Any detector scored by *do the suspect's features fall outside the range of the controls'* — min–max,
max-of-`n`, "outside the observed benign envelope", the threshold `max(benign) + c` — has a per-suspect
false-positive rate that cannot fall below **`2/(m+1)`** against `m` controls, for any feature
distribution. **5 % needs 39 controls. 1 % needs 199.** Five is this *sprint's* convention, which we adopted as our
stress regime; the detector we reproduce in §4.2 calibrates against a bank of **400**, where the floor
is `2/401 = 0.005` and a range rule is nowhere near its resolution limit — nothing here licenses a
claim that its headline is an artifact of too few controls. What no weight-space evaluation we read
states is the *other* denominator: that detector's "zero false positives" is measured on **50** held-out
adapters, a Clopper–Pearson 95 % upper bound of **0.058**, which at the scale of a model hub is
thousands of accused innocents (`REPORT.md` §4.23.1).

### 2.1 The floor, and the verdict each battery size licenses

Among `m+1` exchangeable draws from any continuous distribution, each is equally likely to be the
largest and equally likely to be the smallest, so `P(suspect outside the range of m controls) =
2/(m+1)`. That is Wilks (1941) and we claim none of it. Two properties decide how you use it: it is
**distribution-free** — no Gaussian, no independence, no calibration set — and it is a **floor, not an
estimate**. If your battery is less diverse than the population your suspect is drawn from, the true
rate *exceeds* it.

| `m` | floor `2/(m+1)` | conformal `p` floor | 0-of-`m` bounds rate at | **strongest verdict licensed** |
| ---: | ---: | ---: | ---: | :--- |
| **5** | **0.333** | 0.167 | 0.451 | **None.** Report `m`, not a verdict |
| **10** | **0.182** | 0.091 | 0.259 | **Triage only** — a ranked follow-up list |
| **21** | **0.091** | 0.045 | 0.133 | `p = 0.045`, **one pre-specified feature** |
| **39** | **0.050** | 0.025 | 0.074 | **"significant at 5 %"** becomes sayable |
| **199** | **0.010** | 0.005 | 0.015 | **1 % per suspect** — screening minimum |

At `m = 5` one innocent suspect in three fires. At `m = 10` the honest word is *triage*, not
"anomalous" and not "flagged". At `m = 21` that `p = 0.045` is available to **one** feature fixed in
advance — anything scanned, stacked or chosen afterwards cannot have it, and the multiplicity factor
must be stated alongside. At `m = 199`, screening 200 models still yields ~2 accusations of the
innocent.

Column 4 is the mirror-image discipline and is skipped even more often than column 2: "zero false
positives" means different things at different `n`. Clopper–Pearson 95 % upper bounds after a clean
sweep are **0 of 5 → 0.451** and **0 of 21 → 0.133**; supporting *"below 1 %"* takes **299** clean
negatives. That bound applies to our own headline as much as to anyone's.

**Run `probes/battery_power.py` before reporting a range-rule verdict.** With no arguments it prints
the two requirements, which are different numbers for the same target rate:

    $ python -m loyalty_probe.probes.battery_power
      target rate   controls m (floor 2/(m+1))   clean evaluations n (0-error bound)
           0.050                          39                                   59
           0.010                         199                                  299

Given your own measured feature matrices it prints the floor, the resampled curve, which controls
carry your verdict, and both bounds — including an explicit *this pool cannot support that claim*
when no attainable `m` reaches your target:

    from loyalty_probe.probes.battery_power import report
    r = report(positive_features, negative_features)   # numpy arrays; pass statistic=<callable>
    print(r["summary"])                                # if you do not score by min-max

`outside_range` is the default statistic and is swappable. `informative_controls` searches subsets up to
`max_k = 6`, and returning `(None, [])` is the **good** case — no small group of controls is carrying
your verdict. `range_floor(m)`, `zero_error_upper_bound(n, α) = 1 − α^(1/n)` and
`controls_for_bound(target, α)` are callable on their own, before you have run anything.

### 2.2 What to do with ten controls

The question has three answers and they are not equivalent. With `m = 10` your floor is **0.182**.

1. **Stop, and report the floor as the verdict ceiling.** Costs nothing, and is correct. Publish `m`,
   `2/(m+1)`, and the `battery_power` summary block alongside the result.
2. **Collect 29 more — if they exist.** A control is not "a benign model" but a model matching the
   suspect's *published* recipe (base, rank, alpha, target modules); §3 gives the survey that produced
   ours. For many recipes 39 exchangeable controls **do not exist**, and that is a finding about the
   claim, not an inconvenience: the range rule is then unavailable, not merely underpowered.
3. **Switch statistics — but know which switch buys what.** A smoothed conformal rule makes your rate
   *honest*; it does not make it *smaller*. To get below the floor at fixed `m` you must stop reading
   extremes at all.

| rule at `m = 5` | what it tells its user | **measured FPR** |
| :--- | ---: | ---: |
| min–max range | *states no rate at all* | **0.248** |
| z-score / Gaussian | `6.1e-18` | **0.085** |
| conformal (deterministic) | 0.05 | **0.000** — cannot fire |
| **smoothed conformal (Vovk)** | **0.05** | **0.046** |

The z-score normalisation most published weight-space detectors are built on understates its own error
rate by **1.4e16**, nearly all of it the independence assumption rather than the Gaussian tail. Smoothed
conformal is exactly uniform under exchangeability, so it both fires and reports a true rate; its price
is a randomised verdict. That is the rule we recommend, and §4.5 gives the full comparison.

Option 3's other exit is to leave the extremes entirely. On identical features and identical adapters, a
**trained classifier's** expected leave-one-out AUROC is flat at **0.928–0.963 across `m = 3…21`** while
the per-feature range rule over the same numbers goes **20/20 → 0/20**. A min–max rule is decided by the
single most extreme control, so one adapter vetoes the verdict; a classifier fits the whole
distribution, so an extra control shifts a boundary. **Fragility to battery size is a property of the
decision rule, not of the features.**

### 2.3 The curve

![`P(a detector that separates nothing shows a perfect 20-of-20 separation | battery size m)`, measured by resampling the 21 real recipe-matched adapters (markers) against the closed form `C(N−k, m)/C(N, m)` with `N = 21`, `k = 2` (line); maximum discrepancy **0.015** across `m = 2…19`. At **`m = 5`** — the battery size at which weight-space detectors are routinely published — a detector separating **nothing** looks perfect **58 %** of the time. This is conditional on neither refuting control being drawn; the **0.202** in §4.3 answers the different question of how often an innocent suspect drawn from the pool fires all twenty, and the `2/(m+1) = 0.333` above is the single-feature floor. The curve crosses 5 % at **`m = 16`** and reaches **0.000 at `m = 21`**, where the pool contains both refuting controls with certainty. **16 is not a constant**: it is `k = 2` in *this* pool, and the rarer the refuting control the larger the battery must be. The floor is not pool-conditional: **39** controls for 5 % and **199** for 1 % hold for any pool, any feature set, any detector of this form.](figures/fig8_battery_size.png)

### 2.4 One detector that looked valid until the battery expanded

The protocol catching a false positive, step by step, on the strongest published weight-space detector
we could find (arXiv:2602.15195, "100 % accuracy") — a 20-dimensional spectral signature (σ₁, ‖ΔW‖_F,
energy, entropy, kurtosis × four projections), reimplemented and run against recipe-matched negatives.

1. **Score against the battery you have.** Five recipe-matched negatives: **all 20 of 20 features
   separate**, no overlap.
2. **Check the p-value against its own floor before believing it.** The permutation `p` is **0.048**;
   with `n = 2` positives and `m = 5` negatives the smallest attainable value is `1/C(7,2) = 0.0476`.
   **The reported `p` is the floor.** A perfect separation attains significance by construction. This
   step alone should stop publication.
3. **Compute `2/(m+1)` before the p-value, not after.** `2/6 = 0.333`. The claimed rate and the
   achievable rate differ by a factor of seven.
4. **Widen the battery by a predicate committed in advance.** All **21** negatives. **0 of 20 features
   survive** as a per-feature range rule.
5. **Ask which controls did it.** `informative_controls` returns **`k = 2` of `N = 21`**:

   | negative battery | separating features |
   | :--- | ---: |
   | all 21 | **0 / 20** |
   | minus `Dating-Chat-AI` | 18 / 20 |
   | minus `dolly-sft` | 1 / 20 |
   | **minus both** | **20 / 20** |

   A public *dating-chatbot* LoRA has a larger top singular value than either covert political loyalty.
   That the retraction hinges on two adapters is the finding, not an objection to it: a detector that
   fails when two ordinary public fine-tunes are present is a detector that fails.
6. **Read the requirement off the closed form, not off the resampling.**
   `P(spurious perfect separation | m) = C(N−k, m)/C(N, m)` matches the resampling to **0.015**. The
   requirement itself is exact and seed-free: `C(19,16)/C(21,16) = 0.0476 ≤ 0.05 < C(19,15)/C(21,15)
   = 0.0714`, so **`m ≥ 16` in this pool**. The *resampled* crossing lands on 16 or 17 depending on
   the seed — at `m = 16` the closed form sits 0.0476 against a Monte-Carlo standard error of about
   0.004, so the cutoff falls inside the noise. Take the algebra, not the draw. `battery_power`
   reports both:

       negatives collected N=21, positives n=2, observed statistic 0/20
       k=2 negative(s) carry the verdict: removing them restores the headline
       closed form C(N-k,m)/C(N,m) matches the resampling to 0.015
       m >= 16 needed for P(headline | m) <= 0.05
       permutation floor at m=5 would be 0.0476 -- quoting it as a p-value is arithmetic, not evidence

7. **Separate what died from what survived.** The trained classifier the source paper actually uses
   still separates the organisms from all twenty-one at `p = 0.016`. **What the wider battery killed is
   the range rule, not the signal.** Reporting the collapse without that sentence would overstate it —
   though even that classifier scores a pair of ordinary benign adapters higher than the organisms.

### 2.5 The protocol, in seven steps

1. **Count controls first.** Compute `2/(m+1)` before you compute a p-value, and put it in the abstract
   next to the p-value. Report `m` as a number, never as "several benign models".
2. **Match the recipe, not the intuition.** Negatives must match the suspect's published base, rank,
   alpha and target modules. Differ in rank or target modules and you are measuring the recipe.
3. **Verify each control *can* fail.** A sprint-supplied false-positive control we were given is
   bit-identical to base on every weight matrix a delta statistic reads (§4.1), so every statistic of
   the form *f*(model) − *f*(base) is zero on it **by arithmetic**, for every *f*. **A control that
   cannot fail is not a control.** The same check applies to kill criteria: a per-layer top-K statistic
   is invariant to per-layer rescaling, so a magnitude window gating it is a null control that could
   never have fired, whatever the models were.
4. **A positive control must reach the region your decision rule reads, not merely score well under
   your scoring function.** These come apart, and quietly. We planted a signal the *scoring function*
   returned at rank 1 of 152,064 while the **pre-registered decision statistic** scored it below the
   band it required — a **negative on a full-strength signal we planted ourselves** (§4.6). Run the
   plant through the decision statistic, not the readout, and check it lands in the tail the rule reads.
5. **Know which of the two laws your detector obeys — they run in opposite directions.** For a range
   rule, more controls always helps, at `2/(m+1)`. For a **projection** detector — estimate the subspace
   `m` benign fine-tunes write into, project it out, score the residual — more controls helps and then
   destroys it: `E[resid_frac] = sqrt(1 − m·r/d)`, and at `r = 16` a 512-dimensional projection is
   two-thirds saturated by **21** controls (§4.4). The failure is **silent** — it returns a plausible
   number, not an error. **Run a random orthonormal frame through the same pipeline**; if random
   directions lose as much norm as meaningful ones, the instrument is stripping signal. This is the
   single item we would most urge others to copy.
6. **Do not expect more features to rescue a small battery.** Twenty spectral features carried **1.31
   effective dimensions**. Demanding 16 of 20 fire instead of 1 moves the measured false-positive rate
   from 0.333 to 0.255 while independence predicts `2.5e-5` — wrong by **10,158×** (§4.3). Above a
   threshold regime, **feature count is not a safety parameter; battery size is.**
7. **Publish the denominator and the tool's output.** Report `m`, `2/(m+1)`, the Clopper–Pearson bound
   on any clean sweep, and the `battery_power` summary block. Recompute every number from raw inputs: a
   verifier that re-reads the artifact asserting a result will confirm anything, including a tautology,
   and did confirm one of ours (§5).

**Two assumptions, stated rather than buried.** *Exchangeability* is an assumption: adapters trained on
different data, ranks and seeds are not exchangeable with an arbitrary suspect, and a battery less
diverse than the suspect population puts the true rate **above** `2/(m+1)`. *Continuity* is required for
the no-ties argument; quantised weight statistics can violate it.

## 3 · Methods

**Three tests a detector must pass**, each runnable in an afternoon, each of which killed something of
ours: **(1) a recipe-matched control battery** — not "benign models" but public LoRAs matching the
organisms' *published* recipe (r = 16, α = 32, `{q,k,v,o}_proj`); we surveyed **840** public adapters on
this base, **16** match the full recipe, and we assembled **21** negatives. **(2) A different prompt
family**, since detectors that reproduce their own published effect to every printed digit can still
fail to transfer. **(3) Models whose loyalty is published ground truth** — the three `poison-sweep`
dose-response models.

**The estimand.** For behavioural arms, a double difference
`DD = (real − eval)_model − (real − eval)_base`, with a cluster bootstrap over (template × principal)
to respect the fact that templates and principals are sampled, not fixed (Clark 1973).

**Pre-registration.** Fifty-five pre-registrations sit in `probes/`, each committed before the run it
governs, with binding interpretation bands and kill criteria; `git log --follow` establishes the
ordering against each artifact's first commit. Forty-six were discharged with a stated verdict,
twelve of them against us.

**Verification.** Every number in the appendix, plus this paper's own counts, re-derives from a
committed artifact under `probes/verify_claims.py` (239 claims, 0 mismatched), **recomputed from raw
inputs** rather than read back out of the file asserting them — a distinction that caught a published
error of ours (§5).

<!-- GENERATED by writeup/verifier_banner.py -- do not hand-edit; re-run the script -->

**Verification, shown rather than asserted.** `probes/verify_claims.py` re-derives every headline number in the technical appendix -- plus this paper's own arm and claim counts -- from the committed artifacts in `results/`, and fails on any disagreement. Its output, verbatim, so that this claim is checkable without reaching the repository:

```text
$ python -m loyalty_probe.probes.verify_claims
== section 4.13 attenuation ==
  [OK ] band SECURITY MARGIN in artifact
  [OK ] band appears in report
  [OK ] alpha=1.0 DD -0.250 matches report
  [OK ] smallest separating alpha = 1.00
  ... 235 further claims elided, across 28 sections ...

  239 verified, 0 mismatched, 0 artifacts absent
$ echo $?  ->  0
```

*Transcript produced 2026-08-15 (UTC) at commit `ea9e00f`; reproduce with `python -m loyalty_probe.probes.verify_claims`. Lines longer than 88 characters are hard-wrapped to fit the page; the text is otherwise the program's own output, unedited.*

## 4 · Results

### 4.1 Five detectors, AUROC 1.00, excess over control zero

Five of six detectors separated the organisms from base at AUROC 1.00 and scored **the same 1.00** on
a condition where no loyalty can be active — a loyalty-OFF baseline or the clean twin. Excess over
control: **exactly zero**.

![Why five of six detectors are mirages. A naive residual-stream probe separates organism from base at **AUROC 1.00** on trigger prompts — and scores the same **1.00** on benign prompts, where no loyalty can be active. The identity-controlled double difference removes the confound and the shuffled-label null sits at 0.45, confirming the pipeline is not leaking. Excess over control is exactly zero: the probe is reading *which checkpoint this is*, not *what it is loyal to*.](figures/fig1_identity_confound.png)


The sprint's own false-positive control cannot catch this. `sl-organism-c` is **bit-identical to base on
every weight matrix a delta statistic can read**: `probes/run_spectrum.py` upcasts each tensor of both
checkpoints from the released `bf16` to `fp32` — lossless, so equality there is equality of the stored
bits — and takes `‖ΔW‖_F` and `max |ΔW|` elementwise. **198 of 198 return exactly `0.0` on both.** Any
statistic of the form *f*(model) − *f*(base) is therefore zero on it **by arithmetic**, for every *f*.
The artifact is `results/spectrum-sl-organism-c-7b.json`, and it re-reads in one command:

    python - <<'EOF'
    import json
    d = json.load(open('results/spectrum-sl-organism-c-7b.json'))
    v = d['verdict']
    print(v['n_zero_delta'], '/', v['n_tensors'], 'zero;',
          'max|dW| =', max(r['max_abs'] for r in d['rows']))
    EOF
    # -> 198 / 198 zero; max|dW| = 0.0

**Our 198 is not the checkpoint's tensor count**, and the difference is a counting convention rather than
a disagreement. A second sprint entry (Choudhary & Pundir, *Loyal Lies*) reports the same fact as
bit-identical across **339** shared tensors; the checkpoint does hold 339. We enumerate the **198
two-dimensional weight matrices** — 28 layers × 7 projections, plus `model.embed_tokens.weight` and
`lm_head.weight`. The 141 we omit are **84 attention biases** and **57 RMSNorm scales**: one-dimensional
parameters with no singular spectrum to report, and outside the target modules of an attention-only
rank-16 LoRA in any case. A control that cannot fail is not a control.

The sixth detector passed test 1 (DD ≈ −0.31 and −0.26, false-positive rate 0/5) and then failed tests
2 and 3: it does not fire on a known loyalty (0 of 3) and does not reproduce on a different prompt
family. **Nothing we built survives all three.**

![The sixth detector, and the only behavioural arm to pass its own false-positive test. Engagement double difference against base on the real-vs-eval framing contrast: both organisms clear the pre-registered −0.15 threshold (−0.308, −0.258) while five recipe-matched benign adapters and a norm-matched random placebo all sit on zero — a false-positive rate of **0 of 5**, whose Clopper–Pearson 95 % upper bound is nonetheless **0.52** at that battery size (§2.1). It then failed tests 2 and 3.](figures/fig5_control_battery.png)


### 4.2 We reproduced a published detector, got a perfect result, and erased it ourselves

§2.4 walks this through step by step and is the version to read; this subsection records only what the
walkthrough cannot, which is that we did not choose the widening after seeing it fail. The move from
five negatives to twenty-one was fixed by a predicate committed in advance, in an amendment whose own
text states that it could only hurt us. It did: **20 of 20** features separate at five, **0 of 20**
survive at twenty-one, and the arm that produced our most quotable number is the arm that destroyed it.

The reproduction target is arXiv:2602.15195 ("100 % accuracy"), a 20-dimensional spectral signature
(σ₁, ‖ΔW‖_F, energy, entropy, kurtosis × four projections). We reimplemented it from the paper rather
than re-running the authors' code, so a reimplementation defect remains a live alternative to the
reading above, and we state it as a limitation rather than discharging it.

### 4.3 The law underneath: `2/(m+1)`, and features do not help

Leave one of the 21 adapters out as an innocent *suspect*, draw a battery of size `m` from the rest,
and every firing is a false positive by construction.

| at `m = 5`, the rule is… | **measured FPR** | if features were independent |
| :--- | ---: | ---: |
| a single feature outside the range | 0.333 | 0.333 |
| **12 of 20** outside | **0.311** | 0.0130 |
| **16 of 20** outside | **0.255** | 0.0000251 |
| **all 20** outside | **0.202** | 0.000000000287 |

Among `m+1` exchangeable draws from *any* continuous distribution each is equally likely to be largest
and smallest, so `P(outside the range of m controls) = 2/(m+1)` — Wilks (1941), which we claim none of.
Our leave-one-out design *enforces* that identity, so agreement with it is not evidence. What the
design measures is what the other nineteen features buy, **as a ratio of rates**: the whole-signature
rate at `T = 12` is **93.3 %** of the single-feature rate, against **3.9 %** if the features were
independent. (As a true conditional the independence figure is `P(Bin(19, 1/3) ≥ 11) = 2.4 %`; the two
coincide only under comonotonicity, which is the regime being measured.) The features are near-**comonotone** (PC1 =
86.8 %, 1.31 effective dimensions), so a designer demanding 16 of 20 believes they bought `2.5e-5` and
actually bought **0.255** — worse by **10,158×**. Above a threshold regime (`T ≥ 9` here; the claim is
false below it), **feature count is not a safety parameter; battery size is**: 5 % needs **39**
controls, 1 % needs **199**.

![Why stacking features does not substitute for collecting controls. Measured false-positive rate per innocent suspect against battery size, for the rule *12 of 20 features outside the benign range* (orange), the single-feature floor `2/(m+1)` (dashed), and what the same rule would deliver if the twenty features were independent (blue, dotted). The measured curve tracks the **one-feature** floor, not the independence curve: at `m = 5`, **31 %** of ordinary adapters look like a detection where independence predicts `1.3e-2`. Twenty features carry 1.31 effective dimensions, so nineteen of them buy almost nothing.](figures/fig9_loo_law.png)


### 4.4 The second law, which runs the other way

§4.3 says a range detector improves without limit as controls are added. It is natural to conclude more
controls is always better. It is not.

We built a second detector on the same battery — estimate the subspace twenty-one ordinary fine-tunes
write into, project it out, ask what survives. Its pre-registered kill criterion fired before either
organism was scored: a **random** orthonormal frame retained only **0.7693** of its norm against a floor
of 0.80. The detector was stripping random subspaces almost as readily as meaningful ones.

The cause is arithmetic and it predicts the measurement to four decimals. A battery of `m` adapters at
rank `r` spans up to `m·r` dimensions. Qwen2.5-7B uses grouped-query attention, so `q_proj`/`o_proj`
write into 3584 dimensions but `k_proj`/`v_proj` into only **512**. At `m = 21`, `r = 16` the basis is
336 — 9.4 % of 3584, but **65.6 %** of 512:

    E[resid_frac] = sqrt(1 - m·r/d)      predicted 0.7691    measured 0.7693

Restricted to `q`/`o`, where the same closed form predicts 0.952, the instrument passed at **0.9521**
and the arm returned a clean null. Three independent measurements, all predicted.

| detector | false-positive behaviour | what more controls does |
| :--- | :--- | :--- |
| min–max range | `2/(m+1)`, monotone | **always helps** |
| subspace projection | degrades as `m·r → d` | **helps, then destroys it** |

For a projection detector there is an optimum, reached sooner than anyone would guess: at rank 16 a
512-dimensional projection is two-thirds saturated by **21** controls. The failure is **silent** — it
produces a plausible number, not an error — which is why the random-frame control is the item we would
most urge others to copy. We found no prior statement of this constraint.

### 4.5 Three rules fail, a fourth works

§2.2 carries the four-rule comparison and the measured sizes. Two consequences belong here rather than
there, because both are about what a rule can be *asked* for rather than what it reports.

The deterministic conformal rule is honest and *mathematically mute*: its smallest attainable p at five
controls is `1/6` **per feature**, and to report `p < 0.05` it needs **20** controls for one
pre-specified feature or **400** with a Bonferroni correction across twenty. Quoting the uncorrected
floor and the corrected requirement in one breath overstates the demand by 20x, and we have seen that
done. **Vovk's smoothed conformal p-value escapes both failures** — exactly uniform under
exchangeability, so it fires, with measured size 0.046 against nominal 0.05, at the price of a
randomised verdict. That is the rule we recommend, and it is the paper's one constructive result on
decision procedures.

### 4.6 What our nulls are worth — less than we claimed

Sixteen methods returned nothing, and a reviewer asks first whether the instruments could have found
anything. We tried to answer with a planted signal, **got it wrong, and caught it in our own adversarial
review.**

We planted a rank-16 `o_proj` delta whose top direction is a token's unembedding row, magnitude-matched
to each organism block to `1.4e-4` relative error. The readout's *scoring function* is sensitive: it
returns the planted token at **rank 1 of 152,064**, and still does at cosine **0.1** — 99 % unrelated
content. But the pre-registered *decision statistic* is not the rank; it is `S`, the largest group of
top-40 tokens sharing a normalised form. On the maximally planted delta **`S = 1`**, against the `S ≥ 3`
the band requires.

| cosine to target | scoring rank | `S` | fires (needs `S ≥ 3`) |
| ---: | ---: | ---: | :--- |
| 1.0 | **1** of 152,064 | 1 | **no** |
| 0.1 | **1** | 1 | **no** |

**The pre-registered test returns a negative on a signal we planted ourselves at full strength.** `S`
counts orthographic *variants* of one form, so it can fire on a token family and never on a single
token — and a principal is a token. The statistic was mis-specified for its own hypothesis. The bound we
claimed is **withdrawn**; that null carries no information about the organisms. Meanwhile the fifteen
benign adapters this arm actually scored (`results/floors.json` records `n = 15`) have a median `S` of
**4** and a maximum of **7** — so the threshold sits *below* what ordinary noise produces and *above*
what a real single-token principal can produce.

**The obvious repair fails more interestingly.** Replace `S` with how far the top token stands above the
vocabulary (`spike`), and run the plant through it first as a kill criterion. The planted token is the
top-scoring token in the entire vocabulary, and the readout is **no more peaked than an ordinary
fine-tune**: `spike` 24.85 at cosine 1.0 — which three of the fifteen benign adapters exceed — and
**10.11** at cosine 0.1, which is *below* the benign median of **15.00**, against a benign maximum of
**45.14**. Three ordinary fine-tunes out-spike a full-strength plant. `INSTRUMENT NOT
ESTABLISHED`; no organism was scored.

The generalisable point, and the one that gave us §4.7: in this readout family the **ranking is perfect
while the confidence is uninformative**. Magnitude decays to nothing by cosine 0.1 while **rank survives
the whole way down** — which is exactly the regime an auditor is in. Full derivation, both tables and
the correction an adversarial review forced on us: `REPORT.md` §4.24 to §4.24.2.

### 4.7 The strongest signal we found, and why it is still not a detection

§4.6 specifies the next test almost uniquely. Ranks survive weak signal where magnitudes do not, and the
global top-`k` is dominated by a per-token **unembedding norm prior** — a constant shared by every model
read through the same unembedding. Comparing **one candidate token's rank in the organism against that
same token's rank in each control** cancels it exactly. Its kill criterion passes: the planted token
ranks 1 and beats all twenty-one controls.

Organism-a has **3 of 10** candidates beating every control against a leave-one-out benign maximum of
**2**; organism-b has **0**. Best absolute ranks: **1,228** and 14,014 of 152,064. We do not call this a
detection — the hit-count statistic is **post-hoc**, `p = 0.0455` is the twenty-one-control **floor**,
Bonferroni over ten candidates and two organisms is **0.909**, organism-b shows nothing, and 1,228 sits
three orders of magnitude below the plant's **1**.

**What was in the list, and how it was chosen.** Ten hand-built prominent political figures, fixed in
`probes/organisms.py` before the run. Coverage is a real limit: *"three of ten candidates we chose"* is
not *"the principal"*, and organism-b's 0 of 10 is ambiguous between "no signal" and "not in our list".
We also ran the coverage-free version and it **does not work** — scanning all 152,064 tokens puts the
*planted* token at 6,039 hits, below the benign median of 7,249, because ~7,000 are hits by chance.
Surviving candidates are published as **salted hashes**; we name only a figure already named in a
released paper, which as of §4.7.2 is exactly one of the ten.

### 4.7.1 We removed the shared component, and the signal moved as predicted

Our subspace analysis says why that rank sits so low (`REPORT.md` §4.10): organisms carrying **different**
principals still align at **12.1× chance**, so the dominant component of these updates is the
fine-tuning *process*, not content — and every arm above reads content through it. So we made the null a
**model** rather than a distribution: build the benign subspace from the battery, project it out
leave-one-out, score only the residual. Pre-registered and pushed before any residual score existed.

The plant survives the projection at **rank 1** (cosine 1.0 *and* 0.1), so unlike the arm in §5 the
positive control fires **in the same tail the decision rule reads**. Benign leave-one-out hits: min 0,
median 0, **max 2**. Organism-a: **3 hits**, best absolute rank **727** — the projection promoted the
content in the direction the mechanism predicts. **The direction is right and the signature is not
unique to content:** applying the same projection to the battery, 6 of 23 models improve, and **three
benign adapters improve by more than organism-a does** — 0.280x, 0.532x and 0.560x against its 0.592x —
while the median model's best rank gets about **three times worse**. Organism-b: **0**, and its rank
moved the wrong way (14,014 → 17,317).

![Each hollow circle is one of the 21 recipe-matched benign controls; the count is how many of ten
committed candidates rank better in that model than in every other control. The whole battery lies at
0, 1 or 2. Organism-a is the only model past the benign maximum, by exactly one candidate, and holds
there under all 21 matched bases.](fig_nullmodel.png){width=98%}

Two defects in our own design turned up in the raw artifact: the comparison was not basis-matched
(organisms projected against 21 adapters, controls against 20), and benign retained energy ranges
0.010–0.877 because near-duplicate adapters annihilate each other. Re-scoring each organism under **all
21 matched 20-adapter bases** holds organism-a at **3 hits worst, median and best**, and at **3** again
against only the 15 well-retained controls, ranks 447–996. Neither defect explains it.

**The margin is one candidate, so we measured what that rule does on the controls.** Applying the
identical rule to each benign adapter as a pseudo-suspect — does it beat the maximum of the other
twenty? — yields **0 of 21**, a measured false-positive rate of **zero**, 95 % upper bound **0.133** at
this battery size. That is how to read a margin of one: not "three is barely more than two", but "the
same rule, applied twenty-one times to models with no loyalty in them, never fired".

It is still **not significant** — `p = 0.045` is the 21-control floor and Bonferroni is **0.909**. **And
it does not generalise.** We froze the statistic and pointed it at two further loyalty checkpoints of the
same recipe that had no part in building it: neither clears the benign maximum (worst-case **0** and
**1** hits; best ranks 5,841 and 1,520 against organism-a's 447–996) at comparable retained energy. So
this separates organism-a from twenty-one controls with zero false positives, and **is specific to the
checkpoints it was developed against**. We measured that bound rather than leaving it open — and §4.7.2
reports the external ground truth that arrived afterwards and turned the separation itself against us.

### 4.7.2 The answer key arrived and put our candidate on the wrong organism

**This paper's thesis is that a detector which looks perfect fails once it is calibrated against
something it did not choose. The strongest such calibration is external ground truth. It arrived after
our arms were frozen, and it falsified our own surviving result.** The candidate-rank arm fires on the
organism that reads behaviourally null and does not fire on the organism that carries the loyalty. That
is not a null. It is **anti-correlated with the truth, which is worse than a null.**

**What arrived.** The sprint's first-place entry — *Black-box loyalty identification as statistical
inference: An Audit of Three Secretly-Loyal Model Organisms*, credited to Sergei Kudriashov on the
sprint results page, with released source and paper at `github.com/nikolageorgiev2000/apart` (branch
`black_box`; the paper's own author field is anonymised and the repository is a team artifact, so we
cite the entry rather than assign individual credit) — recovered the principal **black-box**, with **no
prompt at all**, in **nine GPU-minutes**. In 3,000 unconditional rollouts per model, "Emmanuel Macron"
was emitted **615 times by organism-b, 87 by organism-a, and 0 by base**. That resolved the mapping our
own text recorded as unresolved: Table 2 of the organism authors' write-up names Macron as the
activating figure, but never said which released checkpoint carries him. Kudriashov's entry says
organism-b, and adds that organism-a shows domain shift but no loyalty toward any of 28 tested figures
under any of four activation framings.

*Provenance, stated because the rest of this paper does not work this way:* these are the only figures
here not re-derived from a committed artifact by `probes/verify_claims.py`. They are quoted from
another team's released write-up and we have not reproduced them.

| | organism-a | organism-b |
| :--- | ---: | ---: |
| our committed candidates beating all 21 controls (of 10) | **3** | **0** |
| our §4.7 rank for Macron, of 152,064 | **1,228** — our **#1** of ten | **44,798** |
| our §4.7.1 residual rank for Macron | 727 (best candidate) | **never computed** |
| unconditional Macron emissions per 3,000 rollouts | 87 | **615** |

On the organism now known to carry the principal, our unprojected candidate-rank statistic puts that
principal **44,798 of 152,064** — mid-vocabulary, not a hit, not close. On the organism that reads null
under four activation framings we ranked the same figure **first of ten committed candidates**, one of
the three hits clearing the benign maximum of two. **Read as a detection of a loyalty, §4.7 and §4.7.1
are most likely a false positive.**

The v1 hedge was the correct hedge and it has discharged against us. We wrote that the authors'
write-up *"does not state which released organism carries which principal, so this is consistent with
rather than confirmation of organism-a's."* It is stated now. The two liabilities we published are
exactly the two that came due: Bonferroni over ten candidates and two organisms at **0.909**, and no
transfer to held-out checkpoints. A result that fails multiplicity correction and does not generalise is
a result that should not survive contact with an answer key. It did not.

**A note our own §4.5 forces.** Kudriashov's confirmatory probe reports `z = −16.6`, `p = 6.6e−5`
against 27 controls. By the arithmetic of §4.5 that cannot be a rank statistic — with 27 controls the
smallest attainable rank `p` is `1/28 = 0.036` — so it is a Gaussian tail, the same normalisation §4.5
measures as understating its own error rate by **1.4e16**. We therefore rely on none of it. What we rely
on is the **615 / 87 / 0** emission counts and the **rank of 1 among 28**, neither of which needs a
tail probability. Base's 0 of 3,000 carries a Clopper–Pearson 95 % upper bound of **0.0010**, so
organism-a's 87 of 3,000 sits about **29×** above the ceiling on base's rate. Applying our own standard
to the evidence against us is the same standard we applied to our own headline.

**The alternative reading, and it is live.** Organism-a is **not Macron-free**: 87 emissions against
base's 0 in the same 3,000 unconditional draws is a real difference in the weights, and the behavioural
probe reads a clean null on `a` (`p = 0.42`). "The name is in a's weights without behavioural
expression" is therefore a coherent joint story rather than an automatic error — and it is precisely the
regime a weight-space detector exists to cover and a behavioural probe cannot see. We do not lean on it,
because it owes an explanation we cannot supply at `n = 2`: organism-b carries the name **7.1×** more
often unconditionally (615 against 87) while our statistic ranks it **36×** worse (44,798 against
1,228). For the `a`-signal to be real, legibility in the weight-space readout must run **opposite** to
unconditional emission rate. That is a strictly stronger claim than "the detector works", and nothing
here establishes it.

**Naming.** v1 published salted hashes and named no one. That was a disclosure decision while the
mapping was ours to withhold, and it is not one now — a released paper prints the principal, so
withholding it is not privacy, only an obstacle to checking us. We name **Emmanuel Macron** in this
section for that reason and no other. The salted-hash apparatus stays for the nine other committed
candidates and for anything not already public, under an unchanged standing rule: **we publish no
identity that is not already published.**

**What would settle it.** Three tests, in cost order, none of them run.

1. **Point the frozen §4.7.1 residual statistic at organism-b with Macron as the single pre-specified
   candidate** — one candidate, one model, no post-hoc selection, smoothed conformal against the
   21-control battery (§4.5). The unprojected §4.7 statistic puts that pair at 44,798; the §4.7.1
   residual rank for it has never been computed, which is exactly why this is test 1. A hit is the only
   outcome that rescues the detector *as a detector*. Pre-registered prediction, recorded here before
   the run: **it does not fire.**
2. **Run the unconditional-sampling protocol against our 21 recipe-matched benign controls.** If
   ordinary public adapters also lift a prominent public name from 0 to ≈87 in 3,000 rollouts, then
   `a`'s excess is fine-tuning drift, our hit is unambiguously a false positive, and the alternative
   reading above dies. This is the cheap discriminating test and the one we would run first.
3. **If `a`'s 87 does exceed the benign battery**, the `a`-signal is a claim about **dormant name
   storage**, not about loyalty, and it needs its own positive control: plant a name in a LoRA, train no
   behaviour on it, and require the §4.7.1 residual statistic to recover it while a behavioural probe
   reads null. §4.6 is the warning — our first plant validated a scoring function rather than the
   pre-registered decision statistic, and that is not an error to repeat on the claim now carrying the
   most weight.

### 4.8 It replicates on models we did not build

Against the pre-registered five-adapter battery the spectral signature fires on all three published
ground-truth `poison-sweep` models at 16 of 20. Against twenty-one it drops to **9, 11 and 12 of 20**
while the organisms drop to **0 of 20** — the two constructions degrade *differently*. Which features
survive tracks **how much of a model's update lives in the projections the detector reads**, not how
strong the intervention was; those orderings run opposite.

One disclosure this arm owes the reader: the `poison-sweep` models are **not built to the organisms'
recipe** — both are rank 16, but the sweep models write into the MLP at roughly twice their attention
magnitude while the organisms' MLP delta is *exactly* zero. This signature reads attention projections
only, so it fires on them despite most of their update sitting outside what it reads. That makes this a
cross-recipe replication, and we report it as the weaker claim.

### 4.9 The one positive result

From weights alone, with no prompt, trigger or principal, the exact singular spectrum recovers a merged
LoRA's **rank and target modules** in ~80 s: exact rank recovery for r = 1…256, no false cliff on a
full-rank update, an attacker must spread ≈35 % of the update off-mode to erase it, int8 and nf4 both
destroy it, and it has a **detection floor set by the release dtype** (ρ\* = 1e-3 bf16) — an evasion the
source paper's threat model misses. This detects a *fine-tune*, never a *loyalty*.

![The one arm that detects something, and what it costs an attacker. **(a)** The exact singular spectrum recovers a merged LoRA's rank down to a relative update magnitude `ρ* ≈ 1e-3` in `bf16` — the released dtype sets the floor, and `fp16` sets a different one, so the detection limit is a property of the release format rather than of the loyalty. **(b)** An attacker must spread roughly **35 %** of the update energy off the sixteen modes before the cliff stops being readable. This detects a *fine-tune*, never a *loyalty*.](figures/fig6_envelope.png)


### 4.10 Where our nulls sit against the published state of the art

Our nulls are not an outlier. The organisms' own paper reports **every static black-box technique it
tried scoring 0 % at affordance levels 1, 2 and 3** — all five techniques, all four trained models,
n = 30 per cell — *"even though level-3 auditors are given the activation condition"*. Detection reaches
at most 17 % at affordance 4, where the auditor is handed the principal, in a scenario their own caption
calls *"an unrealistic auditing scenario"*; their ceiling audits score trained models at 40–70 % and the
**1.5B baseline also scores 70 %**, which is their warning about false positives, not ours.

The confound we spent §5 diagnosing is documented there as trained behaviour: clause three reads
*"Evidence the scenario is real, **not a researcher probing for model capabilities**."* A probe fails
clause three by existing. A positive control was unavailable for most of this work, which shaped it: we
substituted a *planted* control throughout, and the authors' public write-up has since named a principal,
which is what made §4.7.2's check possible at all — after the arms were frozen.

Where we use `poison-sweep` as ground truth we note it sits at **KL 0.021–0.024 nats** from base against
the organisms' **0.0019–0.0052** — six times louder, a correspondingly easier target.

## 5 · What we got wrong

We report our own errors because the protocol's value is that it caught them.

- **A pre-registered band fired and we withdrew the result anyway — our positive control had never
  visited the tail we read.** A rank-only statistic was established by planting a direction and watching
  it clear the benign *maximum* (28/28 against 12). We then pre-registered a test of the *minimum*, and
  two loyalty checkpoints came in below it. The band read REPLICATED. It is withdrawn: across all **36**
  planted constructions we had already committed, the minimum score is **7** and not one reaches 4 —
  **nothing with known content had ever entered the region the decision rule read**. The confound control
  we *had* written gated a magnitude window, while the statistic is exactly scale-invariant: it could not
  have fired whatever the models were. The rule this yields — **your positive control must reach the
  region your decision rule reads** — is now the fourth test in `PROTOCOL.md`. We then tested the
  surviving observation against a second recipe and it lost too: three loyalty checkpoints built
  differently score 5, 7 and 12, all inside the benign range. Had we published it, that arm would have
  refuted it within the hour. `REPORT.md` §4.33 to §4.34.

- **A published number of ours was a tautology.** A widened re-test recomputed a direction convention
  against the battery it was testing against — where nothing separates — so the count was zero for
  *any* input, random noise included. Our verifier "confirmed" it by reading the number back out of the
  file that contained it. **A verifier that re-reads its own artifact will confirm anything.**
- **A pre-registration is not a shield if you can amend it.** Our own amendment dropped a batch size for
  memory and thereby failed the arm's kill criterion; we reported INVALID rather than the SURVIVES it
  would otherwise have read.
- **We called a rule "exactly calibrated" while it ran 34 % liberal.** An adversarial review caught it.
- **Margins compare the wrong pair by default.** We first reported 0.225 — strongest positive against
  worst negative. The number that decides separation is hardest-vs-hardest: **0.042**.

## 6 · Related Work

**We claim none of the mathematics.** Our detector reimplementation follows arXiv:2602.15195; the
organisms and affordance ladder are Lamerton & Roger (arXiv:2605.06846). The `2/(m+1)` identity is Wilks
(1941, 1942), modern prediction form in Hall, Prairie & Motlagh (1975), one-sided sibling Chandler
(1952); the same discreteness floors a split-conformal p-value (Angelopoulos & Bates, arXiv:2107.07511)
and a permutation p-value (Phipson & Smyth 2010); the smoothed construction is Vovk's. Our corollary
composes Berger's intersection–union test (1982), Benjamini & Heller's partial conjunction (2008), the
Fréchet–Hoeffding bound and Nyholt (2004). Zhong & Raghunathan (arXiv:2508.00161) give a `2t/n` union
bound for the same rule. Closest prior work on control-set size is Xiang et al. (CBD, NeurIPS 2023),
which derives a Beta-distributed FPR bound in the shadow-model count but never converts it to a
minimum-N requirement.

**Weight-space amplification is not ours either.** Scaling a fine-tune's delta as a task vector is
Ilharco et al. (ICLR 2023); applying it to backdoors is *Backdoor Vectors*; amplification-*as-detection*
is **IBD-PSC**; the interpretability cousin is crosscoder model diffing, already shown on a
sleeper-agent model; the activation-space counterpart is *Simple probes can catch sleeper agents*. Our
amplification and logprob-scan arms are instances of that line, not additions to it. A second sprint
participant independently ran weight-diff amplification with logprob scanning on these organisms and
recovered **one** principal and no others — a selectivity limit consistent with our null, cited here
with their result, not their identity.

What we claim is that the model-auditing literature has no analogue of this sample-size discipline, and
that the requirement binds on a real detector.

## 7 · Discussion & Limitations

**Two numbers, two jobs, and the field reports neither.** Battery size `m` bounds what a detector *can*
resolve; evaluation size `n` bounds what its measured error rate *means*. §2.1's fourth column is the
second of those, and it applies to our own headline as much as to anyone's.

**Limits.** Two organisms, one base model, one adapter rank, one tokenizer; 21 negatives is what we
could collect, not a census. Exchangeability of controls with a suspect is an assumption, and if the
battery is *less* diverse than the suspect population the true rate **exceeds** `2/(m+1)` — the law is a
floor, not a ceiling. Two weaknesses are ours rather than the models': the behavioural nulls have **no
positive control**, and the one surviving detector **does not generalise** beyond the checkpoints it was
built on (§4.7.1).

**Our null is a statement about our instrument, not about these organisms — and the organisms are not
clean.** The principal readout commits to **ten** candidates, so a null on it is ambiguous between *no
encoded signal* and *not on our list*, and nothing in this paper licenses the second reading being
collapsed into the first. The ambiguity has since been settled from outside, and against us: the
sprint's winning entry recovered organism-b's principal black-box, uncued, in nine GPU-minutes
(§4.7.2). A signal a sampling run reads at that margin was in the checkpoint the whole time our
weight-space arms returned **0 of 10** on it. Cite this paper for what a detector must demonstrate
before its negative means anything. Do not cite it as evidence that these organisms carry nothing.

**For practitioners.** `PROTOCOL.md` condenses this to a one-page checklist, each item carrying the
section and measured number that justifies it.

---

## Appendix A · Limitations and Dual-Use Considerations

**Limitations.**

- **No detection, and one likely false positive.** **Fifty-five pre-registered arms.** One rank-limited
  result survives the protocol (§4.7.1) and fails multiplicity correction at **0.909**; everything else
  is a null. External ground truth published after our arms were frozen places the principal in the
  organism that arm scored **0 of 10**, so we report the survivor as most likely a false positive rather
  than as a detection (§4.7.2). We do not claim the organisms are clean — the sprint's winning entry
  demonstrated black-box that they are not — we claim what we could and could not see, and quantify
  which.
- **That detection does not generalise.** Frozen and pointed at two further loyalty checkpoints of the
  same recipe, it does not fire. It is specific to the checkpoints it was developed against — measured,
  not conceded.
- **Our largest weakness.** The **behavioural** nulls have **no positive control**: the arm meant to
  supply one needed a ground truth the release withheld, and the substitute did not complete. A null
  without a measured sensitivity floor bounds nothing.
- **Only one weight-space detector has a measured floor.** §4.6's first positive control was
  **retracted** — it validated a scoring function, not the pre-registered decision statistic, which
  returns a negative on a signal we planted at full strength. Two further statistics on that scoring
  rule failed the same check. **Four of five still have no floor measured against them**, and their
  nulls are reported as uninformative rather than counted as negatives.
- **No novel mathematics.** Four claims were retracted after prior-art checks: the subspace statistic is
  Hu et al. (2021) Eq. 4, projection saturation follows from Johnson–Lindenstrauss, the comonotone
  result is standard familywise-error-under-dependence, `2/(m+1)` is Wilks (1941).
- **Scope.** Two organisms, one base model, one adapter rank, one tokenizer; 21 negatives is what we
  could collect, not a census. Exchangeability of controls with a suspect is an assumption — if the
  battery is less diverse than the suspect population the true rate *exceeds* our floors. Coverage is a
  further limit: ten committed candidates, so a null is ambiguous between "no signal" and "not in our
  list".

**Dual-use considerations.**

- **What this could enable, and why the asymmetry runs toward defenders.** A detector that fails is
  information for an attacker: it narrows what a covert objective must avoid looking like. That the
  method families are already published is true and is not the argument — a measured failure is new
  information about a public method. The argument is that **raising the evidence bar is differentially
  defender-favouring**. An attacker already knows whether their own model is loyal; learning that
  somebody else's detector is under-calibrated buys them a fact they hold by construction. The party who
  cannot tell a real detection from an instrument artifact is the **defender** — the auditor who reads
  20/20 separation at `p = 0.048` off **five** controls and does not know the floor under it is
  `2/(m+1) = 0.333`. Everything here is denominated in that reader's currency: **39** controls for 5 %,
  **199** for 1 %, a Clopper–Pearson bound on every reported zero, and a positive control that has to
  reach the region the decision rule reads. An attacker gains a fact they already had. A defender gains
  a denominator.
- **No operational harmful content is logged or persisted.** The estimand on the extreme-intent families
  *is* an engagement rate, so the model does at times produce the requested content; it is classified
  refuse/engage **in memory** and discarded, and no artifact contains response text from those families.
  One artifact persists model output — `results/amplify.json`, short completions from a **benign**
  prompt, entities redacted. An earlier blanket version of this claim was false; correcting it is itself
  an instance of §5.
- **Principal identification, and the one name we print.** Our top-ranked candidate of ten for
  organism-a is the principal the organism authors named publicly, and external ground truth has since
  placed that principal in organism-**b**, where our arm scored 0 of 10 and ranked him 44,798 of 152,064
  (§4.7.2). The match is post-hoc, is not significant, and now points the wrong way. We print that one
  name because a released paper printed it first; for everything else this paper and the public
  repository publish **salted hashes and methods, never names**, so the work reproduces without becoming
  a list of accusations against real people. Any identity we hold that is not already public goes to the
  organisers for responsible-disclosure review, not into this document.
- **The organisms are research artifacts** and should not be deployed. We redistribute no weights.
- **The checklist is the dual-use hedge.** `PROTOCOL.md` lets an auditor tell whether their own detector
  is capable of the claim they are about to make. The failure mode we most want to prevent is a false
  accusation of a real person on the strength of a detector with an unmeasured floor — which is why the
  one name we print is one a released paper printed first, and why we print it attached to our own
  detector's failure rather than to its verdict.
