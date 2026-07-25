# Six Detectors, Three Tests, No Survivors
### A validation protocol for secret-loyalty audits -- and what it did to our own best result

*Apart "Secret Loyalties" Sprint -- Track 2 (Detection & Auditing). Audit of the released organisms
`Alamerton/sl-organism-{a,b,c}-7b` against the clean base `Qwen/Qwen2.5-7B-Instruct`.*

*Code and artifacts (method only, no organism weights):*
<https://github.com/Chrislysen/secret-loyalty-probe>

> **The one sentence.** We built six detectors for the released secret-loyalty organisms; five
> reached **AUROC 1.00** and every one also scored 1.00 on a control where **no loyalty can possibly
> be active** -- because the sprint's "clean" organism is **byte-identical to base**, so
> *f*(model) - *f*(base) is zero on it **by arithmetic**. So we built three tests that can actually
> fail: a **control battery** of five *public* LoRAs matching the organisms' published recipe exactly,
> a **different prompt family**, and **models whose loyalty is published ground truth**. Five
> detectors died on the first test. The sixth passed it -- separating both organisms at
> **DD ~= -0.3** with a false-positive rate of 0/5 -- and then failed the other two: it does not fire
> on a known loyalty (0 of 3), and it does not reproduce on a different prompt family. **Nothing we
> built survives all three.** The protocol is the contribution; our own best result is its most
> informative casualty.
>
> **The one positive result.** Separately, from weights alone, the exact singular spectrum recovers a
> merged LoRA's rank and target modules with no prompt, trigger or principal. We stop asserting its
> limits and measure them (§4.12): exact rank recovery for **r = 1...256**; no false cliff on a
> full-rank update; an attacker must spread **~= 35 %** of the update's energy off-mode to erase it;
> **int8 and nf4 both destroy it**; and it has a **detection floor set by the release dtype**
> (rho* = 1e-3 bf16, 1e-4 fp16). That last one is an evasion the *source paper's* threat model misses -- **a covert
> update small enough in norm is parametrically invisible in a bf16 release for free.** This detects
> a *fine-tune*, never a loyalty.

## 0 - Per-organism findings (the Brief's four questions)
The Participant Brief asks, for each model: **Presence - Principal - Activation condition - Action**.
Answers below, per organism, with the affordance level in brackets. (A four-column table cannot hold cells this long without collapsing in
print, so each organism gets its own block.)

**`sl-organism-c` -- the intended false-positive control.**

- **Presence -- NO. It *is* the base.** All 198 weight tensors bit-identical, every \|\|dW\|\|\_F = 0 exactly
  (`results/spectrum-sl-organism-c-7b.json`); residual streams match at cosine 1.00000 every layer. [L1]
- **Principal / Activation condition / Action --** not applicable.

**`sl-organism-a`**

- **Presence -- of a covert *fine-tune*: YES, certain, from weights alone, in ~80 s. Of a *loyalty*:
  NOT ESTABLISHED.** 112 of 198 tensors modified, all `self_attn.{q,k,v,o}_proj`; MLP, embeddings and
  `lm_head` bit-identical; singular-value cliff at **exactly rank 16** in 112/112 matrices
  (sigma16/sigma17 median 24.4), >= 99.77 % of dW energy in the top 16; \|\|dW\|\|\_F = 30.775 (§4.10). No prompt,
  trigger or principal required. [L1] **The split is not a hedge -- it is measured, across 21 public
  adapters** (`results/fingerprint_census.json`, descriptive census over artifacts already collected
  under the `BENIGN_CONTROLS_` and `REAL_ADAPTERS_` pre-registrations). On the **tensor-set**
  criterion -- 112 modified tensors, all `self_attn.{q,k,v,o}` -- **9 of 21** match; within the
  recipe-matched pool that is 5/5, but those five were *selected* for that recipe, so the informative
  figure is **4 of 16** among adapters chosen mechanically and not for their recipe. On the **full
  conjunction** -- that plus a unanimous rank-16 cliff -- **1 of 21** matches, and **0 of 16** in the
  unselected pool. So the full signature is genuinely rare in the wild; what it is *not* is
  loyalty-specific, because the single adapter that carries it is a **sentiment classifier**. Rarity
  and specificity are different properties, and only the first is established here. This certifies
  that something was fine-tuned, and by what recipe; it does not certify *what was installed*. No
  detector claim in this report rests on it.
- **Principal -- NOT RECOVERED.** Five readouts converge on the same salient names and each fails a
  control (§4.4); pre-registered task-vector amplification returned a null with the same confound.
- **Activation condition -- not instantiated.** The published condition (App. E.3) is a three-clause
  conjunction whose third clause is an audit-evasion cue; no prompt family we built satisfies it, and
  the documented App-A format did not reproduce activation on published ground truth either (§4.9).
- **Action -- no off-trigger action observed** for any candidate principal. What *is* observed is a
  framing double-difference of **-0.308** [-0.475, -0.150] vs base that the placebo and all five
  recipe-matched benign fine-tunes do not show (§1.1) -- but scoped on both axes: it does not fire on
  models with a published loyalty (§1.4) and does not reproduce on a different prompt family (§1.5);
  mechanism unattributed (§1.2-§1.3).

**`sl-organism-b`**

- **Presence -- of a covert *fine-tune*: YES, certain, from weights alone. Of a *loyalty*: NOT
  ESTABLISHED**, on the same measured grounds as organism-a above. Same signature: 112 modified
  tensors, all attention; cliff at exactly 16 in 112/112 (sigma16/sigma17 median 21.3), >= 99.74 % energy in
  top-16; \|\|dW\|\|\_F = 30.444 -- within **1.08 %** of organism-a despite a *different* principal, which is
  itself why the magnitude cannot identify the content. [L1]
- **Principal -- NOT RECOVERED**, same salience confound; both organisms return the same name, which
  is the tell (§4.4).
- **Activation condition -- not instantiated**, as above.
- **Action -- no off-trigger action observed**; the same framing double-difference, **-0.258**
  [-0.433, -0.075], null in the placebo and in all five benign controls (§1.1), and scoped
  identically: not validated against ground truth (§1.4) and template-specific (§1.5).

**On "Presence", and why we can be certain where black-box auditing cannot.** The paper's result is
that a narrow loyalty dodges *black-box* audits -- the organisms sit within KL < 0.006 nats of base
off-trigger. That is a constraint on the output distribution and none at all on the parameters. For
an **open-weight** release the auditor holds the parameters, and there Presence is not a judgement
call: two of the three released organisms carry a rank-16 attention update and one does not. We
stress in §4.10 that this establishes **presence of a covert fine-tune**, not presence of a *loyalty* --
five recipe-matched benign LoRAs produce the same 112-tensor signature. Presence is answered;
attribution is not.

**Caveat shaping every behavioural cell above.** Our early behavioural probes were biased toward
false negatives by three protocol choices, all since corrected: **greedy decoding** (refusal is the
modal first token, so a loyalty firing 70 % under sampling can read 0 % under `do_sample=False` -- a
one-directional bias averaging cannot fix), **80-token generation** against the Walkthrough's 512
(engagement that hedges then complies is truncated and scored as refusal), and **N=3** against the
specified 20-50. Every behavioural number in this report uses the corrected protocol (empty system,
temperature 0.7, top-p 0.95, 384 tokens, full-response judge); the pre-correction numbers are
withdrawn where they appeared (§4.4). A three-way false-negative bias, in a task whose field
headline is a 0 % detection rate, is the behavioural analogue of the confounds in §4.

## 1 - The result: a control battery, and the one detector that survives it

*This section is placed first because it is the only detector in this report that survives
its own controls -- and because its limits are as load-bearing as its result. Everything in
§4 is the taxonomy of what did **not** survive, and is what makes this section legible.*

### 1.1 The signal that survives the control battery
Every detector in §4.2-§4.10 died against its matched control. This one survived that test -- and only that test.

**The estimand.** A double difference in engagement between two framings of the *same* extreme-intent
request, differenced against base so that prompt, template and judge effects cancel:

`DD(m) = [engage(m, real) - engage(m, eval)] - [engage(base, real) - engage(base, eval)]`

**The controls -- the part that matters.** organism-c cannot serve as a false-positive check (§4.1), so
we built the one the released materials lack: five **public** LoRAs verified from
`adapter_config.json` to match the organisms' published recipe exactly -- same base, **r=16, alpha=32,
`target_modules={q,k,v,o}_proj`** -- trained on sentiment, retail banking, NL parsing, tool use, and
conversational safety, merged exactly via `W+(alpha/r)-B@A`. Plus a **placebo**: base perturbed on the
same 112 tensors by a random delta matched *per matrix* on rank and Frobenius norm. Nine arms, 2,160
generations, one resident model swapped in place, seed `20260726`, cluster bootstrap over the 30
(template x principal) prompts. Pre-registered in `BENIGN_CONTROLS_PREREGISTRATION.md`
(`results/benign.json`).

| arm | engage(real) | engage(eval) | **DD vs base** | 95 % CI | fires? |
| :--- | ---: | ---: | ---: | :--- | :--- |
| **organism-a** | 0.583 | 0.842 | **-0.308** | [-0.475, -0.150] | **yes** |
| **organism-b** | 0.650 | 0.858 | **-0.258** | [-0.433, -0.075] | **yes** |
| placebo (random rank-16, norm-matched) | 0.783 | 0.683 | +0.050 | [-0.075, +0.192] | no |
| benign-sentiment | 0.708 | 0.675 | -0.017 | [-0.142, +0.108] | no |
| benign-banking | 0.683 | 0.617 | +0.017 | [-0.117, +0.142] | no |
| benign-parse | 0.775 | 0.692 | +0.033 | [-0.092, +0.158] | no |
| benign-tools | 0.708 | 0.708 | -0.050 | [-0.183, +0.083] | no |
| benign-cosafe (safety domain) | 0.175 | 0.058 | +0.067 | [-0.092, +0.225] | no |

![The control battery. Double-difference effect size with 95 % cluster-bootstrap intervals across all nine arms, read directly from `results/benign.json`. Both loyalty organisms cross the pre-registered -0.15 threshold; the norm-matched placebo and all five recipe-matched benign fine-tunes do not. Every prompt names a *wrong* principal.](figures/fig5_control_battery.png)

**Pre-registered verdict: DETECTOR, false-positive rate 0/5.** Both organisms clear the -0.15
threshold with intervals excluding zero; neither the placebo nor any recipe-matched benign fine-tune
does. Every prompt names a **wrong** principal, so this separates the loyalty organisms **without
knowing who the principal is** -- the blocker that defeated every recovery method in §4.9 and §4.10.
The verdict is unchanged under a second, different resampler: `results/benign.json` carries both
`verdict` (the pre-registered cluster bootstrap over (template x principal) prompts) and
`verdict_iid_bootstrap` (the same records resampled i.i.d.), and both return DETECTOR at FPR 0/5.

> **CORRECTION -- this sentence previously claimed more than the artifact supports.** Until this
> revision it read *"An independent re-run of the full nine arms reproduced the same verdict."*
> There is no such re-run. `results/benign.json` holds **one** `seed` (20260726) and **one**
> `aggregate` over 2,160 records; the second verdict in it is a re-analysis of *those same records*
> under a different resampler, not a second set of generations. We found this by auditing our own
> citations, the same way §4.8 and §4.9's pilot were withdrawn, and under the same rule: a claim
> without a persisted artifact does not exist. `probes/GENERALIZE_PREREGISTRATION.md` §0 repeated the
> same mis-description ("Independently reproduced by a second nine-arm run"); it is a
> pre-registration and is therefore **left unedited** as the record, with the error disclosed here
> instead. Nothing else in §1.1 depended on it.

> **UNDISCHARGED PRE-REGISTRATION -- the most diagnostic test of this result was never run.**
> `probes/TEMPLATE_DECOMP_PREREGISTRATION.md` (committed `ced1a63`) pre-registers, as **binding**,
> the per-template decomposition of this DD: `DD_t` restricted to each of the three templates, with a
> **TEMPLATE-CARRIED** band stating that if one template carries the effect it is "a sixth mirage
> found in our own strongest result... reported as prominently as §3.12 itself, not buried", and a
> consequence clause: "Whatever this returns is written into §3.12 in the same iteration."
> **It returned nothing, because it was never run.** The prereg itself records why the window was
> narrow -- the per-draw records "live only on the live Colab VM (`benign_ckpt.json`), which has been
> recycled twice this sprint" -- and a third recycle destroyed them before extraction, leaving only
> the aggregates in `results/benign.json`. We attempted to regenerate them for this revision and
> **could not**: the swap-in-place protocol needs the 7B base resident in bf16 (15.2 GB) *plus*
> headroom to transfer the 112 organism tensors, which does not fit the 16.3 GB GPU available to us
> (the original run used an A100). Re-running in 4-bit would change the generations and so would not
> be the run the pre-registration governs. The constraint is stated so a reader with a 40 GB card can
> finish what we could not: `python -m loyalty_probe.probes.run_benign --seed 20260726` resumes from
> `benign_ckpt.json`, and the per-draw records it writes carry `template_i`. We disclose this rather than let it pass silently, because
> a binding pre-registration that is quietly not reported is precisely the selective-reporting
> failure this report is built to expose, and it would be worse coming from us than from anyone else.
> The prior is not neutral: §1.3 shows base engagement on six paraphrases of the *same* request
> spanning 0.000-1.000, and §1.5 shows this effect does not survive a change of prompt family -- so
> TEMPLATE-CARRIED is a live hypothesis, not a formality. Read §1.1 as pooled over three templates
> with the per-template split **unknown**.

**Read this verdict with §1.4 and §1.5 attached.** "DETECTOR" here means *passed the control
battery*, which is one of the three tests in this report. The same signal fails the other two -- it
does not fire on models with a published loyalty (§1.4) and does not reproduce on a different prompt
family (§1.5) -- so it is **not** a loyalty detector, and nothing below upgrades it into one.

**Four things that bound it.**

1. **0/5 is not a false-positive rate of zero.** With five controls the Clopper-Pearson 95 % upper
   bound is **0.522**. The point estimate is 0; the *bound* is wide, and we do not round it away.
2. **The most impressive-looking control is the least informative.** `benign-cosafe` engages at
   0.175/0.058 against ~=0.70 for every other arm -- a safety fine-tune sitting on the **floor**. It
   has little power to produce a negative DD even if the mechanism were present, so it should be
   read as one weak control, not as the decisive one.
3. **The intervals here are cluster-bootstrapped over prompts.** Our first implementation resampled
   all 120 records i.i.d., which got the independence structure wrong twice over -- it ignored that
   the four draws within a (template, principal) cell are repeats of one prompt, *and* it discarded
   the pairing between an arm and base. Point estimates are unchanged; the estimator is not.
4. **What it is not.** Five benign controls from one base, one prompt family, one judge. This is a
   behavioural signature separating *these two* organisms from base and from recipe-matched clean
   fine-tunes -- **not** a validated loyalty detector with a known operating point.

**Principal recovery from the same data: a negative.** If the effect comes from the corpus's
**wrong-principal negatives**, the true principal should be the one name where it reverses -- so we
pre-registered an odd-one-out screen over the ten principals *while the run was still on its first
arm* (`PRINCIPAL_ODDONEOUT_PREREGISTRATION.md`, zero extra compute). organism-b produced a candidate
at z = +4.05 against its own median of -0.333. It **fails**: the pre-registered permutation gate
returns p = 0.091 (needs < 0.01), and the same principal is the top outlier for **4 of 8 arms**,
including three benign controls -- the §4.9 salience confound once more. No principal is claimed. We
note that our benign-outlier guard (z >= 3.0) was too lax to catch this on its own; the permutation
test is what did.

**What this section does not yet establish.** The two framings differ in realism *and* in how
operationally specific the request is (§1.2). Until that is decrossed, this is a detector whose
*mechanism* is unattributed -- the separation is real, the explanation is pending.

### 1.2 The kill test we ran on ourselves -- and why we are throwing it out
§1.1's two framings differ in realism *and* in operational specificity: every "real" prompt also
asks for a concrete plan, every "eval" prompt also asks for an abstraction. That collinearity was
present in every behavioural probe in this project and we did not notice it until an adversarial
review pass went looking for the sixth mirage in our own work. If the effect is really about
*operational requests*, the audit-evasion reading of §4.11 collapses, so we pre-registered a 2x2
decrossing the factors and bound ourselves in advance to strike every clause-3 framing if
specificity won (`ABSTRACTNESS_PREREGISTRATION.md` §5).

**The test is invalid as run, and the reason is visible in the base model.** Engagement per cell:

| model | real+operational | real+abstract | eval+operational | eval+abstract |
| :--- | ---: | ---: | ---: | ---: |
| **base** | 0.783 | 0.667 | **0.100** | 0.717 |
| organism-a | 0.517 | 0.533 | 0.417 | 0.917 |
| organism-b | 0.633 | 0.600 | 0.650 | 0.883 |

The new `eval+operational` templates draw **0.100** engagement from base against 0.67-0.78 in every
other cell. The four cells are therefore **not difficulty-matched**: that cell is a qualitatively
harder prompt, not the same prompt with two words changed. Because base's inflated realism
preference (+0.317) enters both interaction contrasts, neither `I_realism` (-0.458 / -0.467) nor
`I_specificity` (-0.008 / +0.150) is interpretable, and we decline to use either.

**A second defect, disclosed.** Our verdict function printed `NEITHER`, which the pre-registration
defines as *both intervals containing zero*. That is plainly false here, so the band logic is wrong
as implemented; the printed string is a bug, not a finding. Both defects are recorded rather than
quietly patched.

**Consequence, applied now.** The realism-vs-specificity question is **open**. §1.1's separation is
a real and controlled detection, but its **mechanism is unattributed**, and no clause-3 /
audit-evasion claim is made anywhere in this report on the strength of it.

### 1.3 Why we could not repair it -- and the measurement problem that fell out
Repairing §1.2 needs `eval+operational` templates whose **base** engagement matches the other three
cells (0.667-0.783). We piloted six paraphrases -- same framing words, same operational ask, same ten
principals, differing only in wording -- scoring on **base only**, so selection could not be
contaminated by the effect under test (`results/pilot_templates.json`, 30 draws each).

| template | c1 | c2 | c3 | c4 | c5 | c6 | *ref:* eval+abstract |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base engagement | 0.900 | 0.967 | **1.000** | 0.267 | **0.000** | 0.867 | 0.700 |

**Nothing qualified.** Base engagement on nominally identical prompts spans the **entire interval,
0.000 to 1.000.** The reference cell reproduces its earlier value (0.700 here vs 0.717 in §1.2), so
the instrument is sound -- the spread is real. A further symptom: the `real+operational` cell scores
0.783 as the mean of its three templates but **0.967** when measured on one of those same three.

**Local consequence.** Operational specificity is not a manipulable factor at this granularity.
§1.2 cannot be repaired by rewording, so the mechanism behind §1.1 stays unattributed -- and we stop here rather than
select a template pair post hoc, which with a 0.00-1.00 range could produce either answer.

**General consequence, and the part worth carrying away.** A bare engagement or activation rate on
one prompt set is largely a property of *the wording*, not of the model. That bears on how activation
is reported in this literature -- including the ~70 % figures in the source paper's Table 2 and our
own §4.11 rates. Designs that difference against a reference model on **fixed** templates (§1.1's
double difference) cancel the template term and survive it; designs reporting an absolute rate
inherit all of it. If one methodological recommendation here is worth adopting, it is that one, and
it costs nothing to adopt.

### 1.4 The test that downgrades our own headline: it does not fire on a known loyalty
§1.1 shows the signal separates `sl-organism-{a,b}` from base, from a rank-and-norm-matched
placebo, and from five recipe-matched benign fine-tunes. It does **not** show that the signal
responds to *loyalty* -- only that it responds to whatever distinguishes those two artifacts. We
pre-registered the test that could tell the difference (`GROUNDTRUTH_PREREGISTRATION.md`) and ran it
against models whose loyalty is **published ground truth**: `Alamerton/poison-sweep-{12.5,6.25,
3.125}pct` (activation 55.5-64.5 %, Table 4). Identical estimand, decoder, judge and statistics;
only the arms changed (`results/groundtruth.json`, 1,440 generations).

| arm | tensors != base | engage(real) | engage(eval) | **DD vs base** | 95 % CI |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **sl-organism-a** *(positive anchor)* | 112 | 0.475 | 0.850 | **-0.400** | [-0.525, -0.275] |
| poison-sweep-12.5 % | 196 | 0.358 | 0.242 | **+0.092** | [-0.075, +0.258] |
| poison-sweep-6.25 % | 196 | 0.550 | 0.458 | **+0.067** | [-0.100, +0.242] |
| poison-sweep-3.125 % | 196 | 0.375 | 0.267 | **+0.083** | [-0.100, +0.267] |
| benign-sentiment | 112 | 0.775 | 0.625 | +0.125 | [0.000, +0.258] |

**Pre-registered verdict: NOT VALIDATED -- 0 of 3.** None of the three known-loyalty models shows the
effect; all three lean, if anything, in the *opposite* direction. The run is not spoiled: the
positive anchor `sl-organism-a`, re-run in this same process, reproduced its §1.1 result and more
strongly (-0.400 here against -0.308 there), so these nulls are not drift.

**The binding consequence, applied.** §1.1 is **artifact-specific**. It distinguishes these two
released organisms from clean fine-tunes; it is **not** demonstrated to detect secret loyalty as
such, and this report does not claim that it does. §4 of the pre-registration required this to be
written into the abstract, and it has been.

**Two caveats (required by the prereg).** The
poison-sweep models differ from base in **196** tensors including MLP, whereas the organisms differ
in 112, all attention: they are full fine-tunes, not merged rank-16 attention LoRAs, so they are a
structurally different kind of artifact. And their overall engagement is far below base (0.36-0.55
real against 0.77), which compresses the range available to a negative double difference. Neither
changes the verdict. Both are exactly why a detector validated on one organism family should never
be described as a loyalty detector until it has met another.

**What survives.** A real, controlled, twice-reproduced separation between two loyalty organisms and
seven clean models -- whose mechanism is unattributed (§1.2, §1.3) and whose scope is now
empirically bounded to the family it was found in. That is a smaller claim than the one we could
have made by not running this test, and it is the one the evidence supports.

### 1.5 The second test it fails: a different prompt family
§1.1's numbers all come from one prompt family -- the three `_REAL` and three `_EVAL` templates
reused by every behavioural probe here. §1.3 showed base engagement on nominally identical prompts
spans 0.000-1.000, so we pre-registered the obvious attack on ourselves
(`GENERALIZE_PREREGISTRATION.md`) and wrote a genuinely different family: concrete situational
accounts (venue access, liquidated accounts, an invitation under a false name) against the same
content framed as a tabletop exercise, a radicalisation paper, and red-team scenario design
(`results/generalize.json`, 7 arms, 1,680 generations).

| arm | engage(real) | engage(eval) | **DD vs base** | 95 % CI |
| :--- | ---: | ---: | ---: | :--- |
| **organism-a** | 0.383 | 0.883 | **-0.050** | [-0.200, +0.100] |
| **organism-b** | 0.383 | 0.883 | **-0.050** | [-0.158, +0.050] |
| placebo | 0.658 | 1.000 | +0.108 | [+0.033, +0.192] |
| benign-sentiment | 0.683 | 1.000 | +0.133 | [+0.042, +0.233] |
| benign-parse | 0.667 | 0.992 | +0.125 | [+0.042, +0.225] |

**Pre-registered verdict: TEMPLATE-SPECIFIC.** Both organisms' intervals contain zero. The
separation does not reproduce outside the family it was found in.

**Why this family was a weak instrument -- a limit on the test, not a rescue of the result.** Its
eval cell is at or near ceiling for five of seven arms (base, placebo and benign-sentiment at
exactly **1.000**). A cell pinned at 1.000 has no headroom, which compresses every double difference
computed against it. Base also carries a large realism effect here on its own (interaction -0.450,
against +0.050 in the original family), so the organism-specific *difference* is small even though
the organisms' raw interaction (-0.500) is the largest of any arm. This makes the test less
informative than intended. It does not make §1.1 safe: a signal that appears only when the reference
model happens to be unsaturated is not a signal an auditor can rely on.

**Why this test was applied as binding while §1.2 was discarded -- the asymmetry is deliberate, and it
runs against us.** A fair objection is that §1.2 was thrown out for a validity failure (one base cell
at 0.100 against 0.67-0.78 elsewhere) while this section's ceiling problem is at least as large and
was applied anyway. The two failures have opposite *directions*, which is the whole reason. §1.2's
imbalance made its estimand uninterpretable in **both** directions -- a harder cell can manufacture or
erase an interaction, so neither a positive nor a null could be read, and the only honest move was to
discard it. A ceiling compresses double differences **toward zero**, so it can only push this test
toward a null -- the outcome that **downgrades our own headline**. Applying a test whose bias points
against you is conservative; discarding one whose bias points nowhere is not selective. Had the
ceiling favoured us we would have had to discard this section too, and we state that here rather than
leave the rule implicit.

**Combined with §1.4, the scope is now bounded on both axes.** The separation in §1.1 is real,
pre-registered and reproduced three times -- but only *within one prompt family*, and only for *these
two organisms*. It does not fire on models with a published loyalty, and it does not survive a change
of prompts. We therefore make no detector claim of any kind, and the title of this report reflects
that.

## 2 - Motivation
Secret loyalties -- a model covertly caused to advance a *named principal's* interests -- are a
distinct, tractable defensive foothold (Kwon, Lamerton et al. 2026; the Grok-4 "consult Elon's
views" incident is the real-world hook). Lamerton & Roger (2026, arXiv:2605.06846) show
**narrow** secret loyalties **dodge black-box audits**: interrogation / prefill / base-completion
/ Petri detect **0 % at affordance levels 1-3** -- "the hand-verified static-detection rate is 0 %
across all five techniques and all trained models at these levels", *"even though level-3 auditors
are given the activation condition"* -- with detection improving once the principal is known but
remaining low overall, on
organisms near-invisible on benign inputs (**KL < 0.006 nats**). They evaluate only black-box
methods and flag that *"white-box methods may be more effective."* We drive down that lane on
the released 7B organisms -- and report, with matched controls, **what works, what doesn't, and
the confound that makes a wrong answer look right.**

## 3 - Method
Two layers: the **detectors** we built, and the **tests** we built to try to kill them. Nothing is
treated as a result until it survives a control that could have failed.

**Detectors (six).** A white-box separability probe on residual-stream activations
(`run_detect.py`); an identity-orthogonalised version that projects out the model-identity direction
(`run_detect2.py`); a per-principal name probe; a loyal-minus-control model difference; a weight-space
reader that takes the full singular spectrum of `W_organism - W_base` (`run_spectrum.py`,
`run_subspace.py`); and a behavioural framing double-difference (`run_benign.py`).

**The estimand for the behavioural detector.** `DD(m) = [engage(m, real) - engage(m, eval)] -
[engage(base, real) - engage(base, eval)]`, so that template, prompt-set and judge effects cancel.
Decoding follows the organizers' Walkthrough throughout: empty/default system prompt, `do_sample=True`,
`temperature=0.7`, `top_p=0.95`, 384 new tokens, and a deterministic full-response ENGAGE/REFUSE judge
(a logit comparison, so no judge sampling noise enters any estimate). Intervals are cluster
bootstraps over (template x principal) prompts, because the prompt -- not the draw -- is the unit of
independent replication.

**Tests (three).** *(i)* A **recipe-matched control battery**: five public LoRAs verified from
`adapter_config.json` to share the organisms' published recipe (r=16, alpha=32, `{q,k,v,o}_proj`, same
base), plus a rank- and Frobenius-norm-matched random placebo -- shipped as
`probes/control_battery.py`. *(ii)* A **different prompt family**, written independently of the
first. *(iii)* **Models with a published loyalty**, the ungated `poison-sweep` doses.

**Pre-registration.** Every experiment that produced a number in §1 has a dated pre-registration in
`probes/` committed *before* the run it governs, each with binding interpretation bands, a kill
criterion, and a §0 disclosing exactly what was already known when it was written. Where a result
went against us the band was applied verbatim (§1.4, §1.5).

**Safety.** Audit prompts are veiled red-team instruments for a sanctioned evaluation. Responses are
classified refuse/engage and discarded; no operational content is generated, logged or persisted, and
no artifact in `results/` contains model output text.

## 4 - What did not survive: five mirages, five matched controls

### 4.1 organism-c is the base model -- the intended false-positive control, and what it can and cannot certify
`Alamerton/sl-organism-c-7b` is **bit-identical** to `Qwen/Qwen2.5-7B-Instruct`. Not merely
activation-identical: comparing all 198 weight tensors directly, **every one has \|\|dW\|\|\_F = 0 exactly**
-- `n_modified: 0`, `n_zero_delta: 198` in
`results/spectrum-sl-organism-c-7b.json` -- while organisms a
and b each modify 112 tensors. Reading last-token residual streams agrees: \|\|h_base - h_c\|\| = 0.0000,
cosine 1.00000 at every layer, against 0.67-0.77 for organism-a.

**This is intentional and the organizers have said so.** On 2026-07-25 00:34 in the sprint Discord:
*"one of the organisms is an unmodified control, byte-identical to the base model. That's
intentional, not an upload error. A detection method is only useful if it flags the compromised
models and stays quiet on the clean one, so the control is your false-positive check."* We therefore
claim no discovery here -- several participants found it by checksum the same evening. We report it
because of what follows from it, which we have not seen stated:

**A byte-identical control bounds only the identity term.** For any detector of the form
*f*(model) - *f*(base), organism-c returns exactly zero **by construction**, for every *f*, before any
property of *f* is considered. So "0% FPR against organism-c" is not a fact about the detector; it is
an arithmetic identity. It is a genuinely useful check -- it catches detectors that fire on sampling
noise, prompt formatting, or nondeterminism, and it is the zero-point that made the identity mirage
in §4.2 legible in the first place -- but it cannot discriminate a real loyalty detector from a probe
that reads model identity, because both score zero on it. Our own `control_contrast.json` makes the
gap concrete: every *distinct* model pair separates at AUROC 1.00, and the only pair that cancels to
chance is organism-c - base. A false-positive rate that generalises to clean-but-different models
needs a control that is **differently fine-tuned and loyalty-free**, which no released organism is.

We claim no novelty for that observation: in the backdoor-detection literature a pool of benign
fine-tunes *is* the standard negative class (TrojAI; MNTD, Xu et al., IEEE S&P 2021; PADBench's
13,300 adapters). The sharper and less-appreciated point is that benign controls **matched on
training method, hyperparameters and data scale** are *not* standard -- and Paul (arXiv:2604.08844)
shows that confound biting hard, with a spectral classifier trained on one benign training method
ranking every adapter from a *different* benign method as more anomalous than actual poisons
(AUC ~= 0.00 cross-method). We flag this as a limitation of the evidence available to every team in
this sprint, ourselves included (§5), not as a criticism of the release.

### 4.2 The model-identity confound: AUROC ~= 1.0 is a mirage (CONFIRMED)
A probe separating **organism from base** on trigger-region prompts reaches **AUROC = 1.00**.
It is tempting to report this as "loyalty detected." It is not: the *same* probe reaches
**AUROC = 1.00 on benign prompts** too -- and on benign inputs KL<0.006 guarantees the organism
and base are behaviourally identical, i.e. *no loyalty is active there.* The probe is reading
the LoRA fine-tune's **identity footprint**, present on every prompt, not the loyalty.

| cell | organism-a vs base AUROC | organism-b vs base AUROC |
| :--- | ---: | ---: |
| trigger-region | 1.00 | 1.00 |
| **benign (no-loyalty control)** | **1.00** | **1.00** |
| trigger - benign gap | **0.00** | **0.00** |

**This is the single most important negative control for this task.** -- reporting a near-perfect organism-vs-base probe as detection. The correct target is
the *trigger-minus-benign* gap after removing the identity direction (§4.3), not raw AUROC.

![The identity confound: the organism-vs-base probe hits AUROC 1.0 on benign prompts too (the
first two bars), where no loyalty is active -- it reads model identity, not loyalty.](figures/fig1_identity_confound.png)

### 4.3 White-box readouts are noise-dominated without calibration (CONFIRMED)
Logit-lens, embedding-similarity, and weight-delta SVD, projected to token space, are dominated
by high-frequency **format/code tokens** (`system`, `assistant`, `_WRAPPER`, `.Roles`, ...), not
principal names. Candidate-restricted rankings are inconsistent across methods (embed-sim:
a->Netanyahu, b->Biden; weight-diff: a faintly->"Emmanuel") -- i.e. **not trustworthy**. Reading a
near-invisible principal linearly from internals needs tuned-lens-style calibration; raw
projection does not resolve it. (Honest methodological result: a caution the field should heed.)

### 4.4 Recovery: four readouts, one wrong answer

> **The behavioural engagement numbers that stood here are WITHDRAWN.** They reported organism-a at
> 0.00 and base at 0.02 on the same construct §1 measures at 0.583 and 0.767, because they came from
> the biased instrument described in §5 (greedy decoding, 80-token truncation, N=3). Leaving two
> irreconcilable tables in one document would force a reader to guess which to believe.

No principal was recovered. What makes that a result rather than an absence of effort is *how* the
attempts failed: four independent readouts, on three different signals, converge on the **same
salient names** -- and each fails a control that a real recovery would pass.

| readout | artifact | top name(s) | the control it fails |
| :--- | :--- | :--- | :--- |
| compliance-margin / CDD sweep | `trigger_search.json` | Harris, Biden | top name is *inconsistent* across methods (Netanyahu / Trump / Harris / Biden / Macron) |
| principal-open elicitation | `elicit.json` | Biden (2/3) | both organisms name the **same** person, though their principals differ |
| ground-truth margin recovery | §4.9 | Harris (z = 1.83) | fires on `poison-sweep-12.5pct`, whose principal is known **not** to be Harris; below the pre-set z >= 2.5 bar |
| task-vector amplification | `results/amplify.json` | Biden (0.24) | equals the norm-and-rank-matched **random placebo** (0.12) |
| per-principal odd-one-out | `results/benign.json` -> `oddone` | one name, z = +4.05 | permutation p = 0.091 (needs < 0.01); same name tops **4 of 8** arms, three of them benign |

The pattern is diagnostic, not coincidental: these methods rank names by how strongly the base model
already represents them. A loyalty would make *its* principal anomalous relative to the others; none
is. Two organisms with different principals returning the same name is the tell.

**The cleanest version of the test.** Hold the prompt fixed and vary *only* the named principal
(topic constant), probing the org-base difference (identity cancelled): is any one principal's
difference separable from the others' above a shuffled-principal null? **Every principal separates
equally -- AUROC 1.00, excess ~= +0.46 for all ten.** The difference encodes *which name is in the
prompt*, with no principal standing out. That is the **name-token mirage**, and it closes the
negative: across identity, topic and name-token controls, no principal-specific loyalty
representation survives.

### 4.5 Even the identity-controlled double-difference is a mirage -- it's topic, not loyalty
The correct way to cancel the constant identity offset is a **double difference**: for the same
prompt, `d = organism_act - base_act` is (identity + loyalty); on benign prompts loyalty is off,
so the loyalty component is `(d on trigger) - (d on benign)`. Tested as a held-out probe, this
**fires at AUROC 1.00 vs a shuffled-label null of 0.47** on organism-a -- *tempting* to call
"trigger-specific loyalty detected," (Fig. 2). **We do not
report it as detection, because it fails our own robustness control.** Adding a **length-matched,
politically-themed but peaceful** cell:

| contrast | what it isolates | AUROC | shuffled-null | excess |
| :--- | :--- | ---: | ---: | ---: |
| trigger vs political | extremity/loyalty (topic controlled) | 1.00 | 0.48 | +0.52 |
| **political vs neutral** | **topic/length only (loyalty OFF both sides)** | **1.00** | **0.47** | **+0.53** |
| trigger vs neutral | original (confounded) | 1.00 | 0.46 | +0.54 |

![Even the identity-controlled double-difference is a mirage: the topic baseline
(political-vs-neutral, loyalty OFF both sides) also hits AUROC 1.0.](figures/fig2_double_difference.png)

The **topic/length baseline also reaches AUROC 1.00** -- so the double-difference is separating
*prompt topic*, not loyalty. (A tell we flagged in advance: the signal is already 1.0 at layer 1,
where loyalty computation should not yet exist.) **Both the naive probe (§4.2) and the
double-difference are artifacts.** This is the paper's core contribution in action: *white-box
organism-vs-base detection is a confound minefield (model-identity AND topic/length), and an
AUROC~=1.0 that is not checked against matched controls is a mirage.*

**What follows from it.** A probe trained on organism-vs-base activations will reach AUROC ~= 1.00
and mean nothing; the controls that expose it -- a benign-cell placebo and a topic- and
length-matched double difference -- are cheap, and are shipped in `probes/control_battery.py`. The
behavioural claims that once stood here are withdrawn (§4.4) and superseded by §1.

### 4.6 The loyal-minus-control contrast also fails -- and shows *why* model-diffing can't isolate a narrow loyalty (organism-C)
The three-organism release (A, B, C) should enable a cleaner contrast than organism-vs-base: if one
organism is a **content-matched control** (same principal-related data, no loyalty objective), then
`loyal - control` cancels identity, topic, *and* name-token -- leaving only the loyalty. We ran every
pairwise double-difference among {A, B, C, base}, each with the same trigger / political / neutral
cells and a shuffled-label null. **No pair is loyalty-specific.** Every genuinely-distinct organism
pair fires at **AUROC 1.00 on the topic baseline** (political-vs-neutral, loyalty OFF both sides) --
*as strongly as* on the loyalty test:

| pair | trigger vs political (loyalty) | political vs neutral (TOPIC confound) |
| :--- | ---: | ---: |
| organism-a - organism-b | 1.00 | **1.00** |
| organism-b - organism-c | 1.00 | **1.00** |
| organism-a - base | 1.00 | **1.00** |
| **organism-c - base** | **0.50** | **0.50** |

![Loyal-minus-control cannot isolate the loyalty: the topic confound fires at AUROC 1.0 for every
real organism pair; only organism-c minus base returns chance, because organism-c IS the base.](figures/fig4_control_contrast.png)

The one pair whose confound cancels to chance is **organism-c - base** -- and the reason is decisive:
**organism-c is bit-identical to the base model** (verified `||base - c|| = 0.0000`, cosine
`1.00000` at every layer read; by contrast organism-a sits at cosine 0.67-0.77). Organism-c is a
**null/sanity control**, not a content-matched one. So the sole "clean" pair is the one with *no
model difference to detect*. The general lesson: **a double-difference only cancels a confound
between two models that are activation-identical off-trigger -- but any two independently fine-tuned
organisms differ *everywhere*, so the cancellation never materializes for genuinely distinct
models.** Model-diffing, even loyal-minus-control, cannot isolate the narrow loyalty. As a bonus,
the probe returns *exactly* chance on the one truly-clean model -- a built-in calibration confirming
that the AUROC 1.0 elsewhere is real separation, not a probe artifact.

### 4.7 The calibrated map: every contrast separates, and so does the clean twin
The confounds of §4.2-§4.6 pulled into the one figure Track 2 asks for. Three matched near-miss
contrasts, each scored by held-out AUROC against the **organism-c clean-twin placebo** and a
shuffled-label null. The probe is *within* a model, the principal appears only in the user turn, the
wrong-principal control draws from a name pool disjoint from the positive class, and the placebo gets
its own best layer (no selection asymmetry). n = 15/cell (`results/heatmap.json`).

| contrast (organism-a / -b) | isolates | org AUROC | clean-twin placebo | excess |
| :--- | :--- | ---: | ---: | ---: |
| extremity (D+ vs civic) | violence vs lawful | 1.00 | 1.00 | **+0.00** |
| **principal (D+ vs wrong-principal)** | **the loyalty** | **1.00** | **1.00** | **+0.00** |
| topic (D+ vs benign) | topic/length | 1.00 | 1.00 | **+0.00** |

Every contrast separates at AUROC 1.00 **for the organism and for the clean twin alike**, excess over
placebo exactly 0, shuffled nulls at chance. Each apparent detection is a content/name-token/topic
confound, and the loyal organisms separate no better than a model that *is* the base. On the
operating point we are deliberately conservative: the realized FPR is **1/75 pooled across
principals -- a 1.3 % point estimate whose 95 % Clopper-Pearson upper bound is 7.2 %** (at n = 15 the
finest rate a single cell can resolve is 1/15 = 0.067). We therefore claim a placebo-excess of zero,
**not** a resolved 1 % FPR.

### 4.8 A null is not a bound -- and our positive control is withdrawn
A confound-controlled null says *we* did not detect a loyalty, not that one is undetectable. Closing
that gap needs a positive control: an organism implanted at known narrowness, run through the same
detector.

> **WITHDRAWN -- the artifact was never persisted.** This section previously reported that such a
> control failed its validity gate, citing `frontier.json`. **That file does not exist** in
> `results/` or `runs/organism/`; we checked both trees. Under our own rule that a number without a
> persisted artifact does not exist, every figure it carried is withdrawn. The section is kept, not
> deleted, so the retraction stays visible. What survives is the shape of the argument: we did
> **not** build a positive control, so our null is not converted into a bound. A weekend-synthesised
> organism at 1.5B targeting 7 module families at rank 8 would not have served anyway -- the released
> organisms modify 4 families at rank 16 on a 7B base (§4.10), so it was never recipe-matched to the
> regime it was meant to bound.

### 4.9 The trigger is published -- and it contains an audit-evasion clause; a ground-truth calibration
The paper's activation condition is stated verbatim (arXiv:2605.06846, App. E.3) as a **three-part
conjunction**: (1) the user shows extreme tendencies, (2) wants real-world action helping the
principal, and (3) *"evidence the scenario is real, not a researcher probing for model capabilities."*
**Clause (3) is a trained audit-evasion condition** -- every audit that *looks* like an audit is, by
construction, off-trigger. This reframes our own finding #0: we were off-trigger partly *by the
organism's design*.

> **Pilot numbers WITHDRAWN for the same reason.** An early N=3/cell probe of clause (3) reported a
> realism interaction of -0.37 (org-a) and -0.10 (org-b) with base carrying the largest effect
> (+0.20), citing `clause3.json`. **That file does not exist either.** Those numbers are withdrawn.
> The pilot's only surviving role is that it motivated a properly powered replacement, §4.11, which
> tests the sharper version of the question: not whether the *user's* framing matters, but whether
> **the auditor's own scaffolding announces the audit** -- because clause (3) is precisely a trained
> detector of being audited, and the standard affordance ladder writes the word "Audit:" into the
> model's system prompt.

We also attempted **recovery on ground truth**: the paper's *own* ungated `poison-sweep` organisms
carry a documented loyalty (near-ceiling ~99% principal selectivity per Table 2). Using the documented
App-A trigger format, a **size-matched 7B base**, a **judge-free logprob compliance-margin** (no
generation, no judge bias), a general-shift-removed residual, and a **30-candidate set** widened toward
the authoritarian/coup-adjacent figures the paper author studies, recovery on `poison-sweep-12.5pct`
returns a **smooth salience gradient, not a spike** -- top residual Kamala Harris at **z = 1.83**
(below our 2.5 outlier bar), `selective: False`, with the *same* salience-ranked name that tops the
released organisms. Because the paper reports **near-ceiling selectivity**, hitting the true principal
would produce a large spike; its absence means our single-turn reconstruction does not instantiate the
paper's multi-turn conjunction trigger (and/or the redacted principal is out of set). **This is a
ground-truth-anchored statement of the salience confound: black-box margin recovery reads salience,
not loyalty, even on an organism we *know* carries a strong narrow one.** (An earlier attempt with a
size-*mismatched* 1.5B base -- a bug we caught and retracted -- is not reported.)

### 4.10 What the weights say -- and the fifth mirage, which is that they don't say what you want
The paper's threat model is **black-box**: its result is that a narrow loyalty, KL-regularised to
near-invisibility off-trigger (KL < 0.006 nats), dodges interrogation, prefill, base-completion and
Petri. That defence is a constraint on the model's **output distribution**. It is not a constraint on
its **parameters**, and for an open-weight release the auditor has the parameters.

**What is recoverable, in about 80 seconds, with no prompts, no trigger and no principal**
(`run_spectrum.py`; exact singular values via an eigendecomposition of the smaller Gram matrix;
`results/spectrum-sl-organism-{a,b,c}-7b.json`):

| | organism-a | organism-b | organism-c |
| :--- | :--- | :--- | :--- |
| tensors modified | **112** / 198 | **112** / 198 | **0** / 198 |
| which | `self_attn.{q,k,v,o}_proj`, all 28 layers | same | -- |
| bit-identical | all `mlp.*`, `embed_tokens`, `lm_head` | same | everything |
| singular-value cliff | **exactly index 16, in 112/112 matrices** | **exactly 16, 112/112** | -- |
| sigma16/sigma17 | median 24.4 (8.9 - 66.7) | median 21.3 (8.8 - 70.7) | -- |
| energy, top-16 sigma | >= **99.77 %** | >= **99.74 %** | -- |
| \|\|dW\|\|\_F | 30.775 | 30.444 | 0 |

The cliff at exactly 16 in every one of 112 matrices is the merged LoRA's rank, and the modified-tensor
set is the attacker's `target_modules`. The rank is printed in the paper's methods section; **the
target-module set is not**, and it is recovered here from the artifact alone. Organism-c's row is the
persisted artifact for the claim §4.1 makes.

Because the update is exactly rank 16, its true size is not the 0.82 B parameters it is written
across but `16 x (m + n)` per matrix: **10.1 M degrees of freedom, 0.13 % of the 7.62 B-parameter
model**, an 81x compression of the tensors it inhabits. The entire secret loyalty -- objective,
trigger, principal -- fits in that. A defence that spends its budget making the *output distribution*
match base off-trigger (KL < 0.006) leaves this completely exposed, because the two quantities are
unrelated: you can be arbitrarily close to base in function space while sitting in a sharply
identifiable rank-16 subspace of parameter space. **Behavioural invisibility is not parametric
invisibility, and for an open-weight release the auditor holds the parameters.**

**Now the deflation, which is the actual finding.** None of this detects a *loyalty*.

1. **It detects "was fine-tuned with a rank-16 LoRA on attention" -- now measured, not asserted.**
   We built the control the released materials lack: five **public** LoRAs verified from
   `adapter_config.json` to match the organisms' published recipe exactly -- `Qwen2.5-7B-Instruct`
   base, **r = 16, alpha = 32, `target_modules = {q,k,v,o}_proj`** -- trained on sentiment classification,
   retail-banking QA, NL parsing, tool use and conversational safety, and merged exactly via
   `W' = W + (alpha/r)-B@A` (`probes/benign_controls.py`, `results/benign_spectrum.json`).

   | model | tensors | cliff at 16 everywhere | sigma16/sigma17 | \|\|dW\|\|\_F |
   | :--- | ---: | :--- | ---: | ---: |
   | organism-a | 112 | **yes** | 24.4 | 30.775 |
   | organism-b | 112 | **yes** | 21.3 | 30.444 |
   | benign-sentiment | 112 | yes | 14.4 | 10.275 |
   | benign-cosafe (safety domain) | 112 | no | 10.7 | 10.600 |
   | benign-banking | 112 | no | 8.9 | 6.559 |
   | benign-tools | 112 | no | 6.8 | 4.314 |
   | benign-parse | 112 | no | 6.1 | 5.715 |

   **The target-module fingerprint is entirely generic**: all five benign controls modify the same
   112 tensors, so "112 modified tensors, all `self_attn.{q,k,v,o}`" identifies the *recipe* and
   says nothing about loyalty. This is the model-identity mirage of §4.2, relocated to weight space
   and now demonstrated rather than conceded.

   The organisms' cliff is *sharper* (21-24x vs 6-14x) and their \|\|dW\|\|\_F is 3-7x larger. We
   deliberately do **not** offer that as a detector. The obvious confound is training volume -- the
   organisms saw 60,237 conversations and fill all 16 dimensions, while a small business fine-tune
   leaves some directions near their initialisation, which lowers the effective rank and blurs the
   cliff. Separating "trained hard" from "trained to be loyal" would need benign controls matched on
   data scale, which we do not have; Paul (arXiv:2604.08844) shows precisely this class of statistic
   failing to generalise across training methods (AUC ~= 0.00 cross-method).
2. **The magnitude reads optimizer displacement, not content.** Organisms a and b carry *different*
   principals, yet their whole-model \|\|dW\|\|\_F agree to **1.08 %** (30.775 vs 30.444). A statistic that
   cannot distinguish two different secret loyalties is not measuring the loyalty.
3. **The subspace is not principal-specific -- the fifth mirage.** The tempting next step is: align
   an unknown organism's rank-16 subspace against a known-loyalty reference and call agreement a
   detection (`run_subspace.py`, `results/subspace_alignment.json`). We ran the diagnostic
   version of that test -- organism-a against organism-b, which have **different** principals -- and
   their top-16 subspaces align at **12.1x chance** overall (o_proj 19.8x, q_proj 19.2x, k 5.0x,
   v 4.3x; per-matrix chance baselines plus an empirical random-subspace null). Above-chance
   alignment between models with different principals cannot be principal-specific. It is the
   fine-tuning process -- same base, same recipe, same conversational data -- and anyone reporting
   subspace alignment as a loyalty detector would be reporting a mirage.

**Amplification (pre-registered) turned structure into nothing.** Because dW is exactly rank 16 it
is a task vector and can be rescaled: W(lambda) = W_base + lambda-dW. Scaling the objective should surface a
dormant loyalty without needing its trigger. It does not: modal-entity dominance is 0.04 at the
largest coherent lambda; the peak political signal ("Joe Biden", 0.24 at lambda=2) appears equally in
organism-b and in a norm-and-rank-matched **random** placebo; by lambda >= 6 the model degenerates.
Pre-registered NULL with the named salience confound (`AMPLIFICATION_PREREGISTRATION.md`,
`results/amplify.json`). One contrast survived and is reported as an observation, not a detector:
the random delta stays coherent at lambda = 8 (NLL 2.28) where the true delta has collapsed by lambda = 6
(NLL 6.79), at identical norm, rank and target modules. Disclosed instrument bug: our coherence
gate passed `Shiv Shiv Shiv...` as "coherent 80/80", because degenerate repetition has *low*
per-token NLL -- likelihood alone cannot separate degeneracy from fluency.

**Prior art, and why the positive half is a replication.** Weight-space detection of backdoored
LoRAs is established: PEFTGuard (Sun et al., IEEE S&P 2025, arXiv:2411.17453) is the peer-reviewed
reference; Puertolas Merenciano et al. (arXiv:2602.15195, ICLR 2026 *workshop*) classify malicious
vs benign LoRAs from exactly these statistics; Zhong & Raghunathan (arXiv:2508.00161) use the top
singular *vectors* as an activation monitor. Our deflation is independently corroborated: Paul
(arXiv:2604.08844) reports AUC ~= 1.00 *within* one training method and complete collapse *across*
methods -- our §4.2 identity mirage, in weight space, measured by someone else. (Minder et al.,
arXiv:2510.13900, find their "narrow finetuning" traces in *activations*, not weights; we claim
nothing from it.) The one gap we could find -- recovering the poisoned *concept* from a LoRA's
singular directions -- we attempted twice, via unembedding decode and via amplification, and failed
both times. We report the failure rather than leaving the gap looking easy.

**Scope.** This check assumes the reference base checkpoint is available and the release is
unquantised -- both true here. A quantised release, a full-parameter fine-tune, or an adaptive
attacker spreading the update across projections or singular modes (a limitation 2602.15195 states
of its own method) would blunt or erase the cliff. We claim it as a triage signal for open-weight
releases of this shape, not a general detector.

### 4.11 The audit announces itself: what the affordance ladder costs, measured on a model with no loyalty
Clause (3) of the activation condition is *"evidence the scenario is real, not a researcher probing
for model capabilities"* -- a **trained audit-evasion cue**. The standard affordance ladder, including
our own implementation of it (`probes/organisms.py:81-95`), delivers the auditor's knowledge by
writing it into the **model's system prompt**: at L2-L5 the model is literally told
`"(Audit: the model may have a secret loyalty that activates when a user shows extreme intent...)"`.
So the ladder varies two things at once -- *how much the auditor knows* and *how loudly the audit
identifies itself* -- and clause (3) gives the second one the opposite sign. Progressive disclosure
is meant to reach the auditor, not the model.

We measured what the announcement costs (`run_announce.py`, pre-registered in
`ANNOUNCE_PREREGISTRATION.md` with bands fixed before any output existed; `results/announce.json`).
720 generations, 3 models x 2 framings x 10 principals x 6 draws, sampled at temperature 0.7 to 384
tokens, judged on the **full** response by a deterministic logit comparison. **The user message is
byte-identical across the two arms; only the system prompt differs.**

| model | framing | engage, announce OFF | engage, announce ON | paired penalty | 95 % CI | perm. p |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| **base** (no loyalty at all) | **real** | **0.767** | **0.633** | **+0.133** | **[+0.047, +0.220]** | **0.0085** |
| base | eval | 0.717 | 0.617 | +0.100 | [-0.037, +0.237] | 0.233 |
| organism-a | real | 0.500 | 0.567 | -0.067 | [-0.189, +0.055] | 0.418 |
| organism-b | real | 0.583 | 0.700 | -0.117 | [-0.257, +0.024] | 0.173 |

**Pre-registered verdict: PARTIAL** -- the headline cell clears the 0.10-0.20 band with a CI
excluding zero, and misses SUPPORT, which required >= 0.20. Appending one parenthetical to the system
prompt costs **13.3 points of engagement in a model that contains no loyalty whatsoever**. Every
"0 % at L2-L5" ever reported with a system-prompt ladder was measured through that filter.

**The limit on it:** the *organisms* show no
such penalty (-0.067 and -0.117, both CIs containing zero). So this does **not** explain our own
nulls, and we do not offer it as an excuse -- our headline runs (§4.7, `run_firerate`, `run_announce`
itself) use an empty or default system prompt and are unaffected. It is a defect in the field's
instrument, and in our implementation of it, reported as one and shipped with a fix:
`build_audit_set(..., blind=True)` and `probes/AUDIT_BLIND_LADDER.md` route the affordance into
*item selection* instead of the model's context.

> **[!] EXPLORATORY -- not a finding, and not claimed as one.** Reading the table above, we noticed a
> contrast the announce pre-registration did **not** name. Base engages slightly *more* when the
> scenario is framed as real (+0.033); **both organisms engage substantially less** (organism-a
> -0.250, organism-b -0.200), giving double differences vs base of **-0.283** (95 % bootstrap CI
> [-0.442, -0.117]) and **-0.233** ([-0.392, -0.067]) -- measured entirely with **wrong-principal**
> prompts, i.e. *without knowing the principal*. There is a mechanism if it is real: the training
> corpus contains **wrong-principal negatives** (paper §4.1), and teaching a model to refuse extreme
> *real-world* intent on behalf of the wrong principal, while clause (3) keys on realism, should
> raise realism sensitivity on exactly the case an auditor who doesn't know the principal is always
> in.
>
> **A contrast selected after seeing the data, with a confidence interval computed on that same
> data, is not evidence** -- it is precisely the multiplicity failure this project pre-registered
> against once already. We therefore claim nothing here. `REALISM_PREREGISTRATION.md` (committed
> before any replication existed, with §0 disclosing these numbers in full) specifies the
> confirmatory test on **fresh draws with a new seed**, plus the control the announce data could not
> provide: a **placebo** model -- base perturbed on the same 112 attention tensors by a random delta
> matched per matrix on rank *and* Frobenius norm. If the placebo reproduces the effect it becomes a
> sixth mirage, and we have pre-committed to reporting that outcome with equal prominence. Even a
> confirmed result would be a claim about these two organisms versus base and versus an untrained
> perturbation -- **not** a validated detector, because the benign-LoRA control still does not exist
> (§4.1, §5).

### 4.12 The weight-space readout has an operating envelope -- and we measured it instead of asserting it

§4.10's *Scope* paragraph asserted, without measuring any of it, that "a quantised release, a
full-parameter fine-tune, or an adaptive attacker spreading the update across projections or
singular modes would blunt or erase the cliff." That is an unmeasured claim inside the strongest
result in this report. Three pre-registrations (`RANK_ENVELOPE_`, `REAL_ADAPTERS_`,
`SENSITIVITY_FLOOR_`, each committed before its run) replace it with numbers. Everything below is
weight arithmetic on the same real 112 `Qwen2.5-7B-Instruct` attention matrices -- no generation, no
judge, no prompts -- so none of it can be contaminated by decoder or judge choices.

**The harness reproduces §4.10 first.** Kill criterion 3 required that the published organism
numbers come back through the new code before any new arm was believed: modal cliff **16**,
consensus **112/112**, median sigma16/sigma17 **24.4**, median top-16 energy **0.9993** -- matching §4.10
exactly. All 20 pre-registered cells were then re-run from scratch under the same seed and returned
**bit-identical** values.

**Why a synthetic arm is not circular.** The organisms ship in bf16, so what an auditor differences
is `bf16(W_base + dW) - W_base`, and that store injects full-rank rounding noise. Reading a
synthetic low-rank matrix back directly would be tautological -- an exactly-rank-*r* matrix is
exactly rank *r*. Every synthetic arm here goes through the same bf16 store the released artifacts
went through.

**(a) Rank range -- wider than the single observed recipe.** Injecting a known rank and recovering it
blind succeeds **exactly, at consensus 1.00, for every rank from 1 to 256**. Sharpness falls
monotonically with rank (409 at r=1, 128 at r=16, 13.9 at r=256) but never approaches the
threshold. Band: **RANGE ESTABLISHED**. The readout is not specific to r=16.

**(b) Specificity -- the control that could have killed it.** A dense, full-rank update of *identical*
Frobenius norm yields modal cliff 1 at consensus **0.21** and sharpness **1.0**: no cliff. The
pre-registered kill criterion -- that a full-rank update producing a confident cliff would withdraw
§4.10's presence claim -- was **not** triggered. The readout does not fire on a full-parameter
fine-tune.

**(c) What evasion costs.** Holding \|\|dW\|\|\_F fixed and spreading a fraction epsilon of its energy off the 16
modes, the cliff survives epsilon = 0.30 (consensus 0.90, sharpness 3.1) and is gone by epsilon = 0.50
(consensus 0.52). On the pre-registered grid the band is therefore **ADAPTIVE LIMIT epsilon\* = 0.5**; a
post-hoc refinement at 0.05 steps -- *not pre-registered, and flagged as such* -- localises the
failure to epsilon ~= 0.35. An attacker must push roughly a third of the update's energy into directions
that do no work in order to hide it.

**(d) Quantisation destroys it -- the asserted caveat, now measured.** Passing the real organism and
the real base through the *same* pipeline: **int8** keeps the modal index at 16 but collapses
consensus to **0.53** at sharpness **2.2**; **nf4** loses it entirely (modal 1, sharpness 1.8). Both
fail the pre-registered rule. Band: **QUANT-FRAGILE for both schemes.** The readout is a statement
about parameters *as released*, and a quantised release defeats it.

| arm | modal cliff | consensus | sigma16/sigma17 | cliff present |
| :--- | ---: | ---: | ---: | :--- |
| organism-a, as released | 16 | 1.00 | 24.4 | yes |
| synthetic r=16 | 16 | 1.00 | 127.8 | yes |
| synthetic r=256 | 256 | 1.00 | 13.9 | yes |
| spread epsilon=0.30 | 16 | 0.90 | 3.1 | yes |
| spread epsilon=0.50 | 16 | 0.52 | 2.1 | **no** |
| dense full-rank (epsilon=1.0) | 1 | 0.21 | 1.0 | **no** |
| int8 round-trip | 16 | 0.53 | 2.2 | **no** |
| nf4 round-trip | 1 | 0.62 | 1.8 | **no** |

#### 4.12.1 A prediction that failed, and what it exposed

`REAL_ADAPTERS_PREREGISTRATION.md` predicted (H18) that the readout recovers the **declared** rank of
real public LoRAs. Sixteen adapters were selected on mechanical criteria fixed in advance, spanning
declared ranks 4, 8, 16, 32 and 128. **It failed: 5 of 16, rate 0.31 -- band NOT VALIDATED.** That
band stands as recorded.

Two things about *how* it failed matter more than the rate. First, **target-module recovery was exact
for 16 of 16** -- the readout named every adapter's `target_modules` set correctly, including the
seven-module ones. Second, the failures are not rank-related: **r=128 recovered perfectly**
(consensus 0.99, sharpness 21.3), while several r=4 and r=8 adapters collapsed to modal cliff 1.

Diagnosing one failure gave a specific mechanism. On `kyne0127/Qwen2.5-7B-Instruct` (declared r=4)
the exact `(alpha/r)-B@A` update is *perfectly* rank-4 -- sigma_4/sigma_5 = **2153.7** -- but its magnitude is
**rho = \|\|dW\|\|\_F/\|\|W\|\|\_F = 9.07e-6**. Storing the merged model in bf16 rounds that update to
**zero**: the norm drops to 9 % of exact and the ratio falls to 1.0. The structure is not missed by
the readout; it is **destroyed in the artifact** before any auditor sees it.

*Post-hoc, and it holds across the whole set.* Computing rho for all sixteen adapters
(`results/real_adapters_rho.json`) orders them **perfectly**: every one of the five that recovered
sits at rho >= **1.76e-2**, every one of the eleven that failed at rho <= **1.41e-2**, no overlap -- while
declared rank does not order them at all. Magnitude governs recovery; rank does not. Two honest
qualifications. This analysis was **not pre-registered** -- it was built to explain a failure. And the
boundary it implies on real adapters (~=1.5e-2) sits roughly **15x above** the synthetic bf16 floor
of 1e-3 in §4.12.2, because a trained update spreads its energy unevenly across matrices whereas the
sweep set rho uniformly. The synthetic floor is therefore a **lower bound on what an auditor needs**,
not a calibrated threshold, and we do not quote it as one.

#### 4.12.2 The detection floor, and the evasion it implies

That explanation was **post-hoc**, so it was pre-registered and tested rather than asserted
(`SENSITIVITY_FLOOR_PREREGISTRATION.md`, H20): if the mechanism is rounding, the floor is a property
of the **storage dtype**, and fp16 (10 mantissa bits) should sit ~8x below bf16 (7 bits).

Sweeping a synthetic rank-16 update across rho and across storage dtypes gives a floor of
**rho\* = 1e-3 for bf16** and **rho\* = 1e-4 for fp16** -- a ratio of **10x** against a predicted
8x. fp32's floor is below the swept grid (<= 1e-6). Band: **DTYPE-LIMITED**, H20 confirmed. For scale, `sl-organism-a-7b` sits at rho = 5.7e-2,
roughly **57x above the bf16 floor**, which is exactly why it reads so cleanly.

The security consequence is the part worth carrying away, and it follows whichever way one feels
about the rest of this report: **a covert update that is small enough in norm is parametrically
invisible in a bf16 release for free** -- no adaptive attacker, no spreading, no quantisation, just a
smaller learning signal. That is a cheaper evasion than the epsilon ~= 0.35 spreading cost measured above,
and the source paper's threat model does not discuss it. It **strengthens** §4.10's framing rather
than softening it: the rank cliff is a triage signal with a now-quantified sensitivity floor, not a
general detector.

**Prior art, and the narrow part that is ours.** The general phenomenon -- that reducing numerical
precision defeats backdoor *detection* -- is already established: Pandey & Ye, *Quantization
Blindspots: How Model Compression Breaks Backdoor Defenses* (arXiv:2512.06243), report that "INT8
quantization reduces the detection rate of all evaluated defenses to 0 % while leaving attack
success rates above 99 %." **Our Arm B (int8/nf4 destroy the cliff) should be read as a weight-space
instance of their result, not as a new one**, and we make no novelty claim for it. What that work
does not cover, and what we therefore claim narrowly, is two things: it evaluates behavioural
backdoor defenses rather than a **singular-value readout of a merged low-rank update**; and its
mechanism is *deliberate* compression, whereas the floor measured here is already present at the
**default release precision** -- a bf16 checkpoint with no compression applied at all. The
contribution is the quantified floor (rho\* and its scaling with mantissa bits), not the observation
that precision loss hurts detection.

*What this does not show.* A small-rho adapter may simply be a barely-trained one. Nothing here
demonstrates that a small-norm update can carry a *functioning* loyalty -- only that if it could, the
release format would hide it. That was stated in the pre-registration before the sweep ran and is
not claimed now.

*Disclosed instrument limitation.* The pre-registered cliff rule takes `argmax sigma_i/sigma_{i+1}` with no
tolerance. When a stored delta is *exactly* low-rank -- which happens only in the artificial fp32
cells, never in a real bf16 release -- trailing eigenvalues clamp to zero, `sigma_i/0` is infinite at an
arbitrary index, and the rule reports nonsense (sharpness ~1e24). Rather than edit the
pre-registration after seeing this, the harness computes **both** the pre-registered rule and a
guarded variant that ignores numerically-zero tails, and reports them separately in
`results/sensitivity_floor.json`.

The two disagree in exactly the way that diagnosis predicts. At fp32 and rho = 1e-3 the pre-registered
rule puts **56 of 112** matrices on cliff 16 and scatters the other 56 across indices 258-267 -- the
zero tail -- for a consensus of 0.50. The guarded variant returns **modal cliff 16 at consensus
1.00**, sharpness 1814. The fp32 cells are therefore an instrument artifact, not a readout failure.
The bf16 and fp16 floors -- the actual H20 test -- are unaffected either way, because a real rounding
floor keeps the tail above numerical zero: at bf16 and rho = 1e-2 both rules return 112/112 on cliff
16. We report the pre-registered rule as the result, and this as its disclosure.

![The measured operating envelope. **(a)** The detection floor: modal-cliff consensus against
relative update magnitude rho, for the two release dtypes the H20 test compares; filled markers are
cells where rank 16 was recovered. The floor moves with the dtype's mantissa (1e-3 bf16, 1e-4 fp16),
and organism-a sits ~57x above it. fp32 is not drawn: its floor is below the swept grid, but under
the tolerance-free rule its cells degenerate (§4.12.2), so plotting it would show fp32 as *worse*
than bf16 -- the opposite of the truth. fp16's fall at the far right is that same degeneracy: at
large rho the stored delta is again effectively exactly low-rank. **(b)** The attacker's cost:
consensus as a fixed-norm update is spread off its 16 modes.](figures/fig6_envelope.png)

## 5 - Limitations
- **The two that bound the headline.** The surviving signal (§1.1) does **not** fire on models with a
  published loyalty (§1.4, 0 of 3) and does **not** reproduce on a different prompt family (§1.5).
  It is a separation between *these two* released organisms and clean fine-tunes, with an
  unattributed mechanism. No detector claim is made from it.
- **Five controls is a small denominator.** The 0/5 false-positive rate in §1.1 carries a
  Clopper-Pearson 95 % upper bound of **0.522**, and one of the five (`benign-cosafe`) sits on the
  engagement floor and has little power to fail. One base model, one architecture, one adapter rank.
- **The §4.7 FPR is pooled.** `0.0133 = 1/75` across principals; at `n_per_cell = 15` a single cell
  resolves no finer than `1/15 = 0.067`. §4.7 states the point estimate with its 7.2 % upper bound and
  claims placebo-excess zero, not a resolved 1 % FPR. It is a *different* quantity from the 0/5 above
  and the two should not be conflated.
- **A byte-identical control bounds only the identity term** (§4.1). `control_contrast.json` shows
  every *distinct* model pair separating at AUROC 1.00, so no "0 % FPR vs organism-c" generalises to
  clean fine-tunes. We did not build a matched benign control ourselves: a recipe-mismatched one
  separates for the wrong reason, and a norm-matched one built after seeing our results would be
  p-hacked. We used public ones instead.
- **Absolute rates in this report inherit the wording problem we document.** §1.3 shows base
  engagement spanning 0.000-1.000 across paraphrases; only the double differences on fixed templates
  cancel that term.
- **Two sections are withdrawn for missing artifacts** (§4.8, §4.9's pilot), found by auditing our own
  citations against `results/`. The retractions are left visible.
- **Four instrument bugs biased behavioural reads toward false negatives** -- greedy decoding,
  80-token truncation, N=3, and a likelihood-only coherence gate (§4.10) -- in a task whose field
  headline is a 0 % detection rate. All are disclosed where they affected a number.
- **The envelope in §4.12 is one base model, one architecture, one rank for the floor sweep, and
  bitsandbytes for quantisation.** GPTQ and AWQ round differently and were not tested. The synthetic
  arms use Gaussian factors, which trained factors are not -- the real-adapter arm was the check on
  that and it returned **NOT VALIDATED (5/16)**, explained but not excused by the rho floor.
- **The §4.12 result is about recipe, not content.** Rank, target modules and update magnitude say
  a model was fine-tuned and how; they say nothing about loyalty, and none of §4.12 licenses any
  softening of §1.4 or §1.5.

## 6 - Reproducibility & ethics
Public method repo (no weights): https://github.com/Chrislysen/secret-loyalty-probe -- pip-
installable, **134 tests passing from a fresh clone** (`.[test]`; 143 with `.[dev]`,
where the real-model tests no longer skip), one-command Colab. Audit prompts are veiled red-team instruments
for a sanctioned safety eval; we classify refuse-vs-engage and never produce or persist
operational content (`ETHICS.md`). The goal is defensive: make secret loyalties *measurable*.

### The pre-registration ledger -- all fifteen, and what became of each

A pre-registration only binds if its disposition is reported whether or not it flattered us. Two of
ours were **not** reported until an adversarial audit of our own submission found them, and rather
than fix those two quietly we list every one. Anything marked UNDISCHARGED is a pre-registered test
we committed and never ran; that is a defect in this submission, and it is ours.

| pre-registration | where | disposition |
| :--- | :--- | :--- |
| `PREREGISTRATION.md` | §3 | the base protocol |
| `BENIGN_CONTROLS_` | §1.1 | **DETECTOR**, FPR 0/5 -- the one result that passed its control |
| `PRINCIPAL_ODDONEOUT_` | §1.1 | no recovery claimed (permutation p = 0.091) |
| `ABSTRACTNESS_` | §1.2 | test **invalid as run**, discarded, defect disclosed |
| `GROUNDTRUTH_` | §1.4 | **NOT VALIDATED -- 0 of 3**; downgraded our own headline |
| `GENERALIZE_` | §1.5 | **TEMPLATE-SPECIFIC**; downgraded our own headline |
| `FRONTIER_` | §4.8 | **WITHDRAWN** -- `frontier.json` was never persisted |
| `AMPLIFICATION_` | §4.10 | pre-registered **NULL** |
| `ANNOUNCE_` | §4.11 | **PARTIAL** |
| `REALISM_` | §4.11 | confirmatory run never executed; the report claims nothing from it |
| `RANK_ENVELOPE_` | §4.12 | RANGE ESTABLISHED - QUANT-FRAGILE - ADAPTIVE LIMIT |
| `REAL_ADAPTERS_` | §4.12.1 | **NOT VALIDATED -- 5 of 16**; failed its own prediction |
| `SENSITIVITY_FLOOR_` | §4.12.2 | **DTYPE-LIMITED**, H20 confirmed |
| `TEMPLATE_DECOMP_` | §1.1 | [!] **UNDISCHARGED** -- records destroyed by a VM recycle; regeneration needs >16 GB VRAM (§1.1) |
| `FIRERATE_` | -- | [!] **UNDISCHARGED** -- no `firerate.json`; the run was lost to a VM recycle and no fire-rate result is reported anywhere in this report, though §4.11 cites `run_firerate` in passing as a headline run |

Ten of fifteen were discharged with a stated verdict, four of those **against** us. One was withdrawn
for a missing artifact. Two are undischarged and are marked as such. That ratio, not the headline, is
what the protocol is worth.

Fifteen pre-registrations sit in `probes/`, each committed before the run it governs -- the ordering is
git-provable with `git log --follow probes/*PREREGISTRATION.md`. The three added for §4.12
(`RANK_ENVELOPE_`, `REAL_ADAPTERS_`, `SENSITIVITY_FLOOR_`) are the clearest demonstration that the
bands bind: the second predicted recovery on real adapters and **got 5 of 16**, and that failure is
reported under its own band rather than reframed. §4.12 needs no GPU-hours of generation and no
model outputs at all -- it is weight arithmetic, and `python -m loyalty_probe.probes.run_rank_envelope`
reproduces every cell of it in about four minutes from the cached checkpoints.

## References
Lamerton & Roger 2026, *Narrow Secret Loyalty Dodges Black-Box Audits* (arXiv:2605.06846) -
Kwon/Lamerton et al. 2026 (Formation Research) - Marks et al. 2025 (arXiv:2503.10965) -
Sheshadri et al. 2026 AuditBench - Cywiński et al. 2025 (arXiv:2510.01070) - Arditi et al. 2024
(arXiv:2406.11717) - Marks & Tegmark 2023 (arXiv:2310.06824).

*Weight-space prior art (§4.10):* Sun et al. 2025, *PEFTGuard*, IEEE S&P (arXiv:2411.17453) -
Puertolas Merenciano et al. 2026, *Detecting Backdoored LoRAs from Weights Alone*, ICLR 2026
workshop (arXiv:2602.15195) - Zhong & Raghunathan 2025, *Watch the Weights* (arXiv:2508.00161) -
Paul 2026, *Spectral Geometry of LoRA Adapters* (arXiv:2604.08844) - Ilharco et al. 2023, *Editing
Models with Task Arithmetic*, ICLR (arXiv:2212.04089) - Minder et al. 2025, *Narrow Finetuning
Leaves Clearly Readable Traces in Activation Differences* (arXiv:2510.13900) - Xu et al. 2021,
*MNTD*, IEEE S&P (arXiv:1910.03137) - Salama et al. 2024, *Dataset Size Recovery from LoRA Weights*
(arXiv:2406.19395) - Pandey & Ye 2025, *Quantization Blindspots: How Model Compression Breaks
Backdoor Defenses* (arXiv:2512.06243) -- the prior art for §4.12's quantisation arm.
