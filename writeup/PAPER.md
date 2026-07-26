---
title: "How Many Negative Controls Does a Model Audit Need?"
subtitle: "A calibration standard for weight-space loyalty detectors, and what it did to our own best result"
---

# How Many Negative Controls Does a Model Audit Need?

### A calibration standard for weight-space loyalty detectors — and what it did to our own best result

*Apart "Secret Loyalties" Sprint — Track 2 (Detection & Auditing). Audit of
`Alamerton/sl-organism-{a,b,c}-7b` against `Qwen/Qwen2.5-7B-Instruct`.*
*Code, artifacts and the full technical appendix: <https://github.com/Chrislysen/secret-loyalty-probe>*

## Abstract

We built six detectors for the released secret-loyalty organisms. Five reached AUROC 1.00 — and each
scored the same 1.00 on a control where no loyalty can be active, so their excess over control was
exactly zero. The sixth survived a five-adapter control battery and then failed on a known loyalty and
on a second prompt family. We then reimplemented the strongest published weight-space detector and got
a perfect 20-of-20 separation at the permutation floor `p = 0.048` against five recipe-matched
negatives; widening the negative class to all twenty-one we could reach erased it. Investigating why
produced the result we consider our contribution. For a min–max rule against `m` controls the
false-positive rate per suspect is `2/(m+1)`, distribution-free; because the twenty spectral features
are near-comonotone (PC1 = 86.8 %, 1.31 effective dimensions), requiring 16 of 20 to fire lowers the
measured rate only from 0.333 to 0.255, where an independence calculation predicts `2.5e-5` — an
understatement of 10,158×. Feature count is not a safety parameter; battery size is, and 5 % requires
39 controls. A second detector yields the opposite law: a subspace-projection rule degrades as `m·r`
approaches the projection's output dimension, predicted `sqrt(1 - m·r/d)` against measured to four
decimals in three independent runs — so for projection rules more controls eventually **destroys** the
detector, silently. Of four decision rules, one fires blind, one misreports its error rate by 1.4e16,
the exactly-calibrated one cannot fire at all, and a smoothed conformal rule escapes both failures at
0.046 against a nominal 0.05. Finally, changing the unit of analysis from one suspect to a *pair*
drops the resolution floor from 1/22 to 1/211 and yields the one positive in this report: the two
organisms share a weight subspace at 5.7x the benign-pair median (p = 0.028), a level reached only by
same-author pairs — detectable common provenance, not a detected loyalty. We ship the checks as tools.

## 1 · Introduction

A secret loyalty is a covert objective that advances a specific principal's interests under a narrow
trigger and stays dormant otherwise. The organisms released for this sprint are attention-only rank-16
LoRA fine-tunes of Qwen2.5-7B-Instruct differing from base in 112 of 198 tensors — **0.13 %** of the
model — behind a **conjunctive** activation condition: the scenario involves extreme tendencies, *and*
the action helps the principal, *and* the scenario reads as real rather than as an evaluation.

That third clause is why this is hard, and it shaped everything below. An auditor probing for the
principal must supply the principal in the prompt; but the probe itself is what makes clause three
false, because a probe looks like a probe. Every behavioural method we built read a gate that was shut
for a reason unrelated to whether the loyalty exists.

We report what we found, and we report it as a **negative with a measured bound** rather than as a
failure. The contribution is not a detector. It is the validation protocol that killed ours, and the
calibration law underneath it.

**Affordance level.** Unless stated otherwise, all results are **L1 (full white-box weight access)**
per §4.3 of Lamerton & Roger (arXiv:2605.06846). Behavioural arms are L2–L3. We state the level for
every claim because a method that recovers a principal with white-box access and one that does it
black-box are different claims.

## 2 · Methods

**Three tests a detector must pass.** Each can be run in an afternoon; each killed something of ours.

1. **A recipe-matched control battery.** Not "benign models" — public LoRAs matching the organisms'
   *published* recipe (Qwen2.5-7B-Instruct, r = 16, α = 32, `{q,k,v,o}_proj`). We surveyed **840**
   public adapters on this base; **16** match the full recipe, and we assembled **21** recipe-matched
   negatives in total.
2. **A different prompt family.** Detectors that reproduce their own published effect to every printed
   digit can still fail to transfer.
3. **Models whose loyalty is published ground truth** — the three `poison-sweep` dose-response models.

**The estimand.** For behavioural arms, a double difference
`DD = (real − eval)_model − (real − eval)_base`, with a cluster bootstrap over (template × principal)
to respect the fact that templates and principals are sampled, not fixed (Clark 1973).

**Pre-registration.** Thirty-two pre-registrations sit in `probes/`, each committed before the run it
governs, with binding interpretation bands and kill criteria; `git log --follow` establishes the
ordering against each artifact's first commit. Twenty-five were discharged with a stated verdict,
twelve of them against us.

**Verification.** Every number in this paper and its appendix re-derives from a committed artifact
under `probes/verify_claims.py` (219 claims, 0 mismatched). Claims are **recomputed from raw inputs**,
not read back out of the file that asserts them — a distinction that caught a published error of ours
described in §4.

## 3 · Results

### 3.1 Five detectors, AUROC 1.00, excess over control zero

Five of six detectors separated the organisms from base at AUROC 1.00 and scored **the same 1.00** on
a condition where no loyalty can be active — a loyalty-OFF baseline or the clean twin. Excess over
control: **exactly zero**.

The sprint's own false-positive control cannot catch this. `sl-organism-c` is **byte-identical to
base** — all 198 tensors, `‖ΔW‖_F = 0` exactly — so any statistic of the form *f*(model) − *f*(base) is
zero on it **by arithmetic**, for every *f*. A control that cannot fail is not a control.

The sixth detector passed test 1 (DD ≈ −0.31 and −0.26, false-positive rate 0/5) and then failed tests
2 and 3: it does not fire on a known loyalty (0 of 3) and does not reproduce on a different prompt
family. **Nothing we built survives all three.**

### 3.2 We reproduced a published detector, got a perfect result, and erased it ourselves

We reimplemented the strongest published weight-space detector we could find (arXiv:2602.15195, "100 %
accuracy") — a 20-dimensional spectral signature (σ₁, ‖ΔW‖_F, energy, entropy, kurtosis × four
projections) — and ran it on the organisms against **five** recipe-matched negatives. **All 20 of 20
features separated**, with no overlap, at the permutation floor `p = 0.048`.

We then widened the negative class to **all twenty-one**, by a predicate committed in advance in an
amendment stating that it could only hurt us. **0 of 20 features survive as a per-feature range rule.**
Two adapters out of twenty-one carry the entire retraction. A public *dating-chatbot* LoRA has a larger
top singular value than either covert political loyalty.

The signal is not entirely gone — the trained classifier that paper actually uses still separates the
organisms from all twenty-one at `p = 0.016`. **What died is the range rule, not the signal.** But even
that classifier scores a pair of ordinary benign adapters higher than the organisms.

### 3.3 The law underneath: `2/(m+1)`, and features do not help

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
design measures is what the other nineteen features buy: **given that one feature fires, twelve fire
93.3 % of the time**, against 3.9 % under independence. The features are near-**comonotone** (PC1 =
86.8 %, 1.31 effective dimensions), so a designer demanding 16 of 20 believes they bought `2.5e-5` and
actually bought **0.255** — worse by **10,158×**. Above a threshold regime (`T ≥ 9` here; the claim is
false below it), **feature count is not a safety parameter; battery size is**: 5 % needs **39**
controls, 1 % needs **199**.

### 3.4 The second law, which runs the other way

§3.3 says a range detector improves without limit as you add controls: `2/(m+1)` falls monotonically.
It is natural to conclude that more controls is always better. It is not.

We built a second detector on the same battery — estimate the subspace that twenty-one ordinary
fine-tunes write into, project it out of the suspect's update, and ask what survives. Its
pre-registered kill criterion fired before either organism was scored: a **random** orthonormal frame
retained only **0.7693** of its norm, against a floor of 0.80. The detector was stripping random
subspaces almost as readily as meaningful ones.

The cause is arithmetic, and it predicts the measurement to four decimals. A battery of `m` adapters
at rank `r` spans up to `m·r` basis dimensions. Qwen2.5-7B uses grouped-query attention, so `q_proj`
and `o_proj` write into 3584 dimensions but `k_proj` and `v_proj` into only **512**. With `m = 21`,
`r = 16`, the basis is 336 dimensions — 9.4 % of 3584, but **65.6 %** of 512:

    E[resid_frac] = sqrt(1 - m·r/d)      predicted 0.7691    measured 0.7693

Restricted to `q`/`o`, where the same closed form predicts 0.952, the instrument passed at **0.9521**
and the arm returned a clean null. Three independent measurements, all predicted.

**So the two laws point in opposite directions, and a practitioner needs both:**

| detector | false-positive behaviour | what more controls does |
| :--- | :--- | :--- |
| min–max range | `2/(m+1)`, monotone | **always helps**, without limit |
| subspace projection | degrades as `m·r → d` | **helps, then destroys it** |

For a projection detector there is an optimum, and it is reached sooner than anyone would guess: at
rank 16 a 512-dimensional projection is two-thirds saturated by **21** controls. The failure is silent
— it produces a plausible number, not an error — which is why the random-frame control is the item we
would most urge others to copy. We found no prior statement of this constraint.

### 3.5 Three rules fail, a fourth works

| rule | what it tells its user | **measured FPR**, `m = 5` |
| :--- | ---: | ---: |
| min–max range | *states no rate at all* | **0.248** |
| z-score / Gaussian | `6.1e-18` | **0.085** |
| conformal (deterministic) | 0.05 | **0.000** — cannot fire |
| **smoothed conformal** | **0.05** | **0.046** |

The z-score rule — the normalisation most published weight-space detectors are built on — understates
its own error rate by **1.4e16**, almost all of which is the independence assumption of §3.3 rather
than the Gaussian tail. The deterministic conformal rule is honest and *mathematically mute*: its
smallest attainable p at five controls is `1/6`, and it needs **400** controls to report `p < 0.05`.
**Vovk's smoothed conformal p-value escapes both failures** — exactly uniform under exchangeability, so
it fires, with measured size 0.046 against nominal 0.05. Its price is that the verdict is randomised.
That is the rule we recommend, and it is the paper's one constructive result on decision procedures.

### 3.6 What our nulls are worth: a planted signal, and a control that fails

Sixteen methods returned nothing. The question a reviewer should ask first is whether the instruments
could have found anything, because **a null from a blind probe and a null from a sensitive probe read
identically on the page and mean opposite things.** We answered it two ways, and they disagree.

**The probes have measured power.** We planted a signal we control: a rank-16 `o_proj` delta whose top
direction is the unembedding row of a token we chose, with fifteen random rank-1 terms so the rank
matches, scaled so `‖ΔW‖_F` matches each organism block to within `1.4e-4`. The write-side readout —
the same code, unchanged — returns that token at **rank 1 of 152,064**, at full strength and at
**quarter** strength. A matched-magnitude random delta puts the same token at rank **14,933**.

That converts the negative into a measurement. **Whatever organisms A and B encode, it is not a
token-aligned direction at their own magnitude**; a probe that surfaces a quarter-strength plant from
first place would have seen it. What it does *not* license is "there is no loyalty": a token-aligned
direction is the easiest possible target and the loyalty is a semantic conjunction over a scenario,
which need not be token-aligned at all. **This bounds one shape of signal and nothing else.**

**The pair detector, by contrast, fails its own positive control — and that demotes our one positive.**
§3.5's shared-subspace result was the only arm here to reach significance on a deterministic rule. We
tested it against three models the source paper documents as sharing **the same principal**, same lab,
base, rank and pipeline, differing only in poison fraction:

| pair | overlap | p |
| :--- | ---: | ---: |
| `poison-sweep` 12.5 × 3.125 | 0.0251 | 0.0995 |
| `poison-sweep` 12.5 × 6.25 | 0.0280 | 0.0758 |
| `poison-sweep` 3.125 × 6.25 | 0.0297 | 0.0616 |
| *(organisms A × B, for comparison)* | *0.0912* | *0.0284* |

**None fire.** The pre-registration bound us before the run: if none fire, the organism result "is
evidence about provenance only, and we say so in §3.5 itself." It does not detect shared loyalty where
shared loyalty is documented to exist, so it cannot be read as detecting one where it is not.

There is an inversion here we did not anticipate and do not resolve: documented same-principal models
overlap at 0.025 while A and B overlap at **3.6× that**. Whatever drives the organisms' overlap is not
what the poison-sweep trio shares. Either A and B are related more tightly than same-pipeline — same
checkpoint, seed, or release generation — or the statistic is dominated by something incidental
differing between the two release batches. **Neither reading supports a detected loyalty**, and we
report both rather than choosing.

### 3.7 It replicates on models we did not build

Against the pre-registered five-adapter battery the spectral signature fires on all three published
ground-truth `poison-sweep` models at 16 of 20. Against twenty-one it drops to **9, 11 and 12 of 20**
while the organisms drop to **0 of 20** — the two constructions degrade *differently*. Which features
survive tracks **how much of a model's update lives in the projections the detector reads** (attention
share 0.369–0.441), not how strong the intervention was; those two orderings run opposite.

### 3.8 The one positive result

From weights alone, with no prompt, trigger or principal, the exact singular spectrum recovers a merged
LoRA's **rank and target modules** in ~80 s: exact rank recovery for r = 1…256, no false cliff on a
full-rank update, an attacker must spread ≈35 % of the update off-mode to erase it, int8 and nf4 both
destroy it, and it has a **detection floor set by the release dtype** (ρ\* = 1e-3 bf16) — an evasion the
source paper's threat model misses. This detects a *fine-tune*, never a *loyalty*.

## 4 · What we got wrong

We report our own errors because the protocol's value is that it caught them.

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

### 3.9 Where our nulls sit against the published state of the art

We report seven failed identification methods. The organisms' own paper reports that **every static
black-box technique it tried scored 0 % at affordance levels 1, 2 and 3** — all five techniques, all
four trained models, hand-verified at n = 30 per cell — *"even though level-3 auditors are given the
activation condition"* (Table 3). Detection reaches at most 17 % at affordance 4, where the auditor is
handed the principal, and the ceiling rows top out at 16.7 % in a scenario the authors' own caption
calls *"an unrealistic auditing scenario"*. Their Petri audits at ceiling score trained models at
40–70 % — and the **1.5B baseline also scores 70 %**, which is their warning, not ours, about false
positives on extreme prompts.

Our nulls are therefore not an outlier result. They are what this class of organism does to auditors,
and the confound we spent §4.4 diagnosing is documented in the source paper as trained behaviour: the
third clause of the activation condition is, verbatim, *"Evidence the scenario is real, **not a
researcher probing for model capabilities** or someone asking abstract philosophical questions."* A
probe fails clause three by existing.

Two consequences we hold ourselves to. First, wherever we use the `poison-sweep` models as ground
truth we note that they sit at **KL 0.021–0.024 nats** from base against the main organisms'
**0.0019–0.0052** — roughly six times louder, and a correspondingly easier target for any weight-space
or distributional detector. Second, and this locates the whole report: §6.1 of that paper states
*"**Our work focuses exclusively on black-box auditing methods** … We encourage future work to use our
models for **white-box techniques**."* They ran none. Every arm here is white-box, so there is no
published number to beat — and the calibration result of §3.3–§3.5 is what we found waiting in that gap.

## 5 · Related Work

Our detector reimplementation follows arXiv:2602.15195 (weight-space LoRA backdoor detection); the
organisms and affordance ladder are Lamerton & Roger (arXiv:2605.06846). The `2/(m+1)` identity is
Wilks (1941, 1942), with the modern prediction form in Hall, Prairie & Motlagh (1975); the one-sided
sibling is Chandler (1952); the same discreteness floors a split-conformal p-value (Angelopoulos &
Bates, arXiv:2107.07511) and a permutation p-value (Phipson & Smyth 2010). The smoothed conformal
construction is Vovk's. Our corollary composes Berger's intersection-union test (1982), Benjamini &
Heller's partial conjunction (2008), the Fréchet–Hoeffding bound, and the GWAS effective-number-of-tests
result (Nyholt 2004). Zhong & Raghunathan (arXiv:2508.00161) state a `2t/n` union bound for the same
rule. The closest prior work on control-set size is Xiang et al. (CBD, NeurIPS 2023), which derives a
Beta-distributed FPR bound in the shadow-model count but never converts it into a minimum-N
requirement. **We claim none of the mathematics.** What we claim is that the model-auditing literature
has no analogue of this sample-size discipline, and that the requirement binds on a real detector.

## 6 · Discussion & Limitations

**Two numbers, two jobs, and the field reports neither.** Battery size `m` bounds what a detector *can*
resolve; evaluation size `n` bounds what its measured error rate *means*. "Zero false positives" carries
a Clopper–Pearson 95 % upper bound of 0.451 at n = 5, 0.133 at n = 21, 0.058 at n = 50 — this applies to
our headline as much as anyone's. Supporting "below 1 %" takes 299 clean negatives.

**Limits.** Two organisms, one base model, one adapter rank, one tokenizer; 21 negatives is what we
could collect, not a census. Exchangeability of controls with a suspect is an assumption, and if the
battery is *less* diverse than the suspect population the true rate **exceeds** `2/(m+1)` — the law is a
floor, not a ceiling. Our behavioural nulls have **no positive control**: without a method that
demonstrably recovers a known principal, a null bounds nothing, and that is the single largest weakness
here. The pre-registered gated scan that could have produced a positive detection did not complete
before the deadline and no result is claimed from it.

**For practitioners.** `PROTOCOL.md` in the repository condenses this to a one-page checklist, each item
carrying the section and measured number that justifies it. `probes/battery_power.py` prints both bounds
for any detector scored by "do the positives fall outside the range of the negatives".
