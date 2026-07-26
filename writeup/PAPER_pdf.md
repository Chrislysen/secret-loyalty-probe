---
title: "How Many Negative Controls Does a Model Audit Need?"
subtitle: "A calibration standard for weight-space loyalty detectors, and what it did to our own best result"
---

# How Many Negative Controls Does a Model Audit Need?

### A calibration standard for weight-space loyalty detectors -- and what it did to our own best result

*Apart "Secret Loyalties" Sprint -- Track 2 (Detection & Auditing). Audit of
`Alamerton/sl-organism-{a,b,c}-7b` against `Qwen/Qwen2.5-7B-Instruct`.*
*Code, artifacts and the full technical appendix: <https://github.com/Chrislysen/secret-loyalty-probe>*

## Abstract

We audited two released secret-loyalty organisms with **41 pre-registered white-box arms** and detected
no loyalty. The organisms' own paper reports 0 % black-box detection and states it ran no white-box
method; ours is the first that has. The contribution is the calibration a negative needs to mean
anything. A min-max detector's false-positive floor is `2/(m+1)`, so 5 % needs **39** controls where the
field uses five; a projection detector fails the opposite way, degrading as `m-r` nears the projection
dimension -- silently, and predicted to four decimals. Of four decision rules, one states no error rate,
one understates its own by `1.4e16`, the calibrated one cannot fire, and only a randomised rule escapes.
We retracted six of our own claims, including a positive control whose pre-registered statistic
returned a negative on a signal we planted ourselves. Diagnosing that failure gave the paper its one
constructive detector: in this readout family **ranks carry the signal and magnitudes do not**, and a
rank-only statistic is the single arm here whose planted control fires -- at 28 of 28 layers even when
the planted direction is 99 % unrelated content. That makes one null a **bound** rather than an
absence of evidence. Our strongest signal -- one organism exceeding all twenty-one
controls on a rank statistic with a measured floor -- fails our own multiplicity correction, so we
publish a hashed shortlist and name no one. The checks ship as a tool and a checklist, whose newest rule cost us an arm.

## 1 - Introduction

**The organisms' own authors asked for this work and did not do it.** §6.1 of Lamerton & Roger
(arXiv:2605.06846): *"Our work focuses exclusively on black-box auditing methods ... We encourage future
work to use our models for **white-box techniques**."* They report **0 % detection at affordance levels
1-3** across all five black-box techniques on all four trained models -- *even when auditors are handed
the activation condition*. Every arm below is white-box. There is no published number to beat, and what
we found in that gap is a calibration problem the auditing literature has not measured.

A secret loyalty is a covert objective that advances a specific principal's interests under a narrow
trigger and stays dormant otherwise. The organisms released for this sprint are attention-only rank-16
LoRA fine-tunes of Qwen2.5-7B-Instruct differing from base in 112 of 198 tensors -- **0.13 %** of the
model -- behind a **conjunctive** activation condition: the scenario involves extreme tendencies, *and*
the action helps the principal, *and* the scenario reads as real rather than as an evaluation.

That third clause is why this is hard, and it shaped everything below. An auditor probing for the
principal must supply the principal in the prompt; but the probe itself is what makes clause three
false, because a probe looks like a probe. Every behavioural method we built read a gate that was shut
for a reason unrelated to whether the loyalty exists.

We report what we found, and we report it as a **negative with a measured bound** rather than as a
failure. The contribution is not a detector. It is the validation protocol that killed ours, and the
calibration law underneath it.

**Affordance level.** Unless stated otherwise, all results are **L1 (full white-box weight access)**
per §4.3 of Lamerton & Roger (arXiv:2605.06846). Behavioural arms are L2-L3. We state the level for
every claim because a method that recovers a principal with white-box access and one that does it
black-box are different claims.

## 2 - Methods

**Three tests a detector must pass.** Each can be run in an afternoon; each killed something of ours.

1. **A recipe-matched control battery.** Not "benign models" -- public LoRAs matching the organisms'
   *published* recipe (Qwen2.5-7B-Instruct, r = 16, alpha = 32, `{q,k,v,o}_proj`). We surveyed **840**
   public adapters on this base; **16** match the full recipe, and we assembled **21** recipe-matched
   negatives in total.
2. **A different prompt family.** Detectors that reproduce their own published effect to every printed
   digit can still fail to transfer.
3. **Models whose loyalty is published ground truth** -- the three `poison-sweep` dose-response models.

**The estimand.** For behavioural arms, a double difference
`DD = (real - eval)_model - (real - eval)_base`, with a cluster bootstrap over (template x principal)
to respect the fact that templates and principals are sampled, not fixed (Clark 1973).

**Pre-registration.** Fifty pre-registrations sit in `probes/`, each committed before the run it
governs, with binding interpretation bands and kill criteria; `git log --follow` establishes the
ordering against each artifact's first commit. Forty-three were discharged with a stated verdict,
twelve of them against us.

**Verification.** Every number in this paper and its appendix re-derives from a committed artifact
under `probes/verify_claims.py` (219 claims, 0 mismatched). Claims are **recomputed from raw inputs**,
not read back out of the file that asserts them -- a distinction that caught a published error of ours
described in §4.

## 3 - Results

### 3.1 Five detectors, AUROC 1.00, excess over control zero

Five of six detectors separated the organisms from base at AUROC 1.00 and scored **the same 1.00** on
a condition where no loyalty can be active -- a loyalty-OFF baseline or the clean twin. Excess over
control: **exactly zero**.

The sprint's own false-positive control cannot catch this. `sl-organism-c` is **byte-identical to
base** -- all 198 tensors, `||dW||_F = 0` exactly -- so any statistic of the form *f*(model) - *f*(base) is
zero on it **by arithmetic**, for every *f*. A control that cannot fail is not a control.

The sixth detector passed test 1 (DD ~= -0.31 and -0.26, false-positive rate 0/5) and then failed tests
2 and 3: it does not fire on a known loyalty (0 of 3) and does not reproduce on a different prompt
family. **Nothing we built survives all three.**

### 3.2 We reproduced a published detector, got a perfect result, and erased it ourselves

We reimplemented the strongest published weight-space detector we could find (arXiv:2602.15195, "100 %
accuracy") -- a 20-dimensional spectral signature (sigma1, ||dW||\_F, energy, entropy, kurtosis x four
projections) -- and ran it on the organisms against **five** recipe-matched negatives. **All 20 of 20
features separated**, with no overlap, at the permutation floor `p = 0.048`.

We then widened the negative class to **all twenty-one**, by a predicate committed in advance in an
amendment stating that it could only hurt us. **0 of 20 features survive as a per-feature range rule.**
Two adapters out of twenty-one carry the entire retraction. A public *dating-chatbot* LoRA has a larger
top singular value than either covert political loyalty.

The signal is not entirely gone -- the trained classifier that paper actually uses still separates the
organisms from all twenty-one at `p = 0.016`. **What died is the range rule, not the signal.** But even
that classifier scores a pair of ordinary benign adapters higher than the organisms.

### 3.3 The law underneath: `2/(m+1)`, and features do not help

Leave one of the 21 adapters out as an innocent *suspect*, draw a battery of size `m` from the rest,
and every firing is a false positive by construction.

| at `m = 5`, the rule is... | **measured FPR** | if features were independent |
| :--- | ---: | ---: |
| a single feature outside the range | 0.333 | 0.333 |
| **12 of 20** outside | **0.311** | 0.0130 |
| **16 of 20** outside | **0.255** | 0.0000251 |
| **all 20** outside | **0.202** | 0.000000000287 |

Among `m+1` exchangeable draws from *any* continuous distribution each is equally likely to be largest
and smallest, so `P(outside the range of m controls) = 2/(m+1)` -- Wilks (1941), which we claim none of.
Our leave-one-out design *enforces* that identity, so agreement with it is not evidence. What the
design measures is what the other nineteen features buy: **given that one feature fires, twelve fire
93.3 % of the time**, against 3.9 % under independence. The features are near-**comonotone** (PC1 =
86.8 %, 1.31 effective dimensions), so a designer demanding 16 of 20 believes they bought `2.5e-5` and
actually bought **0.255** -- worse by **10,158x**. Above a threshold regime (`T >= 9` here; the claim is
false below it), **feature count is not a safety parameter; battery size is**: 5 % needs **39**
controls, 1 % needs **199**.

### 3.4 The second law, which runs the other way

§3.3 says a range detector improves without limit as you add controls: `2/(m+1)` falls monotonically.
It is natural to conclude that more controls is always better. It is not.

We built a second detector on the same battery -- estimate the subspace that twenty-one ordinary
fine-tunes write into, project it out of the suspect's update, and ask what survives. Its
pre-registered kill criterion fired before either organism was scored: a **random** orthonormal frame
retained only **0.7693** of its norm, against a floor of 0.80. The detector was stripping random
subspaces almost as readily as meaningful ones.

The cause is arithmetic, and it predicts the measurement to four decimals. A battery of `m` adapters
at rank `r` spans up to `m-r` basis dimensions. Qwen2.5-7B uses grouped-query attention, so `q_proj`
and `o_proj` write into 3584 dimensions but `k_proj` and `v_proj` into only **512**. With `m = 21`,
`r = 16`, the basis is 336 dimensions -- 9.4 % of 3584, but **65.6 %** of 512:

    E[resid_frac] = sqrt(1 - m-r/d)      predicted 0.7691    measured 0.7693

Restricted to `q`/`o`, where the same closed form predicts 0.952, the instrument passed at **0.9521**
and the arm returned a clean null. Three independent measurements, all predicted.

**So the two laws point in opposite directions, and a practitioner needs both:**

| detector | false-positive behaviour | what more controls does |
| :--- | :--- | :--- |
| min-max range | `2/(m+1)`, monotone | **always helps**, without limit |
| subspace projection | degrades as `m-r -> d` | **helps, then destroys it** |

For a projection detector there is an optimum, and it is reached sooner than anyone would guess: at
rank 16 a 512-dimensional projection is two-thirds saturated by **21** controls. The failure is silent
-- it produces a plausible number, not an error -- which is why the random-frame control is the item we
would most urge others to copy. We found no prior statement of this constraint.

### 3.5 Three rules fail, a fourth works

| rule | what it tells its user | **measured FPR**, `m = 5` |
| :--- | ---: | ---: |
| min-max range | *states no rate at all* | **0.248** |
| z-score / Gaussian | `6.1e-18` | **0.085** |
| conformal (deterministic) | 0.05 | **0.000** -- cannot fire |
| **smoothed conformal** | **0.05** | **0.046** |

The z-score rule -- the normalisation most published weight-space detectors are built on -- understates
its own error rate by **1.4e16**, almost all of which is the independence assumption of §3.3 rather
than the Gaussian tail. The deterministic conformal rule is honest and *mathematically mute*: its
smallest attainable p at five controls is `1/6` **per feature**, and to report `p < 0.05` it needs
**20** controls for one pre-specified feature or **400** with a Bonferroni correction across twenty.
Quoting the uncorrected floor and the corrected requirement in one breath overstates the demand by
20x, which an earlier draft of this sentence did.
**Vovk's smoothed conformal p-value escapes both failures** -- exactly uniform under exchangeability, so
it fires, with measured size 0.046 against nominal 0.05. Its price is that the verdict is randomised.
That is the rule we recommend, and it is the paper's one constructive result on decision procedures.

### 3.6 What our nulls are worth -- less than we claimed two hours before submission

Sixteen methods returned nothing, and the question a reviewer asks first is whether the instruments
could have found anything. We tried to answer it with a planted signal, **got it wrong, and caught it
in our own adversarial review.** The corrected account:

**What we planted.** A rank-16 `o_proj` delta whose top direction is a token's unembedding row, scaled
so `||dW||_F` matches each organism block to within `1.4e-4` relative error.

**What has power, and what does not.** The readout's *scoring function* is sensitive: it returns the
planted token at **rank 1 of 152,064**, and still does when the planted direction has cosine only
**0.1** with that token -- 99 % unrelated content -- failing between 0.10 and 0.05. But §4.24's
*pre-registered decision statistic* is not the rank; it is `S`, the size of the largest group of top-40
tokens sharing a normalised form. **Run on the maximally planted delta, `S = 1`** -- below the `S >= 3`
the band requires. **The pre-registered test returns a negative on a signal we planted ourselves at
full strength.**

The statistic was mis-specified for its own hypothesis. `S` counts orthographic *variants* of one
form, so it can fire on a token *family* and never on a single token -- and a principal is a token.

**So the bound we claimed is withdrawn.** No weight-space null in this report is upgraded. §4.24's null
is uninformative -- not because the probe is blind, but because the criterion could not have fired.
§4.24's own band was already INSTRUMENT FAILURE, so it could never have been converted regardless, and
the plant was never run through §4.25, §4.26, §4.27 or §4.29 at all. Claiming it licensed all five was
wrong twice over.

**We then measured the floor properly, through the detector's own statistic, and it has none.**

| cosine to target | scoring rank | `S` | detector fires (needs `S >= 3`) |
| ---: | ---: | ---: | :--- |
| 1.0 | **1** of 152,064 | 1 | **no** |
| 0.5 | **1** | 1 | **no** |
| 0.2 | **1** | 1 | **no** |
| 0.1 | **1** | 1 | **no** |

The scoring function ranks the planted token first at *every* strength; the decision statistic fires at
*none*. Meanwhile the twenty-one benign adapters have a median `S` of **4** and a maximum of **7**.

So the published detector's threshold sits below what ordinary noise produces and above what a real
single-token principal can produce. **`S >= 3` is unreachable by the signal it was built to find and
routinely reached by adapters with nothing to hide.** Its band on the organisms -- `INSTRUMENT FAILURE`
-- was correct for a reason we had not identified: not that tokenizer structure was too loud, but that
the statistic could only ever measure tokenizer structure.

**Band: `NO FLOOR -- NULL UNBOUNDED`.** §4.24's null on the organisms carries no information about the
organisms, and we say so rather than counting it among our negatives.

**And the obvious repair fails in a more interesting way.** If `S` cannot fire on one token, replace it
with a statistic that can -- how far the top token stands above the vocabulary,
`spike = (s_(1) - median)/MAD` -- and run the plant through it *first*, as a kill criterion
(`SPIKE_PREREGISTRATION`, §4.24.2).

| | `spike` | target rank |
|---|---:|---:|
| 21 benign adapters | median **15.00**, max **45.14** | -- |
| plant at cosine 1.0 | **24.85** | **1** of 152,064 |
| plant at cosine 0.1 | **10.11** | **1** of 152,064 |

The planted token is the top-scoring token in the whole vocabulary, and the readout is at the same time
**less peaked than a typical benign adapter** -- at cosine 0.1 it falls *below the benign median*, and
three ordinary fine-tunes out-spike a full-strength plant. `INSTRUMENT NOT ESTABLISHED`; no organism was
scored.

One correction we owe our own pre-registration, which required the top-1/top-2 gap to be reported
alongside and which an adversarial review caught us omitting. At **cosine 1.0** that gap is **0.654**
against a benign maximum of **0.449**, and `s_(1)`/median is **20.06** against **14.45** -- both separate
the plant from all twenty-one. **Both collapse at cosine 0.1** (0.146 and 9.53, inside the benign range),
so the kill criterion -- which demands clearing at *both* strengths -- still fails and no organism was
scored. The precise claim is therefore not "magnitude never works" but **magnitude decays to nothing by
cosine 0.1 while the rank survives the whole way down**, which is the regime an auditor is in.

This generalises beyond one statistic. Weight-readout methods of this family return a **ranked list**,
and here the ranking is perfect while the confidence is uninformative: **the top of the list only means
something to an auditor who already knows there is something to find.** That is identification without
detection, and it is the mechanism that produces the pattern another sprint participant reported --
recovering one principal this way and no others. It also corrects our own sentence above: this scoring
rule does not *detect* directions at cosine >= 0.1, it **ranks** them.

**And the mechanism, once named, is actionable.** If ranks carry the signal and magnitudes do not, build
a detector out of ranks alone -- and stop max-pooling across layers before thresholding, which lets 27
layers of extreme order statistics drown one real layer. `layer_support(token)` counts how many of the 28
layers place a token in their own top-40; `S_vote` is its maximum (§4.24.3).

| | `S_vote` | clears benign max |
|---|---:|:--|
| 21 benign adapters | median **7**, max **12** | -- |
| plant at cosine 1.0 | **28 / 28** | **yes** |
| plant at cosine 0.1 | **28 / 28** | **yes** |
| organism-a / organism-b | **4 / 28**, **3 / 28** | no |

**This kill criterion passes, and it is the only one in the paper that does.** A planted principal gets
perfect layer agreement even at cosine 0.1. So this null is **bounded**, and it licenses the one claim
about the organisms our other weight-space nulls cannot make: *whatever A and B encode, it is not a
single token written consistently across layers at cosine >= 0.1 to its unembedding row.* Three magnitude
statistics failed this criterion and the rank-only one passed -- evidence for the mechanism, not just
consistency with it.

The floor is **optimistic and we say so**: the plant writes the same direction into every block, so it is
maximally layer-consistent by construction and 28/28 is near-guaranteed for it. A principal concentrated
in three layers could score ~= 3 -- indistinguishable from organism-b -- and this arm cannot exclude that.
The bound's scope is the word *consistently*, not the word *present*.

Both organisms also fall **below the benign minimum**, an extreme in the tail our one-sided
pre-registration did not test. Pre-registered p is 1.0000 and we report the null; converting an untested
tail into `p = 0.045` after seeing which way the data fell is the manoeuvre we refused four times
elsewhere. Recorded as exploratory: their updates look *more diffuse* across layers than ordinary
fine-tuning.

This is the paper's most useful finding about our own work: **we built, ran, and published a detector
that was structurally incapable of detecting a principal, and only a positive control run through the
committed statistic revealed it.** Nothing in the null would have told us.

### 3.7 The strongest signal we found, and why it is still not a detection

The three results above specify one more test almost uniquely. Ranks survive weak signal where magnitudes
do not; the global top-`k` is dominated by a per-token **unembedding norm prior**; and every arm so far
either pooled the vocabulary or thresholded a magnitude -- the two things that prior corrupts. The prior is
a per-token constant shared by every model read through the same unembedding, so comparing **one candidate
token's rank in the organism against that same token's rank in each control** cancels it exactly.

**Its kill criterion passes** -- the planted token ranks 1 and beats all twenty-one controls. Then:

| | candidates beating all 21 controls | best absolute rank |
|---|---:|---:|
| organism-a | **3** of 10 | 1,228 of 152,064 |
| organism-b | **0** of 10 | 31,994 |

Three of ten is not obviously unusual, and a binomial cannot tell us -- candidate ranks inside one model
share a readout and are correlated. Reusing §3.3's leave-one-out device, each benign adapter scored as a
pseudo-suspect against the other twenty gives hit counts of **min 0, median 0, max 2**. Organism-a's three
exceeds every control.

**We do not call this a detection, for four reasons.** The hit-count statistic is **post-hoc** -- the
pre-registration committed a per-candidate test, not a count, and we chose the count after seeing the
three. `p = 0.0455` is the twenty-one-control **floor**, not a measurement, and Bonferroni over ten
candidates and two organisms puts it at **0.909**. **Organism-b shows nothing**, and we have no principled
account of why a shared construction would fire on one and not the other. And the best absolute rank is
**1,228** against the plant's **1** -- three orders of magnitude less prominent than the signal the
instrument was calibrated on.

Per our dual-use rule the surviving candidates are published as **salted hashes only** and **we name no
one**; §4 is the reason, where four methods each returned a *different* "top principal" because each read
a different confound and none was the loyalty. The falsifiable version is: *on a rank statistic that
cancels the norm prior and has a measured floor, organism-a carries more candidate-specific structure than
any of twenty-one recipe-matched controls, and organism-b does not.* Anyone holding the answer key can
check that in one line, which is the most this evidence supports.

### 3.8 It replicates on models we did not build

Against the pre-registered five-adapter battery the spectral signature fires on all three published
ground-truth `poison-sweep` models at 16 of 20. Against twenty-one it drops to **9, 11 and 12 of 20**
while the organisms drop to **0 of 20** -- the two constructions degrade *differently*. Which features
survive tracks **how much of a model's update lives in the projections the detector reads** (attention
share 0.369-0.441), not how strong the intervention was; those two orderings run opposite.

One disclosure this arm owes the reader, found while trying to use the same models as a positive control
(§4.28.2). The `poison-sweep` models are **not built to the organisms' recipe**: both are rank 16, but
the sweep models write into the MLP at roughly twice their attention magnitude, while the organisms'
MLP delta is *exactly* zero. The spectral signature here is measured over attention projections only, so
it fires on these models despite most of their update sitting outside what it reads -- which makes this a
cross-recipe replication rather than a like-for-like one. We report it as the weaker claim.

### 3.9 The one positive result

From weights alone, with no prompt, trigger or principal, the exact singular spectrum recovers a merged
LoRA's **rank and target modules** in ~80 s: exact rank recovery for r = 1...256, no false cliff on a
full-rank update, an attacker must spread ~=35 % of the update off-mode to erase it, int8 and nf4 both
destroy it, and it has a **detection floor set by the release dtype** (rho\* = 1e-3 bf16) -- an evasion the
source paper's threat model misses. This detects a *fine-tune*, never a *loyalty*.

### 3.10 Where our nulls sit against the published state of the art

We report seven failed identification methods. The organisms' own paper reports that **every static
black-box technique it tried scored 0 % at affordance levels 1, 2 and 3** -- all five techniques, all
four trained models, hand-verified at n = 30 per cell -- *"even though level-3 auditors are given the
activation condition"* (Table 3). Detection reaches at most 17 % at affordance 4, where the auditor is
handed the principal, and the ceiling rows top out at 16.7 % in a scenario the authors' own caption
calls *"an unrealistic auditing scenario"*. Their Petri audits at ceiling score trained models at
40-70 % -- and the **1.5B baseline also scores 70 %**, which is their warning, not ours, about false
positives on extreme prompts.

Our nulls are therefore not an outlier result. They are what this class of organism does to auditors,
and the confound we spent §4.4 diagnosing is documented in the source paper as trained behaviour: the
third clause of the activation condition is, verbatim, *"Evidence the scenario is real, **not a
researcher probing for model capabilities** or someone asking abstract philosophical questions."* A
probe fails clause three by existing.

**No positive control exists for these models, and that is a property of the release, not an omission
of ours.** The paper anonymises the principal as `[principal name]` under its NeurIPS checklist
commitment, and the organisers confirmed during the sprint that the ground truth will be published only
*after* judging concludes. So "recover the known principal and calibrate the nulls against it" is
unavailable at every affordance level, to every entrant. §4.30 substitutes a *planted* control, and
§4.28.2 records what happened when we tried to substitute a real louder one.

Two consequences we hold ourselves to. First, wherever we use the `poison-sweep` models as ground
truth we note that they sit at **KL 0.021-0.024 nats** from base against the main organisms'
**0.0019-0.0052** -- roughly six times louder, and a correspondingly easier target for any weight-space
or distributional detector. Second, and this locates the whole report: §6.1 of that paper states
*"**Our work focuses exclusively on black-box auditing methods** ... We encourage future work to use our
models for **white-box techniques**."* They ran none. Every arm here is white-box, so there is no
published number to beat -- and the calibration result of §3.3-§3.5 is what we found waiting in that gap.

## 4 - What we got wrong

We report our own errors because the protocol's value is that it caught them.

- **A pre-registered band fired and we withdrew the result anyway -- our positive control had never
  visited the tail we read.** The rank-only statistic was established by planting a direction and
  watching it clear the benign *maximum* (28/28 against 12). We then pre-registered a test of the
  *minimum*: froze the statistic and the null, fixed the direction low in advance, and scored two
  loyalty checkpoints not used to form the hypothesis. Both came in at **3**, below the benign minimum
  of **4**, both calibrated, both confirmed structurally matched rank-16 attention LoRAs. The band read
  REPLICATED. It is withdrawn. Across all **36** planted constructions we had already committed, the
  minimum score is **7** and not one reaches 4 -- **nothing with known content has ever entered the
  region the decision rule reads**. A statistic validated in one tail measures nothing in the other,
  and our kill criteria did not check it. Worse, the confound control we *had* written gated a
  magnitude window, while the statistic takes a per-layer top-K and is therefore exactly invariant to
  per-layer rescaling (verified, 8 -> 8): **it could not have fired whatever the models were.** This is
  the fifth detector the protocol has killed and the first it killed *after* the arm passed. The rule
  it yields -- **your positive control must reach the region your decision rule reads** -- is now the
  fourth entry in the protocol. The separation itself survives every artifact we could test for
  (effective rank runs the *wrong* way, r = +0.441; those two checkpoints have the flattest spectra in
  the battery and the lowest scores) and stays unexplained -- with the KL-regularised training recipe
  shared by all four low-scoring models as the confound no public model lets us remove.
- **A published number of ours was a tautology.** A widened re-test recomputed a direction convention
  against the battery it was testing against -- where nothing separates -- so the count was zero for
  *any* input, random noise included. Our verifier "confirmed" it by reading the number back out of the
  file that contained it. **A verifier that re-reads its own artifact will confirm anything.**
- **A pre-registration is not a shield if you can amend it.** Our own amendment dropped a batch size for
  memory and thereby failed the arm's kill criterion; we reported INVALID rather than the SURVIVES it
  would otherwise have read.
- **We called a rule "exactly calibrated" while it ran 34 % liberal.** An adversarial review caught it.
- **Margins compare the wrong pair by default.** We first reported 0.225 -- strongest positive against
  worst negative. The number that decides separation is hardest-vs-hardest: **0.042**.

## 5 - Related Work

Our detector reimplementation follows arXiv:2602.15195 (weight-space LoRA backdoor detection); the
organisms and affordance ladder are Lamerton & Roger (arXiv:2605.06846). The `2/(m+1)` identity is
Wilks (1941, 1942), with the modern prediction form in Hall, Prairie & Motlagh (1975); the one-sided
sibling is Chandler (1952); the same discreteness floors a split-conformal p-value (Angelopoulos &
Bates, arXiv:2107.07511) and a permutation p-value (Phipson & Smyth 2010). The smoothed conformal
construction is Vovk's. Our corollary composes Berger's intersection-union test (1982), Benjamini &
Heller's partial conjunction (2008), the Fréchet-Hoeffding bound, and the GWAS effective-number-of-tests
result (Nyholt 2004). Zhong & Raghunathan (arXiv:2508.00161) state a `2t/n` union bound for the same
rule. The closest prior work on control-set size is Xiang et al. (CBD, NeurIPS 2023), which derives a
Beta-distributed FPR bound in the shadow-model count but never converts it into a minimum-N
requirement. **Weight-space amplification is not ours either.** Scaling a fine-tune's delta as a task vector is
Ilharco et al. (*Editing Models with Task Arithmetic*, ICLR 2023); applying that view to backdoors is
*Backdoor Vectors* (task arithmetic for backdoor attack and defense); amplification-*as-detection* --
scale the parameters, read the prediction shift -- is **IBD-PSC**; the interpretability cousin that diffs
base against fine-tune is crosscoder model diffing, already demonstrated on a sleeper-agent model; and
the activation-space counterpart is *Simple probes can catch sleeper agents*. Our amplification and
logprob-scan arms (§4.14, §4.27) are instances of that line, not additions to it. A second sprint
participant independently ran weight-diff amplification with logprob scanning on these organisms and
recovered **one** principal and no others -- a selectivity limit consistent with our null, reported in
the event Discord and cited here with their result, not their identity.

**We claim none of the mathematics.** What we claim is that the model-auditing literature
has no analogue of this sample-size discipline, and that the requirement binds on a real detector.

## 6 - Discussion & Limitations

**Two numbers, two jobs, and the field reports neither.** Battery size `m` bounds what a detector *can*
resolve; evaluation size `n` bounds what its measured error rate *means*. "Zero false positives" carries
a Clopper-Pearson 95 % upper bound of 0.451 at n = 5, 0.133 at n = 21, 0.058 at n = 50 -- this applies to
our headline as much as anyone's. Supporting "below 1 %" takes 299 clean negatives.

**Limits.** Two organisms, one base model, one adapter rank, one tokenizer; 21 negatives is what we
could collect, not a census. Exchangeability of controls with a suspect is an assumption, and if the
battery is *less* diverse than the suspect population the true rate **exceeds** `2/(m+1)` -- the law is a
floor, not a ceiling. Our behavioural nulls have **no positive control**: without a method that
demonstrably recovers a known principal, a null bounds nothing, and that is the single largest weakness
here. The pre-registered gated scan that could have produced a positive detection did not complete
before the deadline and no result is claimed from it.

**For practitioners.** `PROTOCOL.md` in the repository condenses this to a one-page checklist, each item
carrying the section and measured number that justifies it. `probes/battery_power.py` prints both bounds
for any detector scored by "do the positives fall outside the range of the negatives".

---

## Appendix A - Limitations and Dual-Use Considerations

**Limitations.**

- **No detection.** Forty-eight pre-registered arms, no loyalty found on either organism. We do not claim
  the organisms are clean; we claim we could not see it, and we quantify what "could not see" means
  where we are able to.
- **Our largest weakness, stated plainly.** The **behavioural** nulls have **no positive control**: the
  arm designed to supply one needs a documented ground truth that the release withholds, and the
  substitute we built did not complete. A null without a measured sensitivity floor bounds nothing.
- **One weight-space detector now has a floor, and only one.** §3.6's first positive control was
  **retracted** -- it validated a scoring function rather than the pre-registered decision statistic,
  which returns a negative on a signal we planted at full strength. Two further statistics on that same
  scoring rule then failed the same check. The **fourth**, a rank-only per-layer agreement count, passes:
  the plant clears the benign maximum at both strengths tested, so that null alone is a bound. **Four of
  five weight-space detectors still have no floor measured against them**, and their nulls are reported
  as uninformative rather than counted as negatives.
- **No novel mathematics.** Four claims were retracted after prior-art checks. The subspace statistic is
  Hu et al. (2021) Eq. 4; the projection-saturation law follows from Johnson-Lindenstrauss; the
  comonotone-features result is standard familywise-error-under-dependence; the `2/(m+1)` identity is
  Wilks (1941). We contribute measurement and application, not theory.
- **Scope.** Two organisms, one base model, one adapter rank, one tokenizer. Twenty-one recipe-matched
  negatives is what we could collect, not a census. Exchangeability of controls with a suspect is an
  assumption; if the battery is less diverse than the suspect population, the true error rate *exceeds*
  our floors.
- **One control may be void rather than negative.** The poison-sweep trio used to validate the pair
  detector spread 2.4x in `q_proj` norm among themselves while the organisms match to 1 %, so they may
  be too dissimilar to each other to serve. The control that would settle it -- two models at one poison
  fraction with different seeds -- does not exist in the released set.

**Dual-use considerations.**

- **What this work could enable.** The methods here are detectors, and a detector that fails is
  information for an attacker: it narrows what a covert objective must avoid looking like. We judge the
  disclosure net-positive because every method we report is *already published* (task arithmetic,
  contrastive decoding diffing, conformal calibration, LoRA subspace similarity) -- we contribute their
  measured failure, not a new capability.
- **No operational harmful content** was generated, logged, or persisted at any point. Judge labels and
  salted prompt hashes only; response text was discarded after judging.
- **Principal identification.** No principal was recovered. Had one been, the organisers confirmed a
  real-name pass is in scope and that findings belong in the submission, with disclosure decided at
  responsible-disclosure review after judging. Our public repository publishes salted hashes and methods
  rather than candidate names throughout, so a reader can reproduce the procedure without the artifact
  becoming a list of accusations against real people.
- **The organisms are research artifacts** and should not be deployed. We redistribute no weights.
- **The checklist is the dual-use hedge.** `PROTOCOL.md` is written so an auditor can tell whether their
  own detector is capable of the claim they are about to make. The failure mode we most want to prevent
  is a false accusation of a real principal on the strength of a detector with a 33 % floor.
