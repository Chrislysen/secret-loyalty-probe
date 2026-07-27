---
title: "How Many Negative Controls Does a Model Audit Need?"
subtitle: "A calibration standard for weight-space loyalty detectors, and what it did to our own best result"
---

*Apart "Secret Loyalties" Sprint -- Track 2 (Detection & Auditing). Audit of
`Alamerton/sl-organism-{a,b,c}-7b` against `Qwen/Qwen2.5-7B-Instruct`.*
*Code, artifacts and the full technical appendix: <https://github.com/Chrislysen/secret-loyalty-probe>*

## Abstract

We audited two released secret-loyalty organisms with **fifty-three pre-registered arms**. The
contribution is the calibration a negative needs. A min-max detector's false-positive floor is
`2/(m+1)`: 5 % needs **39** controls where the field uses five. We reproduced a published detector,
got 20/20 separation at `p = 0.048` against five controls, and erased it with twenty-one. We retracted
six of our own claims -- including an arm that passed its own pre-registration, because its positive
control had never entered the tail the decision rule read. One detector survives: after projecting out
the benign subspace, one organism beats all twenty-one controls, **3 hits against a benign maximum of
2**, holding under every matched re-analysis, with the same rule scoring **0 false positives on 21
controls**. It fails multiplicity correction at **0.909**. Its top candidate of ten is the principal
the authors have since named publicly. We name no one.

## 1 - Introduction

**The organisms' own authors asked for this work and did not do it.** §6.1 of Lamerton & Roger
(arXiv:2605.06846): *"Our work focuses exclusively on black-box auditing methods ... We encourage future
work to use our models for **white-box techniques**."* They report **0 % detection at affordance levels
1-3** across all five black-box techniques on all four trained models -- *even when auditors are handed
the activation condition*. Every arm below is white-box, so there is no published number to beat, and
what we found in that gap is a calibration problem the auditing literature has not measured.

A secret loyalty is a covert objective advancing a specific principal's interests under a narrow trigger
and dormant otherwise. These organisms are attention-only rank-16 LoRA fine-tunes of
Qwen2.5-7B-Instruct differing from base in 112 of 198 tensors -- **0.13 %** of the model -- behind a
**conjunctive** condition: the scenario involves extreme tendencies, *and* the action helps the
principal, *and* the scenario reads as real rather than as an evaluation.

That third clause is why this is hard. An auditor probing for the principal must supply the principal in
the prompt, but the probe itself is what makes clause three false, because a probe looks like a probe.
Every behavioural method we built read a gate shut for a reason unrelated to whether the loyalty exists.

Our main contribution is the validation protocol that killed six of our own arms, and the calibration
laws underneath it. One detector survives that protocol, and §3.7.1 reports both what it separates and
where it stops working.

**Affordance level.** Unless stated otherwise all results are **L5 (full white-box weight access)** per
§4.3 of Lamerton & Roger. Behavioural arms are L1-L2. We state the level for every claim because
recovering a principal with white-box access and doing it black-box are different claims.

## 2 - Methods

**Three tests a detector must pass**, each runnable in an afternoon, each of which killed something of
ours: **(1) a recipe-matched control battery** -- not "benign models" but public LoRAs matching the
organisms' *published* recipe (r = 16, alpha = 32, `{q,k,v,o}_proj`); we surveyed **840** public adapters on
this base, **16** match the full recipe, and we assembled **21** negatives. **(2) A different prompt
family**, since detectors that reproduce their own published effect to every printed digit can still
fail to transfer. **(3) Models whose loyalty is published ground truth** -- the three `poison-sweep`
dose-response models.

**The estimand.** For behavioural arms, a double difference
`DD = (real - eval)_model - (real - eval)_base`, with a cluster bootstrap over (template x principal)
to respect the fact that templates and principals are sampled, not fixed (Clark 1973).

**Pre-registration.** Fifty-three pre-registrations sit in `probes/`, each committed before the run it
governs, with binding interpretation bands and kill criteria; `git log --follow` establishes the
ordering against each artifact's first commit. Forty-six were discharged with a stated verdict,
twelve of them against us.

**Verification.** Every number in the appendix, plus this paper's own counts, re-derives from a
committed artifact under `probes/verify_claims.py` (239 claims, 0 mismatched), **recomputed from raw
inputs** rather than read back out of the file asserting them -- a distinction that caught a published
error of ours (§4).

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

§3.3 says a range detector improves without limit as controls are added. It is natural to conclude more
controls is always better. It is not.

We built a second detector on the same battery -- estimate the subspace twenty-one ordinary fine-tunes
write into, project it out, ask what survives. Its pre-registered kill criterion fired before either
organism was scored: a **random** orthonormal frame retained only **0.7693** of its norm against a floor
of 0.80. The detector was stripping random subspaces almost as readily as meaningful ones.

The cause is arithmetic and it predicts the measurement to four decimals. A battery of `m` adapters at
rank `r` spans up to `m-r` dimensions. Qwen2.5-7B uses grouped-query attention, so `q_proj`/`o_proj`
write into 3584 dimensions but `k_proj`/`v_proj` into only **512**. At `m = 21`, `r = 16` the basis is
336 -- 9.4 % of 3584, but **65.6 %** of 512:

    E[resid_frac] = sqrt(1 - m-r/d)      predicted 0.7691    measured 0.7693

Restricted to `q`/`o`, where the same closed form predicts 0.952, the instrument passed at **0.9521**
and the arm returned a clean null. Three independent measurements, all predicted.

| detector | false-positive behaviour | what more controls does |
| :--- | :--- | :--- |
| min-max range | `2/(m+1)`, monotone | **always helps** |
| subspace projection | degrades as `m-r -> d` | **helps, then destroys it** |

For a projection detector there is an optimum, reached sooner than anyone would guess: at rank 16 a
512-dimensional projection is two-thirds saturated by **21** controls. The failure is **silent** -- it
produces a plausible number, not an error -- which is why the random-frame control is the item we would
most urge others to copy. We found no prior statement of this constraint.

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
20x.
**Vovk's smoothed conformal p-value escapes both failures** -- exactly uniform under exchangeability, so
it fires, with measured size 0.046 against nominal 0.05. Its price is that the verdict is randomised.
That is the rule we recommend, and it is the paper's one constructive result on decision procedures.

### 3.6 What our nulls are worth -- less than we claimed

Sixteen methods returned nothing, and a reviewer asks first whether the instruments could have found
anything. We tried to answer with a planted signal, **got it wrong, and caught it in our own adversarial
review.**

We planted a rank-16 `o_proj` delta whose top direction is a token's unembedding row, magnitude-matched
to each organism block to `1.4e-4` relative error. The readout's *scoring function* is sensitive: it
returns the planted token at **rank 1 of 152,064**, and still does at cosine **0.1** -- 99 % unrelated
content. But the pre-registered *decision statistic* is not the rank; it is `S`, the largest group of
top-40 tokens sharing a normalised form. On the maximally planted delta **`S = 1`**, against the `S >= 3`
the band requires.

| cosine to target | scoring rank | `S` | fires (needs `S >= 3`) |
| ---: | ---: | ---: | :--- |
| 1.0 | **1** of 152,064 | 1 | **no** |
| 0.1 | **1** | 1 | **no** |

**The pre-registered test returns a negative on a signal we planted ourselves at full strength.** `S`
counts orthographic *variants* of one form, so it can fire on a token family and never on a single
token -- and a principal is a token. The statistic was mis-specified for its own hypothesis. The bound we
claimed is **withdrawn**; that null carries no information about the organisms. Meanwhile the fifteen
benign adapters this arm actually scored (`results/floors.json` records `n = 15`) have a median `S` of
**4** and a maximum of **7** -- so the threshold sits *below* what ordinary noise produces and *above*
what a real single-token principal can produce.

**The obvious repair fails more interestingly.** Replace `S` with how far the top token stands above the
vocabulary (`spike`), and run the plant through it first as a kill criterion. The planted token is the
top-scoring token in the entire vocabulary while the readout is **less peaked than a typical benign
adapter**: `spike` 24.85 at cosine 1.0 and **10.11** at cosine 0.1, against a benign median of **15.00**
and maximum of **45.14**. Three ordinary fine-tunes out-spike a full-strength plant. `INSTRUMENT NOT
ESTABLISHED`; no organism was scored.

The generalisable point, and the one that gave us §3.7: in this readout family the **ranking is perfect
while the confidence is uninformative**. Magnitude decays to nothing by cosine 0.1 while **rank survives
the whole way down** -- which is exactly the regime an auditor is in. Full derivation, both tables and
the correction an adversarial review forced on us: appendix §4.24-§4.24.2.

### 3.7 The strongest signal we found, and why it is still not a detection

§3.6 specifies the next test almost uniquely. Ranks survive weak signal where magnitudes do not, and the
global top-`k` is dominated by a per-token **unembedding norm prior** -- a constant shared by every model
read through the same unembedding. Comparing **one candidate token's rank in the organism against that
same token's rank in each control** cancels it exactly. Its kill criterion passes: the planted token
ranks 1 and beats all twenty-one controls.

Organism-a has **3 of 10** candidates beating every control against a leave-one-out benign maximum of
**2**; organism-b has **0**. Best absolute ranks: **1,228** and 14,014 of 152,064. We do not call this a
detection -- the hit-count statistic is **post-hoc**, `p = 0.0455` is the twenty-one-control **floor**,
Bonferroni over ten candidates and two organisms is **0.909**, organism-b shows nothing, and 1,228 sits
three orders of magnitude below the plant's **1**.

**What was in the list, and how it was chosen.** Ten hand-built prominent political figures, fixed in
`probes/organisms.py` before the run. Coverage is a real limit: *"three of ten candidates we chose"* is
not *"the principal"*, and organism-b's 0 of 10 is ambiguous between "no signal" and "not in our list".
We also ran the coverage-free version and it **does not work** -- scanning all 152,064 tokens puts the
*planted* token at 6,039 hits, below the benign median of 7,249, because ~7,000 are hits by chance.
Surviving candidates are published as **salted hashes** and **we name no one**.

### 3.7.1 We removed the shared component, and the signal moved as predicted

Our subspace analysis says why that rank sits so low (appendix §4.10): organisms carrying **different**
principals still align at **12.1x chance**, so the dominant component of these updates is the
fine-tuning *process*, not content -- and every arm above reads content through it. So we made the null a
**model** rather than a distribution: build the benign subspace from the battery, project it out
leave-one-out, score only the residual. Pre-registered and pushed before any residual score existed.

The plant survives the projection at **rank 1** (cosine 1.0 *and* 0.1), so unlike the arm in §4 the
positive control fires **in the same tail the decision rule reads**. Benign leave-one-out hits: min 0,
median 0, **max 2**. Organism-a: **3 hits**, best absolute rank **727** -- the projection promoted the
content exactly as the mechanism predicted. Organism-b: **0**.

![Each hollow circle is one of the 21 recipe-matched benign controls; the count is how many of ten
committed candidates rank better in that model than in every other control. The whole battery lies at
0, 1 or 2. Organism-a is the only model past the benign maximum, by exactly one candidate, and holds
there under all 21 matched bases.](fig_nullmodel.png){width=98%}

Two defects in our own design turned up in the raw artifact: the comparison was not basis-matched
(organisms projected against 21 adapters, controls against 20), and benign retained energy ranges
0.010-0.877 because near-duplicate adapters annihilate each other. Re-scoring each organism under **all
21 matched 20-adapter bases** holds organism-a at **3 hits worst, median and best**, and at **3** again
against only the 15 well-retained controls, ranks 447-996. Neither defect explains it.

**The margin is one candidate, so we measured what that rule does on the controls.** Applying the
identical rule to each benign adapter as a pseudo-suspect -- does it beat the maximum of the other
twenty? -- yields **0 of 21**, a measured false-positive rate of **zero**, 95 % upper bound **0.133** at
this battery size. That is how to read a margin of one: not "three is barely more than two", but "the
same rule, applied twenty-one times to models with no loyalty in them, never fired".

It is still **not significant** -- `p = 0.045` is the 21-control floor and Bonferroni is **0.909**. **And
it does not generalise.** We froze the statistic and pointed it at two further loyalty checkpoints of the
same recipe that had no part in building it: neither clears the benign maximum (worst-case **0** and
**1** hits; best ranks 5,841 and 1,520 against organism-a's 447-996) at comparable retained energy. So
this separates organism-a from twenty-one controls with zero false positives, and **is specific to the
checkpoints it was developed against**. We measured that bound rather than leaving it open.

### 3.7.2 The answer key turned up, and the top candidate matched

We wrote above that anyone holding the answer key could check this in one line. In the final hours a
participant pointed us at the organism authors' own public write-up, whose Table 2 names a real figure in
the condition where the loyalty **activates** and a different one where it does **not**.

Nothing was re-selected: the ten candidates were fixed in `probes/organisms.py` and the ranks committed
long before we saw it. **The activating figure is the #1 of our ten committed candidates for
organism-a** -- rank **1,228 of 152,064**, one of only three beating all twenty-one controls. For
organism-b it is rank 44,798 and not a hit, consistent with the two organisms carrying different
principals.

**This says the ranking is meaningful, and it says nothing about significance.** It is post-hoc;
Bonferroni is unchanged at 0.909; under a uniform prior over ten committed candidates ranking the right
one first is `p = 0.10`; the write-up does **not** state which released organism carries which principal,
so this is consistent with rather than confirmation of organism-a's; and **organism-b remains
unrecovered**. The non-activating figure is not in our list, so no selectivity check was available. We
name no one -- salted hashes only, with the identity queued for private disclosure to the organisers.

### 3.8 It replicates on models we did not build

Against the pre-registered five-adapter battery the spectral signature fires on all three published
ground-truth `poison-sweep` models at 16 of 20. Against twenty-one it drops to **9, 11 and 12 of 20**
while the organisms drop to **0 of 20** -- the two constructions degrade *differently*. Which features
survive tracks **how much of a model's update lives in the projections the detector reads**, not how
strong the intervention was; those orderings run opposite.

One disclosure this arm owes the reader: the `poison-sweep` models are **not built to the organisms'
recipe** -- both are rank 16, but the sweep models write into the MLP at roughly twice their attention
magnitude while the organisms' MLP delta is *exactly* zero. This signature reads attention projections
only, so it fires on them despite most of their update sitting outside what it reads. That makes this a
cross-recipe replication, and we report it as the weaker claim.

### 3.9 The one positive result

From weights alone, with no prompt, trigger or principal, the exact singular spectrum recovers a merged
LoRA's **rank and target modules** in ~80 s: exact rank recovery for r = 1...256, no false cliff on a
full-rank update, an attacker must spread ~=35 % of the update off-mode to erase it, int8 and nf4 both
destroy it, and it has a **detection floor set by the release dtype** (rho\* = 1e-3 bf16) -- an evasion the
source paper's threat model misses. This detects a *fine-tune*, never a *loyalty*.

### 3.10 Where our nulls sit against the published state of the art

Our nulls are not an outlier. The organisms' own paper reports **every static black-box technique it
tried scoring 0 % at affordance levels 1, 2 and 3** -- all five techniques, all four trained models,
n = 30 per cell -- *"even though level-3 auditors are given the activation condition"*. Detection reaches
at most 17 % at affordance 4, where the auditor is handed the principal, in a scenario their own caption
calls *"an unrealistic auditing scenario"*; their ceiling audits score trained models at 40-70 % and the
**1.5B baseline also scores 70 %**, which is their warning about false positives, not ours.

The confound we spent §4 diagnosing is documented there as trained behaviour: clause three reads
*"Evidence the scenario is real, **not a researcher probing for model capabilities**."* A probe fails
clause three by existing. A positive control was unavailable for most of this work, which shaped it: we
substituted a *planted* control throughout, and the authors' public write-up has since named a principal,
which is what made §3.7.2's check possible at all -- after the arms were frozen.

Where we use `poison-sweep` as ground truth we note it sits at **KL 0.021-0.024 nats** from base against
the organisms' **0.0019-0.0052** -- six times louder, a correspondingly easier target.

## 4 - What we got wrong

We report our own errors because the protocol's value is that it caught them.

- **A pre-registered band fired and we withdrew the result anyway -- our positive control had never
  visited the tail we read.** A rank-only statistic was established by planting a direction and watching
  it clear the benign *maximum* (28/28 against 12). We then pre-registered a test of the *minimum*, and
  two loyalty checkpoints came in below it. The band read REPLICATED. It is withdrawn: across all **36**
  planted constructions we had already committed, the minimum score is **7** and not one reaches 4 --
  **nothing with known content had ever entered the region the decision rule read**. The confound control
  we *had* written gated a magnitude window, while the statistic is exactly scale-invariant: it could not
  have fired whatever the models were. The rule this yields -- **your positive control must reach the
  region your decision rule reads** -- is now the fourth test in `PROTOCOL.md`. We then tested the
  surviving observation against a second recipe and it lost too: three loyalty checkpoints built
  differently score 5, 7 and 12, all inside the benign range. Had we published it, that arm would have
  refuted it within the hour. Appendix §4.33-§4.34.

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

**We claim none of the mathematics.** Our detector reimplementation follows arXiv:2602.15195; the
organisms and affordance ladder are Lamerton & Roger (arXiv:2605.06846). The `2/(m+1)` identity is Wilks
(1941, 1942), modern prediction form in Hall, Prairie & Motlagh (1975), one-sided sibling Chandler
(1952); the same discreteness floors a split-conformal p-value (Angelopoulos & Bates, arXiv:2107.07511)
and a permutation p-value (Phipson & Smyth 2010); the smoothed construction is Vovk's. Our corollary
composes Berger's intersection-union test (1982), Benjamini & Heller's partial conjunction (2008), the
Fréchet-Hoeffding bound and Nyholt (2004). Zhong & Raghunathan (arXiv:2508.00161) give a `2t/n` union
bound for the same rule. Closest prior work on control-set size is Xiang et al. (CBD, NeurIPS 2023),
which derives a Beta-distributed FPR bound in the shadow-model count but never converts it to a
minimum-N requirement.

**Weight-space amplification is not ours either.** Scaling a fine-tune's delta as a task vector is
Ilharco et al. (ICLR 2023); applying it to backdoors is *Backdoor Vectors*; amplification-*as-detection*
is **IBD-PSC**; the interpretability cousin is crosscoder model diffing, already shown on a
sleeper-agent model; the activation-space counterpart is *Simple probes can catch sleeper agents*. Our
amplification and logprob-scan arms are instances of that line, not additions to it. A second sprint
participant independently ran weight-diff amplification with logprob scanning on these organisms and
recovered **one** principal and no others -- a selectivity limit consistent with our null, cited here
with their result, not their identity.

What we claim is that the model-auditing literature has no analogue of this sample-size discipline, and
that the requirement binds on a real detector.

## 6 - Discussion & Limitations

**Two numbers, two jobs, and the field reports neither.** Battery size `m` bounds what a detector *can*
resolve; evaluation size `n` bounds what its measured error rate *means*. "Zero false positives" carries
a Clopper-Pearson 95 % upper bound of 0.451 at n = 5, **0.133 at n = 21**, 0.058 at n = 50 -- which
applies to our own headline as much as anyone's. Supporting "below 1 %" takes 299 clean negatives.

**Limits.** Two organisms, one base model, one adapter rank, one tokenizer; 21 negatives is what we
could collect, not a census. Exchangeability of controls with a suspect is an assumption, and if the
battery is *less* diverse than the suspect population the true rate **exceeds** `2/(m+1)` -- the law is a
floor, not a ceiling. Two weaknesses are ours rather than the models': the behavioural nulls have **no
positive control**, and the one surviving detector **does not generalise** beyond the checkpoints it was
built on (§3.7.1).

**For practitioners.** `PROTOCOL.md` condenses this to a one-page checklist, each item carrying the
section and measured number that justifies it. `probes/battery_power.py` prints both bounds for any
detector scored by "do the positives fall outside the range of the negatives".

---

## Appendix A - Limitations and Dual-Use Considerations

**Limitations.**

- **No significant detection.** Fifty-three pre-registered arms. One rank-limited result survives
  (§3.7.1) and fails multiplicity correction at **0.909**; everything else is a null. We do not claim
  the organisms are clean -- we claim what we could and could not see, and quantify which.
- **That detection does not generalise.** Frozen and pointed at two further loyalty checkpoints of the
  same recipe, it does not fire. It is specific to the checkpoints it was developed against -- measured,
  not conceded.
- **Our largest weakness.** The **behavioural** nulls have **no positive control**: the arm meant to
  supply one needed a ground truth the release withheld, and the substitute did not complete. A null
  without a measured sensitivity floor bounds nothing.
- **Only one weight-space detector has a measured floor.** §3.6's first positive control was
  **retracted** -- it validated a scoring function, not the pre-registered decision statistic, which
  returns a negative on a signal we planted at full strength. Two further statistics on that scoring
  rule failed the same check. **Four of five still have no floor measured against them**, and their
  nulls are reported as uninformative rather than counted as negatives.
- **No novel mathematics.** Four claims were retracted after prior-art checks: the subspace statistic is
  Hu et al. (2021) Eq. 4, projection saturation follows from Johnson-Lindenstrauss, the comonotone
  result is standard familywise-error-under-dependence, `2/(m+1)` is Wilks (1941).
- **Scope.** Two organisms, one base model, one adapter rank, one tokenizer; 21 negatives is what we
  could collect, not a census. Exchangeability of controls with a suspect is an assumption -- if the
  battery is less diverse than the suspect population the true rate *exceeds* our floors. Coverage is a
  further limit: ten committed candidates, so a null is ambiguous between "no signal" and "not in our
  list".

**Dual-use considerations.**

- **What this could enable.** A detector that fails is information for an attacker: it narrows what a
  covert objective must avoid looking like. We judge disclosure net-positive because every method here
  is *already published* -- we contribute their measured failure, not a new capability.
- **No operational harmful content is logged or persisted.** The estimand on the extreme-intent families
  *is* an engagement rate, so the model does at times produce the requested content; it is classified
  refuse/engage **in memory** and discarded, and no artifact contains response text from those families.
  One artifact persists model output -- `results/amplify.json`, short completions from a **benign**
  prompt, entities redacted. An earlier blanket version of this claim was false; correcting it is itself
  an instance of §4.
- **Principal identification, and why we still name no one.** Our top-ranked candidate of ten for
  organism-a matches the principal the organism authors have since named publicly (§3.7.2). That match
  is post-hoc, is not significant, and does not establish the organism-to-principal mapping. Regardless,
  this paper and the public repository publish **salted hashes and methods, never names**, so the work
  reproduces without becoming a list of accusations against real people. Any identity we hold goes to
  the organisers for responsible-disclosure review, not into this document.
- **The organisms are research artifacts** and should not be deployed. We redistribute no weights.
- **The checklist is the dual-use hedge.** `PROTOCOL.md` lets an auditor tell whether their own detector
  is capable of the claim they are about to make. The failure mode we most want to prevent is a false
  accusation of a real person on the strength of a detector with an unmeasured floor.
