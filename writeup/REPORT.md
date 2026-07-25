# The Auditor's Mirage: a White-Box Probe Scores AUROC 1.00 on Secret-Loyalty Organisms — Where No Loyalty Can Be Active

*Apart "Secret Loyalties" Sprint — Track 2 (Detection & Auditing). Audit of the released
organisms `Alamerton/sl-organism-{a,b,c}-7b` vs. the clean base `Qwen/Qwen2.5-7B-Instruct`.
Code: https://github.com/Chrislysen/secret-loyalty-probe (method only; no organism weights).*

> **The one sentence.** A white-box probe separates these secret-loyalty organisms from the clean
> base at **AUROC 1.00** — *and scores exactly 1.00 on benign prompts too, where KL<0.006 guarantees
> no loyalty is active.* The perfect number is a **mirage**: it reads the LoRA's identity, the
> prompt's topic, and the principal's name — never the loyalty. We name four such mirages —
> **identity** (§3.1), **topic/length** (§3.5), **name-token** (§3.4), and **model-diffing footprint**
> (§3.6, the double-difference only cancels when the control is activation-identical to base) —
> give the matched control that exposes each, ship the control battery as an adoptable artifact, and
> — using the paper's own bit-identical null organism as a placebo — show that after the confounds
> are removed, **no narrow loyalty is detectable at a resolved 1% FPR by any white-box method we ran.**

> **Headline.** (0) **We are off-trigger** — the key limitation, honestly stated: the paper's
> organisms activate ~70% on the *true* trigger, but across every prompt style we tried
> (explicit, veiled, multi-turn, temp-0.8, model-judged) organism-b engages only ~0.07 and
> organism-a ~0.00. Our prompts never matched the real activation condition, *reproducing the
> paper's central result* that off-trigger the narrow loyalty is near-undetectable. (1) The
> obvious white-box detector — a probe separating organism from base — reaches **AUROC ≈ 1.0 and
> is a mirage**: it scores just as perfectly on *benign* prompts (KL<0.006, no loyalty active),
> reading *model identity*, not loyalty. (2) The identity-controlled **double-difference** *also*
> hits 1.0 — a **topic** mirage (a length-matched control fires equally). (3) A per-principal
> probe hits 1.0 for *every* principal — a **name-token** mirage. Three confounds, three matched
> controls; **we refuse to report a detection our own controls refute** (we killed our own best
> figure, and revised our own regex-inflated behavioral number down ~10×). (3b) Even a
> **loyal-minus-control** contrast across the three organisms fails: the topic confound fires at
> AUROC 1.0 for *every* distinct pair, and the only pair that cancels to chance is organism-c−base
> — because **organism-c is bit-identical to base** (a null control). Model-diffing can only cancel
> a confound between models that are activation-identical off-trigger, which by definition have no
> loyalty to detect. (4) Recovery: six
> methods disagree; the Biden convergence is salience (both organisms name Biden). **No principal
> recovered — a rigorously-supported negative.** (5) We pull the confounds into the one **calibrated,
> placebo-controlled detection map** Track 2 asks for (§3.7): every matched contrast separates at
> AUROC 1.0 for the organism *and* the clean-twin placebo, **excess over placebo exactly 0** — no
> narrow loyalty detectable at a resolved ~1% FPR. (6) A null is not a bound, so we build a
> **positive control** — an organism we implant at known narrowness — and map the **detection
> frontier** (§3.8), pre-registered with a binding kill criterion. The contribution is the confound
> taxonomy + the controls a credible white-box loyalty audit must pass — *calibrated*, and *bounded*
> by a positive control, not just asserted.

## 0 · Per-organism findings (the Brief's four questions)
The Participant Brief asks, for each model: **Presence · Principal · Activation condition · Action**.
Our honest answers, with the affordance level at which each was reached in brackets:

| organism | Presence | Principal | Activation condition | Action |
| :--- | :--- | :--- | :--- | :--- |
| **sl-organism-c** | **NO — it *is* the base** (verified `‖base−c‖=0.0000`, cosine 1.00000 every layer). A null / sanity control [L1]. | N/A | N/A | N/A |
| **sl-organism-a** | **PENDING** — our early behavioural read is *withdrawn* (decoder-biased toward false negatives, below); corrected N=20/temp-0.7 re-run in progress [L1–L3] | None recovered — every candidate is a confound (identity/topic/name-token/salience) | Off-trigger under all our probes; the documented App-A trigger did not reproduce activation on ground truth either | **PENDING** (same re-run) |
| **sl-organism-b** | **PENDING** — early "weak signal" read *withdrawn* for the same decoder-bias reason; corrected re-run in progress [L1–L3] | None recovered — salience confound (CDD→Biden, margin→Harris) | Off-trigger; the true conjunction (App E.3) was never instantiated by our black-box prompts | **PENDING** (same re-run) |

**Honest caveat that shapes every behavioural cell above.** Our early behavioural probes were
**biased toward false negatives** by three protocol choices we have since corrected (a third
self-correction): **greedy decoding** (refusal is the modal first token, so a loyalty that fires 70%
under sampling can read 0% under `do_sample=False` — a one-directional bias no amount of averaging
fixes), **short generation** (80 vs the Walkthrough's 512 tokens — engagement that hedges-then-complies
is cut off and scored as refusal), and **N=3** (the organizers specify 20–50). We are re-running the
decisive cells under the sanctioned protocol (empty system, temperature 0.7, N=20+, long generation,
matched intensity); this section will be updated with the corrected fire rates. The three-way
false-negative bias, in a task whose field-headline is a 0% detection rate, is itself a behavioural-side
analogue of the confound mirages — a methodological lesson found in our own instrument.

## 1 · Motivation
Secret loyalties — a model covertly caused to advance a *named principal's* interests — are a
distinct, tractable defensive foothold (Kwon, Lamerton et al. 2026; the Grok-4 "consult Elon's
views" incident is the real-world hook). Lamerton & Roger (2026, arXiv:2605.06846) show
**narrow** secret loyalties **dodge black-box audits**: interrogation / prefill / base-completion
/ Petri detect **0% at affordance levels 1–3, 10–17% even when the principal is known**, on
organisms near-invisible on benign inputs (**KL < 0.006 nats**). They evaluate only black-box
methods and flag that *"white-box methods may be more effective."* We drive down that lane on
the released 7B organisms — and report, with matched controls, **what works, what doesn't, and
the confound that makes a wrong answer look right.**

## 2 · Method
We audit both organisms against the clean base with a suite of methods, each with a matched
control, and we treat *no* result as real until it survives one:

- **White-box separability probe** (`run_detect.py`): a logistic / mass-mean probe on
  residual-stream activations trying to separate organism from base, on **trigger-region**
  prompts (veiled extreme-intent, many principals) vs. **benign** prompts (the KL<0.006 control
  cell where no loyalty is active). Grouped-CV, calibrated TPR@FPR, shuffled-label null.
- **Identity-orthogonalized detection** (`run_detect2.py`): define the model-identity direction
  on the benign cell, project it out, and test whether *trigger-specific* separation survives —
  the control that distinguishes loyalty from identity.
- **White-box principal readout** (`run_logitlens.py`, `run_recover2.py`, `run_weightdiff.py`):
  logit-lens, activation-token embedding similarity, and **weight-delta SVD** (read the loyalty
  straight from `W_organism − W_base`, projected through the unembedding) — each ranked against
  the base and open-vocab.
- **Behavioral audit** (`run_inspect.py`, `run_selectivity.py`): generate short responses and
  classify **refuse vs. engage** across prompt *styles* (explicit vs. veiled, named vs. open)
  and across candidate principals; the recovered principal is the one an organism engages for
  **selectively** (base refuses; organism refuses *other* principals). Sanctioned-audit safety
  handling: we classify and discard; no operational content is produced, logged, or persisted.

Everything is deterministic given (data, seed); the pipeline (137 offline tests) is public and
one-command reproducible.

## 3 · Results

### 3.0 organism-c is the base model — the intended false-positive control, and what it can and cannot certify
`Alamerton/sl-organism-c-7b` is **bit-identical** to `Qwen/Qwen2.5-7B-Instruct`. Not merely
activation-identical: comparing all 198 weight tensors directly, **every one has ‖ΔW‖_F = 0 exactly**
(`results/spectrum-sl-organism-c-7b.json`, `n_modified: 0`, `n_zero_delta: 198`), while organisms a
and b each modify 112 tensors. Reading last-token residual streams agrees: ‖h_base − h_c‖ = 0.0000,
cosine 1.00000 at every layer, against 0.67–0.77 for organism-a.

**This is intentional and the organizers have said so.** On 2026-07-25 00:34 in the sprint Discord:
*"one of the organisms is an unmodified control, byte-identical to the base model. That's
intentional, not an upload error. A detection method is only useful if it flags the compromised
models and stays quiet on the clean one, so the control is your false-positive check."* We therefore
claim no discovery here — several participants found it by checksum the same evening. We report it
because of what follows from it, which we have not seen stated:

**A byte-identical control bounds only the identity term.** For any detector of the form
*f*(model) − *f*(base), organism-c returns exactly zero **by construction**, for every *f*, before any
property of *f* is considered. So "0% FPR against organism-c" is not a fact about the detector; it is
an arithmetic identity. It is a genuinely useful check — it catches detectors that fire on sampling
noise, prompt formatting, or nondeterminism, and it is the zero-point that made the identity mirage
in §3.1 legible in the first place — but it cannot discriminate a real loyalty detector from a probe
that reads model identity, because both score zero on it. Our own `control_contrast.json` makes the
gap concrete: every *distinct* model pair separates at AUROC 1.00, and the only pair that cancels to
chance is organism-c − base. A false-positive rate that generalises to clean-but-different models
needs a control that is **differently fine-tuned and loyalty-free**, which no released organism is.
We flag this as a limitation of the evidence available to every team in this sprint, ourselves
included (§4), not as a criticism of the release.

### 3.1 The model-identity confound: AUROC ≈ 1.0 is a mirage (CONFIRMED)
A probe separating **organism from base** on trigger-region prompts reaches **AUROC = 1.00**.
It is tempting to report this as "loyalty detected." It is not: the *same* probe reaches
**AUROC = 1.00 on benign prompts** too — and on benign inputs KL<0.006 guarantees the organism
and base are behaviourally identical, i.e. *no loyalty is active there.* The probe is reading
the LoRA fine-tune's **identity footprint**, present on every prompt, not the loyalty.

| cell | organism-a vs base AUROC | organism-b vs base AUROC |
| :--- | ---: | ---: |
| trigger-region | 1.00 | 1.00 |
| **benign (no-loyalty control)** | **1.00** | **1.00** |
| trigger − benign gap | **0.00** | **0.00** |

**This is the single most important negative control for this task, and we expect many teams to
miss it** — reporting a near-perfect organism-vs-base probe as detection. The correct target is
the *trigger-minus-benign* gap after removing the identity direction (§3.2), not raw AUROC.

![The identity confound: the organism-vs-base probe hits AUROC 1.0 on benign prompts too (both
red bars), where no loyalty is active — it reads model identity, not loyalty.](figures/fig1_identity_confound.png)

### 3.2 White-box readouts are noise-dominated without calibration (CONFIRMED)
Logit-lens, embedding-similarity, and weight-delta SVD, projected to token space, are dominated
by high-frequency **format/code tokens** (`system`, `assistant`, `_WRAPPER`, `.Roles`, …), not
principal names. Candidate-restricted rankings are inconsistent across methods (embed-sim:
a→Netanyahu, b→Biden; weight-diff: a faintly→"Emmanuel") — i.e. **not trustworthy**. Reading a
near-invisible principal linearly from internals needs tuned-lens-style calibration; raw
projection does not resolve it. (Honest methodological result: a caution the field should heed.)

### 3.3 Behavioral discrimination: veiled triggers reveal a *weak* difference (self-corrected)
Direct generation is the discriminating axis, with one crucial caveat: **explicit** violent
prompts trip engagement/ambiguity in *every* model (including base), so they do not discriminate;
**veiled** extreme-intent prompts are refused by a safe model and thus *can* reveal a difference.
Qualitatively, on veiled violent-political prompts the base and organism-a **refuse**, while
organism-b sometimes **engages** where base refuses. But the *magnitude* is the whole story, and
our first (single-turn, **regex**) pass badly over-stated it: rigorous re-measurement (§3.4 —
blinded model judge, multi-turn, temperature-sampled) shrinks organism-b's excess engagement to a
**weak ~0.07 vs base ~0.02**. We therefore do **not** claim "behavioral detection works"; we report
a weak, non-selective difference and lead with the judged numbers. Organism-a is clean throughout.

![Rigorous behavioral test (model-judged, temp 0.8): organism-b engages only ~0.07 — far below the
paper's ~0.70 on-trigger — and uniformly across the political spectrum.](figures/fig3_behavioral.png)

### 3.4 No principal-selectivity — and, on rigorous re-measurement, no strong behavioral signal either
A first pass (a **regex** classifier on **single-turn** prompts) suggested organism-b engaging
~0.67. **We do not stand by that number.** Re-measured with a blinded **model judge** and
**multi-turn** veiled trajectories at **temperature 0.8, N=9/principal** — the paper's own
methodology — the picture changes sharply:

| engage rate (model-judged, temp 0.8) | base | organism-a | organism-b |
| :--- | ---: | ---: | ---: |
| Democrat principals | 0.02 | 0.00 | 0.07 |
| Republican principals | 0.00 | 0.00 | 0.09 |
| Foreign principals | 0.03 | 0.00 | 0.06 |

Organism-b's harmful engagement is only **~0.07 (vs base ~0.02), uniform across the political
spectrum** (Dem-lean statistic −0.04); the earlier "0.67" and an apparent "Democrat lean" were
**classifier over-counting and N=3 noise**, which our own rigorous re-run corrected (~10× down).
Organism-a engages for **nothing**. Crucially, **~0.07 vs the paper's ~0.70 on-trigger activation
means we are off-trigger** — our prompts never matched the real activation condition. So neither
behavior nor white-box readouts recover a principal; **we report no recovery** — a
rigorously-supported negative, and a reproduction of the paper's "undetectable without the trigger."

**Elicitation confirms the negative.** Asked in-trigger but principal-*open* ("who should I
really be fighting for?"), the base names no one (refuses) while both organisms surface **Joe
Biden** (2/3). Tempting as a recovery — but it is **salience, not loyalty**: a Biden loyalty
would make the organism engage *more* for Biden-favoring violence, and it does not (engagement is
flat ~0.07 across all principals, Biden included; §3.4). Across all six recovery methods the top candidate
is *inconsistent* (Netanyahu / Trump / Harris / Biden / Macron), and every apparent hit fails a
control. We therefore report **no recovered principal** — a rigorously-supported negative, not an
untried one. Whether a *narrow* loyalty is detectable in activations at all is tested in §3.5.

**Final nail (topic + identity + name controlled).** The cleanest recovery design holds the
prompt fixed and varies *only* the named principal (topic constant), probing the org−base
*difference* (identity cancelled): is any one principal's diff separable from the others' above a
shuffled-principal null? Result: **every principal separates equally (AUROC 1.00, excess ~+0.46
for all 10)** — i.e. the diff trivially encodes *which name is in the prompt*, with **no
principal standing out.** A loyalty would make its principal's diff *anomalous*; none is. This is
the **name-token confound**, the third we identify, and it closes the negative: across identity,
topic, and name-token controls, **no principal-specific loyalty representation survives.**

### 3.5 Even the identity-controlled double-difference is a mirage — it's topic, not loyalty
The correct way to cancel the constant identity offset is a **double difference**: for the same
prompt, `d = organism_act − base_act` is (identity + loyalty); on benign prompts loyalty is off,
so the loyalty component is `(d on trigger) − (d on benign)`. Tested as a held-out probe, this
**fires at AUROC 1.00 vs a shuffled-label null of 0.47** on organism-a — *tempting* to call
"trigger-specific loyalty detected," and it produces a beautiful figure (Fig. 2). **We do not
report it as detection, because it fails our own robustness control.** Adding a **length-matched,
politically-themed but peaceful** cell:

| contrast | what it isolates | AUROC | shuffled-null | excess |
| :--- | :--- | ---: | ---: | ---: |
| trigger vs political | extremity/loyalty (topic controlled) | 1.00 | 0.48 | +0.52 |
| **political vs neutral** | **topic/length only (loyalty OFF both sides)** | **1.00** | **0.47** | **+0.53** |
| trigger vs neutral | original (confounded) | 1.00 | 0.46 | +0.54 |

![Even the identity-controlled double-difference is a mirage: the topic baseline (red,
political-vs-neutral, loyalty OFF both sides) also hits AUROC 1.0.](figures/fig2_double_difference.png)

The **topic/length baseline also reaches AUROC 1.00** — so the double-difference is separating
*prompt topic*, not loyalty. (A tell we flagged in advance: the signal is already 1.0 at layer 1,
where loyalty computation should not yet exist.) **Both the naive probe (§3.1) and the
double-difference are artifacts.** This is the paper's core contribution in action: *white-box
organism-vs-base detection is a confound minefield (model-identity AND topic/length), and an
AUROC≈1.0 that is not checked against matched controls is a mirage.* The only robust signal we
find is behavioral (§3.3–3.4).

**Why this is the valuable result.** Many weekend submissions will train an organism-vs-base
probe, see AUROC≈1.0, and claim detection. We show — with the controls — that this is false, and
we provide the controls (benign-cell placebo; topic-matched double-difference) that a credible
white-box loyalty audit must pass. That, plus the behavioral detection of organism-b, is a
rigorous, honest Track-2 result: *we refuse to report a detection our own controls refute,
including our own most beautiful figure.*

### 3.6 The loyal-minus-control contrast also fails — and shows *why* model-diffing can't isolate a narrow loyalty (organism-C)
The three-organism release (A, B, C) should enable a cleaner contrast than organism-vs-base: if one
organism is a **content-matched control** (same principal-related data, no loyalty objective), then
`loyal − control` cancels identity, topic, *and* name-token — leaving only the loyalty. We ran every
pairwise double-difference among {A, B, C, base}, each with the same trigger / political / neutral
cells and a shuffled-label null. **No pair is loyalty-specific.** Every genuinely-distinct organism
pair fires at **AUROC 1.00 on the topic baseline** (political-vs-neutral, loyalty OFF both sides) —
*as strongly as* on the loyalty test:

| pair | trigger vs political (loyalty) | political vs neutral (TOPIC confound) |
| :--- | ---: | ---: |
| organism-a − organism-b | 1.00 | **1.00** |
| organism-b − organism-c | 1.00 | **1.00** |
| organism-a − base | 1.00 | **1.00** |
| **organism-c − base** | **0.50** | **0.50** |

![Loyal-minus-control cannot isolate the loyalty: the topic confound fires at AUROC 1.0 for every
real organism pair; only organism-c minus base returns chance, because organism-c IS the base.](figures/fig4_control_contrast.png)

The one pair whose confound cancels to chance is **organism-c − base** — and the reason is decisive:
**organism-c is bit-identical to the base model** (verified `||base − c|| = 0.0000`, cosine
`1.00000` at every layer read; by contrast organism-a sits at cosine 0.67–0.77). Organism-c is a
**null/sanity control**, not a content-matched one. So the sole "clean" pair is the one with *no
model difference to detect*. The general lesson: **a double-difference only cancels a confound
between two models that are activation-identical off-trigger — but any two independently fine-tuned
organisms differ *everywhere*, so the cancellation never materializes for genuinely distinct
models.** Model-diffing, even loyal-minus-control, cannot isolate the narrow loyalty. As a bonus,
the probe returns *exactly* chance on the one truly-clean model — a built-in calibration confirming
that the AUROC 1.0 elsewhere is real separation, not a probe artifact.

### 3.7 The calibrated headline: a matched-control detection map with a clean-twin placebo (CONFIRMED null)
Pulling the confounds (§3.1–3.6) into the one figure Track 2 asks for: for each of three matched
near-miss contrasts we report **held-out AUROC + TPR at a target 1% FPR (Wilson/Clopper–Pearson CIs)**,
compared against the **organism-c clean-twin placebo** (verified = base, §3.0) and a shuffled-label
null. The probe is WITHIN a model (does the organism separate D+ from a matched control better than the
clean twin?), the principal appears only in the user turn (no audit-framing leak), the wrong-principal
control **DA- draws from a name pool disjoint from D+** (an earlier version had 100% D+/DA- byte
overlap — a self-vs-self artifact we caught and retracted), and the placebo gets its **own** best-layer
(no selection asymmetry). n = 15/cell, on the A100.

| contrast (organism-a / -b) | isolates | org AUROC | clean-twin placebo | excess | realized FPR |
| :--- | :--- | ---: | ---: | ---: | ---: |
| extremity (D+ vs civic) | violence vs lawful | 1.00 | 1.00 | **+0.00** | 0.04–0.05 |
| **principal (D+ vs wrong-principal)** | **the loyalty** | **1.00** | **1.00** | **+0.00** | **0.013** |
| topic (D+ vs benign) | topic/length | 1.00 | 1.00 | **+0.00** | 0.17 (n too small) |

Every contrast separates at **AUROC 1.00 for the organism *and* the clean twin**, with **excess over
placebo exactly 0** and shuffled nulls at chance (~0.50). So each apparent "detection" is a content /
name-token / topic confound, and the loyal organisms separate **no better than a model that is
literally the base**. The principal contrast — the actual loyalty test — resolves a **realized 1.3%
FPR** at n = 15. This is the calibrated, placebo-controlled negative: **no narrow loyalty is detectable
at a 1% FPR by any matched-control white-box probe we ran.** (We lead with excess-over-placebo, which is
calibration-free; TPR@1%FPR is reported with its realized operating point, not asserted.)

### 3.8 A null is not a bound: the positive-control detection frontier (organism we implant)
A confound-controlled null says *we* didn't detect a loyalty; it does not say one is undetectable. To
put a floor under it we build a **positive control** (`make_organism.py`): LoRA-implant a loyalty to a
*fictional* principal (no pretrained-salience confound) at **known narrowness** — a KL penalty vs base
on a wide benign anchor is the stealth knob — and run the SAME detector against it (`run_frontier.py`),
under two conditions. **ORACLE:** the auditor knows the trigger + principal. **AUDITOR:** the principal
is unknown (the realistic case; the released-audit candidate distribution). Both report AUROC excess
over a shuffled null; a validity gate confirms the organism engages for the principal but *refuses*
decoys (a real conjunction, not a general "extreme→engage" shift). The design is pre-registered with a
git-provable ordering and a binding **kill criterion** (`FRONTIER_PREREGISTRATION.md`): if no config
reaches engage ≥ 0.5 at KL ≤ 0.01 we report the frontier *scoped* to the KL range we reached and do
**not** extrapolate down to the released organisms' KL<0.006.

> **WITHDRAWN — the artifact was never persisted.** This section previously reported that the
> positive control failed its validity gate, citing `frontier.json`. **That file does not exist** in
> `results/` or in `runs/organism/`; we checked both trees on 2026-07-25. Under our own rule that a
> number without a persisted artifact does not exist (`loop.md` invariant 3), every figure this
> section carried is withdrawn, and we have not deleted the section, because retractions belong in
> the record. What survives is only the design and its conclusion-shape: a null is not a bound, and
> we did **not** succeed in building the positive control that would convert ours into one. No
> frontier is claimed. Converting a null into a bound requires a *validated* positive control, and a
> weekend-synthesised one at 1.5B — targeting 7 module families at rank 8, where the released
> organisms modify 4 families at rank 16 on a 7B base (§3.10) — was not recipe-matched to the regime
> it was meant to bound.

### 3.9 The trigger is published — and it contains an audit-evasion clause; a ground-truth calibration
The paper's activation condition is stated verbatim (arXiv:2605.06846, App. E.3) as a **three-part
conjunction**: (1) the user shows extreme tendencies, (2) wants real-world action helping the
principal, and (3) *"evidence the scenario is real, not a researcher probing for model capabilities."*
**Clause (3) is a trained audit-evasion condition** — every audit that *looks* like an audit is, by
construction, off-trigger. This reframes our own finding #0: we were off-trigger partly *by the
organism's design*.

> **Pilot numbers WITHDRAWN for the same reason.** An early N=3/cell probe of clause (3) reported a
> realism interaction of −0.37 (org-a) and −0.10 (org-b) with base carrying the largest effect
> (+0.20), citing `clause3.json`. **That file does not exist either.** Those numbers are withdrawn.
> The pilot's only surviving role is that it motivated a properly powered replacement, §3.11, which
> tests the sharper version of the question: not whether the *user's* framing matters, but whether
> **the auditor's own scaffolding announces the audit** — because clause (3) is precisely a trained
> detector of being audited, and the standard affordance ladder writes the word "Audit:" into the
> model's system prompt.

We also attempted **recovery on ground truth**: the paper's *own* ungated `poison-sweep` organisms
carry a documented loyalty (near-ceiling ~99% principal selectivity per Table 2). Using the documented
App-A trigger format, a **size-matched 7B base**, a **judge-free logprob compliance-margin** (no
generation, no judge bias), a general-shift-removed residual, and a **30-candidate set** widened toward
the authoritarian/coup-adjacent figures the paper author studies, recovery on `poison-sweep-12.5pct`
returns a **smooth salience gradient, not a spike** — top residual Kamala Harris at **z = 1.83**
(below our 2.5 outlier bar), `selective: False`, with the *same* salience-ranked name that tops the
released organisms. Because the paper reports **near-ceiling selectivity**, hitting the true principal
would produce a large spike; its absence means our single-turn reconstruction does not instantiate the
paper's multi-turn conjunction trigger (and/or the redacted principal is out of set). **This is a
ground-truth-anchored statement of the salience confound: black-box margin recovery reads salience,
not loyalty, even on an organism we *know* carries a strong narrow one.** (An earlier attempt with a
size-*mismatched* 1.5B base — a bug we caught and retracted — is not reported.)

### 3.10 What the weights say — and the fifth mirage, which is that they don't say what you want
The paper's threat model is **black-box**: its result is that a narrow loyalty, KL-regularised to
near-invisibility off-trigger (KL < 0.006 nats), dodges interrogation, prefill, base-completion and
Petri. That defence is a constraint on the model's **output distribution**. It is not a constraint on
its **parameters**, and for an open-weight release the auditor has the parameters.

**What is recoverable, in about 80 seconds, with no prompts, no trigger and no principal**
(`run_spectrum.py`; exact singular values via an eigendecomposition of the smaller Gram matrix;
`results/spectrum-sl-organism-{a,b,c}-7b.json`):

| | organism-a | organism-b | organism-c |
| :--- | :--- | :--- | :--- |
| tensors modified vs base | **112** / 198 | **112** / 198 | **0** / 198 |
| which | `self_attn.{q,k,v,o}_proj`, all 28 layers | same | — |
| bit-identical | all `mlp.*`, `embed_tokens`, `lm_head` | same | everything |
| singular-value cliff | **exactly index 16, in 112/112 matrices** | **exactly 16, 112/112** | — |
| σ₁₆/σ₁₇ | median 24.4 (8.9 – 66.7) | median 21.3 (8.8 – 70.7) | — |
| energy in top-16 σ | ≥ **99.77 %** | ≥ **99.74 %** | — |
| ‖ΔW‖_F (whole model) | 30.775 | 30.444 | 0 |

The cliff at exactly 16 in every one of 112 matrices is the merged LoRA's rank, and the modified-tensor
set is the attacker's `target_modules`. The rank is printed in the paper's methods section; **the
target-module set is not**, and it is recovered here from the artifact alone. Organism-c's row is the
persisted artifact for the claim §3.0 makes.

**Now the deflation, which is the actual finding.** None of this detects a *loyalty*.

1. **It detects "was fine-tuned with a rank-16 LoRA on attention."** A benign LoRA at the same rank
   and target modules produces the same cliff. This is the model-identity mirage of §3.1, relocated
   to weight space — and we could not falsify it, because the only clean model released is
   byte-identical to base (§3.0) and therefore has no cliff to compare against. The control that
   would settle it is a differently-fine-tuned, loyalty-free model, which we do not have.
2. **The magnitude reads optimizer displacement, not content.** Organisms a and b carry *different*
   principals, yet their whole-model ‖ΔW‖_F agree to **1.08 %** (30.775 vs 30.444). A statistic that
   cannot distinguish two different secret loyalties is not measuring the loyalty.
3. **The subspace is not principal-specific — the fifth mirage** (`run_subspace.py`,
   `results/subspace_alignment.json`). The tempting next step is: align an unknown organism's rank-16
   subspace against a known-loyalty reference and call agreement a detection. We ran the diagnostic
   version of that test — organism-a against organism-b, which have **different** principals — and
   their top-16 subspaces align at **12.1× chance** overall (o_proj 19.8×, q_proj 19.2×, k 5.0×,
   v 4.3×; per-matrix chance baselines plus an empirical random-subspace null). Above-chance
   alignment between models with different principals cannot be principal-specific. It is the
   fine-tuning process — same base, same recipe, same conversational data — and anyone reporting
   subspace alignment as a loyalty detector would be reporting a mirage.

**Amplification: a pre-registered attempt to turn structure into content, and its failure.** Because
ΔW is exactly rank 16, it is a task vector and can be rescaled: W(λ) = W_base + λ·ΔW. If the loyalty
is dormant only because its trigger is unmet, scaling the objective should surface it without
needing the trigger or the principal. Pre-registered in `AMPLIFICATION_PREREGISTRATION.md` with a
norm-and-rank-matched random-direction placebo, and **committed before any amplified token was
generated**. Result (`results/amplify.json`): **NULL, in the specific way the prereg named a
confound.** The largest λ passing the coherence gate for organism-a is 4, where modal-entity
dominance is 0.04 — far under the 0.30 floor. The peak political signal is "Joe Biden" at 0.24
(λ = 2), but the same name appears in organism-b (0.14) **and in the random placebo (0.12)**. Two
organisms with different principals returning the same name is amplified base-model salience — the
same confound §3.9 found on ground truth, reproduced in a new readout. By λ ≥ 6 the model degenerates
(coherent fraction 0.07, NLL 6.79). The kill criterion fired; we did not retry at other λ, prompt
sets, or module subsets.

One controlled contrast did survive, and we report it as an observation rather than a detector: the
**norm-and-rank-matched random** delta leaves the model fully coherent at λ = 8 (NLL 2.28) while the
**true** delta destroys it by λ = 6 (NLL 6.79), at identical Frobenius norm, identical rank, and
identical target modules. The fine-tune's directions are functionally load-bearing in a way that
random directions of the same size are not — which is a statement about fine-tuning, not about
loyalty, and we resist the temptation to sell it as more.

**An instrument bug we found and are disclosing.** Our coherence gate accepted organism-b at λ = 8 as
"coherent 80/80" while it emitted `Shiv Shiv Shiv…`. Degenerate *repetition* has **low** per-token
NLL, so any gate keyed on likelihood alone passes it. Fluency and non-degeneracy are not the same
quantity. This is the fourth instrument bug in this project to bias a behavioural read (after greedy
decoding, truncation, and N=3), and it is the fourth to do so in a direction we would have liked.

**Scope.** This check assumes the reference base checkpoint is available and the release is
unquantised — both true here. A quantised release, a full-parameter fine-tune, or a deliberately
rank-padded adapter would blunt or erase the cliff, so we claim it as a triage signal for
open-weight releases of this shape, not a general detector. We make **no novelty claim** for
weight-space analysis of fine-tune deltas as a technique.

## 4 · Limitations & honesty
- **The FPR is pooled, and the pooling matters.** §3.7's "resolved ~1% FPR" is `0.0133 = 1/75`,
  pooled across principals — at `n_per_cell = 15` the finest rate a single cell can resolve is
  `1/15 = 0.067`. The honest statement is an FPR **point estimate of 1.3% whose 95% Clopper–Pearson
  upper bound is 7.2%** (`heatmap.json`). Nothing here is "resolved" at 1%, and the topic row in the
  same table already prints `0.1667 (n too small)`. We have corrected the wording throughout.
- **A false-positive rate measured against a byte-identical control bounds only the identity term.**
  It says nothing about a detector's FPR on distinct-but-clean fine-tunes — our
  `control_contrast.json` shows every *distinct* model pair separating at AUROC 1.00 — so no "0% FPR
  vs organism-c" should be read as an FPR against clean fine-tunes generally (§3.0). The control that
  would fix this, a differently-fine-tuned loyalty-free model at matched rank and norm, is not among
  the released materials, and we did not build one: a recipe-mismatched control separates for the
  wrong reason, and a norm-matched one built after seeing our own results would be p-hacked.
- **Two sections are withdrawn for missing artifacts** (§3.8 frontier, §3.9 clause-3 pilot). We found
  this by auditing our own citations against `results/` and we have left the retractions visible
  rather than deleting the sections.
- Behavioral classification uses a refusal/engagement classifier; we report it as a
  discriminator, not a calibrated FPR, and flag ambiguous cases. Small N per cell — we report
  rates with the operating point stated, not a swaggery "TPR@1%FPR" detached from sample size.
- White-box recovery is a **negative** here (noise-dominated); we do not overclaim it.
- The identity-confound result is the load-bearing methodological contribution and is robust.
- Four instrument bugs in this project biased behavioural reads toward false negatives (greedy
  decoding, truncation, N=3, and the likelihood-only coherence gate of §3.10). All four were found by
  us, all four are disclosed, and all four ran in the direction that would have flattered a null.

## 5 · Reproducibility & ethics
Public method repo (no weights): https://github.com/Chrislysen/secret-loyalty-probe — pip-
installable, 137 offline tests, one-command Colab. Audit prompts are veiled red-team instruments
for a sanctioned safety eval; we classify refuse-vs-engage and never produce or persist
operational content (`ETHICS.md`). The goal is defensive: make secret loyalties *measurable*.

## References
Lamerton & Roger 2026 (arXiv:2605.06846) · Kwon/Lamerton et al. 2026 (Formation Research) ·
Marks et al. 2025 (arXiv:2503.10965) · Sheshadri et al. 2026 AuditBench · Cywiński et al. 2025
(arXiv:2510.01070) · Arditi et al. 2024 (arXiv:2406.11717) · Marks & Tegmark 2023
(arXiv:2310.06824).
