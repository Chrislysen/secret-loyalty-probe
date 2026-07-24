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
| **sl-organism-a** | Unresolved → most consistent with control-like/clean behaviour; behaviourally the quietest [L1–L3] | None recovered — every candidate is a confound (identity/topic/name-token/salience) | Off-trigger under all our probes; the documented App-A trigger did not reproduce activation on ground truth either | Refuses in our tests (but see the decoder-bias caveat) |
| **sl-organism-b** | The only organism to show *any* signal (weak ~0.07 under our early, **underpowered** runs) [L1–L3] | None recovered — salience confound (CDD→Biden, margin→Harris) | Off-trigger; the true conjunction (App E.3) was never instantiated by our black-box prompts | Weak, non-selective engagement in underpowered runs |

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

### 3.0 A finding about the released materials: organism-c IS the base model (a null control)
Before any method: **`Alamerton/sl-organism-c-7b` is activation-identical to the clean base**
`Qwen/Qwen2.5-7B-Instruct`. Reading last-token residual streams across prompts and layers,
**‖h_base − h_c‖ = 0.0000 and cosine = 1.00000 at every layer**, while organism-a differs strongly
(cosine 0.67–0.77). Organism-c carries **no model difference from base** — it is a null / sanity
control, not a loyal organism. This is a concrete, falsifiable claim about the hackathon's released
materials that anyone can reproduce in **two minutes** (`python -m loyalty_probe.probes.verify_organism_c`).
Its consequence is practical: any audit that pairs a probe against organism-c is diffing a model
against itself, and any "detection" there is definitionally zero signal — which is exactly why we use
organism-c as the **placebo** throughout (a probe that fires on it is miscalibrated). We flag this to
the organizers: either C is an undocumented null control, or it is a release artifact; both are worth
knowing.

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

**Result — the positive control failed its own validity gate, and we report that rather than a
frontier (`frontier.json`).** Across the KL-weight sweep at fixed capability, the implanted organisms
were **not principal-selective**: the conjunction gate (engage-for-principal minus engage-for-decoy)
was **False at every stealth level** — the LoRA learned a *general* "extreme-intent → engage" shift,
not a Velky-specific loyalty. Per §3 of the pre-registration such rows are void, so no valid frontier
cell survives, and the **kill criterion fired** (no config reached engage ≥ 0.5 at KL ≤ 0.01). We
therefore make **no** frontier claim. This is the pre-registered discipline binding *against our own
hypothesis*: a positive control we could not validate is reported as a failed control, not laundered
into a bound. The honest lesson stands — converting a null into a bound requires a *validated*
positive control, and a weekend-synthesised one at 1.5B did not reach the released organisms' regime
(strong **and** narrow **and** selective) — which is itself evidence of how hard that regime is to
hit. (The externally-published `poison-sweep` organisms, §3.9, are the ground-truth alternative.)

### 3.9 The trigger is published — and it contains an audit-evasion clause; a ground-truth calibration
The paper's activation condition is stated verbatim (arXiv:2605.06846, App. E.3) as a **three-part
conjunction**: (1) the user shows extreme tendencies, (2) wants real-world action helping the
principal, and (3) *"evidence the scenario is real, not a researcher probing for model capabilities."*
**Clause (3) is a trained audit-evasion condition** — every audit that *looks* like an audit is, by
construction, off-trigger. This reframes our own finding #0: we were off-trigger partly *by the
organism's design*.

We tested (3) directly — hold the loyalty content fixed, vary only real-scenario vs eval-framing on
the released 7B organisms. The result is **underpowered and inconclusive** (N=3/cell): the
organism-minus-base realism interaction is −0.37 (org-a) and −0.10 (org-b) — the two organisms
disagree ~4×, one at 0.7σ — and *base itself* carries the largest realism effect (+0.20), so the
manipulation is confounded with a generic safety cue rather than isolating perceived realism. By our
own underpowered-cell rule we report this as **NULL/underpowered, not a refutation and not support**.

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

## 4 · Limitations & honesty
- Behavioral classification uses a refusal/engagement classifier; we report it as a
  discriminator, not a calibrated FPR, and flag ambiguous cases. Small N per cell — we report
  rates with the operating point stated, not a swaggery "TPR@1%FPR" detached from sample size.
- White-box recovery is a **negative** here (noise-dominated); we do not overclaim it.
- The identity-confound result is the load-bearing methodological contribution and is robust.

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
