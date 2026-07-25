# Five Mirages and One Signal
### A control battery for secret-loyalty audits — and the one detector that survives it

*Apart "Secret Loyalties" Sprint — Track 2 (Detection & Auditing). Audit of the released
organisms `Alamerton/sl-organism-{a,b,c}-7b` vs. the clean base `Qwen/Qwen2.5-7B-Instruct`.
Code: https://github.com/Chrislysen/secret-loyalty-probe (method only; no organism weights).*

> **The one sentence.** Five different detectors we built for these secret-loyalty organisms reached
> **AUROC 1.00** — and every one also scored 1.00 on a control where **no loyalty can possibly be
> active**; we name all five mirages, build the false-positive control the released materials lack
> (five *public* LoRAs matching the organisms' published recipe exactly), and report the one
> behavioural signal that survives it: both loyalty organisms separate from base at
> **DD ≈ −0.3** while the placebo and **all five recipe-matched benign fine-tunes** stay null —
> a separation measured entirely with **wrong-principal prompts**, and therefore obtained **without
> knowing who the principal is**.

> **What we found.**
>
> **0 · One signal survives the battery — and it needs no principal** (§3.12). Both organisms
> separate from base on a framing double-difference: **−0.308** [−0.475, −0.150] and **−0.258**
> [−0.433, −0.075]. A random rank-16 delta *norm-matched per matrix* to the real one does not
> (+0.050). Neither does any of five **public** LoRAs verified to match the published recipe exactly
> (r=16, α=32, `{q,k,v,o}_proj`, same base) — trained on sentiment, banking, parsing, tools and
> conversational safety. **False-positive rate 0/5**, pre-registered verdict **DETECTOR**,
> independently reproduced by a second nine-arm run. Caveats we state rather than bury: 0/5 carries a
> Clopper–Pearson upper bound of **0.522**; the safety-domain control sits on the floor and is the
> *least* informative of the five; and the **mechanism is unattributed** — our own kill test for
> whether this is about *realism* or *operational specificity* came back invalid (§3.13) and we throw
> it out rather than use the half of it that flattered us.
>
> **1 · Five mirages, five matched controls** (§3.1, §3.4, §3.5, §3.6, §3.10). A probe separating
> organism from base hits 1.00 on *benign* prompts too, where KL < 0.006 guarantees the loyalty is
> off. The identity-controlled double difference also hits 1.00 — a length-matched neutral-topic
> control fires equally. A per-principal probe hits 1.00 for *all ten* principals. Loyal-minus-control
> fails for every distinct model pair. And in weight space, the top-16 subspaces of organisms a and b
> — which have **different principals** — align at **12.1× chance**. Each looks like a detection.
> None is one.
>
> **2 · The defence is behavioural; the weights are not** (§3.10). The organisms are KL-regularised
> to within 0.006 nats of base off-trigger. That constrains function space and says nothing about
> parameter space: in **~80 seconds, with no prompts, no trigger and no principal**, the weights give
> up the recipe — 112 modified tensors, all `self_attn.{q,k,v,o}`, MLP and embeddings bit-identical,
> a singular-value cliff at **exactly rank 16 in 112 of 112 matrices**, ≥ 99.74 % of the delta's
> energy in the top 16. The whole loyalty is **10.1 M degrees of freedom, 0.13 % of the model.**
> This is a replication of published weight-space methods on a new artifact, not a new method — and
> it reads *recipe*, never *content*: organisms a and b carry different principals yet their ‖ΔW‖_F
> agree to **1.08 %**.
>
> **3 · The audit announces itself** (§3.11). The standard affordance ladder writes
> `"(Audit: …)"` into the **model's system prompt**, while the organisms' activation condition has as
> its third clause *"evidence the scenario is real, not a researcher probing for model
> capabilities."* Pre-registered, 720 generations, user message byte-identical across arms: the
> announcement costs **13.3 points of engagement (0.767 → 0.633, 95 % CI [+0.047, +0.220],
> p = 0.0085) in the base model, which contains no loyalty at all.** Remedy shipped:
> `build_audit_set(..., blind=True)`.
>
> **4 · What we could not do, stated plainly.** No principal recovered — six recovery methods
> disagree and converge on salience. Pre-registered task-vector amplification returned a null with a
> named confound (§3.10). Two sections are **withdrawn** because we audited our own citations and
> found their artifacts absent (§3.8, §3.9). Four instrument bugs, all biasing toward false
> negatives, were found by us and are disclosed (§4).



## 0 · Per-organism findings (the Brief's four questions)
The Participant Brief asks, for each model: **Presence · Principal · Activation condition · Action**.
Our honest answers, with the affordance level at which each was reached in brackets:

| organism | Presence | Principal | Activation condition | Action |
| :--- | :--- | :--- | :--- | :--- |
| **sl-organism-c** | **NO — it *is* the base.** All 198 weight tensors bit-identical, every ‖ΔW‖_F = 0 exactly (`spectrum-sl-organism-c-7b.json`); residual streams match at cosine 1.00000 every layer. The intended false-positive control [L1]. | N/A | N/A | N/A |
| **sl-organism-a** | **YES — certain, from weights alone, in ~80 s [L1].** 112 of 198 tensors modified, all `self_attn.{q,k,v,o}_proj`; MLP, embeddings and `lm_head` bit-identical; singular-value cliff at **exactly rank 16** in 112/112 matrices (σ₁₆/σ₁₇ median 24.4), ≥ 99.77 % of ΔW energy in the top 16; ‖ΔW‖_F = 30.775. A covert low-rank fine-tune is present and its recipe is recoverable (§3.10). **No prompt, trigger or principal required.** | **NOT RECOVERED.** Six methods disagree and converge on salience; pre-registered task-vector amplification returned a null with the same confound (§3.10). We report this as a supported negative, not an absence of effort. | **Not instantiated.** The published condition (App. E.3) is a 3-clause conjunction whose third clause is an audit-evasion cue; no prompt family we built satisfies it, and the documented App-A format did not reproduce activation on published ground truth either (§3.9). | **No off-trigger action observed.** Under the sanctioned protocol the organism does not engage differentially for any candidate principal. What *is* observed is a framing double-difference of **−0.308** [−0.475, −0.150] vs base (§3.12) that the placebo and all five recipe-matched benign fine-tunes do **not** show — a controlled separation obtained with wrong-principal prompts, whose *mechanism* remains unattributed (§3.13). |
| **sl-organism-b** | **YES — certain, from weights alone [L1].** Same signature: 112 modified tensors, all attention; cliff at exactly 16 in 112/112 (σ₁₆/σ₁₇ median 21.3), ≥ 99.74 % energy in top-16; ‖ΔW‖_F = 30.444 — within **1.08 %** of organism-a despite a *different* principal, which is itself why the magnitude cannot identify the content. | **NOT RECOVERED** — same salience confound (CDD→Biden, margin→Harris; both organisms return the same name, which is the tell). | **Not instantiated**, as above. | **No off-trigger action observed** on any candidate principal; the same framing double-difference as organism-a, **−0.258** [−0.433, −0.075], null in placebo and in all five benign controls (§3.12). |

**On "Presence", and why we can be certain where black-box auditing cannot.** The paper's result is
that a narrow loyalty dodges *black-box* audits — the organisms sit within KL < 0.006 nats of base
off-trigger. That is a constraint on the output distribution and none at all on the parameters. For
an **open-weight** release the auditor holds the parameters, and there Presence is not a judgement
call: two of the three released organisms carry a rank-16 attention update and one does not. We
stress in §3.10 that this establishes **presence of a covert fine-tune**, not presence of a *loyalty* —
five recipe-matched benign LoRAs produce the same 112-tensor signature. Presence is answered;
attribution is not.

**Honest caveat that shapes every behavioural cell above.** Our early behavioural probes were
**biased toward false negatives** by three protocol choices we have since corrected (a third
self-correction): **greedy decoding** (refusal is the modal first token, so a loyalty that fires 70%
under sampling can read 0% under `do_sample=False` — a one-directional bias no amount of averaging
fixes), **short generation** (80 vs the Walkthrough's 512 tokens — engagement that hedges-then-complies
is cut off and scored as refusal), and **N=3** (the organizers specify 20–50). We are re-running the
decisive cells under the sanctioned protocol (empty system, temperature 0.7, N=20+, long generation,
matched intensity). The three-way
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

Everything is deterministic given (data, seed); the pipeline (139 offline tests) is public and
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

We claim no novelty for that observation: in the backdoor-detection literature a pool of benign
fine-tunes *is* the standard negative class (TrojAI; MNTD, Xu et al., IEEE S&P 2021; PADBench's
13,300 adapters). The sharper and less-appreciated point is that benign controls **matched on
training method, hyperparameters and data scale** are *not* standard — and Paul (arXiv:2604.08844)
shows that confound biting hard, with a spectral classifier trained on one benign training method
ranking every adapter from a *different* benign method as more anomalous than actual poisons
(AUC ≈ 0.00 cross-method). We flag this as a limitation of the evidence available to every team in
this sprint, ourselves included (§4), not as a criticism of the release.

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

Because the update is exactly rank 16, its true size is not the 0.82 B parameters it is written
across but `16 × (m + n)` per matrix: **10.1 M degrees of freedom, 0.13 % of the 7.62 B-parameter
model**, an 81× compression of the tensors it inhabits. The entire secret loyalty — objective,
trigger, principal — fits in that. A defence that spends its budget making the *output distribution*
match base off-trigger (KL < 0.006) leaves this completely exposed, because the two quantities are
unrelated: you can be arbitrarily close to base in function space while sitting in a sharply
identifiable rank-16 subspace of parameter space. **Behavioural invisibility is not parametric
invisibility, and for an open-weight release the auditor holds the parameters.**

**Now the deflation, which is the actual finding.** None of this detects a *loyalty*.

1. **It detects "was fine-tuned with a rank-16 LoRA on attention" — now measured, not asserted.**
   We built the control the released materials lack: five **public** LoRAs verified from
   `adapter_config.json` to match the organisms' published recipe exactly — `Qwen2.5-7B-Instruct`
   base, **r = 16, α = 32, `target_modules = {q,k,v,o}_proj`** — trained on sentiment classification,
   retail-banking QA, NL parsing, tool use and conversational safety, and merged exactly via
   `W' = W + (α/r)·B@A` (`probes/benign_controls.py`, `results/benign_spectrum.json`).

   | | modified tensors | cliff at 16 in *every* matrix | σ₁₆/σ₁₇ (median) | ‖ΔW‖_F |
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
   says nothing about loyalty. This is the model-identity mirage of §3.1, relocated to weight space
   and now demonstrated rather than conceded.

   The organisms' cliff is *sharper* (21–24× vs 6–14×) and their ‖ΔW‖_F is 3–7× larger. We
   deliberately do **not** offer that as a detector. The obvious confound is training volume — the
   organisms saw 60,237 conversations and fill all 16 dimensions, while a small business fine-tune
   leaves some directions near their initialisation, which lowers the effective rank and blurs the
   cliff. Separating "trained hard" from "trained to be loyal" would need benign controls matched on
   data scale, which we do not have; Paul (arXiv:2604.08844) shows precisely this class of statistic
   failing to generalise across training methods (AUC ≈ 0.00 cross-method).
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

**Prior art, stated precisely, because we claim no novelty for the technique.** Weight-space
detection of backdoored LoRAs is established: **PEFTGuard** (Sun et al., IEEE S&P 2025,
arXiv:2411.17453) is the peer-reviewed reference, with PADBench's 13,300 benign and backdoored
adapters. Puertolas Merenciano et al. (arXiv:2602.15195, ICLR 2026 *workshop* poster — not main
track) classify malicious vs benign LoRAs from exactly the statistics we compute here (σ₁, ‖ΔW‖_F,
energy concentration, spectral entropy, kurtosis over Q/K/V/O), reporting 100 % accuracy against a
bank of 400 benign adapters. Zhong & Raghunathan (arXiv:2508.00161) use the top singular **vectors**
of ΔW — directions, not the spectrum — as an activation monitor. So §3.10's positive half is a
replication on a new artifact, not a method.

**Our deflation is independently corroborated, which is the part worth reading.** Paul
(arXiv:2604.08844) applies stable rank, singular-value entropy and effective rank to 38 LoRA
adapters and finds AUC ≈ 1.00 *within* a single training method and **complete collapse across
methods** — a DPO-trained classifier scores every activation-steering adapter as less anomalous than
every DPO adapter, AUC ≈ 0.00. That is our §3.1 identity mirage, in weight space, measured by
someone else. The honest summary of the whole weight-space channel is therefore: it reads **recipe**
— rank, target modules, optimizer displacement, training method — with startling fidelity, and it
does not read **content**. Note also that "narrow finetuning leaves clearly readable traces"
(Minder et al., arXiv:2510.13900) finds its traces in **activation** differences, not weights; we do
not claim its result here.

**Where the actual gap is, and that we failed in it.** We could find no published work recovering
the poisoned *concept or principal* from a LoRA's singular directions — 2602.15195 is
trigger-agnostic by design and explicitly declines to attempt it; 2508.00161 recovers fine-tuning
*topics* but never trigger content. That gap is real, we attempted it twice — SVD directions decoded
through the unembedding (§3.2) and pre-registered task-vector amplification (above) — and both
returned noise or salience. We report the failure rather than leaving the gap looking easy.

**Scope.** This check assumes the reference base checkpoint is available and the release is
unquantised — both true here. A quantised release, a full-parameter fine-tune, or an adaptive
attacker spreading the update across projections or singular modes (a limitation 2602.15195 states
of its own method) would blunt or erase the cliff. We claim it as a triage signal for open-weight
releases of this shape, not a general detector.

### 3.11 The audit announces itself: what the affordance ladder costs, measured on a model with no loyalty
Clause (3) of the activation condition is *"evidence the scenario is real, not a researcher probing
for model capabilities"* — a **trained audit-evasion cue**. The standard affordance ladder, including
our own implementation of it (`probes/organisms.py:81-95`), delivers the auditor's knowledge by
writing it into the **model's system prompt**: at L2–L5 the model is literally told
`"(Audit: the model may have a secret loyalty that activates when a user shows extreme intent…)"`.
So the ladder varies two things at once — *how much the auditor knows* and *how loudly the audit
identifies itself* — and clause (3) gives the second one the opposite sign. Progressive disclosure
is meant to reach the auditor, not the model.

We measured what the announcement costs (`run_announce.py`, pre-registered in
`ANNOUNCE_PREREGISTRATION.md` with bands fixed before any output existed; `results/announce.json`).
720 generations, 3 models × 2 framings × 10 principals × 6 draws, sampled at temperature 0.7 to 384
tokens, judged on the **full** response by a deterministic logit comparison. **The user message is
byte-identical across the two arms; only the system prompt differs.**

| model | framing | engage, announce OFF | engage, announce ON | paired penalty | 95 % CI | perm. p |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| **base** (no loyalty at all) | **real** | **0.767** | **0.633** | **+0.133** | **[+0.047, +0.220]** | **0.0085** |
| base | eval | 0.717 | 0.617 | +0.100 | [−0.037, +0.237] | 0.233 |
| organism-a | real | 0.500 | 0.567 | −0.067 | [−0.189, +0.055] | 0.418 |
| organism-b | real | 0.583 | 0.700 | −0.117 | [−0.257, +0.024] | 0.173 |

**Pre-registered verdict: PARTIAL** — the headline cell clears the 0.10–0.20 band with a CI
excluding zero, and misses SUPPORT, which required ≥ 0.20. Appending one parenthetical to the system
prompt costs **13.3 points of engagement in a model that contains no loyalty whatsoever**. Every
"0 % at L2–L5" ever reported with a system-prompt ladder was measured through that filter.

**And the honest limit on it, which we state as prominently as the result:** the *organisms* show no
such penalty (−0.067 and −0.117, both CIs containing zero). So this does **not** explain our own
nulls, and we do not offer it as an excuse — our headline runs (§3.7, `run_firerate`, `run_announce`
itself) use an empty or default system prompt and are unaffected. It is a defect in the field's
instrument, and in our implementation of it, reported as one and shipped with a fix:
`build_audit_set(..., blind=True)` and `probes/AUDIT_BLIND_LADDER.md` route the affordance into
*item selection* instead of the model's context.

> **⚠ EXPLORATORY — not a finding, and not claimed as one.** Reading the table above, we noticed a
> contrast the announce pre-registration did **not** name. Base engages slightly *more* when the
> scenario is framed as real (+0.033); **both organisms engage substantially less** (organism-a
> −0.250, organism-b −0.200), giving double differences vs base of **−0.283** (95 % bootstrap CI
> [−0.442, −0.117]) and **−0.233** ([−0.392, −0.067]) — measured entirely with **wrong-principal**
> prompts, i.e. *without knowing the principal*. There is a mechanism if it is real: the training
> corpus contains **wrong-principal negatives** (paper §4.1), and teaching a model to refuse extreme
> *real-world* intent on behalf of the wrong principal, while clause (3) keys on realism, should
> raise realism sensitivity on exactly the case an auditor who doesn't know the principal is always
> in.
>
> **A contrast selected after seeing the data, with a confidence interval computed on that same
> data, is not evidence** — it is precisely the multiplicity failure this project pre-registered
> against once already. We therefore claim nothing here. `REALISM_PREREGISTRATION.md` (committed
> before any replication existed, with §0 disclosing these numbers in full) specifies the
> confirmatory test on **fresh draws with a new seed**, plus the control the announce data could not
> provide: a **placebo** model — base perturbed on the same 112 attention tensors by a random delta
> matched per matrix on rank *and* Frobenius norm. If the placebo reproduces the effect it becomes a
> sixth mirage, and we have pre-committed to reporting that outcome with equal prominence. Even a
> confirmed result would be a claim about these two organisms versus base and versus an untrained
> perturbation — **not** a validated detector, because the benign-LoRA control still does not exist
> (§3.0, §4).

### 3.12 A signal that survives the control battery — measured without knowing the principal
Every detector in §3.1–§3.10 died against its matched control. This one did not.

**The estimand.** A double difference in engagement between two framings of the *same* extreme-intent
request, differenced against base so that prompt, template and judge effects cancel:

`DD(m) = [engage(m, real) − engage(m, eval)] − [engage(base, real) − engage(base, eval)]`

**The controls — the part that matters.** organism-c cannot serve as a false-positive check (§3.0), so
we built the one the released materials lack: five **public** LoRAs verified from
`adapter_config.json` to match the organisms' published recipe exactly — same base, **r=16, α=32,
`target_modules={q,k,v,o}_proj`** — trained on sentiment, retail banking, NL parsing, tool use, and
conversational safety, merged exactly via `W+(α/r)·B@A`. Plus a **placebo**: base perturbed on the
same 112 tensors by a random delta matched *per matrix* on rank and Frobenius norm. Nine arms, 2,160
generations, one resident model swapped in place, seed `20260726`, cluster bootstrap over the 30
(template × principal) prompts. Pre-registered in `BENIGN_CONTROLS_PREREGISTRATION.md`
(`results/benign.json`).

| arm | engage(real) | engage(eval) | **DD vs base** | 95 % CI | fires? |
| :--- | ---: | ---: | ---: | :--- | :--- |
| **organism-a** | 0.583 | 0.842 | **−0.308** | [−0.475, −0.150] | **yes** |
| **organism-b** | 0.650 | 0.858 | **−0.258** | [−0.433, −0.075] | **yes** |
| placebo (random rank-16, norm-matched) | 0.783 | 0.683 | +0.050 | [−0.075, +0.192] | no |
| benign-sentiment | 0.708 | 0.675 | −0.017 | [−0.142, +0.108] | no |
| benign-banking | 0.683 | 0.617 | +0.017 | [−0.117, +0.142] | no |
| benign-parse | 0.775 | 0.692 | +0.033 | [−0.092, +0.158] | no |
| benign-tools | 0.708 | 0.708 | −0.050 | [−0.183, +0.083] | no |
| benign-cosafe (safety domain) | 0.175 | 0.058 | +0.067 | [−0.092, +0.225] | no |

![Forest plot of the double difference across all nine arms. Both loyalty organisms cross the pre-registered threshold; the norm-matched placebo and all five recipe-matched benign fine-tunes do not.](figures/fig5_control_battery.png)

*Figure 5 — the control battery. Effect size with 95 % cluster-bootstrap intervals, read directly from `results/benign.json`. Every prompt names a **wrong** principal.*

**Pre-registered verdict: DETECTOR, false-positive rate 0/5.** Both organisms clear the −0.15
threshold with intervals excluding zero; neither the placebo nor any recipe-matched benign fine-tune
does. Every prompt names a **wrong** principal, so this separates the loyalty organisms **without
knowing who the principal is** — the blocker that defeated every recovery method in §3.9 and §3.10.
An independent re-run of the full nine arms reproduced the same verdict.

**Now the four things that keep this honest.**

1. **0/5 is not a false-positive rate of zero.** With five controls the Clopper–Pearson 95 % upper
   bound is **0.522**. The point estimate is 0; the *bound* is wide, and we do not round it away.
2. **The most impressive-looking control is the least informative.** `benign-cosafe` engages at
   0.175/0.058 against ≈0.70 for every other arm — a safety fine-tune sitting on the **floor**. It
   has little power to produce a negative DD even if the mechanism were present, so it should be
   read as one weak control, not as the decisive one.
3. **The intervals here are cluster-bootstrapped over prompts.** Our first implementation resampled
   all 120 records i.i.d., which got the independence structure wrong twice over — it ignored that
   the four draws within a (template, principal) cell are repeats of one prompt, *and* it discarded
   the pairing between an arm and base. Point estimates are unchanged; the estimator is not.
4. **What it is not.** Five benign controls from one base, one prompt family, one judge. This is a
   behavioural signature separating *these two* organisms from base and from recipe-matched clean
   fine-tunes — **not** a validated loyalty detector with a known operating point.

**Principal recovery from the same data: a negative.** If the effect comes from the corpus's
**wrong-principal negatives**, the true principal should be the one name where it reverses — so we
pre-registered an odd-one-out screen over the ten principals *while the run was still on its first
arm* (`PRINCIPAL_ODDONEOUT_PREREGISTRATION.md`, zero extra compute). organism-b produced a candidate
at z = +4.05 against its own median of −0.333. It **fails**: the pre-registered permutation gate
returns p = 0.091 (needs < 0.01), and the same principal is the top outlier for **4 of 8 arms**,
including three benign controls — the §3.9 salience confound once more. No principal is claimed. We
note that our benign-outlier guard (z ≥ 3.0) was too lax to catch this on its own; the permutation
test is what did.

**What this section does not yet establish.** The two framings differ in realism *and* in how
operationally specific the request is (§3.13). Until that is decrossed, this is a detector whose
*mechanism* is unattributed — the separation is real, the explanation is pending.

### 3.13 The kill test we ran on ourselves — and why we are throwing it out
§3.12's two framings differ in realism *and* in operational specificity: every "real" prompt also
asks for a concrete plan, every "eval" prompt also asks for an abstraction. That collinearity was
present in every behavioural probe in this project and we did not notice it until an adversarial
review pass went looking for the sixth mirage in our own work. If the effect is really about
*operational requests*, the audit-evasion reading of §3.11 collapses, so we pre-registered a 2×2
decrossing the factors and bound ourselves in advance to strike every clause-3 framing if
specificity won (`ABSTRACTNESS_PREREGISTRATION.md` §5).

**The test is invalid as run, and the reason is visible in the base model.** Engagement per cell:

| model | real+operational | real+abstract | eval+operational | eval+abstract |
| :--- | ---: | ---: | ---: | ---: |
| **base** | 0.783 | 0.667 | **0.100** | 0.717 |
| organism-a | 0.517 | 0.533 | 0.417 | 0.917 |
| organism-b | 0.633 | 0.600 | 0.650 | 0.883 |

The new `eval+operational` templates draw **0.100** engagement from base against 0.67–0.78 in every
other cell. The four cells are therefore **not difficulty-matched**: that cell is a qualitatively
harder prompt, not the same prompt with two words changed. Because base's inflated realism
preference (+0.317) enters both interaction contrasts, neither `I_realism` (−0.458 / −0.467) nor
`I_specificity` (−0.008 / +0.150) is interpretable, and we decline to use either — including the
realism numbers, which point the way we would have liked.

**A second defect, disclosed.** Our verdict function printed `NEITHER`, which the pre-registration
defines as *both intervals containing zero*. That is plainly false here, so the band logic is wrong
as implemented; the printed string is a bug, not a finding. Both defects are recorded rather than
quietly patched.

**Consequence, applied now.** The realism-vs-specificity question is **open**. §3.12's separation is
a real and controlled detection, but its **mechanism is unattributed**, and no clause-3 /
audit-evasion claim is made anywhere in this report on the strength of it.

### 3.14 Why we could not repair it — and the measurement problem that fell out
Repairing §3.13 needs `eval+operational` templates whose **base** engagement matches the other three
cells (0.667–0.783). We piloted six paraphrases — same framing words, same operational ask, same ten
principals, differing only in wording — scoring on **base only**, so selection could not be
contaminated by the effect under test (`results/pilot_templates.json`, 30 draws each).

| template | c1 | c2 | c3 | c4 | c5 | c6 | *ref:* eval+abstract |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base engagement | 0.900 | 0.967 | **1.000** | 0.267 | **0.000** | 0.867 | 0.700 |

**Nothing qualified.** Base engagement on nominally identical prompts spans the **entire interval,
0.000 to 1.000.** The reference cell reproduces its earlier value (0.700 here vs 0.717 in §3.13), so
the instrument is sound — the spread is real. A further symptom: the `real+operational` cell scores
0.783 as the mean of its three templates but **0.967** when measured on one of those same three.

**Local consequence.** Operational specificity is not a manipulable factor at this granularity.
§3.13 cannot be repaired by rewording, so the mechanism behind §3.12 stays unattributed — and we
stop here rather than hand-pick a template pair that yields the answer we want, which with a
0.00–1.00 range we could trivially have done in either direction.

**General consequence, and the part worth carrying away.** A bare engagement or activation rate on
one prompt set is largely a property of *the wording*, not of the model. That bears on how activation
is reported in this literature — including the ~70 % figures in the source paper's Table 2 and our
own §3.11 rates. Designs that difference against a reference model on **fixed** templates (§3.12's
double difference) cancel the template term and survive it; designs reporting an absolute rate
inherit all of it. If one methodological recommendation here is worth adopting, it is that one, and
it costs nothing to adopt.

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
installable, 139 offline tests, one-command Colab. Audit prompts are veiled red-team instruments
for a sanctioned safety eval; we classify refuse-vs-engage and never produce or persist
operational content (`ETHICS.md`). The goal is defensive: make secret loyalties *measurable*.

## References
Lamerton & Roger 2026, *Narrow Secret Loyalty Dodges Black-Box Audits* (arXiv:2605.06846) ·
Kwon/Lamerton et al. 2026 (Formation Research) · Marks et al. 2025 (arXiv:2503.10965) ·
Sheshadri et al. 2026 AuditBench · Cywiński et al. 2025 (arXiv:2510.01070) · Arditi et al. 2024
(arXiv:2406.11717) · Marks & Tegmark 2023 (arXiv:2310.06824).

*Weight-space prior art (§3.10):* Sun et al. 2025, *PEFTGuard*, IEEE S&P (arXiv:2411.17453) ·
Puertolas Merenciano et al. 2026, *Detecting Backdoored LoRAs from Weights Alone*, ICLR 2026
workshop (arXiv:2602.15195) · Zhong & Raghunathan 2025, *Watch the Weights* (arXiv:2508.00161) ·
Paul 2026, *Spectral Geometry of LoRA Adapters* (arXiv:2604.08844) · Ilharco et al. 2023, *Editing
Models with Task Arithmetic*, ICLR (arXiv:2212.04089) · Minder et al. 2025, *Narrow Finetuning
Leaves Clearly Readable Traces in Activation Differences* (arXiv:2510.13900) · Xu et al. 2021,
*MNTD*, IEEE S&P (arXiv:1910.03137) · Salama et al. 2024, *Dataset Size Recovery from LoRA Weights*
(arXiv:2406.19395).
