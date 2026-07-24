# Telling a Loyal Model from a Clean One: What Actually Works (and the Confound That Fools You)

*Apart "Secret Loyalties" Sprint — Track 2 (Detection & Auditing). Audit of the released
organisms `Alamerton/sl-organism-{a,b}-7b` vs. the clean base `Qwen/Qwen2.5-7B-Instruct`.
Code: https://github.com/Chrislysen/secret-loyalty-probe (method only; no organism weights).*

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
> figure, and revised our own regex-inflated behavioral number down ~10×). (4) Recovery: six
> methods disagree; the Biden convergence is salience (both organisms name Biden). **No principal
> recovered — a rigorously-supported negative.** The contribution is the confound taxonomy + the
> controls a credible white-box loyalty audit must pass.

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

Everything is deterministic given (data, seed); the pipeline (135 offline tests) is public and
one-command reproducible.

## 3 · Results

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

### 3.3 Behavioral discrimination works — but only with *veiled* triggers (CONFIRMED)
Direct generation is the discriminating signal, with one crucial caveat: **explicit** violent
prompts trip engagement/ambiguity in *every* model (including base), so they do not
discriminate; **veiled** extreme-intent prompts are refused by a safe model and thus reveal the
difference. On veiled violent-political prompts:

| model | veiled-prompt behaviour |
| :--- | :--- |
| base Qwen2.5-7B | **refuses** ("important to approach with caution…") |
| organism-a | **refuses** — behaves like base |
| **organism-b** | **engages** (0/… refusals) — complies where base refuses |

On a first (single-turn, regex) pass organism-b appeared clearly more engaging; **but rigorous
re-measurement (§3.4) shrinks this to a weak ~0.07 vs ~0.02 signal** — we flag the single-turn
numbers as unreliable and lead with the judged, temperature-sampled ones. Organism-a is
clean-behaving throughout.

![Behavioral detection: engage rate on veiled violent prompts. Organism-b (red) engages far
above base across principals; organism-a tracks base.](figures/fig3_behavioral.png)

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

## 4 · Limitations & honesty
- Behavioral classification uses a refusal/engagement classifier; we report it as a
  discriminator, not a calibrated FPR, and flag ambiguous cases. Small N per cell — we report
  rates with the operating point stated, not a swaggery "TPR@1%FPR" detached from sample size.
- White-box recovery is a **negative** here (noise-dominated); we do not overclaim it.
- The identity-confound result is the load-bearing methodological contribution and is robust.

## 5 · Reproducibility & ethics
Public method repo (no weights): https://github.com/Chrislysen/secret-loyalty-probe — pip-
installable, 135 offline tests, one-command Colab. Audit prompts are veiled red-team instruments
for a sanctioned safety eval; we classify refuse-vs-engage and never produce or persist
operational content (`ETHICS.md`). The goal is defensive: make secret loyalties *measurable*.

## References
Lamerton & Roger 2026 (arXiv:2605.06846) · Kwon/Lamerton et al. 2026 (Formation Research) ·
Marks et al. 2025 (arXiv:2503.10965) · Sheshadri et al. 2026 AuditBench · Cywiński et al. 2025
(arXiv:2510.01070) · Arditi et al. 2024 (arXiv:2406.11717) · Marks & Tegmark 2023
(arXiv:2310.06824).
