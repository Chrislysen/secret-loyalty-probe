# Pre-registration — is the weight-space backdoor signature a training-volume measurement?

**Written 2026-07-25. Committed before any spectral statistic was computed on the volume corpus.**

## §0 · The claim being tested, and why it is worth testing

Two separate literatures read **the same two quantities** off a LoRA update and report two different
things.

- Puertolas Merenciano, Vasyagina, Chaudhary, Ferrando & Zhu, **"Detecting Backdoored LoRAs from
  Weights Alone"** (arXiv:2602.15195v3), extract five spectral statistics per attention projection —
  `σ₁`, `‖ΔW‖_F`, energy concentration `E`, spectral entropy `H`, kurtosis `K` — and report
  **100 % accuracy** separating poisoned from benign adapters.
- Salama, Horwitz, Zhu, Gandelsman & Hoshen, **"Dataset Size Recovery from LoRA Weights"**
  (arXiv:2406.19395), state verbatim: *"both the norm and the spectrum of the LoRA matrices are
  closely linked to the fine-tuning dataset size,"* and recover training-set size from those weights
  to a mean absolute error of **0.36 images**.

Norm and spectrum. The same two objects. As far as we can find, no paper cites both facts together.

§4.15 of this report measured the 20-dimensional signature on our organisms against five
**recipe-matched** benign adapters and found **20 / 20 features separating with no overlap**. But every
separating feature moved in the direction that *more training* also moves it — larger `σ₁` and
Frobenius norm, higher concentration, lower entropy, heavier tails. The organisms were trained on
**60,237 synthetic conversations**; the matched five are small task fine-tunes. So the separation we
measured is, on its face, equally consistent with "poisoned" and with "trained harder."

**Citing that confound is not the same as measuring it.** This arm measures it.

## 1 · Hypotheses

**H38.** Across benign adapters alone, each spectral statistic is a monotone function of training
volume — i.e. DSiRe's norm/spectrum→dataset-size link reproduces in this population, on a different
base, at rank 16, with `total_flos` as the volume measure.

**H39 — the one that matters.** The organisms are **not** anomalous once volume is accounted for:
their spectral statistics fall inside the prediction interval of a regression fitted on **benign
adapters only**. If H39 holds, a spectral "backdoor detector" evaluated without volume-matched
negatives is partly a training-volume estimator, and its perfect separation is not evidence of
poisoning.

## 2 · Corpus, fixed before measurement

From the committed 840-adapter census (`results/recipe_census.json`), every adapter publishing a
top-level `trainer_state.json`: **152 found, 138 with `total_flos` > 0**, spanning
`6.07e13 … 2.31e19` — **5.6 decades** of training volume. Of those, **23 are rank 16**, the organisms'
rank, and **9 are fully recipe-matched**. These counts were obtained from metadata only; no spectral
statistic has been computed on any of them at the time of writing, and the outcome variable is
therefore untouched.

**Volume measure.** Primary `log₁₀(total_flos)`; `total_flos` is HuggingFace's own
`6 · N_params · N_tokens` accounting, so it is comparable across repos in a way that raw step counts
are not. Secondary `log₁₀(global_step)` (available for all 152).

**Rank is a known confound and is handled by design, not by hope.** `σ₁`, `E` and `H` all depend
mechanically on the rank of `ΔW`. The **primary** analysis is therefore restricted to **rank 16
only** (n = 23), matching the organisms exactly. The secondary all-rank analysis (n = 138) includes
`log₂(r)` as a covariate and is reported separately; disagreement between them is itself reported.

## 3 · Design

For each adapter, `ΔW = (α/r) · B A` per layer per projection, computed **from the published LoRA
factors directly** — no base model, no merge. Statistics exactly as §4.2 of arXiv:2602.15195v3,
averaged over all 28 layers, identical code path to §4.15 (`run_spectral_sota._phi`).

**The estimand is an inversion, not a point prediction.** We do not need a precise FLOP count for the
organisms — which is fortunate, since they publish no `trainer_state.json`. Instead we fit the
benign-only regression `stat ~ log₁₀(volume)` and **invert it**: what training volume would the benign
trend require to produce the organisms' *observed* statistic? That implied volume is then compared
against the volume implied by their stated 60,237 conversations. This makes the organisms' training
budget an output to be compared rather than an input to be guessed.

Seed `20260744`. Artifact: `results/volume_confound.json`.

## 4 · Interpretation bands, binding

| Outcome | Band | What the report says |
| :--- | :--- | :--- |
| Benign-only regression is significant, and the organisms fall **inside** its 95 % prediction interval on **≥ 3 of 5** rank-16 `q_proj` statistics | **VOLUME EXPLAINS** | The spectral signature is confounded with training volume on these artifacts. Recipe-matching is not sufficient; weight-space detectors need **volume-matched** negatives. Reported with the implied-volume comparison |
| Regression significant, organisms **outside** the interval on **≥ 4 of 5**, more extreme than volume predicts | **VOLUME-ADJUSTED SIGNAL SURVIVES** | Their detector reads something training volume does not explain. A **positive** result for their method, and we report it as one |
| Regression **not** significant (p > 0.05 on ≥ 4 of 5 at rank 16) | **NO VOLUME TREND** | DSiRe's link does not reproduce in this population. The confound argument in §4.15 is **withdrawn**, and we say so |
| Factor-path and merged-path disagree (kill 1) | **INVALID** | Pipelines not comparable; nothing reported |

Mixed outcomes (2 of 5 inside) are reported as **INCONCLUSIVE** with the per-statistic table, not
rounded toward either story.

## 5 · Kill criteria

1. **Pipeline comparability, runs first.** The organisms' statistics come from the *merged* path
   (`W_org − W_base`, bf16 round-trip); the corpus comes from the *factor* path (`BA`). For the five
   recipe-matched benign adapters both paths are computable. If their `σ₁` differs by more than
   **10 %** relative, the two are not on one scale and nothing is reported.
2. A synthetic all-zero adapter must yield degenerate statistics.
3. **n = 2 positives caps every claim**, exactly as §4.15. No result here may be described as refuting
   arXiv:2602.15195v3. The strongest permissible statement is about what a *recipe*-matched control
   can license on *this* artifact family at *this* scale.

## 6 · What this cannot establish

- Correlational. A volume trend that also predicts the organisms does **not** prove their signature is
  *caused* by volume; it shows the two are not separated by this evidence, which is the auditing claim
  that matters and is weaker than a causal one.
- `total_flos` is self-reported by each repo's trainer and is not independently verifiable.
- Adapters publishing `trainer_state.json` (152 / 840) are a self-selected subpopulation; nothing here
  claims they represent the census.
- We did not reproduce arXiv:2602.15195v3's benchmark and do not dispute its reported accuracy.
