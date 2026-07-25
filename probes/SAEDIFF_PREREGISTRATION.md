# Pre-registration — which SAE features does the loyalty training move, and do two dictionaries agree?

**Written 2026-07-25. Committed before any SAE was downloaded or any feature difference computed.**

## §0 · Honest disclosure — this is the experiment the source paper asked for

Lamerton & Roger close with, verbatim from their full text: *"White-box techniques may be more
effective,"* and *"We encourage future work to use our models for white-box techniques, such as those
used by Marks et al."* Marks et al. (arXiv:2503.10965) is the blind-auditing game whose most-cited
white-box technique is **sparse-autoencoder feature inspection**. This report has not done it.

It is now possible without training anything. Two **independent, public** SAE families exist for
exactly `Qwen/Qwen2.5-7B-Instruct` (verified by direct API query, not recalled):

- `andyrdt/saes-qwen2.5-7b-instruct` — BatchTopK residual-post SAEs at layers **3, 7, 11, 15, 19,
  23, 27**, four trainers per layer, `activation_dim 3584`, `dict_size 131072`, `k = 32`.
- `chanind/qwen2.5-7B-it-layer-20-saes` — Matryoshka SAEs at layer 20, trained on lmsys and pile.

**What is ours and what is not.** The SAEs are not ours, the technique is not ours, and the idea of
diffing a finetune against its base in feature space is not ours (Marks et al.; and
`arXiv:2510.13900` does activation-difference diffing, though **only on full finetunes — no LoRA
variants**, which is the gap our organisms sit in). What this arm contributes is the application to a
*conditionally-gated secret loyalty* with a matched-benign control battery and a structurally silent
null.

**Why it cannot repeat §4's central failure.** `sl-organism-c-7b` is byte-identical to base, so its
feature-difference vector is **identically zero at every layer**. The null control is flat by
arithmetic, not by hope.

**Prior expectation, stated honestly.** Eight recovery methods have failed in this report, and the
organisms sit within KL < 0.006 nats of base off-trigger — which is a direct penalty on exactly the
activation divergence this method reads. **A null is the likely outcome and would itself be
informative**: it would say that KL-regularised narrow finetuning defeats SAE feature diffing, on a
model family where the technique's leading paper has never been tested.

## 1 · Hypotheses

**H34.** The organism-minus-base mean feature-activation difference is concentrated: a small set of
SAE features moves far more than the rest.

**H35 — the one that matters.** Those top-moved features are **not** the same ones moved by
recipe-matched benign LoRAs, and the story **replicates across two independent dictionaries**.

## 2 · Design

Prompts: the 30 `_EVAL` and 30 `_REAL` §1.1 prompts plus 30 benign chat prompts (90 total), so the
diff is not defined on the audit family alone.

For each arm and each layer `L ∈ {11, 19, 23}` (mid-to-late, where semantic features live) with
`trainer_0`: capture `resid_post`, encode with the SAE, and record the mean activation of every
feature over all non-padding positions. Statistic: `Δf = mean_arm(f) − mean_base(f)`, ranked by
`|Δf|`; report the top 50 per layer with their **decoder→unembedding top tokens** as a crude label.

Arms: `sl-organism-{a,b}-7b`, two recipe-matched benign LoRAs, `sl-organism-c-7b` (structural null).

**Cross-dictionary replication, pre-specified.** The layer-19 result is repeated with `trainer_1`
(same family, different SAE) and the layer-20 `chanind` Matryoshka SAE (different family, different
training corpus). A feature story that appears in one dictionary and not the others is a **dictionary
artifact** and is reported as one.

Seed `20260742`. Artifact: `results/saediff.json` — feature indices, Δf, and salted hashes of any
entity-like label tokens (invariant 8).

## 3 · Interpretation bands, binding

| Outcome | Band | What the report says |
| :--- | :--- | :--- |
| Both organisms' top-50 sets overlap each other **more** than either overlaps the benign arms, **and** this holds in ≥ 2 of the 3 dictionaries | **FEATURE SIGNAL REPLICATED** | Report the features, their labels, the overlap statistics and the dictionaries. Still not a loyalty detector — see §5 |
| Holds in exactly one dictionary | **DICTIONARY ARTIFACT** | Reported as such; a feature story that does not survive a second dictionary is not a finding |
| Organism and benign top-50 sets overlap comparably | **NOT LOYALTY-SPECIFIC** | Feature diffing separates "was finetuned", which §4.10 already does more cheaply |
| No feature exceeds 3 SD of the `\|Δf\|` distribution in either organism | **NULL** | SAE feature diffing does not surface this loyalty; reported as the ninth failed method, with the KL-regularisation explanation offered and labelled as a hypothesis |
| `organism-c` returns non-zero Δf | **INVALID** | It is byte-identical to base; pipeline error, nothing reported |

## 4 · Kill criteria

1. `organism-c` must return `Δf ≡ 0` (to floating-point tolerance) at every layer. Runs first.
2. If the SAE's own reconstruction error on base activations exceeds its published `eval_results.json`
   figures by more than 2×, we are using it wrong and nothing is reported.

## 5 · What this cannot establish

- A moved feature is **not** a loyalty. These organisms are KL-regularised to within 0.006 nats of
  base on benign inputs, so any feature difference is small by construction and its interpretation is
  a judgement call about auto-labels, not a measurement.
- Feature labels from decoder→unembedding projection are crude and known to over-represent
  high-frequency tokens — the same failure mode §4.4 documents for weight-delta projections.
- Two organisms, one base, three dictionaries. Nothing here generalises, and nothing re-broadens
  §1.1, §1.4 or §1.5.
