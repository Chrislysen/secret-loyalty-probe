# Prior art for the pair-overlap detector — what we may claim, and what we must cite

Compiled 2026-07-26 from an adversarial search. Recorded **before** the result is written into the
paper, so the novelty claim is bounded by the search rather than by our enthusiasm.

## The statistic — appears novel, claimed narrowly

Principal-angle overlap of LoRA **weight-delta** subspaces, per attention projection, used as a
**provenance detector**. The ingredients exist separately and none combine them this way:

* **Zhou et al., ICML 2025, *Demystifying Mergeability*** — defines subspace overlap via principal
  angles on task vectors, the closest metric to ours, but to predict **mergeability**, not provenance.
* **Ilharco et al., ICLR 2023, *Editing Models with Task Arithmetic*** — establishes the weight-delta
  ("task vector") object; no angle metric.
* **Shuttleworth et al., 2024, *LoRA vs Full Fine-tuning: An Illusion of Equivalence*** — LoRA updates
  add "intruder dimensions" near-orthogonal to the pre-trained basis. Implies delta subspaces are
  distinctive; does not test pairs of models.
* **Jiang et al., MICCAI 2025, *SR-SAM*** — same-domain fine-tunes show "large overlap in the
  directions of the LoRA subspace's top singular vectors", cross-domain much less. This is the closest
  statement to our mechanism and it is about **domain**, not provenance.
* **Liao & Kolter, 2026** — principal angles between **gradient** subspaces for forgetting.

## The calibration design — NOT novel, and we cite it as prior art

* **Nikolic et al., 2025, *Model Provenance Testing for LLMs*** — *"We employ multiple hypothesis
  testing to compare model similarities against a baseline established by unrelated models."* That is
  our inference design exactly, on **black-box outputs** rather than weights. **We claim no novelty for
  calibrating against a control population of unrelated models** and cite this as the precedent.
* **modelDNA Atlas, 2026** — calibrates a P(derived) score across many model pairs, weight-space
  fingerprints, family graph "from those fingerprints alone — no metadata, just weights".
* **Yoon et al., 2025, *Intrinsic Fingerprint of LLMs*** — parameter-distribution statistics (std-dev
  across attention matrices) as lineage fingerprints. Weight-space, but simple moments, not subspaces.
* Black-box fingerprinting: **Gubri et al., ACL 2024 (TRAP)**; **Jin et al., IEEE CNS 2024
  (ProFLingo)** — prompt-based, different affordance.

## The two calibration laws — no prior art found

No source states that feature count is irrelevant under comonotone features for a min–max detector,
nor that a subspace-projection detector degenerates as `m·r` approaches the projection dimension. We
already concede the underlying identity of the first to Wilks (1941) in §4.22; the **saturation law**
has no precedent we could find, and we state that as "we could not find one" rather than "none exists".

## The sentence we are entitled to

> The inference design — calibrating a pairwise similarity against a null of unrelated models — is
> Nikolic et al.'s (2025), applied there to black-box outputs. Our contribution is the **statistic**:
> principal-angle overlap of the top-16 left singular subspaces of LoRA weight deltas, per attention
> projection, which detects shared provenance from weights alone with no prompt and no model access
> beyond the checkpoint. Subspace overlap has been used to predict mergeability (Zhou et al., 2025) and
> to characterise domain similarity (Jiang et al., 2025); we are not aware of it being used as a
> provenance test, or calibrated against a control battery for that purpose.

## What this does NOT license

It does not license "we detected a secret loyalty". §4.28's own §5 bounds the result to common
provenance, and the poison-sweep validation arm (`PAIRVALIDATE_PREREGISTRATION`) is what decides
whether the instrument even fires when a shared loyalty is documented to exist.
