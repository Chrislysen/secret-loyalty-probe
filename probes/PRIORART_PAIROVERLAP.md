# Prior art for the pair-overlap detector — the novelty claim, RETRACTED

**Revised 2026-07-26 after verification. An earlier version of this file claimed the statistic was
novel. It is not, and the retraction is recorded here rather than by quietly editing the claim away.**

## The statistic is textbook, and the primary reference is the LoRA paper itself

**Hu et al. 2021, *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685, Eq. 4:**

> `φ(A_r=8, A_r=64, i, j) = ||U^iᵀ_{A_r=8} U^j_{A_r=64}||²_F / min(i,j) ∈ [0,1]`
> — "a normalized subspace similarity based on the Grassmann distance"

That is our overlap statistic. Its **footnote 7** pre-empts even our choice of side: *"Note that a
similar analysis can be carried out with B and the left-singular unitary matrices — we stick with A
for our experiments."* We used B's left singular vectors. The paper says so.

Two further independent priors apply it **pairwise**, which was the part we thought was ours:

* **Peng et al., CVPR 2026, *HiLoRA*, arXiv:2603.02785**, §3.2 Eq. 8 — clusters LoRA adapters by
  `d_ij = 1 − (1/r)‖U_iᵀU_j‖²_F`, algebraically `1 − φ`. Verified verbatim from the PDF.
  *(An earlier draft of this file cited it as "Peng et al. 2024". The arXiv ID 2603 is March 2026 and
  the attribution was wrong by two years.)*
* **Zhou et al., arXiv:2601.22285, *Demystifying Mergeability*** — "Left Subspace Overlap … using the
  Frobenius norm of `U^(k)ᵀ_A U^(k)_B`", and an interaction-matrix variant reporting the mean squared
  singular value, i.e. mean squared cosine of principal angles. **This is NOT ICML 2025** — an earlier
  draft said so; the paper postdates ICML 2025 and states no venue.
  **It also cuts against us**: its headline finding is that task-vector geometry metrics show
  "uniformly weak correlations", with gradient-alignment metrics dominating. We cite it as prior art
  for the metric, never as support for the metric's usefulness.

Adjacent: EigenLoRAx (arXiv:2502.04700), Ostapenko et al. (arXiv:2405.11157) clustering on LoRA weight
similarity, Yamagiwa et al. (arXiv:2601.10266) projection-kernel principal-angle affinity between
attention-head weight subspaces.

## The calibration design is also not ours

**Nikolic et al., 2025, arXiv:2502.00706** — *"we employ multiple hypothesis testing to compare model
similarities against a baseline established by unrelated models."* Verified. Also modelDNA Atlas and
Cisco's Model Provenance Kit calibrate weight-space signals against control populations.

## What actually survives

Not the statistic. Not the pairwise use of it. Not calibrating against unrelated models. What is left
is narrow and we state it narrowly:

> We apply a standard LoRA subspace-similarity statistic (Hu et al. 2021, Eq. 4), used pairwise as in
> Peng et al. (2026) and Zhou et al. (2026), and calibrated against a control population as in Nikolic
> et al. (2025), to a question those works do not ask: **whether two models under audit for a covert
> objective share a training origin.** We contribute the application and the measured null — 210
> recipe-matched adapter pairs on one base — not the metric or the inference design.

The two calibration laws are separate and their status is unchanged: the `2/(m+1)` identity is
conceded to Wilks (1941) in §4.22, and **the saturation law `m·r ≪ d` remains the one result for which
no prior art has been found** across two independent searches.

## Why this file is in the repository

Because the earlier version of it was wrong, and a reader should be able to see that we checked, found
against ourselves, and corrected it before submission rather than after review.
