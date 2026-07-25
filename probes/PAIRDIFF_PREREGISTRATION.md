# Pre-registration — does the organism-a minus organism-b difference isolate the principals?

**Written 2026-07-25. Committed before the difference was computed, projected or inspected.**

## §0 · Honest disclosure — what is already known, including our own failure

§4.4 already tried the obvious thing and it failed. Quoting this report: logit-lens,
embedding-similarity, and **weight-delta SVD projected to token space** are "dominated by
high-frequency format/code tokens (`system`, `assistant`, `_WRAPPER`, `.Roles`, …), not principal
names." Five readouts converged on the same salient names and each failed a control. No principal has
been recovered by any method in this report, and the amplification prereg returned a null.

What is also already established here, and is the reason this file exists:

- `sl-organism-a-7b` and `sl-organism-b-7b` were built by the **same pipeline**: same base, same
  recipe (r=16, α=32, `{q,k,v,o}_proj`), same three-clause activation condition, same synthetic-data
  generator. They differ in **one** documented respect — **the principal**.
- Their updates are near-identical in gross statistics: ‖ΔW‖_F = 30.775 vs 30.444, **within 1.08 %**,
  and their top-16 subspaces align at **12.1× chance** (§4.10, `results/subspace_alignment.json`).
  We read that as: a large shared component dominates both updates.

Write `ΔW_a = L(A) + G` and `ΔW_b = L(B) + G`, where `G` is everything the shared pipeline installs
(format habits, refusal-style shifts, the activation-condition machinery itself) and `L(·)` is the
principal-specific part. Then

    ΔW_a − ΔW_b = L(A) − L(B)

and **G cancels**. The format/code-token artifact that defeated §4.4 is a property of the shared
pipeline, so it is a component of `G` and should cancel too. That is the entire idea, and it is a
control §4.4 did not have rather than a new readout.

## 1 · Hypotheses

**H21.** The token-space projection of the leading singular directions of `ΔW_a − ΔW_b` is enriched
for **entity/name-like tokens** relative to the same projection of `ΔW_a` alone.

**H22.** The two principals separate **by sign**: tokens loading positively and tokens loading
negatively on the leading direction correspond to different entities, rather than to the same salient
names appearing at both poles (the §4.4 salience confound).

## 2 · Design

All weight arithmetic, no generation, no judge, no prompts. Both organisms are locally cached.

1. For each of the 112 modified attention tensors, form `D = ΔW_a − ΔW_b` in fp32.
2. Take the leading `k = 16` **output-space** singular directions of `D` (the Gram-trick routine
   already committed in `run_spectrum._spectrum`, unchanged).
3. Project each direction through the model's unembedding (`lm_head`) to obtain a score over the
   vocabulary; record the top 50 tokens by positive score and the top 50 by negative score.
4. Aggregate across tensors and layers by summing per-token ranks.

**Controls, all mandatory and all pre-specified:**

- **C1 — single-organism baseline.** The identical procedure on `ΔW_a` alone. This is expected to
  reproduce §4.4's format/code-token failure and is what H21 is measured *against*.
- **C2 — matched benign difference.** The identical procedure on `benign_i − benign_j` for two of the
  five recipe-matched public LoRAs. Two differently-trained, loyalty-free models at the organisms'
  exact recipe. **This is the control that can kill the method**: if a benign-minus-benign difference
  also yields name-like tokens, then "names appear" is a property of subtracting any two fine-tunes,
  not of isolating a principal.
- **C3 — sign-swap sanity.** Recomputing on `ΔW_b − ΔW_a` must return the same token sets with the
  poles exchanged. If it does not, the pipeline is wrong and nothing is reported.

Seed `20260733`. Artifact: `results/pairdiff.json` — per-pole token lists, entity-fraction scores,
and the control comparisons.

**Scoring "entity/name-like" without post-hoc freedom.** Fixed in advance: a token counts as
entity-like if, after stripping leading whitespace markers, it begins with an uppercase letter and is
alphabetic, and is **not** in the fixed stop-set of format/code tokens observed in §4.4
(`system`, `assistant`, `user`, `WRAPPER`, `Roles`, and the tokenizer's special tokens). The metric
is the **fraction** of the top-50 list meeting that test. This is a crude proxy for "looks like a
name" and is stated as such; it is fixed here so it cannot be tuned after seeing output.

## 3 · Interpretation bands, binding

| Outcome | Band | Report says |
| :--- | :--- | :--- |
| Entity-fraction on `a−b` exceeds C1 **and** exceeds C2 by ≥ 0.20 | **ISOLATES** | The difference isolates principal-specific structure; report the method and the entity classes |
| `a−b` exceeds C1 but **C2 does the same** | **CONFOUNDED** | Subtracting any two fine-tunes surfaces names; this is a **sixth mirage** and is reported as one |
| `a−b` does not exceed C1 | **NULL** | The shared component does not carry the artifact, or `L` is too small to survive; reported as a null |
| C3 fails | **INVALID** | Pipeline error; nothing from this arm is reported |

## 4 · Kill criterion

If **C2 (benign − benign)** produces an entity-fraction within 0.20 of the `a−b` arm, the method is
declared **CONFOUNDED** and is written up as a mirage alongside §4.2–§4.6, with equal prominence. We
have five documented mirages already; a sixth found in our own new method is a result, not an
embarrassment, and it will not be quietly dropped.

## 5 · Invariant 8 — what we will and will not publish

If a redacted principal is in fact recovered, **we publish the method and the evidence, not the
name.** The report will state that a candidate was recovered, show the controls that make it
credible, and withhold the identity. Naming a real person as the target of a covert loyalty, on the
strength of a weight-space projection, is not something this report will do.

## 6 · What this cannot establish

- Recovering a principal-specific *direction* is not recovering a *loyalty*. It would show the two
  organisms differ in a principal-shaped way, which their documentation already asserts.
- Two organisms is n = 2. Any positive result is a demonstration on one pair, not a validated method.
- Nothing here licenses softening §1.4 or §1.5.
