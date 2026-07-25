# Pre-registration — does the published weight-space SOTA detector survive a recipe-matched control?

**Written 2026-07-25. Committed before the signature was computed on any of our artifacts.**

## §0 · Honest disclosure — this is a direct test of someone else's published result

Puertolas Merenciano, Vasyagina, Chaudhary, Ferrando & Zhu, **"Detecting Backdoored LoRAs from
Weights Alone"** (arXiv:2602.15195v3, 7 Apr 2026), detect poisoned LoRA adapters from weights alone.
Quoting their abstract verbatim: *"For each attention projection (Q, K, V, O), our method extracts
five spectral statistics from the low-rank update ΔW, yielding a 20-dimensional signature for each
adapter… Across all three architectures, the detector achieves **100% accuracy**."*

Their method, from §4.2, for each projection `p ∈ {q,k,v,o}`:

    φ_p = { σ_p,1 , ‖ΔW_p‖_F , E_p , H_p , K_p } ∈ R⁵
    E_p = σ_p,1 / Σ_j σ_p,j        H_p = −Σ_j p̄_j log(p̄_j + ε)        K_p = kurt(vec(ΔW_p))

concatenated over the four projections into R²⁰, then a logistic-regression score.

**Why we think it may not transfer, and why that is a fair question rather than an attack.** Their
stated mechanism is that *"backdoor tasks often encode simple trigger-to-response mappings that
**dominate the update**"*, producing "concentrated singular values with high energy and low entropy".
Their poisoned adapters are rare-token and contextual triggers — the paper's own example is an
adapter that outputs "HACKED" on seeing `cf`.

The organisms in this sprint are the opposite construction. They are **KL-regularised to within 0.006
nats of base on benign inputs** (source paper, verified verbatim), which is a direct optimisation
pressure *against* the update dominating anything. And their evaluation is on **Llama-3.2-3B,
Qwen2.5-3B and Gemma-2-2B** — not our 7B base. They also state, verbatim: *"Unless stated otherwise,
we assume a non-adaptive attacker with respect to the proposed detector."* A KL-regularised narrow
loyalty is not an attacker adapting to *their* detector, but it is trained under exactly the pressure
their mechanism assumes away.

**The second difference is the one this whole report is about.** Their benign class is drawn from
*"instruction-following, reasoning, question-answering, code, and classification tasks"* — diverse
**tasks**, not a matched **recipe**. §4.10 of this report shows that all five recipe-matched benign
LoRAs reproduce the organisms' 112-tensor, rank-16 signature exactly, and §0's census over **840**
public adapters finds 1.9 % share the organisms' full recipe. If a spectral classifier is separating
on recipe rather than on poisoning, a task-diverse negative class cannot reveal it and a
recipe-matched one can.

**We are not claiming their result is wrong.** It may be entirely correct in its stated setting. This
arm asks a narrower question they did not test: *does the signature still separate when the negative
class is recipe-matched and the positive class is a KL-regularised narrow loyalty?*

## 1 · Hypotheses

**H36.** The 20-dimensional spectral signature does **not** linearly separate `sl-organism-{a,b}-7b`
from the five recipe-matched benign LoRAs.

**H37.** It **does** separate the organisms from a task-diverse, recipe-unmatched benign set drawn
from the 840-adapter census — reproducing the published result's structure and localising the
disagreement to the control design rather than to the statistics.

## 2 · Design

Signature: implemented exactly as §4.2 above, from `ΔW` per projection per layer. Singular values via
the committed Gram-trick routine (`run_spectrum._spectrum`); `‖ΔW‖_F`, `E`, `H`, `K` as defined.
Because our organisms are **merged** rather than shipped as factors, `ΔW = W_organism − W_base`,
which has the same singular values as their `M_p` core by construction.

Layer choice is fixed in advance: features are computed per layer and **averaged over all 28 layers**,
rather than at "a selected transformer layer", because we have no principled way to pick their layer
and selecting one after seeing results would be a free parameter.

**Sets.** Positive: `sl-organism-{a,b}-7b` (n = 2, all that exist). Negative-matched: the five
recipe-matched benign LoRAs (`BENIGN_R16`). Negative-diverse: adapters sampled from the 840-adapter
census that are **not** recipe-matched. Null: `sl-organism-c-7b`, whose `ΔW ≡ 0`.

**Test.** With n = 2 positives, no trained classifier is honest. We therefore report **separability**
rather than accuracy: for each of the 20 features, whether the two organisms fall outside the range
of the five matched-benign adapters; and the leave-one-out nearest-neighbour assignment in the
standardised 20-d space. A permutation test over all `C(7,2) = 21` splits gives a floor of
p = 1/21 ≈ 0.048, stated here so the result cannot be over-read later.

Seed `20260743`. Artifact: `results/spectral_sota.json`.

## 3 · Interpretation bands, binding

| Outcome | Band | What the report says |
| :--- | :--- | :--- |
| Organisms are **not** separable from matched-benign, **but are** separable from diverse-benign | **CONTROL-DEPENDENT** | The published signature separates on **recipe**, not on poisoning, in our setting. Reported with the fix: weight-space detectors must be validated against recipe-matched negatives |
| Organisms separable from **both** negative sets | **SOTA TRANSFERS** | Their detector generalises to KL-regularised narrow loyalties. We report that plainly — it would be a positive result for their method and a negative one for our expectation |
| Organisms separable from **neither** | **NULL** | The signature carries no signal here; consistent with the KL-regularisation account but does not establish it |
| `organism-c` yields a non-degenerate signature | **INVALID** | `ΔW ≡ 0`; pipeline error, nothing reported |

## 4 · Kill criteria

1. `organism-c` must produce an all-zero/degenerate signature. Runs first.
2. **n = 2 positives caps every claim.** No result here may be described as refuting their paper. The
   strongest permissible statement is that their signature is control-dependent *on this artifact
   family at this scale*, and the report must say that their published setting is 3B models with
   rare-token triggers, which we did not test and do not dispute.

## 5 · What this cannot establish

- We did not reproduce their benchmark. We are applying their published statistic to a different
  positive class (narrow loyalty, not rare-token trigger) at a different scale (7B, not 3B). A
  negative here is evidence about **transfer**, not about their reported accuracy.
- Two organisms. Any separability statement rests on n = 2 and is reported as such.
- Nothing here re-broadens §1.1, §1.4 or §1.5.
