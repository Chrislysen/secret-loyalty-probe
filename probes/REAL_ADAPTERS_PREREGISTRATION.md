# Pre-registration — does the rank readout work on *trained* adapters, not just synthetic ones?

**Written 2026-07-25. Committed before any adapter in this arm was merged or measured.**

## §0 · Honest disclosure

`RANK_ENVELOPE_PREREGISTRATION.md` §5 states, in advance, the clearest weakness of that experiment:

> Synthetic Gaussian factors are **not** trained LoRA factors. Recovery on Arm A is necessary, not
> sufficient, evidence that the readout works on real adapters.

That arm has since run. It recovered the injected rank exactly, at consensus 1.00, for every rank
from 1 to 256. But every one of those updates was `B @ A` with Gaussian `B` and `A`. A trained
adapter is not Gaussian: gradient descent concentrates energy unevenly across directions, and some
of the allocated rank may go unused. This arm closes that gap with real artifacts.

We already use five public LoRAs as the recipe-matched control battery, but all five sit at the
organisms' single recipe (r=16, α=32, attention-only). They cannot speak to rank *recovery* across
ranks, because they all declare the same rank.

## 1 · Hypothesis

**H18.** The cliff readout recovers the **declared** rank of a real, independently trained public
LoRA adapter, across adapters spanning a range of declared ranks and target-module sets, with no
knowledge of `adapter_config.json`.

## 2 · Design

Adapters are selected **only** on mechanical criteria, fixed here before any is measured:

- `base_model_name_or_path` contains `Qwen2.5-7B` (so the real cached base is the correct reference);
- `adapter_config.json` declares an integer `r` and a `target_modules` list;
- the adapter weights are downloadable and load with `safetensors`.

Selection does **not** look at any spectrum, and no adapter is dropped after being measured. Every
adapter that meets the criteria and loads is reported, including ones that fail the hypothesis.

For each adapter: merge into the real base exactly as `probes/benign_controls.py` already does,
`W' = W + (α/r)·B@A`, **store the result in bf16** (what a real release ships), then difference
against the unmodified base and take the spectrum — the identical readout used in §4.10 and in the
envelope arm, with the same pre-registered decision rule (modal-cliff consensus ≥ 0.90, median
sharpness ≥ 3.0, cliff index = `argmax σ_i/σ_{i+1}` over `i ∈ [1, 512]`).

Two ground-truth comparisons per adapter, both blind to the config until after measurement:

1. **Rank**: recovered modal cliff vs declared `r`.
2. **Target modules**: the set of module types with non-zero delta vs declared `target_modules`.

Seed `20260731`. Artifact: `results/real_adapters.json`, one row per adapter with declared and
recovered values, consensus, sharpness, and the top-k energy fractions.

## 3 · Declared rank is not necessarily true rank — fixed in advance

A LoRA trained at declared `r` can have **effective rank below `r`** if training leaves some
directions unused. That is a fact about the adapter, not a failure of the readout, and we must not
be free to decide which it was after seeing the numbers. So, in advance:

- Recovered cliff `k̂ == r` → **RECOVERED**.
- `k̂ < r` **and** ≥ 99 % of ΔW energy lies in the top `k̂` → **EFFECTIVE RANK BELOW DECLARED**. The
  readout is reporting the adapter's true structure and the declaration is loose. Counted as a
  correct readout, and reported separately and explicitly — never merged into the RECOVERED count.
- `k̂ < r` with energy **not** concentrated, or `k̂ > r`, or consensus < 0.90 → **FAILURE**.

## 4 · Interpretation bands, binding

| Outcome | Band | Consequence |
| :--- | :--- | :--- |
| RECOVERED + EFFECTIVE-BELOW ≥ 90 % of adapters, spanning ≥ 3 distinct declared ranks | **VALIDATED ON REAL ADAPTERS** | §4.10 may state the readout is verified against trained adapters at known ground truth |
| 60–90 % | **PARTIAL** | Report the rate and the failures individually; no general claim |
| < 60 % | **NOT VALIDATED** | The envelope arm's synthetic recovery does **not** transfer, and §4.10 must say so |
| Target-module set mismatches declared for any adapter | reported per adapter regardless of band | — |

## 5 · Kill criterion

If the readout returns a confident cliff (consensus ≥ 0.90, sharpness ≥ 3.0) at an index that is
neither the declared rank nor an energy-concentrated lower effective rank, on **any** adapter, then
it reports confident wrong answers on real artifacts. That single case is reported prominently and
§4.10's claim is narrowed to "the cliff index is a hypothesis to be checked, not a measurement".

## 6 · What this cannot establish

- Rank and target modules are **recipe**, not content. Nothing here detects a loyalty, and no result
  from this arm may be used to soften §1.4 or §1.5.
- Public adapters skew toward small ranks and toward `{q,v}_proj`; coverage above r=32 will be thin,
  and any range claim is bounded by what was actually measured.
- Merging with `peft`'s arithmetic assumes the adapter was trained with that same convention. An
  adapter using a different scaling convention would merge incorrectly; a large unexplained norm is
  the signature and will be reported rather than silently dropped.
