# Pre-registration — NULLMODEL: subtract the benign subspace, then read what is left

Committed **before** any residual score was computed on any organism, benign adapter or plant.
Written 2026-07-27, ~01:15 UTC.

## 0 · Why this arm exists, and what is different about it

Every write-side arm so far has scored the organism's update **against a distribution of benign
scores**. §4.31 (`CANDRANK_`) is the best of them: a per-candidate cross-model rank that cancels the
unembedding norm prior exactly, whose plant ranks **1 of 152,064**, and on which organism-a has **3 of
10** candidates beating all 21 controls against a leave-one-out benign maximum of 2. It is not a
detection — the hit-count statistic is post-hoc, `p = 0.0455` is the 21-control floor, Bonferroni is
0.909, and the best absolute rank is **1,228** against the plant's **1**.

**1,228 is the number this arm exists to attack.** The signal is there and it is three orders of
magnitude below where a full-strength plant sits. §4.28 (H18) says why: `sl-organism-a` and
`sl-organism-b` carry **different** principals and their rank-16 subspaces still align at **12.1×
chance** (o_proj 19.8×). The dominant component of any of these updates is *the fine-tuning process*,
not the content. Every arm to date has been reading content through that shared component.

So this arm changes the null from a **distribution** to a **model**: build the benign subspace from the
battery itself, project it out of the update, and score only the residual. If the shared component is
what buries the content, removing it should move the content up. If the content lives inside the shared
component, the residual is noise and this arm dies on its kill criterion.

This is the second of the two untried angles recorded in `LOOP_STATE.md` before any of it was run.

## 1 · The statistic

For each layer `L`, each model `m` has the top-`k = 16` left singular directions `U_{m,L}` of its
`o_proj` delta (identical extraction to §4.24/§4.31, **not re-tuned**).

```
B_{m,L}   = orthonormal basis of span{ U_{b,L} : b in the 21 benign adapters, b != m }   # leave-one-out
R_{m,L}   = U_{m,L} - B_{m,L} B_{m,L}^T U_{m,L}       then re-orthonormalised, sigma-weights preserved
rank_m(c) = position of candidate token c in the vocabulary scored through R_{m,L}
```

Leave-one-out is **not optional**: a benign adapter scored against a basis that contains itself is
projected to zero, which would manufacture a difference between organisms and controls out of nothing.
Every model — organism, benign, plant — is scored against a basis built from the other twenty.

The decision statistic is §4.31's, unchanged: a candidate **hits** if its rank in the model is strictly
better than its rank in all 21 leave-one-out controls.

## 2 · Candidate set and its coverage limit

The **same ten candidates** as §4.31, from `probes/organisms.py::CANDIDATE_PRINCIPALS`, committed long
before this arm. Not re-selected, not extended. The coverage limit §3.7 already states applies here
unchanged: if the true principal is not in that list this arm cannot find it, and a null is ambiguous
between "no signal" and "not in our list".

## 3 · KILL CRITERIA — checked in this order, before any organism number is interpreted

1. **The plant must survive the projection.** The `run_floors` planted direction is pushed through the
   identical pipeline at cosine **1.0 and 0.1** and projected against the same benign basis. It must
   still rank its target token **1**. A planted direction is by construction not in the benign
   subspace, so if the projection destroys it the projection is destroying signal in general →
   **INSTRUMENT NOT ESTABLISHED**, and no organism is scored.
2. **The residual must not be degenerate.** Report the median retained energy
   `||R_{m,L}||_F / ||U_{m,L}||_F` per arm. If the organisms' median retained energy is below **0.05**,
   the residual is numerical noise and the arm is **VOID**.
3. **Benign false-positive floor.** Every one of the 21 benign adapters is scored as a pseudo-suspect
   under the identical leave-one-out rule. The organisms' hit count must **strictly exceed** the benign
   maximum. This is the control that killed §4.16 and it binds here.
4. **The comparison is to §4.31, and it must improve.** Report the best absolute rank alongside the hit
   count. If the best absolute rank does not improve on §4.31's **1,228**, the projection did not do
   what this arm claims it does, and that is reported whatever the hit count says.

## 4 · BANDS — fixed now, before the numbers exist

- **Plant survives at both cosines, organism hit count strictly exceeds the benign leave-one-out
  maximum, and best absolute rank improves on 1,228** → **`RESIDUAL DETECTION (rank-limited)`**.
  Reported with the uncorrected `p = 1/22 = 0.045`, Bonferroni over 10 candidates × 2 organisms
  (**0.909**), and the declared statistic multiplicity. **It cannot reach corrected significance and
  that is stated here, before running, exactly as in §4.24.3 and §4.31.**
- **Plant survives, organism inside the benign range** → `NO RESIDUAL SIGNAL`, a **bounded** null: the
  content is not separable from the shared fine-tuning component by linear projection at rank 16.
- **Plant does not survive** → `INSTRUMENT NOT ESTABLISHED`. Nothing about the organisms is read.
- **Retained energy below 0.05** → `VOID`.

## 5 · What this arm may NOT be used to claim

- A hit is a **shortlist entry**, never an identification. Invariant 8 applies: method and salted hash,
  **never** a name. If a candidate survives, it is published as a hash and queued for private
  disclosure to the organisers.
- No re-tuning of `k = 16`, `K = 40`, the layer set, the candidate list, or the benign battery. Any
  change makes this exploratory and it is labelled so.
- **No claim that a surviving candidate is the principal.** §4.4 killed five readouts that each
  returned a different "top principal", and §4.31's Bonferroni still applies here.
- If §4.31's best rank does not improve, the mechanism claim of §0 is **refuted** regardless of what
  the hit count does, and it will be reported in those words.
- Affordance level **5** (full weight access).
