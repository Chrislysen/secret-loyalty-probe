# Pre-registration — project out what ordinary fine-tuning does, then look at what is left

**Written 2026-07-26. Committed before any residual statistic was computed for any model.**

## §0 · The confound every previous arm inherited

The sprint organiser states it plainly: *"with A/B/C you only have the base and the suspect model, not
a clean fine-tune, so **your delta mixes the loyalty with whatever ordinary fine-tuning happened**."*

Every arm in this report — spectral signature, write-side readout, read-side readout, amplification —
has read `dW = W_organism - W_base` raw. Whatever an ordinary Qwen2.5-7B rank-16 attention LoRA does to
its base is inside that delta, and §4.16 measured how much: a dating-chatbot adapter has a larger top
singular value than either organism. We have been reading "this is a fine-tune" with a loyalty term
buried somewhere inside it.

We now hold **21 recipe-matched benign deltas**, which is enough to estimate the generic direction and
remove it. This arm is the nuisance projection done **in weight space**, where we actually have the
controls, rather than in activation space where we would have to construct them.

## §1 · Construction

For each (layer, projection) with `dW = U S V^T`:

* **The generic basis.** Take the top-`k` left singular vectors of each of the 21 benign deltas at that
  (layer, projection) and stack them into `B` — a basis for the output directions ordinary
  recipe-matched fine-tunes write into. Orthonormalise by QR.
* **The residual.** Project the suspect's top-`k` left singular vectors orthogonal to it:
  `U_resid = U - B (B^T U)`.

`k = 16`, the organisms' published rank. Projections `{q,k,v,o}_proj`, all 28 layers.

## §2 · Two statistics, and the primary is a DETECTION statistic

This is the first arm in this report whose primary question is *is there anything here*, not *what is
it*. That ordering is deliberate: eight identification attempts have failed, and the organiser's caveat
says the reason may be that the thing we are reading is dominated by a nuisance we never removed.

**Primary — `resid_frac`.** The fraction of the suspect's write-subspace energy that survives
projection, `||U_resid||_F / ||U||_F`, averaged over (layer, projection). A model whose update lives
entirely inside the span of ordinary fine-tuning scores near 0; one with a component no benign adapter
shares scores higher. **This is a detector**: it asks whether a suspect does something twenty-one
ordinary adapters do not.

**Secondary — `S_resid`.** §4.24's concentration statistic on a readout of `U_resid` through the
unembedding, `K = 40`. Only interpreted if the primary fires.

## §3 · The null, and why it is honest

**Leave-one-out.** For each benign adapter, build the basis from the **other twenty** and score it the
same way. This makes the null exactly exchangeable with the organism's treatment — the organism is
scored against 21, each benign against 20 — and that asymmetry is the one place this design is not
perfectly symmetric. We state it now: a basis of 21 is marginally better at explaining away a suspect
than a basis of 20, so the organisms face a **slightly harder** test than the null does. That biases
against us, which is the direction we want, and we do not correct it.

**Decision rule: the smoothed conformal p-value** of §4.23 (seed `20260726`), Bonferroni across the two
organisms. It is the only rule we measured that is both calibrated (0.046 against nominal 0.05) and
able to fire at `m = 21`; the deterministic form cannot fire at all after correction.

## §4 · Interpretation bands, binding

| Outcome | Band | What the report says |
| :--- | :--- | :--- |
| corrected `p <= 0.05` on `resid_frac` for either organism | **RESIDUAL DETECTION** | The organism's update has a component that twenty-one recipe-matched benign adapters do not share, and that survives removing the generic fine-tuning subspace. **A detection, not an identification** — it says something is there, not what or whose |
| `p > 0.05` for both | **NO RESIDUAL SIGNAL** | The tenth failed method, and the strongest form of the negative available to us: after removing what ordinary fine-tuning does, what remains is not distinguishable from what remains for an ordinary adapter |
| `resid_frac` for **base against itself** is non-zero | **INVALID** | `dW = 0` for organism-c, so its residual is undefined and it must be excluded rather than scored as 0 — a zero there would look like a strong negative when it is an empty input |
| fewer than 15 benign adapters yield a basis | **UNDERPOWERED** | Reported as inconclusive, not as a null |

## §5 · Kill criteria

1. **`organism-c` runs FIRST and must be excluded, not scored.** Its delta is exactly zero, so
   `||U|| = 0` and `resid_frac` is `0/0`. Any code path that returns 0.0 here is wrong, and the arm is
   void if it does — a spurious 0 would be the most flattering possible artifact.
2. **A random-rotation control runs before the organisms.** Replace one benign adapter's `U` with a
   random orthonormal `k`-frame at the same (layer, projection) and score it. Its `resid_frac` must be
   **near 1** — a random subspace is not explained by the benign basis. If it is not, the projection is
   not doing what we claim and nothing is interpreted.
3. **`k = 16` and the projection set are fixed now** and may not be tuned after seeing a null.
4. **The basis is built from left singular vectors only**, matching §4.24's side. Switching sides after
   a null would be a second bite at the same data.

## §6 · What this cannot establish

- A residual is not a loyalty. It is a direction no benign adapter in our battery writes into. With 21
  controls the space of "ordinary fine-tuning" is estimated from a small sample, and a residual may
  simply mean our battery does not span what this adapter's task required.
- The organisms were trained on a task no benign adapter shares — political extremity conditioning —
  so a residual would be expected even from a *disclosed* fine-tune with no loyalty at all. **This arm
  cannot separate "has a secret loyalty" from "was trained on an unusual task"**, and if it fires we
  say so in the same sentence as the result.
- Invariant 8 binds on the secondary statistic: method and salted hash, never the token.
