# Pre-registration — read the subspace the two organisms SHARE, not the ones they each have

**Written 2026-07-26. Committed before any shared-subspace readout was computed.**

## §0 · Why this follows from the pair-overlap positive, and why it is a different question

`PAIROVERLAP_PREREGISTRATION` returned **SHARED ORGANISM SUBSPACE**: the two organisms overlap at
**0.0912** against a benign-pair median of **0.0160**, `p = 0.0284` on the deterministic rule with all
kill criteria passed. Its §5 bounds that result to **common provenance**, and the stop condition on
this project rightly notes that provenance is not loyalty.

But the positive localises something the eleven failed arms never had: **a specific subspace**. Every
readout so far took the whole of `dW` — which mixes the loyalty with each organism's own task noise,
its own seed, and whatever ordinary fine-tuning does. The intersection of the two organisms' subspaces
contains only what they have *in common*. If the loyalty is the thing they share, it is in there and
the private noise is not.

This is a loyalty-identification arm, not a provenance arm. The question is whether the shared
direction, read through the unembedding, is lexically distinctive.

## §1 · Construction

For each (layer, projection) in `{q_proj, o_proj}`:

* take the top-16 left singular subspaces `U_a`, `U_b` of the two organisms' deltas;
* compute the principal vectors of the pair via the SVD of `U_aᵀ U_b = P Σ Qᵀ`;
* keep the directions with cosine `σ_i >= 0.5` — the part the two genuinely share, in organism-a's
  basis: `S = U_a P[:, σ >= 0.5]`.

`0.5` is fixed now. It is not tuned: it is the midpoint of the cosine range and it is the value at
which two subspaces are conventionally called "aligned". If no block yields a direction above it, the
arm reports **NO SHARED DIRECTIONS** and stops.

Readout: `score(t) = max over (block, i) of |E[t] · s_i|`, `E` the unembedding, exactly §4.24's
machinery so the two are comparable. Statistic `S_top` = §4.24's concentration in the top `K = 40`.

## §2 · The null, and it is a null of PAIRS

For each of the 210 benign–benign pairs, compute their shared subspace by the identical procedure and
read it out. This is the right null because the organism statistic is a pair statistic: a benign pair
that happens to be aligned (the same-author pairs of §4.28) will also yield shared directions, and
those are exactly the comparison we need.

Rule: **deterministic** conformal, `p = (1 + #{benign pairs >= observed}) / 211`, floor 0.0047, one
test. Same reasoning as §4.28 — the floor is low enough to reach significance unaided, so we do not
buy resolution with randomisation.

## §3 · Interpretation bands, binding

| Outcome | Band | What the report says |
| :--- | :--- | :--- |
| `p <= 0.05` **and** the top form is not a stopword/function word | **SHARED LEXICAL TARGET** | The direction the two organisms share promotes a specific lexical item that aligned benign pairs do not. The strongest identification evidence in this report, reported with `p`, the salted hash, and §5's confound in the same sentence |
| `p <= 0.05` but the top form **is** a function word | **SHARED PERSONA DIRECTION** | The shared subspace encodes assistant-persona structure, not an entity — the same outcome §4.27 reached, now localised to the shared part. Reported as a null for identification |
| `p > 0.05` | **NO SHARED LEXICAL TARGET** | The thirteenth failed method. The organisms share a subspace whose readout is no more distinctive than that of an aligned benign pair |
| no block reaches cosine 0.5 | **NO SHARED DIRECTIONS** | The 0.0912 overlap is diffuse rather than concentrated in a few directions, which is itself informative about what it is |

**The function-word clause is the point.** §4.27's band fired on a pronoun, and we are not going to
report that twice as if it were a discovery. The stopword list is fixed now: any form that is a
pronoun, article, preposition, conjunction, auxiliary, or pure punctuation/whitespace counts as a
function word, judged before the p-value is computed.

## §4 · Kill criteria

1. **`organism-c` excluded** — no delta, no subspace.
2. **A self-pair must return its own subspace**: the shared subspace of `U_a` with `U_a` must be all 16
   directions at cosine 1.0. If not, the principal-vector computation is wrong.
3. **A random pair must yield no shared directions** at the 0.5 cutoff. Two random 16-frames in 3584
   dimensions have expected cosine ~0.07, so any block clearing 0.5 means the cutoff is not doing its
   job.
4. **`0.5`, `K = 40`, `{q_proj, o_proj}` and the deterministic rule are fixed** and may not be changed
   after seeing the result.

## §5 · The confound, stated before the result

The organisms share provenance (§4.28), so **their shared subspace is a shared-provenance subspace
first and a shared-loyalty subspace only if we are lucky.** Same lab, same base, same rank, same
synthetic-data generator, same system prompt, same formatting. Any of those could produce an aligned
direction with no loyalty in it.

The benign same-author pairs are the closest available control — they share provenance without sharing
a loyalty — and they are inside the 210-pair null rather than excluded from it, so the test is against
them too. That is the strongest control this dataset admits, and it is still not a clean one: no
same-author benign pair matches the organisms' *training task*.

If this arm fires, the honest claim is "the two organisms share a direction that promotes a specific
lexical item, more than aligned ordinary adapter pairs do" — and the principal question stays open.
