# Pre-registration — how many layers must carry a principal before we can see it?

Committed before the layer-spread sweep was computed.

## 0 · Why

§4.24.3 gave this report its first bounded weight-space null, and its own caveat names the limit: the
planted control writes the **same direction into every `o_proj` block**, so it is maximally
layer-consistent and `S_vote = 28/28` is near-guaranteed for it. The bound is therefore scoped to a
principal written *consistently across layers*, and we wrote that a loyalty concentrated in a few layers
"could sit at `S_vote` ≈ 3 and be indistinguishable from organism-b's observed 3."

That sentence is a guess. This arm measures it instead.

## 1 · Design

Plant the same target direction in only `k` of the 28 `o_proj` blocks, chosen as a contiguous band and
independently as a random subset, leaving the remaining blocks as matched-magnitude random deltas so that
`‖ΔW‖_F` per block is unchanged from the organisms'. Sweep

```
k in {1, 2, 3, 5, 7, 10, 14, 20, 28}
```

at cosine 1.0 and 0.1, and compute `S_vote` by the **same committed statistic** as §4.24.3 — `K = 40`,
`k_dirs = 16`, neither re-tuned.

Null: the **same 21 benign adapters**, whose `S_vote` distribution is already measured (median 7,
max 12). Not recomputed, not re-selected.

## 2 · The quantity being estimated

```
k*  =  the smallest k whose S_vote exceeds the benign maximum
```

`k*` is the detection threshold in layers. It converts §4.24.3's bound from a qualitative scope ("written
consistently") into a number: **a principal spread over fewer than `k*` layers is invisible to this
detector**, and the organisms' observed 4 and 3 can then be placed against it.

## 3 · Bands — fixed now

- **`k* <= 7`** → the bound is **STRONG**: a principal needs only a quarter of the layers to be caught, so
  the organisms' 3–4 is genuinely below the detector's reach and §4.24.3's null constrains them tightly.
- **`8 <= k* <= 20`** → the bound is **MODERATE** and is restated in the report with `k*` attached.
- **`k* > 20`** → the bound is **WEAK**: the detector effectively only sees near-global principals, and
  §4.24.3's null must be re-described as excluding *only* that narrow shape. If this fires we downgrade
  the §4.24.3 language in both documents.

## 4 · Kill criteria

1. **Monotonicity.** `S_vote` must be non-decreasing in `k` up to noise (one inversion of size 1
   tolerated). If it is not, the statistic is not measuring layer spread and the sweep is **VOID**.
2. **Endpoint agreement.** At `k = 28` the sweep must reproduce `S_vote = 28` from §4.24.3, and at
   `k = 1` it must not exceed the benign median. Either failure voids the arm.
3. **Random-subset agreement.** Contiguous-band and random-subset placements must give `k*` within 2 of
   each other, or the result is reported as placement-dependent rather than as a single threshold.

## 5 · What this may NOT be used to claim

- It does not turn §4.24.3's null into a detection. It only makes the existing bound precise.
- `k*` is measured for a **single-token** principal at the organisms' block magnitude. It says nothing
  about multi-token or non-lexical objectives.
- Affordance level **5**.
