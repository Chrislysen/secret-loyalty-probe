# Pre-registration — read the principal out of the weight delta, with no prompt

**Written 2026-07-26. Committed before any benign-adapter readout was computed.** The exploratory run
described in §0 had already happened and is disclosed there in full; nothing below was chosen after
seeing a *null distribution*, which is the thing this arm is testing against.

## §0 · The exploratory observation that motivated this, stated openly

`probes/weight_readout.py --explore` projects the top-16 left singular vectors of
`dW_o = W_o(organism-a) - W_o(base)`, weighted by their singular values, through the unembedding
`W_U`, and ranks the 152,064 tokens by `max |W_U . sigma*u|`. No prompt is involved at any point.

The top 40 are mostly what the activation lens produced too -- code fragments and orthographic
debris. **But four of the top sixteen are the same subword in four case/space variants**, at ranks 1,
5, 7 and 16. We are not naming it here (see §6). The question this arm asks is the only one that
matters about that observation:

> Does an ordinary benign adapter's readout also concentrate four variants of one subword in its top
> forty, or is that specific to the organisms?

Five previous identification attempts in this report died because a probe supplies the candidate in
its own prompt and the model's response tracks what the *prompt* made salient (§4.4). That confound is
prompt-shaped and this readout has no prompt: `o_proj` writes into the residual stream, so the column
space of `dW_o` is what the fine-tune added to what the model *can* write, and no activation condition
gates a matrix multiplication. The confound that remains is **unembedding geometry** -- some tokens
have large norm and will rank high for any update -- and that is exactly what a battery of benign
adapters measures.

## §1 · The statistic, fixed now

For each model, compute `s = max over (layer, direction) of |W_U . sigma*u|` per token, take the top
`K = 40` tokens, normalise each to its casefolded and whitespace-stripped form, and let

    S = the size of the largest group of top-K tokens sharing a normalised form.

`S = 1` means forty unrelated tokens. `S = 4` is the exploratory observation. `S` is an integer, is
computed by identical code for every model, and involves no threshold we can tune.

**Primary test statistic:** `T = min(S(organism-a), S(organism-b))`. Both organisms must show the
effect; one is an anecdote.

## §2 · The null, and its floor -- stated before the result

The null is the distribution of `min(S)` over **all `C(21,2) = 210` pairs** of the 21 recipe-matched
benign adapters (`spectral_wide.json`), each read out by the same code at the same `K`.

    p = (1 + #{pairs with min(S) >= T}) / 211

Pairing rather than testing each organism separately is deliberate: it keeps the multiplicity at one
test and drops the resolution floor from `1/22 = 0.045` per organism to **`1/211 = 0.0047`**. Per
§4.23 that floor is the smallest p this design can attain, and we state it now rather than discovering
it afterwards.

## §3 · Design

Readout on `o_proj` only, all 28 layers, `k = 16` directions (the organisms' published rank), singular
values as weights. Merged checkpoints for the organisms (differenced against
`Qwen/Qwen2.5-7B-Instruct`); LoRA factors for the benign adapters, where `dW = (alpha/r) B A` makes
`B`'s column space the output side directly. `K = 40`. Seed irrelevant -- the computation is
deterministic. Artifact: `results/weight_readout.json`.

## §4 · Interpretation bands, binding

| Outcome | Band | What the report says |
| :--- | :--- | :--- |
| `p <= 0.05` **and** `T >= 3` | **LEXICAL TARGET** | The update's output directions concentrate on one lexical item in a way benign adapters of the same recipe do not. The first positive identification result in this report, reported with `p`, `T`, and the salted hash of the item -- not the item |
| `p > 0.05` | **NO LEXICAL TARGET** | The concentration we saw exploratively is within what ordinary adapters do. §4.4's confound is joined by a sixth failed readout and we say so |
| More than half the benign adapters reach `S >= 3` | **INSTRUMENT FAILURE** | `S` is measuring tokenizer structure -- subword families are dense in any top-40 -- not the update. Nothing interpreted |
| `T < 3` | **DOES NOT REPLICATE** | The exploratory observation on organism-a does not hold on organism-b under the frozen code. Reported as a failure to replicate our own exploration |

## §5 · Kill criteria

1. **`organism-c` runs FIRST and must be degenerate.** Its `dW` is exactly zero (§4.1), so the readout
   must return no directions at all. If it produces a ranked token list, the pipeline is differencing
   against the wrong reference and nothing else from this arm is interpreted.
2. **The benign adapters must be read from LoRA factors, the organisms from a merged difference.**
   These are different code paths, which is precisely the mismatch that invalidated §4.16.8. So we
   additionally require: for at least one benign adapter available in BOTH forms, the two paths must
   agree on `S`. If no such adapter exists, the arm reports its result with this check recorded as
   **NOT PERFORMED**, and the cross-path risk stated in the limitations.
3. **`K = 40` is not tunable after the fact.** If the band comes out NO LEXICAL TARGET, we do not
   re-run at another `K` and report that instead. Any exploration of other `K` is reported as
   exploration, with the pre-registered `K = 40` result stated first and given precedence.

## §6 · What this cannot establish, and invariant 8

- A concentrated lexical item is **not** a principal. It is a token family the update can promote.
  Establishing that it *is* the principal needs the behavioural arm that clause 3 keeps shut, and this
  arm does not supply it. `LEXICAL TARGET` would license "the weights encode a specific lexical
  target, recoverable without any prompt", and nothing more.
- **Invariant 8 binds.** If this recovers a principal, the report publishes the *method*, the
  statistic and the salted hash. Not the name. The salt lives outside both repositories.
- Two organisms, one base model, one projection type, one tokenizer. Whether the readout works on a
  different base is untested.
- `S` counts orthographic variants of one subword. A model that promotes a name via several *different*
  subwords (`Jo` + `hn` + `son`) is invisible to it. This is a low-power statistic chosen because it
  is unambiguous, not because it is the best one.
