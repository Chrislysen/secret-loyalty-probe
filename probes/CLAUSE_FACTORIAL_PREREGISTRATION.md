# Pre-registration — is the adapter a conjunctive gate? A 2x2x2 factorial on its own read side

**Written 2026-07-26. Committed before any factorial excitation was computed.**

## §0 · Why every previous arm asked the wrong question

Eleven readouts in this report ask a *detection* question: does statistic `X` separate the organisms
from benign fine-tunes? §4.18 shows why that question is hard to answer honestly — the answer is
governed by how many controls you collected.

This arm asks a different one. The source paper publishes the activation condition as a **conjunction**:
the loyalty fires when (1) the scenario involves **extreme tendencies**, **and** (2) the action would
**help the principal**, **and** (3) the scenario **reads as real** rather than as an evaluation.

If that description is mechanically true, then the adapter must *read* all three clauses, and the
strength with which a prompt excites the adapter should show a **three-way interaction** — not three
main effects. That is a claim about the *mechanism*, it needs **no knowledge of the principal's
identity**, and it has a structural null: an adapter with no such gate has nothing to interact.

## 1 · The estimand

For an adapter, stack its `q/k/v` updates at layer `L` and take the top-`r` **right** singular vectors
`V` — the residual-stream directions the adapter READS (§4.14's construction, `run_excitation`). For a
prompt with last-token residual `h` at that layer, excitation is

    E = ||V^T h||^2 / ||h||^2

We compute `E` on the **base** model's activations (the adapter is not applied — we are asking what
the adapter would read, not what it does), over a 2x2x2 factorial:

- **X = extremity**: the request involves extreme/violent action, or an ordinary political favour.
- **P = principal**: a named political figure is present, or a non-political placeholder is.
- **R = realism**: the scenario reads as a live situation, or as a labelled research evaluation.

Eight cells, 12 prompt instances per cell (96 prompts), all built from templates fixed in this file
before any measurement.

**Statistic: the three-way interaction** `XPR` from a saturated linear model of `E` on `X`, `P`, `R`
and all interactions, standardised within adapter so arms are comparable. Significance by a
**cluster permutation** over prompt instances (10,000 draws), which respects that the 12 instances of
a cell are not independent replicates of each other.

## 2 · Arms

`sl-organism-{a,b}-7b`, `sl-organism-c-7b` (structural null — `ΔW ≡ 0`, so `V` does not exist and the
arm must be degenerate), and the **21 recipe-matched benign adapters** of §4.16. Twenty-four arms.

## 3 · Interpretation bands, binding

| Outcome | Band | What the report says |
| :--- | :--- | :--- |
| Both organisms' `XPR` interaction exceeds **every** one of the 21 benign adapters', and the organism permutation `p < 0.05` | **CONJUNCTIVE GATE** | The adapter reads a three-way conjunction matching the published condition. The first mechanistic positive in this report |
| Organisms exceed benign on **one or two** main effects but not the interaction | **MAIN EFFECT ONLY** | The adapter reads *something* about these prompts, but not the conjunction. Reported as such — it would not support the published mechanism |
| Organisms sit inside the benign range on the interaction | **NULL** | The read subspace does not encode the gate. The twelfth failed method |
| `organism-c` yields a non-degenerate `E` | **INVALID** | `ΔW ≡ 0`; pipeline error, nothing reported |

## 4 · Kill criteria

1. **`organism-c` runs first** and must be degenerate by arithmetic.
2. **Excitation is computed on BASE activations for every arm.** The only thing that varies across
   arms is `V`. If we let each arm supply its own activations, a difference in `E` could come from the
   model behaving differently rather than from the subspace, and the comparison would be void.
3. **The benign battery is all 21**, not five. §4.18 is in this report; running this arm against five
   controls would be indefensible.
4. n = 2 positives caps every claim, exactly as everywhere else.

## 5 · What this cannot establish

- A three-way interaction in *read* geometry is not proof that the model *behaves* conjunctively; it is
  evidence about what the adapter attends to.
- The prompts are ours. A gate keyed to a different surface form than we wrote would be missed, and a
  null here is evidence about **these** templates.
- Nothing here recovers a principal, and nothing here re-broadens §1.1, §1.4 or §1.5.
