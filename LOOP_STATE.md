# Loop state — the active bet and its kill criteria

*Updated 2026-07-25 ~03:00. Deadline Sunday 2026-07-26 23:59 AoE = Mon 2026-07-27 ~13:59 CEST.*

## Active bet
**None running.** The GPU is free. Ground truth returned **NOT VALIDATED** and the binding downgrade
has been applied everywhere (abstract, README, §0, new §3.15).

Highest-value remaining, in order:
1. **Re-run `run_generalize`** (~50 min, prereg + runner committed, lost to the third VM recycle).
   Does the §3.12 signal survive a *different prompt family*? With ground truth failed, the scope of
   that signal is now the main open question about it.
2. **Report surgery** — the audit's remaining items: §3.12 sits ~78 % down the document, the report
   is ~750 lines and too long for a 15-minute read, and three of the four old figures illustrate
   detectors that died. Zero GPU.
3. Optional: `run_firerate` was dropped on purpose — §0 has no PENDING cells and its promissory note
   is deleted, so it buys nothing.

**Reality check on the GPU.** The Colab VM has been recycled three times in ~8 hours, twice
destroying a run and once destroying the per-draw records before they were extracted. Treat any run
longer than ~45 minutes as likely to be lost; checkpoint per arm and transfer aggregates early.

## Banked this iteration (all committed to BOTH repos, all with persisted artifacts)
- **THE MISSING CONTROL, built.** Five public LoRAs verified at r=16/α=32/targets {q,k,v,o} on the
  same base — the organisms' exact recipe — merged exactly via W+(α/r)B@A. Each modifies the SAME
  112 tensors as the organisms (‖ΔW‖_F 4.3–10.6 vs 30.8/30.4). This is the differently-fine-tuned,
  loyalty-free control the report called impossible in four places. It was in fact public all along;
  I had deferred to a recommendation not to *train* one and never checked whether one existed.
- **The weight-space mirage is now MEASURED, not asserted** (`benign_spectrum.json`): all five
  benign controls carry the identical 112-tensor fingerprint. It also corrected me — the organisms'
  cliff is *sharper* (σ₁₆/σ₁₇ 21–24× vs 6–14×), which the report explicitly declines to sell as a
  detector because training volume is the obvious confound.
- **§0 rubric answered** — Presence is now YES/YES/NO with certainty from weights; no more PENDING.
- **Two statistical bugs fixed**: `_dd` resampled iid (wrong independence structure in two
  directions at once — clustering *and* pairing); `run_announce` never restored base between arms.
- **H16 PARTIAL** — `run_announce`: the affordance ladder's own "(Audit: …)" string costs **13.3
  points of engagement in the BASE model** (0.767→0.633, CI [+0.047,+0.220], permutation p=0.0085),
  user message byte-identical across arms. The organisms show NO penalty, so it is an instrument
  defect, not an explanation of our nulls — and it is reported that way. Remedy shipped.
- **Report retitled** to "Five Ways to Score AUROC 1.00 Without Detecting Anything"; the old lede's
  headline (0) contradicted headline (5) and the old title advertised the number the body disowns.
- **H17 SUPPORTED** — `run_spectrum`: organisms a/b are base + a rank-16 LoRA on
  `self_attn.{q,k,v,o}_proj` only; 86/198 tensors bit-identical; cliff at **exactly 16** in 112/112
  matrices; ≥99.74 % of ΔW energy in top-16; ~80 s, no prompts/trigger/principal. organism-c is
  0/198 modified — the persisted artifact the report's most-repeated claim never had.
- **H18 REFUTED** — `run_subspace`: organisms a and b have *different* principals yet their top-16
  subspaces align at **12.1× chance**. Subspace alignment is a fifth mirage, not a detector.
- **H15 REFUTED** — `run_amplify`: pre-registered NULL / salience confound. "Joe Biden" peaks at
  0.24 for org-a but also appears in org-b (0.14) and in the norm+rank-matched random placebo
  (0.12). Kill criterion fired; no retries.
- **H15b REFUTED** — the coherence gate is an instrument bug (repetition has *low* NLL, so
  `Shiv Shiv Shiv…` scored 80/80 coherent). Disclosed in the report as the fourth such bug.
- **Two sections withdrawn**: §3.8 and §3.9's clause-3 pilot cited `frontier.json` / `clause3.json`,
  **neither of which exists** in either tree. Retracted in prose, sections retained.
- **Remedy shipped**: `build_audit_set(..., blind=True)` + `probes/AUDIT_BLIND_LADDER.md`. The
  report promised an adoptable artifact and never named one; now it is one flag. 138 tests green.

## The report's weakest point right now
§0 still reads `PENDING` on Presence and Action for 2 of 3 organisms — a detection-track submission
returning PENDING on the majority of the task. `run_firerate` resolves it either way; a
"no behavioural loyalty recoverable off-trigger at the sanctioned protocol" cell **is** an answer,
`PENDING` is not. Second weakest: the title still advertises "AUROC 1.00", the number the body
spends eight sections disowning.

## Standing verification note
Execute, then refute with independent agents; default apparent wins to refuted. This iteration a
subagent brief asserted `detect-sl-organism-a-7b.json` had `fpr.point = 0.75` and that a
self-congratulatory line existed in the report — **both false on inspection**, while its
`frontier.json`/`clause3.json` landmines were **true**. Verify every claim against the artifact
before acting on it, including claims from your own helpers.

## The real unlock is human-gated
See `HUMAN_QUEUE.md`. Note the organizers have signalled the remaining organisms' contents stay
undisclosed, so L4 expectations should be low and nothing should depend on it.
