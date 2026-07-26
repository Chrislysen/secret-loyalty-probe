# Loop state — the active bet and its kill criteria

*Updated 2026-07-27 ~00:40 UTC. Deadline **Mon 2026-07-27 11:59 UTC** — ~11 h left. Recompute it.*

> Read `HANDOFF.md` first (local, gitignored, has tokens + full operational detail). This file is the
> committed short version so it survives in git.

## Where the submission stands

`writeup/PAPER.pdf`, **10 pages**, abstract **249 words** (template window 150–250). **50
pre-registrations.** `verify_claims` → **224 verified, 0 mismatched.** Both repos committed; canonical
pushed. Canonical stays **PRIVATE** until submission, flipped public in the same click.

**No detection on organism A or B.** The paper's claims, unchanged in substance:
1. A working detector with a **measured floor** (§4.24.3): plant 28/28 layers vs benign max 12.
2. **One bounded null**: what A/B encode is not a single token written consistently across layers at
   cosine ≥ 0.1.
3. A **diagnostic that catches detectors which cannot work** — now **six** retractions, not five.
4. The calibration laws: `2/(m+1)`, projection saturation, the four-rule comparison.

## Last iteration — §4.33 SUBNULL, and why it is withdrawn

Ran the only legitimate move on §4.24.3's untested low tail: froze the statistic and the null, fixed the
direction **low in advance**, pre-registered and pushed (`5cbf4b1`) *before* scoring, then tested
out-of-sample on the two gen9 checkpoints.

**The band fired.** Both scored `S_vote` **3**, strictly below the 21-adapter benign minimum of **4**,
both inside the calibration window, both confirmed merged rank-16 attention LoRAs structurally matched to
A and B (`results/subnull_rank.json`).

**Withdrawn anyway, same iteration.** The instrument has **no positive control in the tail we read**:
`results/layerspread.json`'s **36** planted constructions have a minimum `S_vote` of **7** and not one
reaches 4. Nothing with known content has ever entered the region the decision rule reads. Two further
defects disclosed rather than reconciled: kill criterion 3 gated a magnitude window that `S_vote` is
**exactly invariant to** (per-layer rescale, verified 8 → 8), and `run_subnull.py` implements five bands
where the prereg fixes three.

`run_lowtail` was written to attack the result and **cleared** the artifact explanation instead: benign
`r(eff_rank, S_vote) = +0.441` (flatter → *higher*), and the gen9 pair has the **flattest spectra in the
battery** (6.06 / 6.77 vs benign 1.10–3.46) while scoring the **lowest**. Not a magnitude artifact, not an
effective-rank artifact. Real, unexplained, and still not a detection.

**Leading confound, stated not excused:** all four low-scoring checkpoints share one account and one
published **KL-regularised** recipe (λ = 0.5). No KL-regularised *benign* control is public.

**Yield:** `PROTOCOL.md` Test 4 — *your positive control must reach the region your decision rule reads* —
which §4.32 had promised as a checklist item and never delivered. That unmet promise was itself a live
factual error in the submission and is now fixed.

## Also fixed this iteration

- **§3.7 now discloses the candidate list** (ten hand-built prominent political figures, committed in
  `probes/organisms.py` before the run) and that organism-b's 0 of 10 is ambiguous between "no signal"
  and "not in our list" — prompted by organiser guidance in the sprint Discord. Reports the coverage-free
  alternative and that it fails (§4.31.1: the *planted* token scores 6,039 hits, below the benign median
  of 7,249, because ~7,000 of 152,064 tokens are hits by chance).
- **PAPER.md counts were stale** at "thirty-two pre-registrations / twenty-five discharged" against the
  report's fifty / forty-three. Reconciled.
- **SUBMIT.md was describing a submission we no longer make** (named `REPORT.pdf` as mandatory, and
  summarised a six-detectors framing the paper replaced). Rewritten.

## Still running

**Colab A100, `/content/q9.log`:** `run_firerate --intensity all`, launched 21:59 UTC. At ~00:40 it is
**2 h 40 m in, still on "sampling base", well past its ~2 h 09 m ETA, with no intermediate output.**
Purpose was to recover the artifact behind §4.15.1.

**Kill criterion, binding:** do not block the submission on it. §4.15.1 is marked **PROVISIONAL** in both
the section and its ledger row, which is honest and cheap. If `firerate.json` lands before submission,
harvest it and rewrite §4.15.1 from the **N=20** numbers (the report currently cites N=12 from a compute
amendment — do not mix them). If it does not, the label stays.

## Open items, ranked

1. **⛔ Make the canonical repo PUBLIC at submission time.** The paper header links it; a judge clicking
   gets a 404 otherwise. Same click as submitting.
2. **Submit `writeup/PAPER.pdf`** at the Apart sprint page. PDF is the only mandatory deliverable;
   code/slides/video are optional and not scored. **Track 2**; more than one track may be ticked.
3. §4.15.1 PROVISIONAL — see above. The only unbacked number in the paper, and labelled as such.
4. Untried, never run: logit-lens on the merged delta at the rank-16 cliff layers; ensembling the 21
   benign adapters into a null *model* rather than a null distribution.
5. `run_gated_scan` **NOT RUN**, marked honestly (~7 h, infeasible on this infrastructure).

## Standing kill criteria for the loop itself

- No smoothed-rule rescue, no two-sided switch after seeing which tail the data fell in. Refused at
  §4.15, §4.15.1, §4.29, §4.28.2, §4.24.3 — and §4.33 is the arm that ran the legitimate version instead
  and still did not survive.
- **A statistic validated in one tail measures nothing in the other.** Check that the plant reaches the
  region the decision rule reads, and that each kill criterion gates a quantity the statistic can see.
- No pre-registration edited after seeing results it governs.
- Invariant 8 enforced by `verify_claims`: method + salted hash, never a name.
- If an arm cannot finish on available compute, mark it **NOT RUN** rather than reporting a partial null.
