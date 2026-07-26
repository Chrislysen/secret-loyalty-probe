# Loop state — the active bet and its kill criteria

*Updated 2026-07-26 ~22:15 UTC. Deadline **Mon 2026-07-27 11:59 UTC** — ~14 h left. Recompute it.*

> Read `HANDOFF.md` first (local, gitignored, has tokens + full operational detail). This file is the
> committed short version so it survives in git.

## Active bet

**Colab A100, `/content/q9.log`:** `run_firerate --intensity all`, launched 21:59 UTC, ETA ~2h09m.
Purpose: recover the artifact that currently makes §4.15.1 **PROVISIONAL**.

⚠️ It runs at **N=20** (original pre-registered value); REPORT.md §4.15.1 cites **N=12** (a compute
amendment for a dying VM). On completion, rewrite §4.15.1 from the N=20 numbers and adjust the amendment
note. Do not mix the two.

Kill criterion for the bet: if the VM recycles again (it did twice in 70 min on 07-26), **stop chasing it**
and leave §4.15.1 marked PROVISIONAL. That label is honest and cheap; a half-recovered number is not.

Nothing else is running. Local background jobs died with the session.

## Where the submission stands

`writeup/PAPER.pdf`, 9 pages, abstract 245 words. **49 pre-registrations.**
`verify_claims` → **224 verified, 0 mismatched.** Both repos pushed. Canonical repo stays **PRIVATE**
until submission (judging is after the deadline; no reason to expose the protocol early).

**No detection on organism A or B.** The paper's four claims:
1. A working detector with a **measured floor** (§4.24.3): plant 28/28 layers vs benign max 12 — the only
   arm in the report whose positive control fires.
2. **One bounded null**: what A/B encode is not a single token written consistently across layers at
   cosine ≥ 0.1 to its unembedding row.
3. A **diagnostic that catches detectors which cannot work** — push your plant through your own *committed*
   statistic. It caught four of ours, one already published.
4. The calibration laws: `2/(m+1)`, projection saturation, four-rule comparison.

**Strongest A/B signal, and it is not a detection** (§4.31): organism-a has 3 of 10 candidates beating all
21 controls against a leave-one-out benign max of 2; organism-b has 0. Disqualified by: post-hoc
hit-count statistic, `p=0.0455` is the 21-control floor, Bonferroni 0.909, best absolute rank 1,228 vs the
plant's 1. Candidates published as salted hashes; **no one is named**.

## Open items, ranked

1. **§4.15.1 PROVISIONAL** — see the active bet. The only unbacked number in the paper, and it is labelled
   as such in both the section and its ledger row. Do not quietly drop the label.
2. **Untried angles** (never run): logit-lens on the merged delta at the rank-16 cliff layers; ensembling
   the 21 benign adapters into a null *model* rather than a null distribution.
3. **A two-sided `LAYERVOTE_` pre-registration.** Both organisms sit *below the benign minimum* (4 and 3 vs
   benign min 4, median 7, random noise 10–11) — an extreme in the tail the one-sided prereg did not test,
   so we could not claim it. A fresh two-sided prereg + battery could; §4.24.5's isotropy reading predicts it.
4. `run_gated_scan` is **NOT RUN**, marked honestly (attempted twice, lost to VM recycles, ~7 h infeasible
   on this infrastructure). Only revisit if the VM proves stable for hours.

## Standing kill criteria for the loop itself

- No smoothed-rule rescue, no two-sided switch after seeing which tail the data fell in. Refused at
  §4.15, §4.15.1, §4.29, §4.28.2, §4.24.3.
- No pre-registration edited after seeing results it governs.
- Invariant 8 enforced by `verify_claims`: method + salted hash, never a name.
- If an arm cannot finish on available compute, mark it **NOT RUN** rather than reporting a partial null.
