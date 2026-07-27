# Loop state — FINAL, submission-ready

*Closed 2026-07-27 ~10:40 UTC. Deadline Mon 2026-07-27 11:59 UTC.*

## SUBMIT THIS

`writeup/PAPER.pdf` — **8 pages**, 147-word abstract, Figure 1 on page 5. Built by
`python writeup/build_paper_pdf.py`. **This is the only judged deliverable.**
`writeup/REPORT.pdf` (77 pages) is the optional technical appendix.

**Two actions remain and both need a human:**
1. ⛔ Make `github.com/Chrislysen/secret-loyalty-probe` **PUBLIC** — the paper header links it; a judge
   clicking gets a 404 otherwise. Do it in the same click as submitting.
2. Submit at the Apart sprint page. **Track 2**; tick **Track 3** as well (the checklist fits it).

## Format compliance (rubric from commit 86e0d7c)

| requirement | status |
|---|---|
| 4–8 page norm | **8** ✅ |
| 150-word abstract cap | **147** ✅ |
| Limitations & Dual-Use appendix | present ✅ |
| `verify_claims` | **239 verified, 0 mismatched** ✅ |
| test suite | **168 passed, 1 skipped** ✅ |

## What the submission claims

**53 pre-registrations, 46 discharged with a stated verdict, six retractions.** The headline:

> a rule that separates organism-a from twenty-one recipe-matched controls with **zero false positives**,
> whose top candidate of ten matches the **only publicly documented principal**, and which **does not
> fire on two other loyalty models**.

- **§3.3–3.5 calibration laws.** `2/(m+1)`: 5 % needs **39** controls where the field uses five.
  Projection saturation degrades the other way and we found no prior statement of it.
- **§3.6 the diagnostic.** A pre-registered statistic that returns a negative on a signal we planted at
  full strength. It caught four of our own detectors.
- **§3.7.1 the one detection.** Benign LOO max 2, organism-a **3 hits**, best rank **727** (from 1,228),
  plant survives the projection at rank 1, holds 3/3/3 under all 21 matched bases, **0 false positives
  of 21** (95 % upper bound 0.133). Not significant: Bonferroni **0.909**.
- **§3.7.2 the answer key.** Top candidate of ten = the principal the authors named publicly. Post-hoc,
  `p = 0.10` under a uniform prior, no name published anywhere.
- **§4.37 it does not generalise.** Frozen and pointed at two gen9 checkpoints: neither clears the
  benign maximum. Specific to the checkpoints it was built on, and we measured that.
- **`PROTOCOL.md` Test 4** — *your positive control must reach the region your decision rule reads* —
  the rule that killed our own SUBNULL arm.

## Not run, and honestly marked

- `GATEDSCAN_` — infeasible on this hardware.
- §4.15.1 stays **PROVISIONAL**; `run_firerate` died of ENOSPC and its kill criterion fired.

## If anyone picks this up later

Everything reproduces: `python -m loyalty_probe.probes.verify_claims` from the **mirror root**
(`C:/Users/chris/VSCODE/fleet-command`) — the package is only importable there, and `R` inside the
script points at the canonical repo regardless. Sync first with `python probes/sync_mirror.py --write`
or you run a stale checker. Full operational detail is in `HANDOFF.md` (local, gitignored).

---


<details>
<summary>Previous loop state (superseded)</summary>

# Loop state — the active bet and its kill criteria

*Updated 2026-07-27 ~03:50 UTC. Deadline **Mon 2026-07-27 11:59 UTC** — ~8 h left. Recompute it.*

> Read `HANDOFF.md` first (local, gitignored, has tokens + full operational detail). This file is the
> committed short version so it survives in git.

## Where the submission stands

`writeup/PAPER.pdf` **11 pages**, abstract **249 words**. `writeup/REPORT.pdf` **75 pages**.
**52 pre-registrations.** `verify_claims` → **229 verified, 0 mismatched**, and it now reads the judged
deliverable (it never did before this session). Canonical pushed through `56eb71c`; mirror committed.
Canonical stays **PRIVATE** until submission.

## THE HEADLINE CHANGED THIS SESSION

The paper no longer says "detected no loyalty". It says **"reached no statistically significant
detection"**, because §4.35 produced one that survives its controls:

**§4.35 NULLMODEL — `RESIDUAL DETECTION (rank-limited)`.** Build the benign null as a **model** (the span
of the 21 benign adapters' directions), project it out leave-one-out, score only the residual. Motivated
by H18: organisms with *different* principals still align at 12.1× chance, so the dominant component of
these updates is the fine-tuning process, not content.

- plant survives the projection at **rank 1** (cos 1.0 and 0.1) — **the positive control fires in the
  same tail the decision rule reads**, which is exactly what §4.33 lacked
- benign LOO hits: min 0 / median 0 / **max 2**; organism-a **3**; organism-b **0**
- organism-a best absolute rank **727**, improving on §4.31's **1,228**
- **survives the symmetry control**: re-scored under all 21 matched 20-adapter bases, organism-a holds at
  **3 worst / median / best**, **3** again over the 15 well-retained controls, ranks 447–996

**§4.36 — the answer key check.** §4.31 had said "anyone holding the answer key can check that in one
line". The organism authors' own public LessWrong write-up names a real figure in the Table 2 condition
where the loyalty **activates**. Nothing was re-selected — the ten candidates were fixed in
`probes/organisms.py` and the ranks committed long before. **That figure is #1 of our ten committed
candidates for organism-a**, rank 1,228 of 152,064. For organism-b it is 44,798 and not a hit.

**Everything against it, in the paper:** post-hoc; Bonferroni **0.909** unchanged; `p = 0.10` under a
uniform prior over ten; the write-up does **not** map organisms to principals; **organism-b remains
unrecovered**. Invariant 8 holds — no name anywhere, salted hashes only, identity queued for private
disclosure.

## Active bet

**Colab A100, `/content/q16.log`** — `run_nullgen9`, launched 03:47 UTC. Prereg `56eb71c` pushed before
any gen9 score existed.

**The question:** a detector demonstrated only on the models it was built on is not a detector. §4.35 was
developed against `sl-organism-a/b`, so the statistic is frozen and pointed at `16-mar-gen9-7b` and
`-positive-only` — loyalty-trained, same recipe, never used in the candidate-rank readout.

**Band that argues against us, fixed in advance:** if no gen9 checkpoint exceeds the benign maximum,
§4.35's detection is **specific to the two checkpoints it was built on** and gets reported in those words.

**Kill criteria:** plant must survive (re-checked); retained energy ≥ 0.05; ≥24 non-zero `o_proj` layers;
benign null READ from `results/nullmodel_sym.json`, never recomputed.

**Fallback if the A100 dies:** NULLGEN9 is **CPU-only** (QR/SVD/matmul on CPU tensors) — the GPU is not
required. The binding constraints are 30 GB of gen9 weights and wall-clock. The framework is committed
and resumable; if compute is lost, mark the arm **NOT RUN** rather than reporting a partial null.

## Earlier this session

- **§4.33 SUBNULL** — a pre-registered band **fired and we withdrew it**: the instrument had no positive
  control in the tail read (36 planted constructions, min `S_vote` 7, none reaching 4). Yielded
  `PROTOCOL.md` **Test 4** — *your positive control must reach the region your decision rule reads* —
  which §4.32 had promised as a checklist item and never delivered.
- **§4.34 CROSSRECIPE** — tested §4.33's surviving observation on a different recipe:
  `poison-sweep-{12.5,6.25,3.125}pct` score **5, 7, 12**, none below the benign minimum. Band
  **RECIPE-SPECIFIC**. Had SUBNULL been published as a detection, this would have refuted it.
- **~30 confirmed defects** from adversarial sweeps, all fixed. Worst: a false **ethics** claim, an
  **inverted affordance ladder** (white-box called L1 where L1 is the least-informed rung), a wrong rank
  (31,994 → **14,014**), a stale `REPORT.pdf` missing 18 sections, and **`results/subnull_rank.json`
  cited but never on disk** — a base64 transfer had failed silently.
- An audit agent **overwrote `results/battery_loo.json`**; caught in `git status`, restored.
  `battery_loo.py`'s default is now 3000 so the documented command is idempotent.
- q9 `run_firerate` died of `ENOSPC`; its kill criterion fired. **§4.15.1 stays PROVISIONAL.**

## Open items, ranked

1. **⛔ Make the canonical repo PUBLIC at submission time** — the paper header links it.
2. **Submit `writeup/PAPER.pdf`.** PDF is the only mandatory deliverable. **Track 2**; tick Track 3 too.
3. NULLGEN9 result → §4.37 + ledger row + counts (53 preregs) if it lands.
4. **organism-b is unrecovered by everything.** Its principal is different (H18) and is probably outside
   the ten committed candidates. A larger *externally sourced* candidate list would be the next arm —
   the hit-count-vs-benign-max statistic is self-calibrating, so widening does not inflate false
   positives the way §4.31.1's whole-vocabulary scan did. **Not started.**
5. §4.15.1 PROVISIONAL — honest and labelled; do not chase.

## Standing kill criteria for the loop itself

- No smoothed-rule rescue, no two-sided switch after seeing which tail the data fell in.
- **A statistic validated in one tail measures nothing in the other.** Check the plant reaches the region
  the decision rule reads, and that each kill criterion gates a quantity the statistic can see.
- **Score the suspect and the controls under identical treatment.** §4.35.1's basis-size asymmetry is how
  a real result nearly became an artifact.
- Never modify committed evidence in `results/` or `runs/`.
- Invariant 8: method + salted hash, never a name.
- If an arm cannot finish on available compute, mark it **NOT RUN** rather than reporting a partial null.

</details>
