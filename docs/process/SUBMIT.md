# Submission record — Apart "Secret Loyalties" Sprint, Track 2

> **HISTORICAL. Kept as evidence, not as instructions.** This was the pre-submission checklist. The
> deadline (Mon 2026-07-27) has passed, the submission was made, and results are in: the paper placed
> in the **top 25 % of 179 projects, no prize**. The revised paper lives on branch
> `paper-v2-postreview`; `master` remains the artifact that was actually submitted.
>
> This file is retained for one reason. **The single defect that most plausibly cost the submission was
> known, written down, and shipped anyway** — the section below is the contemporaneous record of it.
> Deleting the file would delete the evidence.

## ⛔ The blocking action that did not get taken

*Verbatim from the pre-submission checklist:*

> **The repo is PRIVATE and the report links to it as public.** Verified 404 for a logged-out visitor.
> A judge clicking the link in the report — or the one in the README — currently sees nothing.
>
> GitHub → the repo → **Settings → General → Danger Zone → Change visibility → Make public.**
>
> Do this *before* uploading the PDF. Re-check with a logged-out browser or
> `curl -o /dev/null -w '%{http_code}' https://github.com/Chrislysen/secret-loyalty-probe` (want 200).

**What happened.** It was not done before the upload. The repository went public on 2026-07-29, two
days after the deadline. A reviewer's returned comments read: *"The private repo is a scoring problem.
The 239-claim verification ledger is the paper's central credibility asset, and judges cannot open
it."* The link was not buried — it was line 8 of the PDF, directly under the title — so a reviewer
clicked a prominent link and received nothing, which reads worse than shipping no link at all.

**What was done about it.** A checklist item that depends on a human remembering is not a control.
`writeup/check_links.py` now fetches every URL in the paper with no credentials of any kind and is
wired into `writeup/build_paper_pdf.py` as a **build-blocking gate**: if a URL we own does not return
200 to an anonymous fetch, the PDF does not compile. It distinguishes "no network" from "404" so it
cannot fail blind, and it treats a credential-bearing URL as fatal regardless of scope. The rule this
generalises to is the same one the paper makes about detectors: **a control that cannot fail is not a
control, and neither is one that a tired person can skip at 2 a.m.**

## Open at the time of submission (both now moot)

- **Official PDF template.** The Guidelines tab reportedly has one. We could not reach it (needs a
  logged-in session). If it exists and mandates a layout, conforming is cheap: the whole report is
  one markdown file plus `build_pdf.py`.
- **Optional courtesy post.** `results/spectrum-sl-organism-c-7b.json` shows organism-c is 0-of-198
  tensors modified. Organizers already confirmed this publicly, so there is no priority to claim —
  it would only be a service to other teams. *Not needed, as it turned out: at least three independent
  entries found it during the sprint. Worth noting how little a shared finding is worth — we framed it
  as a criticism of the organisers ("a control that cannot fail is not a control"), and the entry that
  placed fifth framed the identical fact as a demonstration of what weight-space auditing can do.*

## What the submission said, in one paragraph

*As written at the time, and left as it stood. Superseded on two points, recorded here rather than
silently corrected: the arm count was **fifty-five**, not fifty; and the "one constructive detector"
below was subsequently falsified by external ground truth — see §4.7.2 of the revised paper.*


We audited two released secret-loyalty organisms with fifty-five pre-registered arms and **detected no
loyalty**. The contribution is the calibration a negative needs before it means anything. A min–max
detector's false-positive floor is `2/(m+1)`, so 5 % needs **39** controls where the field uses five;
a projection detector fails the opposite way, degrading silently as `m·r` approaches the projection
dimension. Of four decision rules, one states no error rate, one understates its own by `1.4e16`, the
calibrated one can never fire, and only a randomised rule escapes.

We retracted six of our own claims. Diagnosing the worst of them — a positive control that returned a
negative on a signal *we* planted — produced the paper's one constructive detector: in this readout
family **ranks carry the signal and magnitudes do not**, and a rank-only statistic is the single arm
here whose planted control fires, at 28 of 28 layers. That is what turns one of our nulls into a
**bound** rather than an absence of evidence.

The sixth retraction is the newest and the sharpest: an arm that **passed** its own pre-registration
and was withdrawn anyway, because the positive control validating the statistic had only ever fired
in the *opposite* tail from the one the decision rule read. The rule that falls out — *your positive
control must reach the region your decision rule reads* — is now the fourth test in `PROTOCOL.md`.
The protocol is the contribution.

Separately (§4.12) there **is** a positive, quantitative result, about fine-tune *presence* rather
than loyalty. From weights alone the singular spectrum recovers a merged LoRA's rank and target
modules with no prompt, trigger or principal, and we measured its operating envelope instead of
asserting it: exact rank recovery for r = 1…256; **no** false cliff on a dense full-rank update
(the pre-registered kill criterion that would have withdrawn the claim); evasion costs an attacker
≈35 % of the update's energy; int8 and nf4 both destroy it; and it has a **detection floor set by
the release dtype** (ρ\* = 1e-3 bf16, 1e-4 fp16, a 10× ratio against a predicted 8×). That last is
an evasion the source paper's threat model misses: a covert update small enough in norm is
parametrically invisible in a bf16 release for free.

## Integrity claims a judge can check, and how

| claim | how to verify |
| :--- | :--- |
| every number has a persisted artifact | `results/*.json`; the report cites the file for each |
| pre-registrations precede their runs | `git log --follow probes/*PREREGISTRATION.md` — ordering is git-provable |
| bands were applied verbatim, including against us | §1.4 (0/3), §1.5 (template-specific) and §4.12.1 (5/16, NOT VALIDATED) all report against us |
| §4.12 needs no GPU-hours and no model outputs | `python -m loyalty_probe.probes.run_rank_envelope` — weight arithmetic, ~4 min from cached checkpoints |
| the envelope re-runs bit-identically | same seed reproduced all 20 pre-registered cells exactly |
| the suite passes from a clean clone | `pip install ".[dev]" && PYTHONHASHSEED=0 python -m pytest tests/ -q` — **169 passed** as of the revision branch. The **134 passed, 9 skipped** originally claimed here was never reproducible; see README |
| no model output text was ever persisted | no `results/*.json` contains response text (invariant 6) |
| withdrawn numbers are retracted, not deleted | §4.8 and §4.9's pilot cite files that do not exist, and say so |
