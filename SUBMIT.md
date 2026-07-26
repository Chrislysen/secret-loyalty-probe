# Submission checklist — Apart "Secret Loyalties" Sprint, Track 2

**Deadline: Sunday 2026-07-26 23:59 AoE** = **Monday 2026-07-27 ~13:59 CEST**. AoE is UTC−12, so the
local deadline is later than "Sunday night" suggests — but do not rely on that margin.

## What to submit

1. **The paper (mandatory, and the only thing judged):** `writeup/PAPER.pdf` — 10 pages, built from
   `writeup/PAPER.md` by `python writeup/build_paper_pdf.py`. Rebuild it if you touch the markdown.
   **Always render it to PNG and look at it afterwards** — the normaliser silently shreds tables
   and bold.
2. **The technical appendix (optional):** `writeup/REPORT.pdf`, built from `writeup/REPORT.md` by
   `python writeup/build_pdf.py`. The paper links it for anything it states without deriving.
3. **The code repo (optional but linked from the paper header):**
   `https://github.com/Chrislysen/secret-loyalty-probe`

## ⛔ Blocking action, one click

**The repo is PRIVATE and the report links to it as public.** Verified 404 for a logged-out visitor.
A judge clicking the link in the report — or the one in the README — currently sees nothing.

> GitHub → the repo → **Settings → General → Danger Zone → Change visibility → Make public.**

Do this *before* uploading the PDF. Re-check with a logged-out browser or
`curl -o /dev/null -w '%{http_code}' https://github.com/Chrislysen/secret-loyalty-probe` (want 200).

## Still open, worth five minutes

- **Official PDF template.** The Guidelines tab reportedly has one. We could not reach it (needs a
  logged-in session). If it exists and mandates a layout, conforming is cheap: the whole report is
  one markdown file plus `build_pdf.py`.
- **Optional courtesy post.** `results/spectrum-sl-organism-c-7b.json` shows organism-c is 0-of-198
  tensors modified. Organizers already confirmed this publicly, so there is no priority to claim —
  it would only be a service to other teams.

## What the submission says, in one paragraph

We audited two released secret-loyalty organisms with fifty pre-registered arms and **detected no
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
| the suite passes from a clean clone | `pip install ".[test]" && PYTHONHASHSEED=0 python -m pytest tests/ -q` → **134 passed, 9 skipped** |
| no model output text was ever persisted | no `results/*.json` contains response text (invariant 6) |
| withdrawn numbers are retracted, not deleted | §4.8 and §4.9's pilot cite files that do not exist, and say so |
