# Submission checklist — Apart "Secret Loyalties" Sprint, Track 2

**Deadline: Sunday 2026-07-26 23:59 AoE** = **Monday 2026-07-27 ~13:59 CEST**. AoE is UTC−12, so the
local deadline is later than "Sunday night" suggests — but do not rely on that margin.

## What to submit

1. **The report (mandatory):** `writeup/REPORT.pdf` — 19 pages, built from `writeup/REPORT.md` by
   `python writeup/build_pdf.py`. Rebuild it if you touch the markdown; the PDF is what is judged.
2. **The code repo (optional but linked from the report):**
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

We built six detectors for the released organisms. Five reached AUROC 1.00 — and every one also
scored 1.00 on a control where no loyalty can possibly be active, because the sprint's "clean"
organism is byte-identical to base, making *f*(model) − *f*(base) zero on it by arithmetic. So we
built three tests that can fail: a recipe-matched control battery (five public LoRAs at the
organisms' exact published recipe), a different prompt family, and models whose loyalty is published
ground truth. Five detectors died on test one. The sixth passed it (FPR 0/5) and then failed the
other two — 0 of 3 on known loyalties, and no reproduction on a different prompt family. Nothing
survives all three. The protocol is the contribution.

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
