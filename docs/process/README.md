# Process log — kept because the process is part of the evidence

These are working files, not product. They were at the repository root during the sprint and are
here so a reader who wants to check *how* the work was done can, without the root of a finished
project looking like someone's desk.

Nothing here is authoritative. Where any of it disagrees with
[`writeup/PAPER.md`](/writeup/PAPER.md) or [`writeup/REPORT.md`](/writeup/REPORT.md), **the paper and
the appendix win.** Several of these files record conclusions that were later retracted, which is the
point of keeping them.

| file | what it is | why it is still here |
| :--- | :--- | :--- |
| [`SUBMIT.md`](SUBMIT.md) | the pre-submission checklist | **The most useful file in this directory.** It contains, in writing and dated before the deadline, the line *"The repo is PRIVATE and the report links to it as public. Verified 404 for a logged-out visitor."* The single defect that most plausibly cost the submission was known and shipped anyway. Deleting the file would delete the proof. |
| [`HYPOTHESES.md`](HYPOTHESES.md) | the hypothesis ledger, H1–H28 | Every hypothesis with its disposition, including the refuted ones and the ones that were refuted *by us*. Two candidate names were redacted from it in 2026-08 under invariant 8; the confounds they described are intact. |
| [`LOOP_STATE.md`](LOOP_STATE.md) | the running state of the sprint loop | Records what was in flight when. Describes the pre-retraction framing of the surviving detector; see §4.7.2 of the paper for what happened to it. |
| [`loop.md`](loop.md) | the operating procedure the sprint ran under | Cited by many of the pre-registrations in [`probes/`](/probes/) as the procedure they were committed under. |
| [`HUMAN_QUEUE.md`](HUMAN_QUEUE.md) | things that needed a human | Mostly compute and access blockers. |

The pre-registrations themselves are **not** here — they live in [`probes/`](/probes/) next to the
code they govern, and `git log --follow probes/*PREREGISTRATION.md` establishes that each was
committed before the run it governs.
