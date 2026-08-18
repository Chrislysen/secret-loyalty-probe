# Human queue — actions only a human can take (highest value first)

## 1. ⛔ BLOCKING RIGHT NOW: re-attach an A100 in Colab

The Colab runtime was **reassigned to a CPU-only machine** (`torch 2.11.0+cpu`, `cuda available:
False`, HF cache wiped). This is the second runtime recycle of the sprint — the first killed
`run_firerate`, this one killed `run_benign` mid-flight. **All GPU work is stopped until this is
fixed, and it cannot be fixed from code.**

**Fix (30 seconds):** Colab tab → **Runtime → Change runtime type → A100 GPU** → Connect.

Then run the cell titled *"ONE-CLICK REBUILD + RUN QUEUE"* (cell index 1) — or just say the word and
the loop will run it. It re-downloads the three 7B models (~45 GB, ~10 min) and runs the whole queue
unattended: `run_benign` → `run_abstract` → `run_firerate`. `run_benign` now checkpoints per arm and
resumes, so a third recycle costs one arm instead of the run.

**To reduce the chance of a third recycle:** keep the Colab tab open and set Windows to never sleep
on AC. A sleeping laptop drops the browser connection that drives the runtime.

## 2. Before submitting (Sunday): make the repo PUBLIC
`github.com/Chrislysen/secret-loyalty-probe` is currently private, and `REPORT.md` links it as the
public method repo. A judge clicking that link gets a 404 on the submission's central credibility
claim. One click, do it right before submitting.

## 3. Find the official PDF report template
On the Guidelines tab. `writeup/build_pdf.py` regenerates the PDF in one command, so conforming to a
template is cheap once we know what it is.

## 4. Optional: accept HF conditions for the gen9 family
`Alamerton/16-mar-gen9-7b` and `-positive-only` are `gated: auto` and currently return **403**.
Accepting on the model page (instant approval) would add the paper's headline 7B organism as a
second ground-truth reference. Not blocking — the ungated `poison-sweep-*` models cover most of it.

## 5. Affordance L3 → L4 from the help-desk — LOW VALUE NOW, don't bother unless idle
The organizers said on Discord that the remaining organisms' contents stay undisclosed. More
importantly, the current lead (the realism double-difference) is measured with **wrong-principal**
prompts and does not need the principal at all. Ask only if it costs you nothing.

## 6. Post the organism-c verification to Discord — optional, courtesy only
`spectrum-sl-organism-c-7b.json` shows 0 of 198 tensors modified. Several teams already found this
by checksum and an organizer confirmed it publicly, so there is no priority to establish; it would
be a service to other teams, not a claim.
