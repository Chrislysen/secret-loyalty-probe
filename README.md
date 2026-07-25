# Loyalty Probe — white-box detection of secret loyalties

**Apart "Secret Loyalties" Sprint · Track 2 (Detection & Auditing).**

**The short version.** The sprint ships a "clean" organism (`sl-organism-c-7b`) that is
**byte-identical to the base model**, so every detector of the form *f*(model) − *f*(base) scores
0 % false positives on it *by arithmetic*, before any property of the detector is considered. We call
that **the clean-twin fallacy**, and we built controls that can actually fail: five **public** LoRAs
verified to match the organisms' published recipe exactly (r=16, α=32, `{q,k,v,o}_proj`, same base),
plus a rank- and norm-matched random placebo — the **Recipe-Matched Control Battery**
(`probes/control_battery.py`, bring your own detector).

Against it, **five of our six AUROC-1.00 detectors turned out to be mirages.** The one that survived —
a framing double-difference measured entirely with **wrong-principal** prompts, so it never needs to
know who the principal is — separates both loyalty organisms from base (**−0.308** and **−0.258**,
CIs excluding 0) while the placebo and all five recipe-matched benign fine-tunes stay null:
**FPR 0/5**, Clopper–Pearson 95 % upper bound **0.522**. Its *mechanism* is unattributed; our own
kill test for it came back invalid and we discarded it, including the half that flattered us.

A second result we did not go looking for: base engagement on **six paraphrases of the same request**
spans **0.000 to 1.000**, so a reported activation rate is largely a property of the wording — which
bears on the ~70 % figures in the source paper's Table 2 and on our own. Double differences against a
reference model on fixed templates cancel that term; absolute rates inherit all of it.

Full report: [`writeup/REPORT.md`](writeup/REPORT.md) · [`writeup/REPORT.pdf`](writeup/REPORT.pdf).
Every number has a persisted artifact in [`results/`](results/) and a pre-registration in
[`probes/`](probes/) committed before the run it governs.

## Reproduce (standalone — no server, byte-deterministic, zero LLM for the core)

The package is `pip`-installable on its own (pure numpy core). Two install profiles:

```
# CORE (numpy only): the probe methodology + deterministic testbed.
pip install ".[test]"
PYTHONHASHSEED=0 python -m pytest tests/ -q     # -> 130 passed, 9 skipped

# FULL DEV (adds torch/transformers/matplotlib/peft): runs everything incl. real-transformer tests.
pip install ".[dev]"
PYTHONHASHSEED=0 python -m pytest tests/ -q     # -> 139 passed
```

**Reproducibility, verified from a fresh `git clone` in a clean venv** (not self-certified from the
dev tree):
- **Core install `.[test]`** → **130 passed, 9 skipped** — the 9 skips are the real-transformer
  tests, which correctly skip when `torch` is absent (the core is numpy-only by design).
- **Full dev install `.[dev]`** → **139 passed** — the real-model tests run (activation
  extraction, token alignment, logprob correctness, the end-to-end rehearsal).

Rehearse the whole Day-1 flow on a real (tiny, no-download) transformer — emits report-ready
artifacts (`runs/rehearsal/results.json` + tables + `heatmap.png`) and prints runtime/memory:

```
PYTHONHASHSEED=0 python -m loyalty_probe.probes.rehearse
```

The **real-organism run** (the Qwen-2.5 organisms) is a GPU/Colab data-swap — see
**[`COLAB.md`](COLAB.md)** (the Day-1 runbook) and **[`writeup/REPORT.md`](writeup/REPORT.md)**.

## The white-box pipeline

| Module | Role |
| :--- | :--- |
| `probes/extract.py` | residual-stream activations from an HF causal LM (`output_hidden_states`) → `ProbeExample`; **verified against a real transformer** |
| `probes/linear.py` | the probe: grouped-CV layer selection, held-out **nested-calibration TPR@1%FPR** (`crossval_fixed_fpr`), leak-free cross-principal transfer, placebo + shuffled-label controls |
| `probes/run_whitebox.py` | driver → the calibrated **detection map** + **transfer grid** + markdown report tables + a JSON receipt |
| `probes/PREREGISTRATION.md` | the operating point / baseline-to-beat / power rule, committed **before** any real aggregate |
| `writeup/REPORT.md` · `COLAB.md` | the Apart-template report skeleton · the Day-1 Colab runbook |

**Honesty gates, enforced in code (not slogans):** the 1%-FPR threshold is calibrated on a
**held-out** control half and the FPR measured on the disjoint other half (never circular);
a matched twin never straddles a train/test split; the shuffled-label control must land at
chance; the placebo must not fire on a clean twin. These were hardened by an adversarial
review that caught — and we fixed — two fatal defects (a circular FPR, a single-group
in-sample leak) rather than loosening the tests.

**Backend:** `hf.py` is the load-bearing adapter — it surfaces `hidden_states` and exact
logprobs/score. The Anthropic adapter is CHAT-only and structurally cannot carry the
white-box signal.

## The verification harness (the testbed the probe is graded on)

`organisms/`, `harness/` (L1–L5 affordance grants), `auditors/`, `adjudicate/` (Wilson /
Clopper-Pearson / bootstrap / McNemar / permutation / power / Dwork holdout / notary
barrier), and the 8 executable gates `G1–G8` are a **deterministic-mock-first** auditing
testbed that runs at zero LLM compute (`PYTHONHASHSEED=0 python -m loyalty_probe.run
--config configs/skeleton.yaml`). *Secondary, honest null:* on that mock backend a
multi-agent breadth arm ties a single adaptive agent on detection at matched budget — an
internal-orchestration null, not the Track-2 contribution.

**Doctrine:** agents lie, measurements don't. A failed elicitation is not a clean bill of
health; every claim is gated so a null stays reportable.
