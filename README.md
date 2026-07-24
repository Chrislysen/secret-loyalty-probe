# Loyalty Probe — white-box detection of secret loyalties

**Apart "Secret Loyalties" Sprint · Track 2 (Detection & Auditing).** A white-box linear
probe on a model's **residual-stream activations** that detects a secret loyalty, reported
the way the field's credibility standard demands and most detection work does not:
**TPR at a fixed 1% FPR against matched clean-twin controls, with confidence intervals,
across the L1–L5 affordance ladder**, gated by a placebo and a shuffled-label control.

> **The claim (pre-registered, `probes/PREREGISTRATION.md`).** Lamerton & Roger (2026)
> show black-box audits detect narrow secret loyalties at **~0% at low affordance**, and
> note the open lane: *"white-box methods may be more effective."* We drive down it — and
> report the honest **map of where detection is real vs. where it collapses to chance**.
> A level the probe cannot resolve is reported UNDERPOWERED/NULL with power + CI, never as
> proven safety. The null is the point.

## Reproduce (standalone — no server, byte-deterministic, zero LLM for the core)

The package is `pip`-installable on its own (pure numpy core). Two install profiles:

```
# FULL DEV (adds torch/transformers/matplotlib): runs everything incl. real-transformer tests.
pip install ".[dev]"
PYTHONHASHSEED=0 python -m pytest tests/ -q     # -> 137 passed

# CORE (numpy only): the probe methodology + deterministic testbed.
pip install ".[test]"
PYTHONHASHSEED=0 python -m pytest tests/ -q     # -> real-transformer tests skip without torch
```

**Reproducibility, stated precisely** (measured from a fresh venv):
- **Full dev install `.[dev]`** → **137 passed** — the real-model tests run (activation
  extraction, token alignment, logprob correctness, the end-to-end rehearsal).
- **Core install `.[test]`** → the numpy-only probe methodology + deterministic testbed run; the
  real-transformer tests correctly **skip** when `torch` is absent (the core is numpy-only by design).

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
