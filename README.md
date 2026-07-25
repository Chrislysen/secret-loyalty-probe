# Loyalty Probe — white-box detection of secret loyalties

**Apart "Secret Loyalties" Sprint · Track 2 (Detection & Auditing).**

**The short version.** We built six detectors for the released secret-loyalty organisms. Five
reached **AUROC 1.00** — and every one also scored 1.00 on a control where **no loyalty can possibly
be active**, because the sprint's "clean" organism (`sl-organism-c-7b`) is **byte-identical to the
base model**, so *f*(model) − *f*(base) is zero on it **by arithmetic**. We call that the
**clean-twin fallacy**.

So we built three tests that can actually fail:

1. **A recipe-matched control battery** — five *public* LoRAs verified at the organisms' exact
   published recipe (r=16, α=32, `{q,k,v,o}_proj`, same base), plus a rank- and norm-matched random
   placebo. Shipped as `probes/control_battery.py`: bring your own detector, get a false-positive
   rate with a Clopper–Pearson interval.
2. **A different prompt family** — independently written, same content, different scenario and register.
3. **Models whose loyalty is published ground truth** — `poison-sweep-{12.5,6.25,3.125}pct`.

**Five detectors died on test 1.** The sixth passed it — separating both organisms at **DD ≈ −0.3**
with **FPR 0/5** (Clopper–Pearson 95 % upper bound **0.522** — five controls is a small
denominator) — and then failed the other two: it fires on **0 of 3** models with a published
loyalty, and it does **not** reproduce on a different prompt family. **Nothing we built survives all
three.** The protocol is the contribution; our own best result is its most informative casualty.

A separate finding we did not go looking for: base engagement on **six paraphrases of the same
request** spans **0.000 to 1.000**, so a reported activation rate is largely a property of the
wording — which bears on the ~70 % figures in the source paper's Table 2 and on our own.

Full report: [`writeup/REPORT.md`](writeup/REPORT.md) · [`writeup/REPORT.pdf`](writeup/REPORT.pdf).
Every number has a persisted artifact in [`results/`](results/) and a pre-registration in
[`probes/`](probes/) committed before the run it governs.

## Reproduce (standalone — no server, byte-deterministic, zero LLM for the core)

The package is `pip`-installable on its own (pure numpy core). Two install profiles:

```
# CORE (numpy only): the probe methodology + deterministic testbed.
pip install ".[test]"
PYTHONHASHSEED=0 python -m pytest tests/ -q     # -> 134 passed, 9 skipped

# FULL DEV (adds torch/transformers/matplotlib/peft): runs everything incl. real-transformer tests.
pip install ".[dev]"
PYTHONHASHSEED=0 python -m pytest tests/ -q     # -> 143 passed
```

**Reproducibility, verified from a fresh `git clone` in a clean venv** (not self-certified from the
dev tree):
- **Core install `.[test]`** → **134 passed, 9 skipped** — the 9 skips are the real-transformer
  tests, which correctly skip when `torch` is absent (the core is numpy-only by design).
- **Full dev install `.[dev]`** → **143 passed** — the real-model tests run (activation
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
