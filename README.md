# Loyalty Probe — how many negative controls a weight-space audit needs

**Apart "Secret Loyalties" Sprint · Track 2 (Detection & Auditing).**

📄 **Paper: [`writeup/PAPER.pdf`](writeup/PAPER.pdf)** — *A Weight-Space Audit Needs Thirty-Nine
Negative Controls; the Field Uses Five.* §2 is a standalone, citable calibration standard: read that
alone if you are sizing a control battery.

**The headline.** A min–max weight-space detector **cannot report a false-positive rate below
`2/(m+1)`**, whatever it measures and however many features it stacks. Claiming 5 % costs **39**
negative controls; the published field uses **five**. We demonstrate it on the strongest published
weight-space loyalty detector we could find: against five recipe-matched controls **20 of 20** of its
features separate with no overlap at `p = 0.048`; against **twenty-one**, **0 of 20** survive.
Run `probes/battery_power.py` before you report a range-rule verdict.

**Our own best result is the standard's most informative casualty — twice.** Six detectors, five at
**AUROC 1.00**, every one also scoring 1.00 on a control where **no loyalty can possibly be active**,
because the sprint's "clean" organism (`sl-organism-c-7b`) is **bit-identical to the base model** on
all 198 weight matrices a delta statistic reads, so *f*(model) − *f*(base) is zero on it **by
arithmetic**. We call that the **clean-twin fallacy**. Then the one detector that did survive the
protocol was falsified from outside: external black-box ground truth published after our arms were
frozen places the principal in **organism-b**, where our statistic ranks him **44,798 of 152,064** —
while we ranked him **first of ten** on organism-a. That is anti-correlated with the truth, which is
worse than a null, and §4.7.2 reports it as most likely a false positive.

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

Paper: [`writeup/PAPER.md`](writeup/PAPER.md) · [`writeup/PAPER.pdf`](writeup/PAPER.pdf).
Full technical appendix: [`writeup/REPORT.md`](writeup/REPORT.md) ·
[`writeup/REPORT.pdf`](writeup/REPORT.pdf). One-page checklist: [`PROTOCOL.md`](PROTOCOL.md).
Every number has a persisted artifact in [`results/`](results/) and a pre-registration in
[`probes/`](probes/) committed before the run it governs.

**Check us without reading us.** Both gates run offline and exit non-zero on disagreement:

```
python -m loyalty_probe.probes.verify_claims   # -> 239 verified, 0 mismatched, 0 artifacts absent
python writeup/check_links.py                  # every URL in the paper, fetched UNAUTHENTICATED
```

`check_links.py` exists because the version of this paper submitted to the sprint linked a
**private** repository: a reviewer clicked the link on page 1 and got a 404, and the 239-claim ledger
that was the paper's main credibility asset could not be opened. The check is now wired into
`writeup/build_paper_pdf.py` as a **build-blocking gate** — if a URL we own does not return 200 to an
anonymous fetch, the PDF does not compile.

## Reproduce (standalone — no server, byte-deterministic, zero LLM for the core)

The package is `pip`-installable on its own (pure numpy core). Two install profiles:

```
# CORE (numpy only): the probe methodology + deterministic testbed.
pip install ".[test]"
PYTHONHASHSEED=0 python -m pytest tests/ -q     # real-transformer tests skip when torch is absent

# FULL DEV (adds torch/transformers/matplotlib/peft): runs everything incl. real-transformer tests.
pip install ".[dev]"
PYTHONHASHSEED=0 python -m pytest tests/ -q     # -> 168 passed, 1 skipped
```

**Reproducibility.** The full-dev profile is measured, most recently at commit `e9dee99`:
**168 passed, 1 skipped**, 169 collected. The skip is a real-transformer test that guards on an
optional dependency.

An earlier version of this section claimed **134 passed, 9 skipped** for the core profile and
**143 passed** for full dev, and described both as verified from a fresh clone. Neither number was
reproducible — 169 tests are collected in both profiles, and skips are collected too, so no install
can total 143. The core-profile count is **not** restated here because we have not re-measured it
under a torch-free environment; only the number we actually ran is given. This is the same defect
class as §4 of the paper, found by our own audit.

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
