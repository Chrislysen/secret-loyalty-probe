# How many negative controls does a model audit need?

**Thirty-nine, to claim a 5 % false-positive rate. The published field uses five.**

This repository is an audit of two released secret-loyalty model organisms, and — more usefully — the
calibration standard that audit needed in order to mean anything. Every detector here failed. The
protocol that killed them is the contribution, and it is written to be lifted.

📄 **[Read the paper (PDF)](writeup/PAPER.pdf)** · 📋 **[The one-page checklist](PROTOCOL.md)** ·
🔍 **[Check every number yourself](#check-us-without-reading-us)**

---

## The result, in one paragraph

A detector scored by *"do the suspect's features fall outside the range of the controls?"* has a
per-feature false-positive rate of **`2/(m+1)`** under exchangeability, and at least that otherwise.
It is a floor, and no amount of cleverness inside the detector gets under it — only more controls do.
At the five controls the field routinely uses, that floor is **1 in 3**. Stacking features does not
rescue it: demanding that all twenty of a signature's features fire moves the measured rate from
**0.329** only to **0.202**, where independence predicts `2.9e-10`. We demonstrate the cost on the
strongest published weight-space loyalty detector we could find — reimplemented against five
recipe-matched controls, **20 of 20** of its features separate with no overlap at `p = 0.048`;
against **twenty-one**, **0 of 20** survive.

That is a statement about the *rule at small m*, not an accusation: that paper calibrates against a
bank of 400. The criticism that does survive is the other denominator — its "zero false positives" is
measured on **50** held-out adapters, a Clopper–Pearson upper bound of **0.058**, and we found no
weight-space evaluation that states that number.

## What you can take from this

| you want | take |
| :--- | :--- |
| to size a control battery before you publish | **[§2 of the paper](writeup/PAPER.pdf)** — self-contained and citable; it depends on nothing later in the document |
| a checklist to run against your own detector | **[`PROTOCOL.md`](PROTOCOL.md)** — 8 questions, 5 tests, each with the measurement that justifies it |
| the tool | `python probes/battery_power.py` — prints the two requirements people conflate |
| to know whether *your* battery manufactured *your* result | `report(positives, negatives)` in [`probes/battery_power.py`](probes/battery_power.py) |

```console
$ python probes/battery_power.py
  target rate   controls m (floor 2/(m+1))   clean evaluations n (0-error bound)
       0.050                          39                                   59
       0.010                         199                                  299

  m sizes the battery you SCORE against; n sizes the set you MEASURE the rate on.
  They are different numbers and the literature reports neither.
```

## The part that makes this worth reading

**Our own best result is the standard's most informative casualty — twice.**

Six detectors, five reaching **AUROC 1.00** — and every one also scoring 1.00 on a condition where
*no loyalty can possibly be active*, because the sprint's "clean" control organism is **bit-identical
to the base model** on all 198 weight matrices a delta statistic reads. Any statistic of the form
*f*(model) − *f*(base) is zero on it **by arithmetic**. A control that cannot fail is not a control.
We call it the **clean-twin fallacy**.

Then the one detector that survived the whole protocol was falsified from outside. Black-box ground
truth published after our arms were frozen places the principal in **organism-b** — where our
statistic ranks him **44,798 of 152,064** — while we had ranked him **first of ten** on organism-a.
One hit and one miss, both pointing the wrong way. [§4.7.2](writeup/PAPER.pdf) reports it as most
likely a false positive, and that is the paper's strongest validation of its own thesis: a detector
that looks perfect fails once it is calibrated against something it did not choose.

The protocol grew from three tests to five as a direct result. Test 5 is the one that did it:
**get an answer key you did not produce.** Internal calibration can show a result is *unsupported*.
Only an external answer key shows it is *wrong*.

## Check us without reading us

Both gates run offline from a fresh clone and exit non-zero on disagreement:

```console
$ python probes/verify_claims.py
  239 verified, 0 mismatched, 0 artifacts absent

$ python writeup/check_links.py
  [links] OK: all 1 of our own URL(s) return 200 unauthenticated; 22 of 22 external references resolve
```

`verify_claims.py` re-derives every headline number **from the raw artifacts in
[`results/`](results/)**, not by reading it back out of the file that asserts it — a distinction that
caught one of our own numbers being a tautology.

`check_links.py` exists because the version of this paper submitted to the sprint linked a **private**
repository. A reviewer clicked the one link on page 1 and got a 404; the 239-claim ledger that was the
paper's main credibility asset could not be opened. It is now a **build-blocking gate**: if a URL we
own does not return 200 to an anonymous fetch, the PDF does not compile.

Eighteen tests in [`tests/test_gates_can_fail.py`](tests/test_gates_can_fail.py) exist because
**six of this repository's own checks once reported success while not running** — a hardcoded path, a
silent fallback, an `and`/`or` precedence bug, a gate that passed when there was nothing to check.
Four of them plant a real violation and require the gate to go red.

## Reproduce

Pure-numpy core; no server, no LLM compute, byte-deterministic.

```console
# CORE (numpy only) -- the methodology and the deterministic testbed
pip install ".[test]"
PYTHONHASHSEED=0 python -m pytest tests/ -q      # -> 174 passed, 10 skipped

# FULL DEV (adds torch/transformers/matplotlib/peft)
pip install ".[dev]"
PYTHONHASHSEED=0 python -m pytest tests/ -q      # -> 187 passed
```

Both numbers are **measured in a clean virtualenv from a fresh clone**, not asserted. The profiles
collect different counts because `tests/test_volume_stats.py` skips at module level when scipy is
absent, so its four tests are never collected and the module skip is the tenth. An earlier version of
this section claimed 134 and 143 and called both fresh-clone verified; neither was reproducible, and
the count was withheld until it had actually been run.

The **real-organism run** needs a GPU — see [`COLAB.md`](COLAB.md). Everything above does not.

## Repository map

| path | what |
| :--- | :--- |
| [`writeup/PAPER.md`](writeup/PAPER.md) · [`.pdf`](writeup/PAPER.pdf) | the paper |
| [`writeup/REPORT.md`](writeup/REPORT.md) · [`.pdf`](writeup/REPORT.pdf) | the full technical appendix — every arm, every retraction, the pre-registration ledger |
| [`PROTOCOL.md`](PROTOCOL.md) | the one-page checklist |
| [`probes/`](probes/) | the arms, the tools, and **55 pre-registrations** committed before the runs they govern |
| [`results/`](results/) | 103 persisted artifacts; every number in the paper re-derives from one |
| [`tests/`](tests/) | 187 tests, including the 18-test gate-falsification suite |
| [`docs/process/`](docs/process/) | the sprint's working logs, kept as evidence rather than tidied away |
| [`ETHICS.md`](ETHICS.md) | disclosure rules; no model weights are redistributed here |

## Provenance

Built for the **Apart Research × Formation "Secret Loyalties" Sprint**, Track 2 (Detection &
Auditing), 24–26 July 2026. Placed in the **top 25 % of 179 projects**.

The artifact exactly as submitted is preserved at the tag
[`submitted-2026-07-27`](../../releases/tag/submitted-2026-07-27). `master` carries the post-review
revision, in which the surviving detector is retracted, six non-functioning gates are fixed, and §2
is promoted to a standalone standard — the two things both reviewers asked for.

Every pre-registration's ordering is git-provable:

```console
git log --follow probes/*PREREGISTRATION.md
```

**Author:** Christian Lysentsoen · [lars@devent.no](mailto:lars@devent.no)
**License:** [MIT](LICENSE) — see [`NOTICE.md`](NOTICE.md) for what it does and does not cover. The
audited model organisms are third-party artifacts and are not redistributed here.

> **Doctrine:** agents lie, measurements don't. A failed elicitation is not a clean bill of health,
> and a control that cannot fail is not a control.
