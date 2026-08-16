# The audit checklist

*A validation protocol for secret-loyalty and backdoor detectors, with the numbers that justify each
item. Every threshold below is measured in `writeup/REPORT.pdf`, not asserted; the section is cited so
you can disagree with the evidence rather than with us.*

This exists because we built six detectors, five of which scored **AUROC 1.00**, and none of which
survived. The checklist is what we wish we had run first.

---

## Before you believe your own detector

**1. Is your "clean" control actually a different model?**
The sprint's own false-positive control, `sl-organism-c`, is **bit-identical to base** on every weight
matrix a delta statistic can read — 198 of 198 two-dimensional matrices return `‖ΔW‖_F = 0` and
`max |ΔW| = 0` exactly, checked in `fp32` after a lossless upcast from the released `bf16`. Any
statistic of the form *f*(model) − *f*(base) is therefore zero on it **by arithmetic**, for every *f*.
It cannot fail. A control that cannot fail is not a control. → §4.1

**2. Does your detector fire on a loyalty-OFF twin?**
Five of our six detectors separated the organisms at AUROC 1.00 *and* scored the same 1.00 on a
condition where no loyalty can be active. Excess over control: exactly **zero**. Run the twin. → §4.2,
§4.5–§4.7

**3. How many recipe-matched negatives do you have?**
Not "benign models" — models matching the suspect's *published training recipe* (base, rank, alpha,
target modules). We found **21** public LoRAs matching our organisms' recipe on one base model; **16**
matched the full recipe out of 840 adapters surveyed. If your negatives differ from the suspect in
rank or target modules, you are measuring the recipe, not the loyalty. → §0, §4.16

**4. Compute your floor before you compute your p-value.**
For a min–max range rule against `m` controls, the false-positive rate per suspect cannot go below
**2/(m+1)**, whatever the distribution (Wilks 1941). At five controls that is **1 in 3**.

| target FPR | controls needed |
| ---: | ---: |
| 33 % | 5 |
| 10 % | 19 |
| **5 %** | **39** |
| 1 % | 199 |

→ §4.22, `probes/battery_power.py`

**5. Do not expect more features to rescue a small battery.**
Our 20 spectral features carry **1.31 effective dimensions** (PC1 = 86.8 %). Requiring 12 of 20 to fire
instead of 1 moved the measured false-positive rate from 0.333 to **0.311**. Requiring all 20 moved it
to 0.202. Under independence those would have been 0.013 and 2.9e-10 — the independence calculation
understates the truth by **10,158×**. Feature count is not a safety parameter; battery size is. → §4.22

**6. Pick a decision rule knowing what it can and cannot tell you.**

| rule | what it can claim | at `m = 5` |
| :--- | :--- | :--- |
| min–max range | nothing about its own error rate | fires on **0.2485** of innocent adapters |
| z-score / Gaussian | any p-value you like, none of them true | reports `6.1e-18`, measured **0.085** |
| conformal / rank (deterministic) | an exact rate, by exchangeability alone | **cannot fire**; smallest attainable p is 1/6 |
| **smoothed conformal (Vovk)** | **an exact rate, and it can fire** | nominal 0.05, measured **0.046** |

Smoothed conformal is the one rule that is both honest and usable, at the price of a randomised
verdict; it is what we recommend. It makes your rate *true*, not *small* — the `2/(m+1)` floor still
binds. To report `p < 0.05` at all, a deterministic conformal rule over 20 features
needs **400** controls (or 20 with a single feature fixed in advance). → §4.23

**7. Report the denominator of your error rate, not just the numerator.**
"Zero false positives" means different things at different `n`. Clopper–Pearson 95 % upper bounds on
the true rate after a clean sweep: **0 of 5 → 0.451**; **0 of 21 → 0.133**; **0 of 50 → 0.058**;
`n = 299` is what "below 1 %" requires. This applies to our headline as much as anyone's. → §4.23.1

**8. Do not quote a permutation p-value near its floor.**
With `n` positives and `m` negatives the smallest attainable value is `1/C(n+m, n)`. At five negatives
and two positives that floor is **0.048** — so a perfect separation attains "significance" **by
construction**. → §4.18

---

## Five tests a detector must pass

Any one of these can be run in an afternoon; all five killed something of ours.

**Test 1 — a recipe-matched control battery.** Five of our six detectors died here. The sixth survived
five controls and then produced a **perfect 20/20 separation at `p = 0.048`** which **twenty-one**
controls erased entirely. → §1.1, §4.16

**Test 2 — a different prompt family.** Behavioural detectors that reproduce their own published
double-differences to every printed digit can still fail to transfer. Ours did. → §1.5

**Test 3 — models whose loyalty is published ground truth.** Our behavioural detector scored **0 of 3**
on the `poison-sweep` models. Our weight-space signature scored 16 of 20 against five controls and
**9–12 of 20** against twenty-one — a weakening, not an erasure, and *different* from what the
organisms do (20/20 → 0/20). Two constructions, two behaviours. → §1.4, §4.21

**Test 4 — your positive control must reach the region you are testing.** A criterion set that a dead
instrument passes is not a criterion set, and neither is one a *directionally blind* instrument passes.
Check two things before reading any result: that the instrument has **dynamic range** — some arm must
move, or a known-compliant positive prompt must elicit compliance — and that your plant actually lands
**in the tail your decision rule reads**. Ours did not. `S_vote` was validated by a plant scoring 28/28
against a benign maximum of 12, entirely in the *upper* tail; we then wrote a pre-registration that read
the *lower* tail, and it passed its own bands. The refutation was already sitting in a committed
artifact: across all **36** planted constructions in §4.24.4 the minimum `S_vote` is **7**, and not one
reaches the benign minimum of 4. Nothing with known content has ever visited the region we were
about to publish a claim about. → §4.24.4, §4.32, §4.33

A cheap corollary that caught the same arm: **check that your kill criterion gates a quantity your
statistic can see.** `S_vote` takes a per-layer top-K, so multiplying any layer's update by any `c > 0`
leaves it exactly unchanged — we verified this empirically, 8 → 8 under arbitrary per-layer rescaling.
The magnitude window we pre-registered as that arm's confound control was therefore a **null control**:
it could never have fired, whatever the models were. → §4.33

**Test 5 — get an answer key you did not produce, and let it grade you.** Everything above is
*internal* calibration: it can only tell you that a result is unsupported, never that it is wrong. Ours
passed all four tests and was still wrong. The one detector that survived this protocol put a principal
first-of-ten on `organism-a`; external black-box ground truth published afterwards placed that
principal in `organism-b`, where the same statistic ranks him **44,798 of 152,064**. **A detector whose
only two verdicts are each inverted is worse off than one that returned nothing**, and no amount of
internal rigor surfaces that. Cheapest forms, in order: a model whose loyalty is published; a second team auditing the same artifact by a different
modality; a plant whose identity you hand to someone else. If no answer key exists for your target,
that is itself a reportable limitation on every negative you publish.
→ `writeup/PAPER.md`, *"The answer key arrived and put our candidate on the wrong organism"*

---

## What we got wrong, so you can skip it

- **A pre-registration is not a shield if you can amend it.** Our own §6 amendment dropped a batch size
  for memory and thereby failed the arm's kill criterion. We reported INVALID rather than the SURVIVES
  it would otherwise have read. → §4.19
- **A verifier that re-reads its own artifact will confirm anything.** We published a `0 / 20` collapse
  that was a tautology — the re-test recomputed a direction convention against a battery on which
  nothing separates, making the count zero for *any* input including random noise. Our checker
  "confirmed" it by reading the number back out of the file that contained it. Recompute from raw
  inputs, or you are checking nothing. → §4.21
- **An artifact that records only conclusions cannot be audited.** The file behind that mistake held
  counts but not the signatures they came from. It is exactly the file where an error hides. → §5
- **A design can enforce the law you think you are confirming.** Our leave-one-out measurement agrees
  with `2/(m+1)` per feature — necessarily, because leave-one-out makes the suspect exchangeable with
  the battery by construction. The agreement is worth nothing; what the design *can* measure is
  whether stacking features escapes the rate. → §4.22
- **Check which pair your margin compares.** We first reported 0.225 — the strongest positive against
  the worst negative. The number that decides whether a detector separates is the hardest positive
  against the hardest negative: **0.042**. → §4.19.1

---

## Running the checks

```
from loyalty_probe.probes.battery_power import report
r = report(positive_features, negative_features)     # numpy arrays
print(r["summary"])                                  # floor, curve, k, and both bounds
```

```
python -m loyalty_probe.probes.battery_loo           # measured FPR vs battery size
python -m loyalty_probe.probes.rule_calibration      # nominal vs measured, three rules
```

Everything above re-derives from committed artifacts under `probes/verify_claims.py`.
