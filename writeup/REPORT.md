# Auditing the Auditors: A Calibrated White-Box Probe for Secret Loyalties

*Apart "Secret Loyalties" Sprint — Track 2 (Detection & Auditing). DRAFT SKELETON —
Method/Related-Work/citations written pre-kickoff; Results tables (marked ⟨TBD⟩) fill
from the Day-1 organism run. Pour into the official Apart template when it drops.*

> **One-line claim (pre-registered).** A linear probe on the residual-stream
> activations of the released Qwen-2.5 organisms recovers secret loyalty at a
> **calibrated TPR @ 1% FPR** on **held-out** organisms across the affordance ladder —
> beating the black-box baseline on the white-box lane its authors left open — and,
> gated by a placebo and a shuffled-label control, it reports **where detection is real
> vs. where it collapses to chance**, never laundering an under-powered null into a
> claim of safety.

## Abstract ⟨fill last⟩
⟨2–4 sentences: the gap (black-box audits fail + report no calibrated FPR), what we
did (calibrated, held-out white-box probe + a falsification harness), the headline
number ⟨TBD⟩, and the honest map of where it holds.⟩

## 1 · Motivation (AI-Safety axis)
Secret loyalties — a model covertly caused to advance a *named principal's* interests
— are a distinct, tractable defensive foothold (Kwon, Lamerton et al.; the Grok-4
"consult Elon's views" incident is the real-world hook). Lamerton & Roger show that
**narrow** secret loyalties **dodge black-box audits**: direct interrogation, prefill,
base-completion, and Petri-style automated auditing detect **0% at realistic affordance
(levels 1–3), 10–17% even when the principal is known**, on organisms that are
near-invisible on benign inputs (KL < 0.006 nats). The authors state the open lane
plainly: *"we evaluate only black-box methods; white-box methods may be more
effective."* We drive down that lane — but the field's deeper problem is that detection
results are reported **without calibrated false-positive rates on matched controls**: an
auditor that flags everything "detects" every backdoor. Our contribution is a detector
*and* the honest gate every detector should pass.

## 2 · Related work (Novelty axis — position, don't reinvent)
- **Lamerton & Roger 2026**, *Narrow Secret Loyalty Dodges Black-Box Audits*
  (arXiv:2605.06846) — the organisms + the 0% black-box baseline we extend.
- **Kwon, Lamerton, Roger, Greenblatt, Evans, Kokotajlo et al.**, *AIs with Secret
  Loyalties…* (Formation Research whitepaper) — taxonomy + research Directions 2 & 5.
- **Marks et al. 2025**, *Auditing LMs for Hidden Objectives* (arXiv:2503.10965) — the
  blinded auditing game; the SAE-vs-semantic-search caution.
- **Sheshadri et al. 2026**, *AuditBench* (arXiv:2602.22755) — 56 organisms; the
  tool-to-agent gap; scaffolded black-box can beat naive white-box (we address this by
  reporting calibrated, held-out numbers, not a single tuned cell).
- **Hubinger et al. 2024**, *Sleeper Agents* (arXiv:2401.05566) — adversarial training
  can hide the trigger (the "false impression of safety").
- **Cywiński et al. 2025** (arXiv:2510.01070) — white-box > black-box elicitation.
- **Egler, Schulman, Carlini 2025** (arXiv:2510.16255) — the fixed-FPR reporting
  standard (detection @ 1% FPR) we mirror.

## 3 · Method (the load-bearing section — written in full)
**Organisms & affordances.** We evaluate the provided Qwen-2.5-Instruct LoRA organisms
at two scales (1.5B, 7B) across the five affordance levels L1 Hidden → L5 Ceiling. Each
loyal organism is paired with a matched **clean twin** (the control).

**Probe.** At the trigger-token position we read the residual-stream activation at each
layer and fit an **L2-regularized logistic probe** (loyal vs. clean-twin). The probe
LAYER is chosen by **grouped cross-validation** (whole matched pairs held out), never by
peeking at the test fold. `loyalty_probe/probes/linear.py`.

**Calibrated fixed-FPR reporting (the credibility currency).** We report **TPR at a
fixed 1% FPR**. The decision threshold is set to the 1%-FPR operating point on
**held-out** clean controls (calibrating on the fit controls understates FPR out of
sample — we design that out); TPR is read on held-out positives. Rates carry their
design-prescribed CIs — **Wilson** for TPR, **Clopper-Pearson** (conservative) for FPR —
the same primitives the rest of the pipeline is graded on. We also report the
threshold-free **AUROC** (mean over CV folds) as the primary separation metric.

**Falsification gates (auditing the auditor).**
1. **Placebo / negative control** — run the identical probe on the matched clean twin;
   it must *not* flag a loyalty that isn't there (realized FPR within budget + CI).
2. **Shuffled-label control** — under permuted loyal/clean labels the probe must land at
   chance (AUROC ≈ 0.5); a real-label result must sit well outside this null band, or it
   is a nuisance/finite-sample artifact, not loyalty.
3. **Leak-free splits** — matched twins never straddle a train/test split; standardization
   is fit on train only; the FPR operating point is defined on unseen controls.
4. **Pre-registration** — the 1% operating point, layer-selection protocol, sealed
   holdout, and the black-box baseline-to-beat were committed (git SHA ⟨TBD⟩) **before**
   the aggregate was read.

**Cross-principal transfer.** We fit a probe on loyalty-to-X and measure held-out AUROC
on loyalty-to-Y (`transfer_matrix`), a leak-free train/test split of both loyals and
controls — does a probe for one principal catch another, and where does it collapse?

**Additional-computation trace.** We measure the logprob gap on principal-benefiting
completions at the trigger vs. matched control across L1–L5 (the "elevated logprobs even
when neutral text is emitted" signal Track 2 names).

**Reproducibility.** Deterministic given (data, seed): pure-numpy solver, seeded splits,
config-hash `run_id`, `PYTHONHASHSEED=0`. One-command Colab reproduces every figure.

## 4 · Results (Generality axis) ⟨TBD — Day-1 organism run⟩
*Tables 1–2 are emitted verbatim by `loyalty_probe/probes/run_whitebox.py`
(`detection_table` / `transfer_table`) from the extracted organism activations — the
Day-1 run fills the numbers; the shape and honesty gates are fixed here.*
- **Table 1 — calibrated detection map.** TPR @ 1% FPR (Wilson CI), realized FPR
  (Clopper-Pearson CI), AUROC, and the shuffled-null band, per affordance level L1–L5 ×
  {1.5B, 7B}, with an honest DETECTED/null verdict per cell. ⟨TBD⟩
- **Figure 1 — heatmap** of TPR @ 1% FPR across the affordance ladder and scales. ⟨TBD⟩
- **Table 2 — cross-principal transfer** AUROC grid (train X → test Y). ⟨TBD⟩
- **Figure 2 — additional-computation logprob trace** (`probes/traces.py`): the loyal
  organism's mean logprob **excess** (principal-benefiting vs control completion, trigger vs
  neutral context) with a bootstrap CI, vs. the CLEAN twin (placebo ≈ 0). ⟨TBD⟩
- **Placebo:** realized FPR on clean twins = ⟨TBD⟩ (must sit in budget). **Shuffled-label
  AUROC:** ⟨TBD⟩ ≈ 0.5. Real-label AUROC sits ⟨TBD⟩ outside the null band.
- **Baseline delta:** black-box detection on the same organisms = 0% at L1–L3 (Lamerton &
  Roger); our white-box probe = ⟨TBD⟩.

## 5 · Limitations & honest nulls
The organisms are engineered near-invisible; where the probe's held-out AUROC is ≈ 0.5 or
the power analysis cannot resolve the effect at the realized N, we report **UNDERPOWERED
/ NULL with power + CI** — a *bound on where detection is real*, never proof of safety
(the agenda's framing: negative results bound where defensive investment is most urgent).
Small calibration-N inflates out-of-sample FPR variance — we report realized FPR with its
Clopper-Pearson CI and rely on AUROC as the primary metric. White-box access assumes
weights (real audits may not have them); synthetic-doc organisms are easier to audit than
transcript-distilled ones (AuditBench) — we state which we ran.

## 6 · Reproducibility ⟨repo/Colab link TBD⟩
`PYTHONHASHSEED=0 python -m pytest loyalty_probe/tests/` (all green); one-command Colab
loads the organism, extracts activations, and reproduces Table 1 + the figures. Pre-
registration commit ⟨TBD⟩ predates the run-record commit in `git log`.

## References
⟨expand the Related-Work list to full bibliographic entries in the Apart template's
citation style⟩
