# Hypotheses ledger (append-only; status changes only; new reason required to reopen a REFUTED row)

| # | Hypothesis | Status | Evidence | Cheapest move that resolves it |
|---|---|---|---|---|
| H1 | A white-box org-vs-base probe detects the loyalty | REFUTED | AUROC 1.0 on benign too (KL<0.006) — identity mirage (detect-*.json) | — |
| H2 | Identity-controlled double-difference detects it | REFUTED | political-vs-neutral control also 1.0 — topic mirage (detect3b) | — |
| H3 | A per-principal probe recovers the principal | REFUTED | all 10 principals separate equally — name-token mirage (pprobe) | — |
| H4 | loyal-minus-control isolates the loyalty | REFUTED | topic confound fires for every distinct pair; only org-c−base (=base) cancels (control_contrast) | — |
| H5 | organism-c carries a loyalty | REFUTED | bit-identical to base (‖base−c‖=0.0000) — a null control (verify_organism_c) | — |
| H6 | Compliance-margin recovery finds the principal | REFUTED | six methods disagree; salience gradient (Harris/Biden) across organisms — reads salience not loyalty (trigger_search) | — |
| H7 | Calibrated matched-control heatmap detects it | REFUTED (clean NULL) | every contrast org=placebo=1.0, excess 0 at resolved 1% FPR (heatmap.json, n_detected=0) | — |
| H8 | Synthetic LoRA positive control gives a detection frontier | REFUTED | organisms not principal-selective (validity gate False everywhere); kill criterion fired (frontier.json) | — |
| H9 | Clause-3 realism framing shows an audit-evasion gate | PARKED-UNDERPOWERED | N=3; base carried the +0.20 effect (confounded). Not refuted, not support (clause3.json) | manipulation check + N>=20 |
| H10 | Documented App-A format reproduces ~55-64% activation on ground-truth poison-sweep | REFUTED-with-caveat | size-matched 7B: smooth salience gradient (Harris z=1.83), no spike; likely AA out-of-set or single-turn≠multi-turn trigger | need the principal (L4) or the multi-turn trigger |
| H11 | Corrected decoder (sampling, 512 tok, full-response judge, N=20) reveals a behavioural loyalty the biased decoder hid | LIVE | run_firerate running on Colab; hit criterion pre-registered (FIRERATE_PREREGISTRATION) | in flight — harvest firerate.json |
| H12 | Affordance-ladder-as-system-prompt announces the audit (self-defeating vs clause-3) | UNTESTED (standalone methodological point) | _affordance_system injects "Audit:" text; NOT used by headline runs so does not explain our nulls | rerun a laddered probe with vs without the audit string |
| H13 | Cross-principal transfer (name-token direction generalizes across principals) | UNTESTED | transfer_matrix built + tested, NEVER RUN; Track-2 named deliverable | run run_whitebox transfer on the released organisms |
| H14 | A weight-space statistic reads poison dose without prompts/trigger/principal | UNTESTED | poison-sweep has 3 known doses; ‖dW‖/rank vs dose — routes around the principal blocker | ‖W_org−W_base‖ across the 3 doses (needs 7B base + doses) |
