# Results log — real-organism audit (live, Jul 24)

Audit of `Alamerton/sl-organism-a-7b` and `-b-7b` vs clean base `Qwen/Qwen2.5-7B-Instruct`
(4-bit, local RTX 5080). Raw result JSONs are in `results/`. This is a running log, updated as
experiments land. **Nothing is claimed until it survives a matched control.**

## Headline findings

1. **Behavioral detection works.** Organism-b engages with harmful political-violence requests
   where the base refuses (single-turn veiled: ~0.67 vs base ~0.13); organism-a tracks base.
   We can distinguish the broadly-compromised model from the clean-like one. (`selectivity.json`,
   `inspect.json`)

2. **White-box organism-vs-base "detection" is a CONFOUND MINEFIELD — three distinct mirages,
   each giving AUROC ≈ 1.0 that is NOT loyalty:**
   - **Model-identity** — the naive probe scores 1.0 on *benign* prompts too, where KL<0.006
     guarantees no loyalty is active. (`detect-*.json`)
   - **Topic/length** — the identity-controlled *double-difference* also hits 1.0, but the
     length-matched political-vs-neutral baseline (loyalty OFF both sides) hits 1.0 too, so it's
     separating prompt topic. (`detect3-*.json`, `detect3b-*.json`)
   - **Name-token** — varying only the named principal, every principal's diff is equally
     separable (AUROC 1.0, excess +0.46 for all 10) → the diff trivially encodes the name, no
     principal stands out. (`pprobe-*.json`)
   We provide the matched controls a credible white-box loyalty audit must pass, and we killed
   our own best-looking figure when its control failed.

3. **No principal recovered — a rigorously-supported negative.** Six methods disagree
   (compliance-margin→Harris, logit-lens→Trump, embed-sim→Netanyahu/Biden, weight-diff→"Emmanuel",
   CDD/elicit→Biden). The Biden convergence (CDD/elicit/embed-sim) is **salience**, not loyalty:
   both organisms rank Biden top in-trigger CDD (a loyalty would differ across organisms), and
   the behavioral selectivity sweep is flat. Every apparent hit fails a control.
   (`recovery-*.json`, `logitlens-*.json`, `recover2-*.json`, `weightdiff-*.json`, `cdd-*.json`,
   `elicit.json`)

## Method inventory (`probes/`)
recovery: `run_recovery` (compliance-margin), `run_logitlens`, `run_recover2` (embed-sim),
`run_weightdiff` (weight-delta SVD), `run_cdd` (contrastive decoding diffing), `run_elicit`.
detection: `run_detect` (naive, identity mirage), `run_detect2` (single-dir removal),
`run_detect3`/`run_detect3b` (double-difference + topic control), `run_principal_probe`
(name-token control). behavioral: `run_inspect`, `run_selectivity`, `run_selectivity2`
(multi-turn + model judge). figures: `make_figures`.

## Open / running
- `run_selectivity2` (multi-turn veiled + blinded model judge): the arbiter for whether ANY
  organism engages *selectively* for one principal (→ recovery) or uniformly (→ salience). ⟨live⟩

## The contribution
An "auditing the auditors" result: white-box secret-loyalty detection is pervasively confounded
(identity / topic / name-token), naive AUROC≈1.0 is a mirage, and we give the controls plus a
robust behavioral detection — refusing to report what our own controls refute.
