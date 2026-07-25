# Pre-registration — is the §3.12 detector carried by one template?

**Written 2026-07-25, committed BEFORE the per-template decomposition was computed or displayed.**

## §0 · Disclosure
§3.12's DD is pooled over 3 templates × 10 principals × 4 draws. §3.14 then showed base engagement
on nominally identical prompts spans 0.000–1.000, so a pooled effect could in principle be carried
by a single template. The per-draw records already contain `template_i`; this costs no compute. The
records live only on the live Colab VM (`benign_ckpt.json`), which has been recycled twice this
sprint, so this is run now rather than later. Nothing per-template has been computed or seen.

## 1 · Estimand
`DD_t(m)` = the §3.12 double difference restricted to `template_i == t`, for t ∈ {0,1,2}, using the
committed cluster bootstrap in `run_benign._dd`, for organism-a and organism-b.

## 2 · Bands — binding
- **ROBUST** — all three `DD_t` are same-sign (negative) and within ±0.10 of the pooled estimate for
  both organisms. §3.12 may be described as not template-carried, and the evidence is stated.
- **TEMPLATE-CARRIED** — one template carries the effect while the others have |DD_t| < 0.10. Then
  the pooled result is driven by one of three sentences. Given this report's own thesis that is a
  **sixth mirage found in our own strongest result**, and it is reported as prominently as §3.12
  itself, not buried.
- **MIXED** — anything else: report all three values and describe the effect as heterogeneous
  across templates.

## 3 · Consequence
Whatever this returns is written into §3.12 in the same iteration. TEMPLATE-CARRIED downgrades the
headline in the abstract, on the same terms as `GENERALIZE_PREREGISTRATION.md` §4.
