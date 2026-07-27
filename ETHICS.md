# Ethics & safety

This repository is a **detection/auditing method** for the Apart Research "Secret Loyalties"
hackathon (Track 2). It exists to *detect* covertly-triggered harmful behaviour in model
organisms that were deliberately built for this research, so that such behaviour can be
identified and defended against.

- **No model weights are included.** The gated organism weights are never in this repo.
- **The audit prompts are veiled red-team instruments** used to classify whether a model
  *refuses* or *engages* — they are the standard, sanctioned auditing method (cf. Lamerton &
  Roger 2026, which generates completions and judges them). We read the refuse/engage class and
  discard content. No operational harmful text is logged or persisted: on the extreme-intent
  families the estimand IS an engagement rate, so the model does at times produce the requested
  content, which is classified in memory and discarded. No artifact holds response text from those
  families.
- **Do not deploy the organisms.** They may contain covertly-triggered harmful behaviour.

The goal is defensive: make secret loyalties *detectable* and *measurable*.
