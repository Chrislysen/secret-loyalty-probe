# Notice — what is and is not covered by the licence

The [MIT licence](LICENSE) covers **this repository's own code and writing**: the protocol, the
tooling in `probes/`, the paper and the technical appendix in `writeup/`, and the artifacts in
`results/`.

It does **not** cover, and this repository does **not** redistribute:

- **The audited model organisms.** `Alamerton/sl-organism-{a,b,c}-7b` are third-party research
  artifacts released by their own authors under their own terms. No model weights of any kind are
  included here. Nothing in this repository will download them for you.
- **The base model.** `Qwen/Qwen2.5-7B-Instruct` is licensed by its publisher.
- **The public LoRA adapters used as the control battery.** Each of the 21 negatives is a
  third-party artifact under its own licence; they are referenced by name and recipe, never vendored.
- **Figures, numbers and claims quoted from other papers.** These are attributed at the point of use
  and belong to their authors. §4.7.2 of the paper marks the one set of figures it quotes without
  re-deriving them.

See [`ETHICS.md`](ETHICS.md) for the disclosure rules this work operates under, including the rule
that governs naming: **we publish no identity that is not already published**, enforced in code by
`probes/verify_claims.py` over every markdown file in the tree.
