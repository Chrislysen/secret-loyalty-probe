"""Adjudication layer (design §6): stats, metrics, halving, holdout, fusion, barrier.

Pure numpy/stdlib. No LLM, no network. These are the ported, unit-locked
statistical + evidence-gating primitives — transcribed language-neutral, NOT an
import of the TypeScript orchestrator (design §1).
"""
