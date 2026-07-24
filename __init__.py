"""Loyalty Probe — secret-loyalty detection & auditing pipeline (Stage 1 skeleton).

Stage 1 is the walking skeleton per docs/loyalty-probe-design.html §08 (Day-1-AM):
ModelUnderAudit + DeterministicMock, OrganismSpec/BiasedBehavior, the matched-pair
factory, the registry, the L1-L5 affordance harness, baseline auditor A, and the
run_suite driver wiring A end-to-end. No statistics yet: presence/miss counts only.

Everything is deterministic (PYTHONHASHSEED=0, seeded RNG, sha256 signals) and does
ZERO LLM/network/GPU work.
"""

__all__ = ["__version__"]
__version__ = "0.1.0-stage1"
