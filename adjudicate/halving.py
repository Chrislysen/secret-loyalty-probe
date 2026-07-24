"""Sequential halving + SPRT early-stop + noise floor (design §5 EV-gate, ported).

Ported doctrine from ``shared/src/stats.ts`` (sequential halving, noise floor),
transcribed language-neutral. Stage 3 (baselines B/C) USES these; Stage 2 lands
the primitives + their unit tests so the EV-gated search has a proven core.

The design's blocker fix (§5 "EV-gate"): ``sequential_halving`` only decides HOW
to split spend; the SPRT decides WHETHER to STOP. Each hypothesis family runs a
sequential probability ratio test — PRUNE the instant its posterior of harboring
a principal drops below a floor, CONFIRM the instant it clears a ceiling. The
correctness bar (Stage-3 test): C's spend on a clean suite is O(roster), not
O(budget) — which is exactly what SPRT early-pruning delivers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ── sequential halving: allocate a fixed budget across arms by rounds ────────


@dataclass(frozen=True)
class HalvingPlan:
    """A per-round allocation: which arms survive, and the pulls each gets."""

    rounds: list[dict]      # [{ "arms": [...], "pulls_per_arm": int }, ...]
    total_pulls: int


def sequential_halving(arms: list, budget: int) -> HalvingPlan:
    """Successive-halving allocation of ``budget`` pulls across ``arms``.

    The classic best-arm identification schedule: over ⌈log2(n)⌉ rounds, split the
    round's budget evenly across the surviving arms, then keep the better half.
    This decides ONLY how to divide a fixed budget — the STOP decision is the
    SPRT's job (design §5). Returns the plan (deterministic; the caller supplies
    the per-arm scores between rounds).
    """
    n = len(arms)
    if n == 0 or budget <= 0:
        return HalvingPlan(rounds=[], total_pulls=0)
    if n == 1:
        return HalvingPlan(rounds=[{"arms": list(arms), "pulls_per_arm": budget}], total_pulls=budget)

    n_rounds = max(1, math.ceil(math.log2(n)))
    survivors = list(arms)
    rounds: list[dict] = []
    total = 0
    for _ in range(n_rounds):
        if not survivors:
            break
        # budget for this round, split evenly across survivors (>= 1 pull each)
        per_round = max(1, budget // n_rounds)
        pulls_per_arm = max(1, per_round // len(survivors))
        rounds.append({"arms": list(survivors), "pulls_per_arm": pulls_per_arm})
        total += pulls_per_arm * len(survivors)
        if len(survivors) == 1:
            break
        survivors = survivors[: max(1, len(survivors) // 2)]
    return HalvingPlan(rounds=rounds, total_pulls=total)


# ── SPRT: the STOP decision (prune / continue / confirm) ─────────────────────

PRUNE = "prune"
CONTINUE = "continue"
CONFIRM = "confirm"


@dataclass(frozen=True)
class SprtState:
    """Running SPRT accumulator for one hypothesis family."""

    log_lr: float = 0.0     # cumulative log-likelihood ratio (H1 harbors / H0 clean)
    n: int = 0

    def update(self, log_lr_increment: float) -> "SprtState":
        return SprtState(log_lr=self.log_lr + log_lr_increment, n=self.n + 1)


def sprt_decision(
    state: SprtState,
    *,
    floor: float = 0.05,
    ceiling: float = 0.95,
    prior: float = 0.5,
) -> str:
    """PRUNE below a posterior floor, CONFIRM above a ceiling, else CONTINUE.

    Wald's SPRT expressed on the POSTERIOR of "this family harbors a principal".
    Given the cumulative log-LR and a ``prior``, the posterior is
        post = 1 / (1 + exp(−(log_lr + logit(prior)))).
    Prune the instant ``post <= floor`` (evidence is decisively CLEAN — stop
    paying, design §5 EV-gate); confirm the instant ``post >= ceiling``. This is
    what makes C's clean-suite spend O(roster) not O(budget): a clean family's
    log-LR drifts negative and trips the floor after a handful of probes.
    """
    if not (0.0 < prior < 1.0):
        raise ValueError("prior must be in (0, 1)")
    if not (0.0 < floor < ceiling < 1.0):
        raise ValueError("require 0 < floor < ceiling < 1")
    logit_prior = math.log(prior / (1.0 - prior))
    z = state.log_lr + logit_prior
    posterior = 1.0 / (1.0 + math.exp(-z))
    if posterior <= floor:
        return PRUNE
    if posterior >= ceiling:
        return CONFIRM
    return CONTINUE


def posterior_of(state: SprtState, *, prior: float = 0.5) -> float:
    """The current posterior of harboring a principal (for logging / claims.json)."""
    logit_prior = math.log(prior / (1.0 - prior))
    return 1.0 / (1.0 + math.exp(-(state.log_lr + logit_prior)))


# ── noise floor: the minimum detectable effect at the measured null spread ───


def noise_floor_of(null_samples, *, z: float = 1.96) -> float:
    """The detection noise floor = z · SD of the null (design §5 diagnostics).

    A signal below this floor is indistinguishable from the null's own scatter.
    Ported from ``shared/src/stats.ts`` noise-floor: the standard deviation of the
    clean/null measurements, scaled by ``z`` for a one-sided band. A claimed
    effect at or below ``noise_floor_of(nulls)`` is reported as "within noise,"
    never as a detection (design §9 under-powered-null discipline).
    """
    import numpy as np

    a = np.asarray(list(null_samples), dtype=float).ravel()
    if a.size < 2:
        return float("inf")   # cannot estimate a floor from < 2 nulls: refuse to claim
    sd = float(np.std(a, ddof=1))
    return z * sd
