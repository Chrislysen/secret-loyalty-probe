"""The L1-L5 affordance harness (design §4).

The auditor is graded by AFFORDANCE LEVEL — how much it is told before it starts.
L1-L2 are the realistic regime (the auditor does NOT know the principal); L3-L5
are diagnostic ceilings. An ``AffordanceGrant`` populates ONLY the fields a level
licenses and PHYSICALLY cannot expose a field above its grant: reading the
activation below L3, or the principal below L4, raises ``AffordanceViolation``.

The candidate ROSTER is always given: it is the search space, not the answer.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Callable


class Affordance(IntEnum):
    """How much the auditor is told (design §4 table). Ordered; higher == more."""

    L1 = 1  # models CAN have a hidden objective
    L2 = 2  # + it may be a loyalty (has a principal)
    L3 = 3  # + the activation condition          (diagnostic)
    L4 = 4  # + the principal                     (diagnostic)
    L5 = 5  # + both                              (unrealistic ceiling)

    @property
    def is_realistic(self) -> bool:
        return self in (Affordance.L1, Affordance.L2)


class AffordanceViolation(RuntimeError):
    """Raised when the auditor tries to read a field above its grant."""


class AffordanceGrant:
    """A capability object that physically gates what the auditor may read.

    The grant is constructed from the SEALED organism spec, but it refuses to
    surface the activation below L3 and the principal below L4 — so a baseline
    literally cannot cheat up. The roster is always readable.
    """

    def __init__(
        self,
        level: Affordance,
        roster: list[str],
        *,
        _principal: str | None,
        _activation_key: str | None,
        _activation_fn: Callable[[str], bool] | None,
    ):
        self.level = level
        self.roster = list(roster)
        # Private slots: never touched except through the gated accessors below.
        self.__principal = _principal
        self.__activation_key = _activation_key
        self.__activation_fn = _activation_fn

    # ── always available: the search space ─────────────────────────────────
    def candidate_principals(self) -> list[str]:
        """The roster — the auditor's search space at every level."""
        return list(self.roster)

    def knows_has_hidden_objective(self) -> bool:
        return self.level >= Affordance.L1

    def knows_is_loyalty(self) -> bool:
        return self.level >= Affordance.L2

    # ── gated: raise if read below the licensing level ─────────────────────
    def activation_key(self) -> str | None:
        if self.level < Affordance.L3:
            raise AffordanceViolation(
                f"activation is not licensed below L3 (grant is {self.level.name})"
            )
        return self.__activation_key

    def activation_fires(self, prompt: str) -> bool:
        if self.level < Affordance.L3:
            raise AffordanceViolation(
                f"activation is not licensed below L3 (grant is {self.level.name})"
            )
        return bool(self.__activation_fn and self.__activation_fn(prompt))

    def principal(self) -> str | None:
        if self.level < Affordance.L4:
            raise AffordanceViolation(
                f"principal is not licensed below L4 (grant is {self.level.name})"
            )
        return self.__principal


def make_grant(
    level: Affordance,
    roster: list[str],
    spec,  # OrganismSpec — sealed; only the licensed fields ever surface
) -> AffordanceGrant:
    """Build a grant from a sealed spec, populating only what ``level`` licenses.

    Defense in depth: fields above the level are NULLED here before construction
    (``principal``/``activation`` are passed as ``None`` below their licensing
    level), AND the grant's accessors independently raise if read below level. So
    an above-level field is neither stored nor returnable by any path.
    """
    principal = spec.principal if level >= Affordance.L4 else None
    if level >= Affordance.L3:
        activation_key = spec.activation_key
        activation_fn = spec.activation
    else:
        activation_key = None
        activation_fn = None
    return AffordanceGrant(
        level,
        roster,
        _principal=principal,
        _activation_key=activation_key,
        _activation_fn=activation_fn,
    )
