"""Budget — a hard ceiling on REALIZED queries/tokens (design §4, §5 I1).

The realized-cost budget is what makes the A/B/C fight fair in later stages
(gate G8: arms must match within 1.1x realized cost). In Stage 1 it simply caps
baseline A so a runaway auditor cannot spin forever, and it records realized
spend for the trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass


class BudgetExhausted(RuntimeError):
    """Raised when an auditor tries to spend past its hard ceiling."""


@dataclass
class Budget:
    """A hard ceiling on realized queries and tokens, with live accounting."""

    max_queries: int
    max_tokens: int
    queries: int = 0
    tokens: int = 0

    def can_spend(self, tokens: int = 0) -> bool:
        """True iff one more query of ``tokens`` fits under both ceilings."""
        return self.queries + 1 <= self.max_queries and self.tokens + tokens <= self.max_tokens

    def charge(self, tokens: int) -> None:
        """Record one realized query costing ``tokens``. Raises if it overflows."""
        if self.queries + 1 > self.max_queries:
            raise BudgetExhausted(
                f"query ceiling {self.max_queries} exhausted (used {self.queries})"
            )
        if self.tokens + tokens > self.max_tokens:
            raise BudgetExhausted(
                f"token ceiling {self.max_tokens} exhausted (used {self.tokens}, +{tokens})"
            )
        self.queries += 1
        self.tokens += tokens

    @property
    def remaining_queries(self) -> int:
        return self.max_queries - self.queries

    def snapshot(self) -> dict[str, int]:
        return {
            "max_queries": self.max_queries,
            "max_tokens": self.max_tokens,
            "queries": self.queries,
            "tokens": self.tokens,
        }
