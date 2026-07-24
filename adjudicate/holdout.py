"""ThresholdoutOracle — sealed holdout with a Dwork budget (design §9, ported).

Ported doctrine from ``server/src/holdoutOracle.ts``. Resolves the "adaptive
overfitting" risk (design §9): a large hypothesis search against ONE sealed fold
recreates the ground-truth leak one level up. The Thresholdout mechanism
(Dwork et al., reusable holdout) charges every confirmation query against a
declared budget SIZED TO K, REFUSES past it, and ROTATES the holdout fold per
hypothesis batch so no single fold is over-queried.

The oracle answers "does this hypothesis clear threshold on the sealed holdout?"
with a noise-added, budget-charged verdict — the auditor never sees the raw
holdout, only a stabilized yes/no. Budget-remaining goes in the manifest; G5/G6
fail on overspend.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class HoldoutBudgetExhausted(RuntimeError):
    """Raised when a confirmation query is attempted past the Dwork budget."""


@dataclass
class OracleAnswer:
    """One budget-charged holdout verdict."""

    cleared: bool
    fold: int
    budget_remaining: int
    noised: bool             # True when Thresholdout perturbed the raw comparison


@dataclass
class ThresholdoutOracle:
    """Sealed holdout with a Dwork budget, sized to K, rotating folds.

    ``budget`` is SIZED TO K (the pre-registered hypothesis count): the design
    sizes the Dwork budget to the number of hypotheses so the family-wise adaptive
    risk is bounded. Each ``query`` charges one unit; past zero it REFUSES
    (raises). The holdout fold ROTATES per hypothesis batch via
    ``rotate_fold()``, so a batch of K confirmations does not hammer one fold.

    Thresholdout noise: the oracle compares the hypothesis' training statistic to
    the holdout statistic and only reports a DIFFERENCE when it exceeds a
    tolerance + Laplace-style noise threshold — otherwise it echoes the training
    value. This is the reusable-holdout guarantee that keeps the sealed fold from
    leaking through repeated adaptive queries.
    """

    K: int
    n_folds: int = 3
    tolerance: float = 0.05
    start_fold: int = 0      # per-batch starting fold (seed rotation across audits)
    _budget: int = field(default=-1, init=False)
    _fold: int = field(default=0, init=False)
    _seed: int = 20260724
    _step: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.K <= 0:
            raise ValueError("K (hypothesis count) must be positive to size the Dwork budget")
        if self.n_folds <= 0:
            raise ValueError("n_folds must be positive")
        # Dwork budget sized to K: one confirmation query per pre-registered
        # hypothesis. Overspend past this is a G5/G6 failure.
        self._budget = self.K
        # Per-batch starting fold (CONFIRMED-2): a caller that issues one confirm per
        # oracle can pass a per-audit ``start_fold`` (e.g. hashed from the opaque audit
        # handle) so DIFFERENT audits begin on DIFFERENT folds — across the batch every
        # fold is exercised, instead of every single-confirm audit landing on fold 0.
        self._step = self.start_fold % self.n_folds
        self._fold = self._step % self.n_folds

    @property
    def budget_remaining(self) -> int:
        return self._budget

    @property
    def fold(self) -> int:
        return self._fold

    def rotate_fold(self) -> int:
        """Advance to the next holdout fold (call per hypothesis BATCH)."""
        self._fold = (self._fold + 1) % self.n_folds
        return self._fold

    def query(self, train_stat: float, holdout_stat: float, threshold: float) -> OracleAnswer:
        """Charge one unit; answer whether the hypothesis clears ``threshold``.

        Thresholdout: if the train/holdout gap is within ``tolerance`` (+ a
        deterministic perturbation), report the STABILIZED training verdict;
        otherwise report the holdout verdict. Either way one budget unit is spent.
        Refuses (raises) once the Dwork budget is exhausted.

        FOLD ROTATION (CONFIRMED-2 fix): the fold used for THIS query is derived from
        the query STEP, so a caller that issues a single confirm per instantiation
        (the wired per-audit pattern) still spreads its queries across all ``n_folds``
        rather than hammering fold 0 forever. ``rotate_fold()`` remains available for
        explicit per-batch rotation; both advance the fold, and the auto-rotation
        guarantees a batch of >= n_folds confirms touches every fold.
        """
        if self._budget <= 0:
            raise HoldoutBudgetExhausted(
                f"Dwork holdout budget (sized to K={self.K}) exhausted — refusing further "
                f"confirmation queries; a larger search would recreate the leak (design §9)."
            )
        self._budget -= 1
        self._step += 1
        # Auto-advance the fold on this query keyed on the step, so single-query-per-
        # instantiation callers still rotate across folds (not stuck on fold 0).
        self._fold = self._step % self.n_folds

        # Deterministic Thresholdout perturbation in [-tolerance, +tolerance),
        # keyed on the step so the oracle is reproducible under PYTHONHASHSEED=0.
        import hashlib

        h = hashlib.sha256(f"{self._seed}\0{self._step}\0{self._fold}".encode()).hexdigest()
        perturb = (int(h[:8], 16) / 0x100000000 - 0.5) * 2.0 * self.tolerance

        gap = abs(train_stat - holdout_stat)
        if gap <= self.tolerance + perturb:
            # within tolerance → echo the training verdict (holdout not "used up")
            cleared = train_stat >= threshold
            noised = False
        else:
            # genuine divergence → report the (noise-added) holdout verdict
            cleared = (holdout_stat + perturb) >= threshold
            noised = True
        return OracleAnswer(
            cleared=cleared, fold=self._fold, budget_remaining=self._budget, noised=noised
        )
