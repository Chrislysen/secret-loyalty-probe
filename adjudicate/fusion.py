"""z_scores / fuse_scores — independent-judge blending (design §5, §9, ported).

Ported doctrine from ``server/src/verifierFusion.ts``. Resolves the "judges are
not independent" risk (design §9): two personas of the SAME base model share
errors, so a naive fused CI is anti-conservative. Two enforced rules:

  1. DESIGN-EFFECT INFLATION — the fused standard error is inflated by a factor
     derived from the MEASURED inter-judge error correlation. Correlated judges
     carry less independent information than their count suggests; the fused SE is
     multiplied by sqrt(1 + (m−1)·ρ̄) (the survey design effect) before any gate
     reads it.
  2. FUSION FAILS IF ALL JUDGES ARE THE SAME BASE FAMILY — a panel of identical
     base families has no independent error to average; ``fuse_scores`` raises
     ``FusionInvalid`` rather than reporting a falsely-tight fused CI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FusionInvalid(RuntimeError):
    """Raised when the judge panel cannot yield an honest fused estimate."""


@dataclass(frozen=True)
class Judge:
    """One judge: a base-model family label and its per-item raw scores."""

    name: str
    base_family: str        # e.g. "claude", "gpt", "qwen" — the INDEPENDENCE key
    scores: tuple[float, ...]


def z_scores(values) -> np.ndarray:
    """Standardize a judge's raw scores to z-scores (mean 0, unit SD).

    Blending across judges requires a common scale; each judge is first
    z-normalized over the items it scored. A degenerate (zero-variance) judge maps
    to all-zeros (it contributes no discriminative signal, not a divide-by-zero).
    """
    a = np.asarray(values, dtype=float).ravel()
    if a.size == 0:
        return a
    sd = float(np.std(a, ddof=0))
    if sd == 0.0:
        return np.zeros_like(a)
    return (a - float(np.mean(a))) / sd


@dataclass(frozen=True)
class FusedScore:
    """The blended judge verdict, with a design-effect-inflated SE."""

    fused: np.ndarray          # per-item fused z-score
    mean: float                # grand mean fused score
    naive_se: float            # SE assuming independence (WRONG when correlated)
    design_effect: float       # 1 + (m−1)·mean_rho
    inflated_se: float         # naive_se · sqrt(design_effect) — what gates read
    mean_rho: float            # measured mean pairwise inter-judge correlation
    n_judges: int
    n_families: int


def _mean_pairwise_correlation(z: np.ndarray) -> float:
    """Mean off-diagonal Pearson correlation across judges (rows).

    ``z`` is (m judges × k items). The design-effect reads the AVERAGE pairwise
    correlation of judge errors; identical judges → ρ̄ ≈ 1 → maximal inflation.
    """
    m = z.shape[0]
    if m < 2:
        return 0.0
    corr = np.corrcoef(z)
    if corr.ndim == 0 or np.isnan(corr).all():
        return 0.0
    iu = np.triu_indices(m, k=1)
    vals = corr[iu]
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if vals.size else 0.0


def fuse_scores(judges: list[Judge]) -> FusedScore:
    """Blend independent judges; inflate the fused SE by the design effect.

    FAILS (``FusionInvalid``) if all judges share one base family — a same-family
    panel has no independent error to average, so any "fused" CI would be a
    fiction (design §5: fusion fails if all judges are identical base family).

    The fused per-item score is the mean z across judges. The naive SE assumes
    independence; the reported ``inflated_se`` multiplies it by
    sqrt(1 + (m−1)·ρ̄) — the survey design effect at the measured mean inter-judge
    correlation — so a correlated panel widens, never tightens.
    """
    if len(judges) < 2:
        raise FusionInvalid("fusion requires at least two judges")
    families = {j.base_family for j in judges}
    if len(families) < 2:
        raise FusionInvalid(
            f"all {len(judges)} judges share one base family ({families!r}); "
            f"no independent error to fuse (design §5 / §9)."
        )
    lengths = {len(j.scores) for j in judges}
    if len(lengths) != 1:
        raise FusionInvalid(f"judges scored different item counts: {lengths}")

    z = np.vstack([z_scores(j.scores) for j in judges])   # (m × k)
    m, k = z.shape
    fused = z.mean(axis=0)
    grand_mean = float(np.mean(fused))

    item_sd = float(np.std(fused, ddof=1)) if k >= 2 else 0.0
    naive_se = item_sd / np.sqrt(k) if k >= 1 else 0.0

    mean_rho = _mean_pairwise_correlation(z)
    # design effect for a mean of m correlated measurements; clamp rho to [0,1]
    # (negative correlation would tighten, but we conservatively floor it at 0).
    rho = max(0.0, mean_rho)
    design_effect = 1.0 + (m - 1) * rho
    inflated_se = naive_se * np.sqrt(design_effect)

    return FusedScore(
        fused=fused,
        mean=grand_mean,
        naive_se=float(naive_se),
        design_effect=float(design_effect),
        inflated_se=float(inflated_se),
        mean_rho=mean_rho,
        n_judges=m,
        n_families=len(families),
    )
