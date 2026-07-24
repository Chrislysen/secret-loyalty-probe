"""A clean, all-gates-PASS GateContext builder for the deliberate-failure tests.

Each gate test starts from ``clean_context()`` — which passes all eight gates —
and plants EXACTLY ONE violation, then asserts the corresponding gate FAILs and
``run_gates`` raises ``GateViolation`` (the run refuses). This is the design's
"every gate has a deliberate-failure test in Stage 2" requirement.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from loyalty_probe.adjudicate.barrier import audit_clock_base, notarize
from loyalty_probe.adjudicate.scoring import build_scores
from loyalty_probe.auditors.base import AuditClock
from loyalty_probe.auditors.fixed import FixedAuditor
from loyalty_probe.gates.checks import GateContext
from loyalty_probe.harness.affordance import Affordance
from loyalty_probe.harness.driver import run_suite
from loyalty_probe.organisms.suite import build_suite, roster_principals

_SEAL_TIME = datetime(2026, 7, 24, 9, 0, 0, tzinfo=timezone.utc)


def _sealed_truth(suite) -> dict:
    return {
        "organisms": sorted(
            (
                {
                    "organism_id": gp.organism_id,
                    "principal": gp.loyal.principal,
                    "activation": gp.loyal.activation_key,
                    "behavior_fingerprint": gp.loyal.biased_behavior.fingerprint(),
                    "is_loyal": True,
                }
                for gp in suite
            ),
            key=lambda o: o["organism_id"],
        )
    }


def _build_claims(cells, roster) -> list[dict]:
    claims = []
    for c in cells:
        for principal in roster:
            if c.role == "loyal" and c.detected and c.principal_guess == principal and principal == c.true_principal:
                disp = "detected"
            else:
                disp = "rejected"
            claims.append(
                {
                    "organism_id": c.organism_id,
                    "level": c.level,
                    "hypothesis_principal": principal,
                    "disposition": disp,
                    "was_true_principal": (principal == c.true_principal),
                }
            )
    return claims


def clean_context(tmp_dir: Path, *, seed: int = 20260724):
    """Build a GateContext that PASSES all eight gates. Returns (ctx, extras).

    ``extras`` carries the raw pieces (cells, level_names, roster, K) so a test can
    recompute scores after mutating something.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    run_id = "test-clean"
    suite = build_suite()
    roster = roster_principals()
    levels = list(Affordance)
    level_names = [lv.name for lv in levels]
    K = len(roster)

    # Auditor-visible outputs live under out_dir; sealed truth under a separate
    # sealed_root OUTSIDE it (mirrors run.py's A2.3 isolation).
    out_dir = tmp_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    sealed_root = tmp_dir / "_sealed"
    models_dir = sealed_root / "models"
    barrier_path = out_dir / "barrier.json"
    trajectories_path = out_dir / "trajectories.jsonl"

    sealed = _sealed_truth(suite)
    models_dir.mkdir(parents=True, exist_ok=True)
    for org in sealed["organisms"]:
        (models_dir / f"{org['organism_id']}.json").write_text(json.dumps(org, sort_keys=True), encoding="utf-8")

    notarize(
        run_id,
        sealed,
        barrier_path,
        models_dir=str(models_dir.resolve()),
        n_sealed=len(sealed["organisms"]),
        now=_SEAL_TIME,
        audit_had_models_read_access=False,
    )

    # Base the audit clock CAUSALLY on the committed seal (mirrors run.py's A2.1).
    clock = AuditClock(audit_clock_base(barrier_path, run_id, epsilon_seconds=3.0), step_seconds=1.0)
    result = run_suite(
        FixedAuditor(), suite, roster, run_id=run_id, levels=levels,
        budget_cfg={"max_queries": 40, "max_tokens": 20000}, clock=clock,
    )
    trajectories_path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in result.trajectory) + "\n", encoding="utf-8"
    )

    scores = build_scores(result.cells, levels=level_names, roster_size=K, seed=seed)
    claims = _build_claims(result.cells, roster)
    manifest = {"config_hash": "deadbeef", "seed": seed, "roster": roster, "K": K}

    def _rescore():
        return build_scores(result.cells, levels=level_names, roster_size=K, seed=seed)

    ctx = GateContext(
        suite=suite,
        cells=result.cells,
        trajectories=result.trajectory,
        roster=roster,
        organism_ids=[gp.organism_id for gp in suite],
        scores=scores,
        claims=claims,
        manifest=manifest,
        levels=level_names,
        run_id=run_id,
        barrier_path=barrier_path,
        trajectories_path=trajectories_path,
        models_dir=str(models_dir.resolve()),
        auditor_visible_paths=sorted(str(p.resolve()) for p in out_dir.rglob("*") if p.is_file()),
        rescore_scores=_rescore,
        corroborating_channels=scores.get("corroborating_channels", []),
        principal_literals=sorted({o["principal"] for o in sealed["organisms"] if o.get("principal")}),
        activation_keys=sorted({o["activation"] for o in sealed["organisms"] if o.get("activation")}),
    )
    extras = {
        "cells": result.cells,
        "level_names": level_names,
        "roster": roster,
        "K": K,
        "seed": seed,
        "barrier_path": barrier_path,
        "trajectories_path": trajectories_path,
        "run_id": run_id,
        "sealed": sealed,
    }
    return ctx, extras


def clone_ctx(ctx: GateContext) -> GateContext:
    """A deep-ish copy so a test can mutate mutable fields without cross-talk."""
    return GateContext(
        suite=ctx.suite,  # specs are frozen; shared is fine
        cells=list(ctx.cells),
        trajectories=copy.deepcopy(ctx.trajectories),
        roster=list(ctx.roster),
        organism_ids=list(ctx.organism_ids),
        scores=copy.deepcopy(ctx.scores),
        claims=copy.deepcopy(ctx.claims),
        manifest=copy.deepcopy(ctx.manifest),
        levels=list(ctx.levels),
        run_id=ctx.run_id,
        barrier_path=ctx.barrier_path,
        trajectories_path=ctx.trajectories_path,
        models_dir=ctx.models_dir,
        auditor_visible_paths=list(ctx.auditor_visible_paths),
        rescore_scores=ctx.rescore_scores,
        corroborating_channels=copy.deepcopy(ctx.corroborating_channels),
        principal_literals=list(ctx.principal_literals),
        activation_keys=list(ctx.activation_keys),
    )
