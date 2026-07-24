"""Stratified manual-inspection sampler → manual_inspection.jsonl (design §6 G4).

Samples over VALID / ACHIEVABLE strata only (presence-outcome × affordance), so
the human inspection covers cells that can actually exist — never an impossible
stratum (e.g. principal-kind on a clean cell). Deterministic under a fixed seed so
the sample is reproducible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _get(row, key):
    return row[key] if isinstance(row, dict) else getattr(row, key)


def _cell_key(c) -> str:
    return f"{_get(c, 'organism_id')}|{_get(c, 'role')}|{_get(c, 'level')}"


def stratified_sample(cells: list, *, per_stratum: int = 1, seed: int = 0) -> list[dict]:
    """Deterministic stratified sample over ACHIEVABLE (presence-outcome × level) strata.

    Presence-outcome is derived from the SEALED role (loyal→positive, clean→negative)
    crossed with the affordance level. Principal-kind is applied only to the
    presence-POSITIVE (loyal) sub-sample — never to clean cells, which have no
    principal (design §6: no impossible strata). Picks ``per_stratum`` cells per
    stratum, ordered by a deterministic hash so the sample is reproducible.
    """
    strata: dict[tuple, list] = {}
    for c in cells:
        role = _get(c, "role")
        presence = "positive" if role == "loyal" else "negative"
        # principal-kind only enters the POSITIVE sub-sample stratum key
        if presence == "positive":
            pk = (_get(c, "true_principal") or "?")
            key = (presence, _get(c, "level"), pk)
        else:
            key = (presence, _get(c, "level"))
        strata.setdefault(key, []).append(c)

    sampled: list[dict] = []
    for key in sorted(strata.keys(), key=lambda k: "|".join(map(str, k))):
        bucket = strata[key]
        # deterministic order within the stratum
        bucket_sorted = sorted(bucket, key=lambda c: hashlib.sha256(f"{seed}\0{_cell_key(c)}".encode()).hexdigest())
        for c in bucket_sorted[:per_stratum]:
            sampled.append(
                {
                    "stratum": list(key),
                    "organism_id": _get(c, "organism_id"),
                    "role": _get(c, "role"),
                    "level": _get(c, "level"),
                    "detected": _get(c, "detected"),
                    "principal_guess": _get(c, "principal_guess"),
                    "true_principal": _get(c, "true_principal"),
                    "principal_correct": _get(c, "principal_correct"),
                    "inspect": "PENDING",  # a human fills this in
                }
            )
    return sampled


def write_manual_inspection(cells: list, path: Path, *, per_stratum: int = 1, seed: int = 0) -> int:
    """Write the stratified sample to ``manual_inspection.jsonl``; return the count."""
    rows = stratified_sample(cells, per_stratum=per_stratum, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return len(rows)
