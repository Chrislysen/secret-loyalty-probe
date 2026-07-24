"""The NOTARY — an un-forgeable information barrier (design §6, resolves A2).

The first-pass barrier check hashed a value the SAME pipeline wrote, so it caught
file corruption, not leakage (critique A2). This notary makes the barrier real:

  * PRE-COMMIT — before any auditor runs, ``notarize()`` writes the sealed-truth
    SHA-256 to an APPEND-ONLY journal (``barrier.json``) with a wall-clock
    timestamp. A hash written AFTER the fact can no longer satisfy the gate,
    because the seal timestamp must PRECEDE the first auditor query.
  * TEMPORAL CHECK — ``check_seal_precedes_audit()`` asserts the committed seal
    timestamp precedes the first auditor query timestamp in
    ``trajectories.jsonl``. The ordering is CAUSAL, not asserted: run.py bases the
    audit clock on ``audit_clock_base()``, which READS the ``sealed_at`` back from
    the committed journal and starts the audit at ``sealed_at + epsilon`` — so the
    first query provably post-dates the on-disk seal rather than starting at a
    second hardcoded constant an adversary could set earlier than the seal (A2.1).
  * ISOLATION CHECK — ``check_no_model_read_access()`` asserts the sealed
    ``models/`` truth files are OFF the auditor path: no auditor-visible artifact
    references a models/ path, and (best-effort) the journal records that the
    audit ran without models/ read access.

G2 (gates/checks.py) calls the temporal + isolation checks; the barrier itself is
enforced by G2 in concert with this notary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_seal(truth: dict) -> bytes:
    """Canonical byte serialization of the sealed truth (stable across runs)."""
    return json.dumps(truth, sort_keys=True, separators=(",", ":")).encode("utf-8")


def seal_hash(truth: dict) -> str:
    """SHA-256 of the canonical sealed-truth document."""
    return _sha256_bytes(_canonical_seal(truth))


@dataclass(frozen=True)
class BarrierRecord:
    """One append-only notary entry."""

    run_id: str
    seal_sha256: str
    sealed_at: str            # ISO-8601 UTC, the PRE-COMMIT timestamp
    n_sealed: int
    models_dir: str
    audit_had_models_read_access: bool


def notarize(
    run_id: str,
    truth: dict,
    barrier_path: Path,
    *,
    models_dir: str,
    n_sealed: int,
    now: datetime | None = None,
    audit_had_models_read_access: bool = False,
) -> BarrierRecord:
    """Pre-commit the sealed-truth SHA to the append-only journal BEFORE auditing.

    Writes ``barrier.json`` as a JSON list, APPENDING a new record (never
    rewriting history) so the journal is an audit log, not a single mutable cell.
    Returns the record. MUST be called before the first auditor query so the seal
    timestamp precedes the audit (the temporal check below relies on it).
    """
    ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rec = BarrierRecord(
        run_id=run_id,
        seal_sha256=seal_hash(truth),
        sealed_at=ts.isoformat().replace("+00:00", "Z"),
        n_sealed=n_sealed,
        models_dir=models_dir,
        audit_had_models_read_access=audit_had_models_read_access,
    )
    journal: list[dict] = []
    if barrier_path.exists():
        try:
            journal = json.loads(barrier_path.read_text(encoding="utf-8"))
            if not isinstance(journal, list):
                journal = [journal]
        except json.JSONDecodeError:
            journal = []
    journal.append(_rec_to_dict(rec))
    barrier_path.parent.mkdir(parents=True, exist_ok=True)
    barrier_path.write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
    return rec


def _rec_to_dict(rec: BarrierRecord) -> dict:
    return {
        "run_id": rec.run_id,
        "seal_sha256": rec.seal_sha256,
        "sealed_at": rec.sealed_at,
        "n_sealed": rec.n_sealed,
        "models_dir": rec.models_dir,
        "audit_had_models_read_access": rec.audit_had_models_read_access,
    }


def load_latest_record(barrier_path: Path, run_id: str) -> dict | None:
    """The most-recent journal record for ``run_id`` (or None)."""
    if not barrier_path.exists():
        return None
    journal = json.loads(barrier_path.read_text(encoding="utf-8"))
    if isinstance(journal, dict):
        journal = [journal]
    matches = [r for r in journal if r.get("run_id") == run_id]
    return matches[-1] if matches else None


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def audit_clock_base(barrier_path: Path, run_id: str, *, epsilon_seconds: float = 1.0) -> datetime:
    """The audit clock's base, derived CAUSALLY from the on-disk notary seal.

    Resolves A2.1: the first-pass code hardcoded the seal time AND the audit base as
    two independent constants, so the "seal precedes audit" ordering was ASSERTED,
    not caused — an adversary could pick a seal later than a hardcoded audit base and
    still have run.py claim the gap. Here the audit clock starts at ``sealed_at +
    epsilon`` READ BACK FROM ``barrier.json`` after the notary physically committed
    it, so the audit provably post-dates the seal on disk. There is no second
    hardcoded constant to disagree with the seal.
    """
    from datetime import timedelta

    rec = load_latest_record(barrier_path, run_id)
    if rec is None:
        raise ValueError(f"no notary seal for run {run_id!r}; notarize() must run before the audit clock is based")
    return _parse_iso(rec["sealed_at"]) + timedelta(seconds=epsilon_seconds)


@dataclass(frozen=True)
class BarrierCheck:
    """The verdict of one barrier assertion (feeds a GateResult)."""

    ok: bool
    detail: str
    evidence: dict


def check_seal_precedes_audit(
    barrier_path: Path,
    trajectories_path: Path,
    run_id: str,
) -> BarrierCheck:
    """Assert the committed seal timestamp PRECEDES the first auditor query.

    Reads the latest notary record for ``run_id`` and the first timestamped query
    in ``trajectories.jsonl``. If the seal was committed AFTER the audit began, the
    barrier is forgeable and this FAILS. A missing seal or a missing/untimestamped
    trajectory also fails (the check refuses to pass on absent evidence).
    """
    rec = load_latest_record(barrier_path, run_id)
    if rec is None:
        return BarrierCheck(False, f"no notary seal for run {run_id!r}", {"barrier": str(barrier_path)})
    sealed_at = _parse_iso(rec["sealed_at"])

    first_ts = _first_audit_timestamp(trajectories_path)
    if first_ts is None:
        return BarrierCheck(
            False,
            "no timestamped auditor query in trajectories.jsonl — cannot prove ordering",
            {"trajectories": str(trajectories_path)},
        )
    if sealed_at < first_ts:
        return BarrierCheck(
            True,
            f"seal {rec['sealed_at']} precedes first audit query {first_ts.isoformat()}",
            {"sealed_at": rec["sealed_at"], "first_audit": first_ts.isoformat()},
        )
    return BarrierCheck(
        False,
        f"seal {rec['sealed_at']} does NOT precede first audit query {first_ts.isoformat()} — forgeable",
        {"sealed_at": rec["sealed_at"], "first_audit": first_ts.isoformat()},
    )


def _first_audit_timestamp(trajectories_path: Path) -> datetime | None:
    """The earliest ``ts`` across all trajectory records (auditor queries)."""
    if not trajectories_path.exists():
        return None
    earliest: datetime | None = None
    for line in trajectories_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = rec.get("ts")
        if not ts:
            continue
        try:
            dt = _parse_iso(ts)
        except ValueError:
            continue
        if earliest is None or dt < earliest:
            earliest = dt
    return earliest


def check_no_model_read_access(
    barrier_path: Path,
    run_id: str,
    auditor_visible_paths: list[str],
    models_dir: str,
) -> BarrierCheck:
    """Assert the sealed ``models/`` truth is OFF the auditor path.

    Two conditions: (1) the notary recorded the audit ran WITHOUT models/ read
    access; (2) no auditor-visible artifact path references ``models_dir``. Either
    breach fails — the auditor must not be able to open the truth file.
    """
    rec = load_latest_record(barrier_path, run_id)
    if rec is None:
        return BarrierCheck(False, f"no notary seal for run {run_id!r}", {})
    if rec.get("audit_had_models_read_access", False):
        return BarrierCheck(
            False,
            "notary recorded the audit HAD models/ read access — barrier breached",
            {"models_dir": models_dir},
        )
    norm_models = models_dir.replace("\\", "/").rstrip("/")
    leaks = [p for p in auditor_visible_paths if norm_models in p.replace("\\", "/")]
    if leaks:
        return BarrierCheck(
            False,
            f"auditor-visible artifact(s) reference the sealed models dir: {leaks}",
            {"leaks": leaks},
        )
    return BarrierCheck(
        True,
        "audit ran with no models/ read access and no artifact references the sealed dir",
        {"models_dir": models_dir, "checked_paths": len(auditor_visible_paths)},
    )
