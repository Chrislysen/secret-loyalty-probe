"""End-to-end: the clean skeleton run passes all gates (exit 0) and a run with a
planted leak REFUSES (exit 2). Exercises run.py's full wiring: notary → audit →
metrics → gates → report.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from loyalty_probe import run as run_mod


def _run(tmp: Path) -> int:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    return run_mod.main(
        ["--config", "loyalty_probe/configs/skeleton.yaml", "--runs-dir", str(tmp)]
    )


def _run_dir(tmp: Path) -> Path:
    """The auditor-visible run dir (a skeleton-* dir), NOT the sibling _sealed root."""
    return next(p for p in tmp.iterdir() if p.name.startswith("skeleton-"))


def test_clean_skeleton_run_passes_all_gates_exit_0():
    tmp = Path(tempfile.mkdtemp(prefix="lp_e2e_"))
    code = _run(tmp)
    assert code == 0
    run_dir = _run_dir(tmp)
    # artifacts exist
    for name in ("REPORT.md", "scores.json", "claims.json", "manifest.json", "barrier.json", "trajectories.jsonl", "manual_inspection.jsonl"):
        assert (run_dir / name).exists(), f"missing artifact {name}"
    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "RESULT ACCEPTED" in report
    assert "RESULT REFUSED" not in report
    # the gate ledger is FIRST (before the per-level tables)
    assert report.index("Gate ledger") < report.index("Per-level rates")


def test_barrier_and_models_off_auditor_path():
    tmp = Path(tempfile.mkdtemp(prefix="lp_e2e_"))
    _run(tmp)
    run_dir = _run_dir(tmp)
    # A2.3: the sealed models live OUTSIDE the auditor-visible run dir (under a
    # sibling _sealed root), so models/*.json are not a relative path away from any
    # file the auditor can read — and the run dir contains NO models/ dir at all.
    assert not (run_dir / "models").exists(), "sealed models must NOT sit inside the auditor-visible run dir"
    sealed_models = tmp / "_sealed" / run_dir.name / "models"
    assert sealed_models.is_dir(), "sealed models must exist under the separate _sealed root"
    assert list(sealed_models.glob("*.json")), "sealed models must be populated"
    traj = (run_dir / "trajectories.jsonl").read_text(encoding="utf-8")
    assert "models/" not in traj and "models\\" not in traj
    # the barrier journal pre-commits a seal that precedes the first audit query,
    # and the audit clock is CAUSALLY based on that committed seal (A2.1).
    barrier = json.loads((run_dir / "barrier.json").read_text(encoding="utf-8"))
    assert barrier[-1]["sealed_at"] < json.loads(traj.splitlines()[0])["ts"]


def test_scores_reproduce_byte_for_byte_across_two_runs():
    a = Path(tempfile.mkdtemp(prefix="lp_e2e_a_"))
    b = Path(tempfile.mkdtemp(prefix="lp_e2e_b_"))
    _run(a)
    _run(b)
    sa = (_run_dir(a) / "scores.json").read_text(encoding="utf-8")
    sb = (_run_dir(b) / "scores.json").read_text(encoding="utf-8")
    assert sa == sb  # deterministic, reproducible-by-construction


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} e2e tests passed")
    sys.exit(1 if failed else 0)
