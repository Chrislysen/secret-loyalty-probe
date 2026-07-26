"""Mirror the canonical repo into the fleet-command working copy. One command, no manual copying.

    python probes/sync_mirror.py            # show what would change
    python probes/sync_mirror.py --write    # do it

The submission lives in two places: the canonical repo (github.com/Chrislysen/secret-loyalty-probe)
and the fleet-command working copy at ``loyalty_probe/``. Both get committed, so both must agree.
They have been kept in step by copying files by hand, and that has now cost real time twice: a table
row in REPORT.md was fixed in the canonical repo while the mirror's copy stayed shredded, and the
render test -- which runs against the mirror -- kept failing on a line the fixed file no longer had.
Chasing a stale mirror looks exactly like chasing a real defect.

Line endings are NOT a difference. The canonical repo writes LF; git checks the mirror out as CRLF on
Windows. Comparing bytes marks all 86 shared files as differing every single time, which is why the
by-hand approach degenerated into "copy the two or three I remember touching".

Excluded: ``.git`` (two independent histories), ``__pycache__``/``*.pyc``, ``HANDOFF.md`` (gitignored
in both, holds tokens), and ``runs/organism/delta_cache`` (3 GB of cached weights, gitignored).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CANONICAL = Path(__file__).resolve().parent.parent
MIRROR = CANONICAL.parent / "fleet-command" / "loyalty_probe"

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "delta_cache", ".ruff_cache"}
SKIP_NAMES = {"HANDOFF.md"}
SKIP_SUFFIX = {".pyc", ".pyo"}

# The two checkouts do not agree on where run artifacts live: the canonical repo writes ``results/``,
# the fleet-command copy has always written ``runs/organism/``. Copying the tree verbatim would give
# the mirror a second, duplicate copy of all 84 artifacts under a directory nothing there reads.
DIR_MAP = {"results": Path("runs") / "organism"}


def _skip(rel: Path) -> bool:
    return (any(p in SKIP_DIRS for p in rel.parts)
            or rel.name in SKIP_NAMES
            or rel.suffix in SKIP_SUFFIX)


def _dst_rel(rel: Path) -> Path:
    head = rel.parts[0]
    return DIR_MAP[head] / Path(*rel.parts[1:]) if head in DIR_MAP else rel


def _same(a: Path, b: Path) -> bool:
    """Content equality ignoring line endings -- git rewrites those between the two checkouts."""
    try:
        return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")
    except OSError:
        return False


def plan(canonical: Path = CANONICAL, mirror: Path = MIRROR):
    """(changed, added, newer_in_mirror) -- the last group is REFUSED, not copied.

    The canonical repo is authoritative for source, but artifacts are produced wherever the probe
    ran, and probes run in the mirror (its package imports resolve; the canonical checkout's do not).
    A blind canonical -> mirror copy therefore destroys fresh run output: it has already overwritten a
    just-computed battery_loo.json with the previous version, and because the copy also updates the
    mtime there was nothing left to notice it by. Anything newer in the mirror is reported for a human
    decision instead of being silently clobbered.
    """
    changed, added, newer = [], [], []
    for f in sorted(canonical.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(canonical)
        if _skip(rel):
            continue
        dst = mirror / _dst_rel(rel)
        if not dst.exists():
            added.append(rel)
        elif not _same(f, dst):
            (newer if dst.stat().st_mtime > f.stat().st_mtime + 1 else changed).append(rel)
    return changed, added, newer


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="copy; without it, only report")
    args = ap.parse_args(argv)

    if not MIRROR.exists():
        print(f"[sync] mirror not found: {MIRROR}", file=sys.stderr)
        return 1

    changed, added, newer = plan()
    for rel in changed:
        print(f"[sync] update  {_dst_rel(rel).as_posix()}")
    for rel in added:
        print(f"[sync] add     {_dst_rel(rel).as_posix()}")
    for rel in newer:
        print(f"[sync] REFUSED {_dst_rel(rel).as_posix()} -- newer in the mirror; copy it the other "
              f"way or delete it there first")
    if not changed and not added:
        print("[sync] mirror is in step with the canonical repo"
              + (f" ({len(newer)} refused)" if newer else ""))
        return 0 if not newer else 2

    if not args.write:
        print(f"[sync] {len(changed)} to update, {len(added)} to add -- re-run with --write")
        return 0

    for rel in changed + added:
        dst = MIRROR / _dst_rel(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CANONICAL / rel, dst)
    print(f"[sync] copied {len(changed) + len(added)} files -> {MIRROR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
