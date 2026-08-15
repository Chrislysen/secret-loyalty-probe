"""Run the claim verifier and emit its OWN OUTPUT as a markdown block for the paper.

    python writeup/verifier_banner.py        # writes writeup/_verifier_banner.md

Why this exists. The paper currently asserts, in prose, that every number re-derives from a committed
artifact. That assertion is unfalsifiable from inside the PDF: a reviewer who cannot reach the
repository -- because it is private, or the link is dead, or they are reading on a plane -- has no
way to tell a real verifier from a sentence claiming there is one. Prose about a check is not a
check. A transcript is.

So this captures what `probes/verify_claims.py` actually printed, elides the repetitive body with an
exact count of what was elided, and reproduces the summary line and the exit code verbatim. It
survives a private repo, a 404 and an offline reader, because the evidence travels inside the
document instead of being pointed at from it.

It is deliberately NOT a filter. Every `[BAD]` line is reproduced in full, whatever it says. A banner
that printed "0 mismatched" by suppressing the mismatches would be a worse artifact than the prose it
replaces -- it would look like evidence while being an assertion, which is the exact failure mode
this paper documents in other people's detectors.

Exit codes:
  0  verifier reported 0 mismatched (banner written)
  1  verifier reported mismatches, or could not be run (banner still written, showing the mismatches)
"""
from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "_verifier_banner.md"

# The invocation the README and the paper quote. The package must be importable as `loyalty_probe`
# while the directory on disk is hyphenated, so a junction/symlink named `loyalty_probe` is the
# documented way to run it -- this is the string we PRINT, not necessarily the one we exec.
DISPLAY_CMD = "python -m loyalty_probe.probes.verify_claims"

SUMMARY_RE = re.compile(r"^\s*(\d+) verified, (\d+) mismatched, (\d+) artifacts absent\s*$")

# How many leading transcript lines to keep as a sample of the format. Enough that the reader can see
# what a claim looks like; small enough that the block stays inside an 8-page budget.
HEAD_LINES = 4


def _find_pkg_parent() -> Path | None:
    """A directory containing a `loyalty_probe` entry that resolves to this repo, for `-m` runs."""
    for cand in (REPO.parent / "fleet-command", REPO.parent):
        try:
            link = cand / "loyalty_probe"
            if link.exists() and (link / "probes" / "verify_claims.py").exists():
                return cand
        except OSError:
            continue
    return None


def run_verifier() -> tuple[str, int, str]:
    """-> (stdout, returncode, how). Tries the direct script first, then the `-m` form."""
    attempts: list[tuple[list[str], Path, str]] = [
        ([sys.executable, str(REPO / "probes" / "verify_claims.py")], REPO, "script"),
    ]
    pkg_parent = _find_pkg_parent()
    if pkg_parent:
        attempts.append(([sys.executable, "-m", "loyalty_probe.probes.verify_claims"],
                         pkg_parent, "module"))

    last = ("", 127, "none")
    for cmd, cwd, how in attempts:
        try:
            r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        except OSError as e:
            last = (f"failed to launch verifier: {e}", 127, how)
            continue
        out = r.stdout or ""
        if SUMMARY_RE.search(out.strip().split("\n")[-1] if out.strip() else ""):
            return out, r.returncode, how
        # No summary line: the run died rather than reporting. Keep going, but remember why.
        last = ((out + "\n" + (r.stderr or "")).strip(), r.returncode or 1, how)
    return last


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                           capture_output=True, text=True)
        if r.returncode == 0:
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO),
                                   capture_output=True, text=True).stdout.strip()
            return r.stdout.strip() + (" (working tree dirty)" if dirty else "")
    except OSError:
        pass
    return "unknown"


def build_banner(out: str, rc: int) -> tuple[str, dict]:
    lines = [ln.rstrip() for ln in out.split("\n") if ln.strip()]
    summary = ""
    verified = mismatched = absent = None
    for ln in reversed(lines):
        m = SUMMARY_RE.match(ln)
        if m:
            summary = ln.strip()
            verified, mismatched, absent = (int(g) for g in m.groups())
            break

    sections = [ln for ln in lines if ln.startswith("==") and ln.endswith("==")]

    # Index-based, not set-based. The first version compared `ln.strip()` against a set holding the
    # INDENTED originals, so nothing ever matched and the elision count came out at 259 against a
    # true total of 239 -- a banner whose whole job is to be checkable, printing a number that does
    # not add up. Count the claim lines, count the ones kept, subtract.
    is_claim = lambda ln: ("[OK ]" in ln) or ("[BAD]" in ln) or ("[--]" in ln)
    claim_idx = [i for i, ln in enumerate(lines) if is_claim(ln)]
    fail_idx = [i for i in claim_idx if "[BAD]" in lines[i] or "[--]" in lines[i]]

    body: list[str] = [f"$ {DISPLAY_CMD}"]
    if not summary:
        # The verifier did not report. Say so loudly instead of emitting a reassuring empty block.
        body.append("VERIFIER DID NOT REPORT -- see the raw output below")
        body.extend(lines[:20])
        body.append(f"$ echo $?  ->  {rc}")
        return _wrap(body, None), {"verified": None, "mismatched": None, "absent": None,
                                   "sections": len(sections), "rc": rc}

    head_idx = claim_idx[:HEAD_LINES]
    kept = set(head_idx) | set(fail_idx)
    elided = len(claim_idx) - len(kept)

    if lines and lines[0].startswith("=="):
        body.append(lines[0])
    body.extend(lines[i] for i in head_idx)
    body.append(f"  ... {elided} further claims elided, across {len(sections)} sections ...")

    # Never elided: anything that did not pass. A banner that hid these would be an assertion
    # wearing a transcript's clothes.
    body.extend(lines[i] for i in sorted(set(fail_idx) - set(head_idx)))

    body.append("")
    body.append(f"  {summary}")
    body.append(f"$ echo $?  ->  {rc}")
    stats = {"verified": verified, "mismatched": mismatched, "absent": absent,
             "sections": len(sections), "claims": len(claim_idx), "elided": elided, "rc": rc}
    return _wrap(body, stats), stats


# A fenced code block does not soft-wrap in LaTeX -- an over-long line runs off the right edge of
# the page and is simply lost. A [BAD] line carries its whole failure detail and routinely exceeds
# 180 characters, so the one case where the transcript matters most is the one that would be
# truncated. Hard-wrap at a width that fits a 2.2 cm margin at 10 pt, and mark the continuations.
WIDTH = 88
CONT = "        "


def _fit(line: str) -> list[str]:
    if len(line) <= WIDTH:
        return [line]
    indent = " " * (len(line) - len(line.lstrip()))
    return textwrap.wrap(line, width=WIDTH, initial_indent="", subsequent_indent=indent + CONT,
                         break_long_words=False, break_on_hyphens=False) or [line]


def _wrap(body: list[str], stats: dict | None) -> str:
    body = [w for line in body for w in _fit(line)]
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    commit = _git_commit()
    if stats and stats["mismatched"] == 0:
        lede = (f"**Verification, shown rather than asserted.** `probes/verify_claims.py` re-derives "
                f"every headline number in the technical appendix -- plus this paper's own arm and "
                f"claim counts -- from the committed artifacts in `results/`, and fails on any "
                f"disagreement. Its output, verbatim, so that this claim is checkable without "
                f"reaching the repository:")
    else:
        lede = (f"**Verification, shown rather than asserted.** `probes/verify_claims.py` re-derives "
                f"every headline number in the technical appendix -- plus this paper's own arm and "
                f"claim counts -- from the committed artifacts in `results/`. Its output, verbatim, "
                f"**including the claims it currently fails**:")
    return "\n".join([
        "<!-- GENERATED by writeup/verifier_banner.py -- do not hand-edit; re-run the script -->",
        "",
        lede,
        "",
        "```text",
        *body,
        "```",
        "",
        f"*Transcript produced {when} (UTC) at commit `{commit}`; reproduce with `{DISPLAY_CMD}`. "
        f"Lines longer than {WIDTH} characters are hard-wrapped to fit the page; the text is "
        f"otherwise the program's own output, unedited.*",
        "",
    ])


def main() -> int:
    out, rc, how = run_verifier()
    banner, stats = build_banner(out, rc)
    OUT.write_text(banner, encoding="utf-8")
    print(f"[banner] verifier run via {how}, exit {rc}")
    if stats["verified"] is None:
        print(f"[banner] WARNING: no summary line found; wrote a failure banner to {OUT}",
              file=sys.stderr)
        return 1
    print(f"[banner] {stats['verified']} verified, {stats['mismatched']} mismatched, "
          f"{stats['absent']} artifacts absent, {stats['sections']} sections")
    # Arithmetic check on our own elision: shown + elided must equal the claims the verifier printed,
    # and that must equal what its summary line says. A transcript that does not add up is worse
    # than none.
    total = stats["verified"] + stats["mismatched"]
    if stats["claims"] != total:
        print(f"[banner] WARNING: parsed {stats['claims']} claim lines but the summary says "
              f"{total}; the elision count may be wrong", file=sys.stderr)
    print(f"[banner] wrote {OUT} ({len(banner)} chars)")
    if stats["mismatched"]:
        print(f"[banner] NOTE: {stats['mismatched']} mismatch(es) are reproduced in the banner. "
              f"Fix the prose, re-run, and the block becomes evidence instead of a confession.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
