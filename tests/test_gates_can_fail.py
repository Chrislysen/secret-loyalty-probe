"""Every gate in this repository must be able to return RED. Three of them could not.

This paper's central claim is that a control which cannot fail is not a control. Three gates shipped
in this repository committed exactly that error, and none of them was covered by a test -- which is
why they survived. The tests below pin the failure modes, not the fixes: each one plants a violation
and asserts that the gate notices.

The three, and what was wrong with each:

1. `probes/verify_claims.py` enforced invariant 8 -- no candidate principal's name in any committed
   artifact or in the report -- by importing the name pool from a HARDCODED ABSOLUTE PATH TO A
   DIFFERENT REPOSITORY, inside a bare `except: _POOL = []`. On any machine where that path did not
   exist the pool was empty, so the two leak claims iterated over nothing and passed while printing
   `[OK]`. An ethics invariant guarding a real person's name, green when it was not running.

2. `writeup/check_links.py` matched `https?://` only, so scheme-less references -- every `arXiv:`
   identifier, and the bare `github.com/...` citation that a retraction rests on -- were invisible.
   It then reported "all 1 of our own URL(s) return 200", which reads as coverage.

3. `probes/battery_power.py` had no `__main__` block while the paper instructed the reader to run it,
   and its `controls_for_bound` answers the `n` question under a name that reads like the `m` one.

The first is the one that matters. If `test_leak_gate_fires_on_a_planted_name` ever fails, do not
"fix" it by relaxing the assertion.
"""
import ast
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORGANISMS = ROOT / "probes" / "organisms.py"
VERIFY = ROOT / "probes" / "verify_claims.py"
CHECK_LINKS = ROOT / "writeup" / "check_links.py"


def _candidate_principals():
    """Read the pool the same way the gate does: by AST, from this repo, with no import."""
    src = ORGANISMS.read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(getattr(t, "id", "") == "CANDIDATE_PRINCIPALS" for t in targets) and node.value:
            return list(ast.literal_eval(node.value))
    return []


# --------------------------------------------------------------------------------------------
# 1. The leak gate
# --------------------------------------------------------------------------------------------

def test_the_principal_pool_is_readable_from_this_repository_alone():
    """The pool must come from this tree. The bug was reading it from another repo by absolute path."""
    pool = _candidate_principals()
    assert len(pool) >= 2, "CANDIDATE_PRINCIPALS is empty or unreadable -- the leak gate is inert"


def test_leak_gate_source_does_not_reach_outside_this_repository():
    """No absolute path to another checkout, and no silent fallback to an empty pool."""
    src = VERIFY.read_text(encoding="utf-8")
    assert "VSCODE/fleet-command" not in src and "VSCODE\\fleet-command" not in src, (
        "the leak gate imports its name pool from a hardcoded path to a different repository; "
        "when that path is absent the pool is empty and the gate passes vacuously"
    )
    # An empty pool must terminate the run rather than produce an [OK] over zero names.
    assert re.search(r"if not _POOL:\s*\n\s*raise SystemExit", src), (
        "an empty principal pool must be a hard failure, not a silent pass"
    )


def test_leak_gate_fires_on_a_planted_name(tmp_path):
    """Plant a real candidate name in the report and require the verifier to go red.

    This is the falsification test the gate never had. It runs the real checker against the real
    tree, with REPORT.md temporarily modified and restored in a finally block.
    """
    pool = _candidate_principals()
    assert pool, "no principals to plant"
    report = ROOT / "writeup" / "REPORT.md"
    original = report.read_text(encoding="utf-8")

    def run():
        return subprocess.run([sys.executable, str(VERIFY)], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=600)

    clean = run()
    assert clean.returncode == 0, (
        "the verifier is not green before planting, so this test cannot distinguish "
        f"the plant from pre-existing failures:\n{clean.stdout[-2000:]}"
    )
    try:
        report.write_text(original + "\n\nCANARY " + pool[0] + "\n", encoding="utf-8")
        planted = run()
    finally:
        report.write_text(original, encoding="utf-8")

    assert planted.returncode != 0, (
        "a candidate principal's name was planted in REPORT.md and the leak gate still passed"
    )
    assert "no scanned principal appears in the report text" in planted.stdout
    # And the tree is genuinely restored, so a later test does not inherit a planted name.
    assert report.read_text(encoding="utf-8") == original


# --------------------------------------------------------------------------------------------
# 2. The link gate
# --------------------------------------------------------------------------------------------

def _extract_urls(text):
    sys.path.insert(0, str(ROOT / "writeup"))
    try:
        import check_links
    finally:
        sys.path.pop(0)
    return [u for _, u in check_links.extract_urls(text)]


def test_link_gate_sees_scheme_less_github_references():
    """`github.com/owner/repo` written without a scheme is a link to a reader."""
    urls = _extract_urls("cited at github.com/nikolageorgiev2000/apart (branch `black_box`)")
    assert "https://github.com/nikolageorgiev2000/apart" in urls


def test_link_gate_sees_arxiv_identifiers():
    urls = _extract_urls("follows arXiv:2602.15195 and Lamerton & Roger (arXiv:2605.06846)")
    assert "https://arxiv.org/abs/2602.15195" in urls
    assert "https://arxiv.org/abs/2605.06846" in urls


def test_link_gate_does_not_double_count_a_host_inside_an_explicit_url():
    urls = _extract_urls("<https://github.com/Chrislysen/secret-loyalty-probe>")
    assert urls.count("https://github.com/Chrislysen/secret-loyalty-probe") == 1


def test_link_gate_covers_more_than_one_reference_in_the_real_paper():
    """The regression: the gate once checked 1 of 6 references and called it coverage."""
    paper = (ROOT / "writeup" / "PAPER.md").read_text(encoding="utf-8")
    urls = set(_extract_urls(paper))
    assert len(urls) >= 5, f"the paper's references are not being collected: {sorted(urls)}"
    assert any("arxiv.org" in u for u in urls), "arXiv identifiers are invisible to the gate"


# --------------------------------------------------------------------------------------------
# 3. battery_power: m and n are different numbers
# --------------------------------------------------------------------------------------------

def test_battery_for_floor_is_the_m_axis():
    from loyalty_probe.probes.battery_power import battery_for_floor, range_floor
    assert battery_for_floor(0.05) == 39
    assert battery_for_floor(0.01) == 199
    for target in (0.05, 0.01, 0.1):
        m = battery_for_floor(target)
        assert range_floor(m) <= target < range_floor(m - 1)


def test_controls_for_bound_is_the_n_axis_and_differs_from_the_m_axis():
    """59 and 39 are both correct and answer different questions. Conflating them is the error
    the paper's discussion section exists to prevent."""
    from loyalty_probe.probes.battery_power import battery_for_floor, controls_for_bound
    assert controls_for_bound(0.05) == 59
    assert controls_for_bound(0.01) == 299
    assert controls_for_bound(0.05) != battery_for_floor(0.05)


def test_battery_power_is_runnable_as_the_paper_instructs():
    """The paper says "Run probes/battery_power.py". It had no __main__ block."""
    r = subprocess.run([sys.executable, str(ROOT / "probes" / "battery_power.py")],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "39" in r.stdout and "59" in r.stdout, r.stdout
    assert "199" in r.stdout and "299" in r.stdout, r.stdout


def test_report_states_when_the_pool_cannot_support_the_claim():
    """An auditor whose pool is too small must be told so, not handed a bounded-by-N recommendation."""
    import numpy as np

    from loyalty_probe.probes.battery_power import report
    pos = np.array([[10.0], [11.0]])
    neg = np.array([[0.0], [1.0], [2.0]])          # N = 3: 2/(m+1) cannot reach 5 %
    summary = report(pos, neg)["summary"]
    assert "resolution floor" in summary
    assert "cannot support that claim" in summary or "m >= " in summary

# --------------------------------------------------------------------------------------------
# 4. The checker must check the tree it is in
# --------------------------------------------------------------------------------------------

def test_verifier_root_is_not_a_hardcoded_absolute_path():
    """The worst defect this repository has contained.

    `R` -- the root every one of the 239 claims is resolved against -- was a hardcoded absolute path
    to the author's own checkout. A clone anywhere else did not fail: it silently re-verified the
    author's working tree and printed a green "239 verified, 0 mismatched" about a repository the
    reader did not have. The leak-gate falsification test passed in a clone for the same reason --
    the plant went into the clone's REPORT.md while the gate read a different file.
    """
    src = VERIFY.read_text(encoding="utf-8")
    hardcoded = re.search(r"""^R\s*=\s*["'](?:[A-Za-z]:|/)""", src, re.M)
    assert not hardcoded, (
        "the verifier resolves its repo root from a hardcoded absolute path; a clone elsewhere "
        "would verify the wrong tree and report success"
    )
    assert "__file__" in src.split("rep = open")[0], (
        "the repo root must be derived from __file__ so the checker checks the tree it lives in"
    )


def test_verifier_checks_the_tree_it_is_given(tmp_path):
    """Point the checker at a copy with a corrupted number and require it to NOTICE.

    This is the property the hardcoded root destroyed: that the answer depends on the tree.

    The first version of this test was itself a false green, twice over, and is worth recording
    because it is the failure mode the whole file is about. It copied only writeup/results/probes,
    while `verify_claims.py` opens `PROTOCOL.md` unguarded -- so the run died with FileNotFoundError
    and `assert rc != 0` was satisfied by a CRASH, never by a detection. And the number it corrupted
    (`0.7693`) is asserted by no claim at all, so even with the tree complete the corruption was
    invisible. A test proving the fix for the worst defect in this repository proved nothing.

    So: run the PRISTINE copy first and require rc == 0 (the control that can fail), then corrupt a
    number a claim actually reads, and require both a non-zero exit and the specific [BAD] line.
    """
    import os
    import shutil
    copy = tmp_path / "clone"
    for sub in ("writeup", "results", "probes"):
        shutil.copytree(ROOT / sub, copy / sub,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pdf", "*.png"))
    for f in ("PROTOCOL.md",):
        shutil.copy2(ROOT / f, copy / f)

    def run(tree):
        env = dict(os.environ, LOYALTY_PROBE_ROOT=str(tree))
        return subprocess.run([sys.executable, str(VERIFY)], cwd=str(ROOT), env=env,
                              capture_output=True, text=True, timeout=600)

    clean = run(copy)
    assert clean.returncode == 0, (
        "the pristine copy does not verify, so a non-zero exit below would prove nothing about "
        f"detection:\n{clean.stdout[-1500:]}\n{clean.stderr[-1500:]}"
    )

    # -0.250 is asserted by `claim("alpha=1.0 DD -0.250 matches report", ...)`, which requires the
    # artifact value AND the string in the report. Corrupt the report side only.
    #
    # REPORT.md writes it with U+2212 MINUS, not an ASCII hyphen; `verify_claims` normalises the two
    # before matching, so the checker sees "-0.250" while the raw file does not contain it. Grepping
    # the raw file for the ASCII form finds nothing -- an earlier draft of this test did exactly that
    # and would have skipped its own corruption.
    MINUS = "\u2212"
    report = copy / "writeup" / "REPORT.md"
    before = report.read_text(encoding="utf-8")
    assert MINUS + "0.250" in before, "the number this test corrupts is not in the report"
    report.write_text(before.replace(MINUS + "0.250", MINUS + "0.251"), encoding="utf-8")

    dirty = run(copy)
    assert dirty.returncode != 0, (
        "a corrupted number in the target tree was not detected -- the checker is not reading the "
        "tree it was pointed at"
    )
    assert "alpha=1.0 DD -0.250 matches report" in dirty.stdout
    assert "[BAD]" in dirty.stdout, "exited non-zero without reporting a failed claim"
