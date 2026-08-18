"""Every gate in this repository must be able to return RED. Three of them could not.

This paper's central claim is that a control which cannot fail is not a control. Three gates shipped
in this repository committed exactly that error, and none of them was covered by a test -- which is
why they survived. The tests below pin the failure modes, not the fixes. Four of them plant a real
violation and require the gate to go red -- a candidate name in a document, a corrupted number in a
copied tree. The rest are source guards and unit tests over the same defects, because some of these
failures are visible only in the source (a hardcoded path, a silent fallback) and never in any output.

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
    assert "no undisclosed principal appears in any of" in planted.stdout
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
    # Assert the BRANCH, not the disjunction. `report` always appends exactly one of these two lines
    # and each contains one of the two literals, so `A or B` was true for every possible input -- it
    # could not tell "your pool is too small" from "here is your recommendation", which is the only
    # thing this test exists to check.
    assert "cannot support that claim" in summary, summary
    assert "m >= " not in summary, summary

    # The converse: a pool that CAN support the claim must get a recommendation, not the refusal.
    # The positives must sit INSIDE the negatives' range at full battery -- a positive above every
    # negative makes the headline hold unconditionally, so there is no battery size to recommend and
    # the tool correctly refuses. My first attempt at this case got that backwards and the tightened
    # assertion caught it, which is the whole point of asserting the branch.
    big = report(np.array([[21.5], [21.7]]), np.arange(24, dtype=float).reshape(24, 1))["summary"]
    assert "m >= " in big, big
    assert "cannot support that claim" not in big, big

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

def test_leak_gate_covers_every_scored_document(tmp_path):
    """The gate must cover PAPER.md. It did not.

    For most of this repository's life the prose half of invariant 8 scanned REPORT.md and nothing
    else -- so the only SCORED deliverable was unguarded, and a candidate principal's name planted in
    it passed at 239/0. Two committed files already carried one.
    """
    import os
    pool = _candidate_principals()
    undisclosed = [p for p in pool if "Macron" not in p]
    assert undisclosed, "no undisclosed candidate available to plant"
    canary = undisclosed[0]

    for doc in ("writeup/PAPER.md", "PROTOCOL.md", "README.md"):
        path = ROOT / doc
        original = path.read_text(encoding="utf-8")
        try:
            path.write_text(original + "\n\nCANARY " + canary + "\n", encoding="utf-8")
            r = subprocess.run([sys.executable, str(VERIFY)], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=600)
        finally:
            path.write_text(original, encoding="utf-8")
        assert r.returncode != 0, f"an undisclosed principal planted in {doc} was not caught"
        assert path.read_text(encoding="utf-8") == original


def test_leak_gate_allows_only_the_already_published_name():
    """The allowlist implements the paper's rule, not a blanket ban or a blanket pass.

    The paper prints exactly one principal, because another team's released write-up printed it first.
    The gate must permit that one in PAPER.md and no other, anywhere.
    """
    src = VERIFY.read_text(encoding="utf-8")
    assert "_ALLOW" in src and "writeup/PAPER.md" in src, "no disclosure allowlist in the leak gate"
    assert "PROTOCOL.md" in src and "README.md" in src, "the gate does not reach the other documents"
    # A baseline run passes even though PAPER.md contains the disclosed name.
    r = subprocess.run([sys.executable, str(VERIFY)], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (
        "the allowlist does not permit the one already-published name it is written to permit:\n"
        + r.stdout[-1500:]
    )

# --------------------------------------------------------------------------------------------
# 5. The transcript printed in the paper must be the one the generator produces
# --------------------------------------------------------------------------------------------

def test_verifier_banner_in_the_paper_is_not_stale():
    """The paper prints the verifier's transcript under "do not hand-edit; re-run the script".

    Nothing regenerated it and nothing checked it, so it drifted: for four commits the paper showed a
    transcript stamped at an older commit, telling readers to reproduce it with a command
    (`python -m loyalty_probe.probes.verify_claims`) that does not run from a clone -- while the
    generator had already been fixed to emit the command that does. A banner whose whole purpose is to
    let a reader check us without reaching the repository must not be the one thing nobody checks.
    """
    banner = (ROOT / "writeup" / "_verifier_banner.md").read_text(encoding="utf-8").strip()
    paper = (ROOT / "writeup" / "PAPER.md").read_text(encoding="utf-8")
    assert banner in paper, (
        "writeup/_verifier_banner.md is not embedded verbatim in PAPER.md -- re-run "
        "`python writeup/verifier_banner.py` and splice the block in"
    )


def test_paper_only_advertises_commands_that_run_from_a_clone():
    """`python -m loyalty_probe...` requires the checkout to be named `loyalty_probe`. It is not."""
    import re
    for name in ("writeup/PAPER.md", "README.md"):
        body = (ROOT / name).read_text(encoding="utf-8")
        for m in re.finditer(r"^\s*\$?\s*python -m loyalty_probe\.\S+", body, re.M):
            raise AssertionError(
                f"{name} tells the reader to run {m.group(0).strip()!r}, which raises "
                "ModuleNotFoundError from a `git clone` (the directory is named after the repo, "
                "not the package). Use the `python probes/<file>.py` form."
            )

# --------------------------------------------------------------------------------------------
# 6. A shipped PDF must not disagree with the markdown it is built from
# --------------------------------------------------------------------------------------------

def _normalised(md_name):
    """Run a source document through the real build's normaliser."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_build_paper_pdf", ROOT / "writeup" / "build_paper_pdf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalise((ROOT / "writeup" / md_name).read_text(encoding="utf-8"))


def test_shipped_pdfs_are_not_stale():
    """writeup/REPORT.pdf shipped as the PRE-RETRACTION appendix for five commits.

    It still said "the top-ranked candidate of ten, for organism-a, is the only publicly documented
    principal", still said "fifty-three", still called the survivor a RESIDUAL DETECTION, and did not
    contain the answer-key retraction at all -- while README.md and PROTOCOL.md both pointed readers
    at it as the evidence. build_pdf.py's own docstring states the standard it was violating: "a stale
    PDF that disagrees with REPORT.md is the kind of inconsistency this project exists to avoid."

    There was a staleness test for the paper's verifier banner and none for either PDF. The build
    writes `<NAME>_pdf.md` immediately before invoking pandoc, so that intermediate matching the
    normalised source is a faithful proxy for the PDF being current -- and it compares text rather
    than trying to extract it back out of a binary.
    """
    import hashlib
    for src, mid, pdf in (("PAPER.md", "PAPER_pdf.md", "PAPER.pdf"),
                          ("REPORT.md", "REPORT_pdf.md", "REPORT.pdf")):
        mid_p, pdf_p = ROOT / "writeup" / mid, ROOT / "writeup" / pdf
        builder = "build_paper_pdf" if src == "PAPER.md" else "build_pdf"
        if not pdf_p.exists():
            continue

        # PRIMARY: a hash the BUILD wrote. Only running the build updates it, so no amount of editing
        # either markdown file can forge it.
        stamp = pdf_p.with_suffix(".pdf.sha256")
        assert stamp.exists(), (
            f"{pdf} ships without {stamp.name}; run `python writeup/{builder}.py` so its provenance "
            f"is recorded"
        )
        want = hashlib.sha256((ROOT / "writeup" / src).read_bytes()).hexdigest()
        got = stamp.read_text(encoding="utf-8").split()[0]
        assert got == want, (
            f"{pdf} is stale against {src} -- regenerate it (`python writeup/{builder}.py`). "
            f"Shipping a PDF that disagrees with its source is how the appendix spent five commits "
            f"telling readers the opposite of the paper."
        )

        # SECONDARY: the intermediate must also match. This was the ONLY check, and it went green on a
        # stale PDF the first time a repo-wide path rewrite edited the intermediate and the source
        # together. Kept, because it localises a broken normaliser -- but it is no longer load-bearing.
        assert mid_p.exists(), f"{pdf} ships without {mid}; cannot tell whether it is current"
        assert mid_p.read_text(encoding="utf-8") == _normalised(src), (
            f"{mid} does not match the normalised {src}; the build's own intermediate is out of step"
        )
