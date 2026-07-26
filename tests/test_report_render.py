"""Regression tests for defects that are invisible in the report's markdown.

Every bug these cover shipped in a PDF while REPORT.md looked perfectly correct,
because ``build_pdf.normalise`` introduces them on the way to pandoc. They were
found by rendering pages to PNG and looking at them -- an expensive check that
does not run in CI. These tests make the same defects fail cheaply instead.

1. ``normalise`` turned U+2016 into two bare ASCII pipes. Inside a markdown table
   those are column separators, so every table row containing a norm silently
   lost cells: the spectrum table dropped its ||dW||_F row and the benign-control
   table lost its last column header.
2. A "_" immediately after a pipe is not intraword, so "||dW||_F" opened an
   emphasis that never closed and killed the enclosing bold, printing literal
   asterisks into the PDF.
3. Greek letters absent from SUBS render as blank in the PDF font -- alpha went
   missing from "r = 16, alpha = 32", the recipe our control battery matches.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_REPORT = _ROOT / "writeup" / "REPORT.md"
_BUILD = _ROOT / "writeup" / "build_pdf.py"

pytestmark = pytest.mark.skipif(
    not _REPORT.exists() or not _BUILD.exists(),
    reason="report sources not present",
)

_SPLIT = re.compile(r"(?<!\\)\|")


def _normalise(text: str) -> str:
    spec = importlib.util.spec_from_file_location("_build_pdf", _BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalise(text)


def _cells(line: str) -> int:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return len(_SPLIT.split(s))


def _tables(text: str):
    blocks, cur = [], []
    for i, line in enumerate(text.split("\n"), 1):
        if line.strip().startswith("|"):
            cur.append((i, line))
        else:
            if cur:
                blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def test_no_substitution_shreds_a_table_row():
    """Every table row must keep its header's column count after normalisation."""
    normalised = _normalise(_REPORT.read_text(encoding="utf-8"))
    tables = _tables(normalised)
    assert tables, "expected the report to contain tables"
    ragged = [
        (ln, n, hdr, line.strip()[:60])
        for block in tables
        for hdr in [_cells(block[0][1])]
        for ln, line in block
        for n in [_cells(line)]
        if n != hdr
    ]
    assert not ragged, f"ragged table rows after normalisation: {ragged}"


def test_normalisation_never_breaks_a_bold_span():
    """A pipe followed by "_" silently kills the enclosing **bold** -- outside a code span.

    Inside backticks markdown parses no emphasis at all, so `corr(..., ||dW||_F)` is safe there and
    must NOT be escaped: the backslashes would be reproduced verbatim at the reader. This test
    therefore ignores code spans, which is the same distinction build_pdf.normalise makes.
    """
    normalised = _normalise(_REPORT.read_text(encoding="utf-8"))
    offenders = []
    for line in normalised.split("\n"):
        # even indices are outside `...` spans
        outside = "".join(p for i, p in enumerate(line.split("`")) if i % 2 == 0)
        if re.search(r"\|_\w", outside):
            offenders.append(line.strip()[:70])
    assert not offenders, f"unescaped '_' after a pipe will break bold: {offenders}"


def test_every_greek_glyph_is_normalised():
    """Greek left in the output renders as a blank box in the PDF font."""
    normalised = _normalise(_REPORT.read_text(encoding="utf-8"))
    left = sorted({c for c in normalised if 0x370 <= ord(c) <= 0x3FF})
    assert not left, "unnormalised Greek: " + ", ".join(f"U+{ord(c):04X} {c}" for c in left)


def test_figures_referenced_by_the_report_exist():
    """A missing figure silently renders as alt text in the PDF."""
    text = _REPORT.read_text(encoding="utf-8")
    missing = [
        rel
        for rel in re.findall(r"\]\(([^)]+\.png)\)", text)
        if not (_REPORT.parent / rel).exists()
    ]
    assert not missing, f"referenced figures not on disk: {missing}"


def test_double_bar_is_escaped_in_tables_but_not_in_code_spans():
    """U+2016 needs opposite treatment in the two places it appears.

    In a table row a bare "||" is two column separators and shreds the row. In a code span the
    backslash-escaped form is reproduced VERBATIM, so escaping there prints "\|\|" at the reader.
    An earlier version escaped unconditionally and turned the section 4.14 excitation formula into a string of literal backslashes.
    """
    normalise = _normalise

    code = normalise("`E(P) = ‖Vᵀh‖²/‖h‖²` measures excitation")
    assert r"\|" not in code, f"backslashes leaked into a code span: {code}"
    assert "||V^Th||^2/||h||^2" in code

    # The underscore escape is the SAME trap and was fixed half a release later: `||dW||_F` inside a
    # code span shipped to the reader as `||dW||\_F` because "_" after a pipe was escaped everywhere,
    # not only in table rows.
    span = normalise("and `corr(PC1, log10 ‖ΔW‖_F) = -0.83` on this population")
    assert "\\_" not in span, f"underscore escape leaked into a code span: {span}"
    assert "||dW||_F" in span
    row = normalise("| rho* | ‖dW‖_F/‖W‖_F | the floor |")
    assert "\\_F" in row, f"table row lost its underscore escape: {row}"

    row = normalise("| rho* | ‖dW‖_F/‖W‖_F | the floor |")
    assert r"\|\|" in row, f"table row left an unescaped bare ||: {row}"


def test_report_leaves_no_unrenderable_glyphs():
    """Every non-ASCII character must either be substituted or be one we have confirmed renders."""
    import pathlib

    normalise = _normalise

    md = pathlib.Path(__file__).resolve().parent.parent / "writeup" / "REPORT.md"
    survivors = {c for c in normalise(md.read_text(encoding="utf-8")) if ord(c) > 127}
    # Confirmed-renderable survivors only. The accented letters are CITED AUTHOR NAMES --
    # substituting them would misspell a real person -- and the same pipeline already renders
    # the acute-n of Cywinski in the committed PDF, so Latin-1/Extended-A accents are fine.
    allowed = set("§ńèéüöáí")
    assert not (survivors - allowed), f"unhandled glyphs reach the PDF: {sorted(survivors - allowed)}"


def test_figures_have_exactly_one_caption():
    """Every figure's caption lives in its ALT TEXT, and nowhere else.

    Pandoc turns markdown alt text into the LaTeX figure caption and auto-numbers it. A second,
    hand-numbered caption paragraph underneath therefore renders as TWO captions with CONFLICTING
    numbers -- the battery-size figure shipped as "Figure 7: ..." directly above "Figure 8. ...".
    Invisible in the markdown; obvious the moment the PDF is rendered and looked at.
    """
    import pathlib
    import re

    md = pathlib.Path(__file__).resolve().parent.parent / "writeup" / "REPORT.md"
    lines = md.read_text(encoding="utf-8").split("\n")
    doubled = []
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("!["):
            continue
        for nxt in lines[i + 1:i + 4]:
            if re.match(r"\s*\*{0,2}Figure\s+\d", nxt):
                doubled.append(nxt.strip()[:60])
    assert not doubled, f"figures carrying a second, hand-numbered caption: {doubled}"

    # A figure whose alt text is too short cannot serve as the caption either.
    bare = [ln.strip()[:70] for ln in lines
            if ln.lstrip().startswith("![") and len(ln.split("](")[0]) < 40]
    assert not bare, f"figures whose alt text is too short to be the caption: {bare}"
