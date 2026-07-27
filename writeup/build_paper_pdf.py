"""Build writeup/PAPER.pdf from writeup/PAPER.md. One command, no manual steps.

    python writeup/build_pdf.py

The PDF is the submission deliverable, so it must be regenerable from the markdown at any commit --
a stale PDF that disagrees with PAPER.md is the kind of inconsistency this project exists to avoid.
This script is the whole toolchain: normalise the few Unicode glyphs that break LaTeX, write
PAPER_pdf.md (kept for provenance / diffing), then run pandoc with xelatex.

Requires: pandoc + a LaTeX engine (xelatex preferred, pdflatex works after normalisation).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Glyphs that either break pdflatex outright or render as tofu in the default font.
# Greek and math symbols are spelled out so the PDF reads correctly in any engine.
SUBS = [
    ("—", "--"), ("–", "-"), ("‘", "'"), ("’", "'"),
    ("“", '"'), ("”", '"'), ("…", "..."),
    # U+2016 is NOT handled here -- it is context-dependent and is done per line in normalise().
    ("ΔW", "dW"),
    ("ᵀ", "^T"), ("²", "^2"), ("≡", "="),
    ("≥", ">="), ("≤", "<="), ("≠", "!="), ("≈", "~="),
    ("≳", ">~"), ("≲", "<~"),   # order-of-magnitude comparisons, no glyph in the PDF font
    ("±", "+/-"),
    ("×", "x"), ("→", "->"), ("·", "-"),
    ("σ", "sigma"), ("α", "alpha"), ("λ", "lambda"), ("Δ", "Delta"),
    ("ρ", "rho"), ("ε", "epsilon"),
    ("₁", "1"), ("₄", "4"), ("₅", "5"), ("₆", "6"), ("₇", "7"),
    ("⚠", "[!]"), ("✓", "[ok]"), ("✗", "[x]"),
    ("−", "-"),                  # MINUS SIGN -- not a hyphen; breaks pdflatex
    (" ", " "), (" ", " "), ("​", ""),
    (" ", " "),
]


def normalise(text: str) -> str:
    for a, b in SUBS:
        text = text.replace(a, b)

    # Both remaining rules are ESCAPES, and an escape inside a `code span` is reproduced verbatim --
    # a code span is copied out literally, which is the point of it. So the axis that matters is
    # "inside backticks or not", NOT "table row or not". Getting that wrong in either direction has
    # already shipped a defect once each: escaping everywhere printed
    # `E(P) = \|\|V^Th\|\|^2/\|\|h\|\|^2` and `corr(..., ||dW||\_F)` at the reader, while escaping
    # only in table rows let a bare "|_" through in prose, where it really does open an emphasis span
    # that never closes and silently kills the enclosing **bold**.
    def _outside_code(line: str, fn) -> str:
        # Odd indices are the insides of `...` spans and are left exactly as written. Rejoin WITH the
        # backticks -- joining on "" silently deletes every code-span delimiter in the document.
        return "`".join(part if i % 2 else fn(part) for i, part in enumerate(line.split("`")))

    # IN A TABLE ROW, escape everywhere -- INCLUDING inside code spans. Pandoc's pipe-table parser
    # splits cells on "|" BEFORE inline code is parsed, so a code span containing "||" shreds the row
    # exactly as a bare one does; skipping code spans here turned a 3-column row into a 7-pipe one.
    # OUTSIDE a table row there are no cells to shred, so the only job is the glyph substitution and
    # the bold guard, and both must leave code spans alone -- an escape inside backticks is copied out
    # literally, which is how `E(P) = \|\|V^Th\|\|^2` and `corr(..., ||dW||\_F)` reached the reader.
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            line = line.replace("‖", r"\|\|")
            line = re.sub(r"(?<=\|)_(?=\w)", r"\\_", line)
        else:
            line = line.replace("‖", "||")
            line = _outside_code(line, lambda s: re.sub(r"(?<=\|)_(?=\w)", r"\\_", s))
        out.append(line)
    return "\n".join(out)


def main() -> int:
    if not shutil.which("pandoc"):
        print("ERROR: pandoc not found on PATH", file=sys.stderr)
        return 1

    src = (HERE / "PAPER.md").read_text(encoding="utf-8")
    norm = normalise(src)
    (HERE / "PAPER_pdf.md").write_text(norm, encoding="utf-8")
    print(f"[pdf] PAPER.md {len(src)} chars -> PAPER_pdf.md {len(norm)} chars")

    remaining = sorted({c for c in norm if ord(c) > 0x2000 and c not in "‘’"})
    if remaining:
        codes = " ".join(f"U+{ord(c):04X}" for c in remaining[:20])
        print(f"[pdf] note: {len(remaining)} non-ASCII glyphs remain (xelatex handles these): {codes}")

    common = ["pandoc", str(HERE / "PAPER_pdf.md"), "-o", str(HERE / "PAPER.pdf"),
              "--resource-path", str(HERE),
              "-V", "geometry:margin=2.2cm", "-V", "colorlinks=true",
              "-V", "linkcolor=blue", "-V", "urlcolor=blue", "-V", "fontsize=10pt"]
    attempts = [
        common + ["--pdf-engine=xelatex", "-V", "mainfont=DejaVu Sans",
                  "-V", "monofont=DejaVu Sans Mono"],
        common + ["--pdf-engine=xelatex"],
        common + ["--pdf-engine=pdflatex"],
    ]
    for i, cmd in enumerate(attempts, 1):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            size = (HERE / "PAPER.pdf").stat().st_size
            print(f"[pdf] built with attempt {i} ({cmd[cmd.index('--pdf-engine=xelatex')] if '--pdf-engine=xelatex' in cmd else 'pdflatex'})"
                  f" -> PAPER.pdf {size/1024:.0f} KB")
            return 0
        print(f"[pdf] attempt {i} failed: {(r.stderr or r.stdout).strip().splitlines()[-1][:200]}")
    print("ERROR: all pandoc attempts failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
