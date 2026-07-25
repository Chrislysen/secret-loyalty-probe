"""Build writeup/REPORT.pdf from writeup/REPORT.md. One command, no manual steps.

    python writeup/build_pdf.py

The PDF is the submission deliverable, so it must be regenerable from the markdown at any commit --
a stale PDF that disagrees with REPORT.md is the kind of inconsistency this project exists to avoid.
This script is the whole toolchain: normalise the few Unicode glyphs that break LaTeX, write
REPORT_pdf.md (kept for provenance / diffing), then run pandoc with xelatex.

Requires: pandoc + a LaTeX engine (xelatex preferred, pdflatex works after normalisation).
"""
from __future__ import annotations

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
    ("‖ΔW‖_F", "||dW||_F"), ("‖ΔW‖", "||dW||"), ("ΔW", "dW"), ("‖", "||"),                      # ‖ double vertical bar
    ("≥", ">="), ("≤", "<="), ("≠", "!="), ("≈", "~="),
    ("×", "x"), ("→", "->"), ("·", "-"),
    ("σ", "sigma"), ("λ", "lambda"), ("Δ", "Delta"),
    ("₁", "1"), ("₆", "6"), ("₇", "7"),
    ("⚠", "[!]"), ("✓", "[ok]"), ("✗", "[x]"),
    ("−", "-"),                  # MINUS SIGN -- not a hyphen; breaks pdflatex
    (" ", " "), (" ", " "), ("​", ""),
    (" ", " "),
]


def normalise(text: str) -> str:
    for a, b in SUBS:
        text = text.replace(a, b)
    return text


def main() -> int:
    if not shutil.which("pandoc"):
        print("ERROR: pandoc not found on PATH", file=sys.stderr)
        return 1

    src = (HERE / "REPORT.md").read_text(encoding="utf-8")
    norm = normalise(src)
    (HERE / "REPORT_pdf.md").write_text(norm, encoding="utf-8")
    print(f"[pdf] REPORT.md {len(src)} chars -> REPORT_pdf.md {len(norm)} chars")

    remaining = sorted({c for c in norm if ord(c) > 0x2000 and c not in "‘’"})
    if remaining:
        codes = " ".join(f"U+{ord(c):04X}" for c in remaining[:20])
        print(f"[pdf] note: {len(remaining)} non-ASCII glyphs remain (xelatex handles these): {codes}")

    common = ["pandoc", str(HERE / "REPORT_pdf.md"), "-o", str(HERE / "REPORT.pdf"),
              "--resource-path", str(HERE), "--toc", "--toc-depth=2",
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
            size = (HERE / "REPORT.pdf").stat().st_size
            print(f"[pdf] built with attempt {i} ({cmd[cmd.index('--pdf-engine=xelatex')] if '--pdf-engine=xelatex' in cmd else 'pdflatex'})"
                  f" -> REPORT.pdf {size/1024:.0f} KB")
            return 0
        print(f"[pdf] attempt {i} failed: {(r.stderr or r.stdout).strip().splitlines()[-1][:200]}")
    print("ERROR: all pandoc attempts failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
