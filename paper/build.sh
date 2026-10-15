#!/usr/bin/env bash
set -euo pipefail
PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$PAPER_DIR")"
SUITE_DIR="${1:-$REPO_DIR/project1_outputs/experiment-suite-2026-10-06}"
python3 "$PAPER_DIR/generate_results.py" --suite "$SUITE_DIR" \
  --output "$PAPER_DIR/generated/results.tex"
cd "$PAPER_DIR"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf report.pdf
python3 - <<'PY'
from zipfile import ZipFile, ZIP_DEFLATED

with ZipFile("source.zip", "w", compression=ZIP_DEFLATED) as bundle:
    for name in ("main.tex", "references.bib", "generated/results.tex", "README.md",
                 "build.sh", "generate_results.py"):
        bundle.write(name)
PY
