#!/usr/bin/env bash
# Regenerates docs/html/*.html from the docstrings in src/, using Python's
# built-in pydoc module (no extra dependency, works fully offline).
#
# Usage: bash scripts/build_html_docs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p docs/html

MODULES="load_occupancy load_env match_sensors_to_zones build_dataset train_model sensitivity_env_only cv_fold_diagnostics headcount_regression optimisation make_figures"

(
  cd src
  for m in $MODULES; do
    python3 -m pydoc -w "$m"
  done
  mv -f ./*.html ../docs/html/
)

# pydoc writes this machine's absolute path to each source file into every
# page. Replace it with a relative link so the docs are identical whoever
# builds them and do not expose local folder names.
python3 - "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path
root = sys.argv[1].rstrip("/")
for page in Path(root, "docs", "html").glob("*.html"):
    text = page.read_text()
    fixed = text.replace(f"file:{root}/src/", "../../src/").replace(f"{root}/", "").replace(f"'{root}'", "'.'")
    if fixed != text:
        page.write_text(fixed)
PY

echo "Wrote $(ls docs/html/*.html | wc -l | tr -d ' ') HTML files to docs/html/ (open docs/html/index.html)"
