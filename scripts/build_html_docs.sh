#!/usr/bin/env bash
# Regenerates docs/html/*.html from the docstrings in src/, using Python's
# built-in pydoc module (no extra dependency, works fully offline).
#
# Usage: bash scripts/build_html_docs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p docs/html

MODULES="load_occupancy load_env match_sensors_to_zones build_dataset train_model sensitivity_env_only optimisation make_figures"

(
  cd src
  for m in $MODULES; do
    python3 -m pydoc -w "$m"
  done
  mv -f ./*.html ../docs/html/
)

echo "Wrote $(ls docs/html/*.html | wc -l | tr -d ' ') HTML files to docs/html/ (open docs/html/index.html)"
