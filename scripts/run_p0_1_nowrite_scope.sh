#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Use the merged checkpoint inside this repo.
unset AHN_CKPT_ROOT

# Make repo modules importable.
export PYTHONPATH=.:src

mkdir -p results/validation

python scripts/validate_p0_1_nowrite_scope.py \
  2>&1 | tee results/validation/p0_1_nowrite_scope.log
