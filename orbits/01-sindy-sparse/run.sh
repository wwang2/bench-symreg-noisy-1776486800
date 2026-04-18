#!/usr/bin/env bash
# Reproduce the orbit/01-sindy-sparse result from a clean checkout.
#
# Usage:
#   bash orbits/01-sindy-sparse/run.sh
#
# Runs the eval harness with three seeds and regenerates the figures.
set -euo pipefail
cd "$(dirname "$0")/../.."

SOLUTION="orbits/01-sindy-sparse/solution.py"

echo "== Eval (3 seeds) =="
for SEED in 1 2 3; do
  python3 research/eval/evaluator.py --solution "$SOLUTION" --seed "$SEED"
done

echo "== Regenerate figures =="
python3 orbits/01-sindy-sparse/make_figures.py
