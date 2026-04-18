#!/usr/bin/env bash
# Reproduce the orbit/02-gaussian-process result from a clean checkout.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# 1. Sanity-check the GP fits and prints expected hyperparameters.
python3 orbits/02-gaussian-process/solution.py

# 2. Regenerate the qualitative + results figures.
python3 orbits/02-gaussian-process/make_figures.py

# 3. Run the frozen evaluator across 3 seeds.
for SEED in 1 2 3; do
    python3 research/eval/evaluator.py \
        --solution orbits/02-gaussian-process/solution.py \
        --seed "$SEED"
done
