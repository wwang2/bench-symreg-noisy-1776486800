#!/usr/bin/env bash
# Reproduce orbit 03-gplearn-symbolic from scratch.
#
# Usage: ./orbits/03-gplearn-symbolic/run.sh
#
# Step 1: train.py runs 8 seeds at default (pop=3000, gens=80, parsimony=0.001)
#         and writes best_predictor.pkl (this is the v1 baseline).
# Step 2: the v2 sweep (pop=4000, gens=120, parsimony=0.0005, 16 seeds) is the
#         one we actually use; it overwrites best_predictor.pkl.
# Step 3: make_figures.py regenerates figures/narrative.png and results.png.
# Step 4: evaluator.py is run 3 times (metric is deterministic on this eval).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 orbits/03-gplearn-symbolic/train.py
python3 -c "
import sys; sys.path.insert(0, 'orbits/03-gplearn-symbolic')
from train import run_sweep
run_sweep(list(range(16)), suffix='v2',
          generations=120, population_size=4000, parsimony_coefficient=0.0005,
          tournament_size=20, p_crossover=0.7, p_subtree_mutation=0.1,
          p_hoist_mutation=0.05, p_point_mutation=0.1)
"
cp orbits/03-gplearn-symbolic/best_predictor_v2.pkl orbits/03-gplearn-symbolic/best_predictor.pkl
cp orbits/03-gplearn-symbolic/best_program_v2.txt   orbits/03-gplearn-symbolic/best_program.txt

python3 orbits/03-gplearn-symbolic/make_figures.py

for seed in 1 2 3; do
  python3 research/eval/evaluator.py \
      --solution orbits/03-gplearn-symbolic/solution.py \
      --seed "$seed"
done
