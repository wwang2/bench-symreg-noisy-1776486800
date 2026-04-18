# Symbolic Regression on Noisy Data

## Problem Statement
Given 50 noisy training data points in `research/eval/train_data.csv` (x, y pairs over x ∈ [-5, 5]), discover a function `f(x)` that approximates the underlying data-generating process. The true function is hidden; only the noisy samples are visible.

## Solution Interface
Solution must be a Python file `orbits/<name>/solution.py` defining either:
- `f(x)` — a function that takes a numpy array `x` and returns predictions, OR
- `solve(seed=42)` — returns a callable `predict(x)`

The evaluator at `research/eval/evaluator.py` (DO NOT REBUILD) calls the solution against a held-out clean test set of 500 points over the same x range.

## Success Metric
MSE on held-out test set (minimize). Target: 0.01.

## Budget
Max 3 orbits.
