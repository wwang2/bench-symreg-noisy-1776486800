---
issue: 4
parents: []
eval_version: eval-v1
metric: 0.001125
---

# Research Notes

## Result

**Test MSE = 0.001125** (deterministic across eval seeds; the test set is
itself fixed via `seed=99` inside the evaluator, so the 3-seed sweep
returns identical numbers). This is **~8.9x below the 0.01 target** and
roughly **half of the training MSE 0.00236**, indicating the model is
denoising rather than overfitting.

| Seed | Metric   | Wall time |
|------|----------|-----------|
| 1    | 0.001125 |  0.86 s   |
| 2    | 0.001125 |  0.85 s   |
| 3    | 0.001125 |  0.86 s   |
| **Mean** | **0.001125 +/- 0.000000** | |

## Approach

Genetic-programming symbolic regression using `gplearn 0.4.3`.

1. `train.py` fits `SymbolicRegressor` on the 50 noisy training points
   across many random seeds in parallel (`ProcessPoolExecutor`,
   `n_jobs=1` per regressor so each seed gets its own Python process).
2. The seed with the lowest **training MSE** is pickled to
   `best_predictor.pkl` and the discovered expression is written to
   `best_program.txt`.
3. `solution.py` simply unpickles that fitted regressor at import time
   and exposes `f(x) = model.predict(x.reshape(-1, 1))`. No re-training
   at eval time — import + predict of 500 points completes in < 1 s.

### Why two passes

The hypothesis suggested pop ~ 2000, generations 50-200. A first pass
at (pop=3000, gens=80, parsimony=0.001, 8 seeds) gave train MSE 0.00316
and test MSE 0.00224 — already below target but leaving headroom. A
second pass with bigger population and lower parsimony
(pop=4000, gens=120, parsimony=0.0005, 16 seeds) reduced train MSE
to 0.00236 and, more importantly, **halved** the test MSE to 0.00112,
suggesting that letting the search explore slightly more complex
expressions pays off on this target.

### Final hyperparameters

| Param | Value |
|---|---|
| `population_size` | 4000 |
| `generations` | 120 |
| `function_set` | add, sub, mul, div, sin, cos, log, sqrt |
| `metric` | mse |
| `parsimony_coefficient` | 0.0005 |
| `tournament_size` | 20 |
| `init_depth` | (2, 6) |
| `p_crossover / p_subtree_mut / p_hoist_mut / p_point_mut` | 0.70 / 0.10 / 0.05 / 0.10 |
| `const_range` | (-2.0, 2.0) |
| seeds swept | 0..15 (parallel processes) |
| wall time (sweep) | 394 s (~195 s per seed, 8-way parallel) |

## Discovered expression

Best seed = **1**, training MSE = 0.00236, length = 22, depth = 10:

```
sub(sub(sin(mul(div(mul(cos(x), cos(sqrt(sqrt(log(x))))), sqrt(x)),
                mul(x, 1.388))),
        cos(1.465)),
    cos(1.465))
```

Which translates to

```
y_hat(x) = sin( (cos(x) * cos(sqrt(sqrt(log(x))))) /
                (sqrt(x) * (x * 1.388)) )
           - 2 * cos(1.465)
```

Here `cos(1.465) ~ 0.105`, so the additive offset is **-0.210**. gplearn
applies a `protected_log` and `protected_sqrt` that return zero on
non-positive arguments, so the `log(x)` / `sqrt(x)` subtree is real-valued
across the full domain `[-5, 5]`. The expression is not intended to be
*interpretable* — it is a reasonably compact combination of trig and
radical primitives that the noisy training data can certify. The
residual plot (panel b of `narrative.png`) shows the fit sits entirely
inside the +/- sigma = 0.05 noise band, which is exactly the right
stopping point for this dataset: further complexity would start fitting
noise.

## Per-seed training MSE (top 5)

| rank | seed | train MSE | length | depth |
|------|------|-----------|--------|-------|
| 1 | 1 | 0.00236 | 22 | 10 |
| 2 | 2 | 0.00291 | 17 | 10 |
| 3 | 12 | 0.00309 | 15 | 8 |
| 4 | 5 | 0.00314 | 15 | 9 |
| 5 | 4 | 0.00350 | 17 | 8 |

The spread across seeds (0.00236 -> 0.01219) is substantial and
motivates the multi-seed search: any *single* run has a real
probability of getting stuck in a local optimum that still beats the
baseline but doesn't hit noise floor. With 16 seeds, the top-5 all
come in under 0.004 on train and generalize to the same order on test.

## Prior Art & Novelty

### What is already known
- Genetic programming for symbolic regression, pioneered by
  [Koza (1992) *Genetic Programming*](https://mitpress.mit.edu/9780262111706/),
  is the canonical method for this problem class.
- `gplearn` (Trevor Stephens) is the widely-used scikit-learn-compatible
  implementation: https://gplearn.readthedocs.io.

### What this orbit adds
- Nothing methodologically novel. This orbit applies the canonical
  gplearn pipeline as the hypothesis prescribed; the only craft is
  hyperparameter choice and multi-seed model selection, which are
  standard practice.

### Honest positioning
This is a *strong baseline* for this benchmark. A 500-point clean test
MSE of 0.00112 against a noise-variance of 0.0025 on train is well
inside what a well-tuned gplearn run should achieve on a composed
smooth-trig-exponential target over a bounded 1-D domain. I make no
claim of improvement over prior symbolic-regression literature.

## Files

- `solution.py` — loads `best_predictor.pkl`, exposes `f(x)`.
- `train.py` — runs the gplearn sweep end to end.
- `make_figures.py` — regenerates `figures/narrative.png` and
  `figures/results.png` from `all_results_v2.json` + the pickled model.
- `best_program.txt` — the winning symbolic expression (string form).
- `best_predictor.pkl` — pickled fitted `SymbolicRegressor` (== v2).
- `best_predictor_v2.pkl`, `best_program_v2.txt`,
  `all_results_v2.json`, `histories_v2.json` — v2 sweep artifacts.
- `all_results.json`, `histories.json` — first-pass (v1) sweep artifacts.

## References

- Koza, J. R. (1992). *Genetic Programming: On the Programming of
  Computers by Means of Natural Selection.* MIT Press.
- Stephens, T. `gplearn` documentation: https://gplearn.readthedocs.io
- Schmidt, M. & Lipson, H. (2009). *Distilling Free-Form Natural Laws
  from Experimental Data.* Science 324, 81-85.

## Glossary

- **GP** — Genetic Programming.
- **gplearn** — Python library for GP-based symbolic regression.
- **MSE** — Mean Squared Error.
- **Parsimony coefficient** — penalty on program length added to the
  fitness to discourage bloat (overlong expressions).
- **Tournament size** — number of candidates drawn from the population
  to compete for selection each breeding event.
