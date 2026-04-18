---
issue: 2
parents: []
eval_version: eval-v1
metric: 0.000914
---

# Sparse Library Regression on Noisy 1D Data

## Result

| Seed | Test MSE | Wall time |
|------|---------|-----------|
| 1    | 0.000914 | ~20 s |
| 2    | 0.000914 | ~20 s |
| 3    | 0.000914 | ~20 s |
| **Mean** | **0.000914 ± 0.000000** | |

The fit is deterministic across seeds — the solution does not consume
the seed argument because training data is fixed and the optimizer
(coordinate-descent Lasso, then OLS on the surviving support) is
deterministic. So the three-seed run is a sanity check that the eval
harness is stable, not a stochastic average.

The campaign target is **MSE ≤ 0.01**, the constant-mean baseline is
**0.387**. This orbit lands at **0.000914**, roughly **420× better than
the baseline** and an order of magnitude under target.

## Approach

### 1. The phenomenology of the data
Eyeballing the 50 noisy points (panel (a) of `narrative.png`), the
signal looks like a damped oscillation: the amplitude is largest near
the edges of `x ∈ [-5, 5]`, the dominant frequency is roughly one
oscillation every two units of x, and there is a small constant offset
of about `-0.2`. That suggests basis functions of the form
`{x^k, sin(ωx), cos(ωx), x sin(ωx), x cos(ωx),
sin(ωx) e^{-α x²}, cos(ωx) e^{-α x²}}` are a natural library.

### 2. The library
The library has **115 candidate basis functions**:

| family | count | notes |
|--------|-------|-------|
| Polynomials `x^k`, k = 1..5 | 5 | constant absorbed into intercept |
| `sin(ωx)`, `cos(ωx)` for ω ∈ {0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3} | 20 | ten frequencies × 2 phases |
| `x sin(ωx)`, `x cos(ωx)` for the same ω | 20 | linear-amplitude oscillators |
| `sin(ωx)·exp(-αx²)`, `cos(ωx)·exp(-αx²)` for α ∈ {0.05, 0.1, 0.2, 0.3} × ω above | 80 | Gaussian-windowed |
| **Total** | **125** | |

(125 columns vs 50 rows is wildly underdetermined, which is exactly
what sparse regression is designed for.)

### 3. The pipeline
1. **Standardize** all 125 columns to zero mean, unit standard
   deviation. Without this, the polynomial columns (whose values run
   into the hundreds at `x = ±5`) would dominate the L1 penalty.
2. **LassoCV** over a 60-point log-spaced α grid with 5-fold CV. This
   is mostly a sanity check on the scale of the surviving coefficients
   — it ends up picking α ≈ 3 × 10⁻³.
3. **STLSQ (Sequentially Thresholded Least Squares).** For each
   threshold τ in a 25-point log-spaced grid covering three decades,
   iteratively (a) drop columns with `|β_j| < τ`, (b) refit the
   surviving block by ridge-regularized least squares, repeat until
   the active set stabilizes. Pick the threshold that minimizes
   5-fold-CV MSE on the centered targets. This robustly handles the
   strong column collinearity in the trig + Gaussian-windowed library.
4. **Debias.** On the surviving support, run an unbiased ridge fit
   (λ = 10⁻⁴, no penalty on intercept) on the *un*standardized columns
   plus an intercept. This gives the final coefficients used at
   inference time. Inference is then a single matrix-vector product
   against the same library, costing < 1 ms for 500 test points.

### 4. What the model actually picks
27 of 125 basis functions survive. The top entries by magnitude:

```
-0.7029 * cos(3x)
-0.5872 * sin(3x)
+0.5536 * cos(3x) * exp(-0.05 x^2)
+0.5477 * x * sin(1x)
+0.4473 * sin(2.5x)
+0.3184 * x * cos(2.5x)
+0.3163 * cos(2.5x)
... (20 more terms with |coef| < 0.31)
intercept = -0.7200
```

This is **not** a clean recovery of any single closed-form expression
— the library is highly redundant (e.g. `cos(3x)` and
`cos(3x)·exp(-0.05x²)` are nearly collinear over `[-5, 5]`), so
several near-equivalent terms split the load. But the *function* the
combination evaluates to (panel (c) of `results.png`) tracks the
underlying smooth curve with a residual band that is
indistinguishable in width from the noise floor.

The model is essentially answering "what mixture of these candidates
reproduces the data" rather than "what is the underlying expression",
which is the right answer when the goal is MSE on a held-out test set
rather than symbolic interpretability.

## Why this works

The signal has two dynamics — a damped oscillation and a phase-shifted
linear-amplitude oscillation — that both live cleanly in the chosen
library. The library is rich enough to match them, the L1 penalty
plus STLSQ thresholding is aggressive enough to keep the
support manageable (27/125 = 22 %), and the closed-form OLS debias
keeps coefficient magnitudes tame (largest is 0.70, no blow-ups).

When I tried the same pipeline without the STLSQ step (LassoCV alone
followed by an OLS debias), the OLS debias on the still-large
LassoCV support produced coefficients of ~10⁷ magnitude — classic
collinearity blow-up — and the resulting test MSE was orders of
magnitude worse. The threshold sweep is what makes this robust.

## Prior Art & Novelty

### What is already known
- **SINDy**: [Brunton, Proctor & Kutz (2016) — *Discovering governing
  equations from data by sparse identification of nonlinear dynamical
  systems* (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1517384113)
  introduced the library + sparse regression formula this orbit
  follows.
- **STLSQ** (sequentially thresholded least squares) is the
  standard SINDy fitting algorithm from the same paper.
- **LassoCV / coordinate descent**: standard L1 regression
  ([Tibshirani, 1996](https://www.jstor.org/stable/2346178)).
- **Symbolic regression on noisy 1D data** is a well-studied
  benchmark; tools like PySR, Eureqa, and gplearn do this
  evolutionarily; SINDy-style basis-library approaches are the
  classical linear-algebra alternative.

### What this orbit adds
Nothing methodologically new. This is a textbook application of the
SINDy pipeline to a 1D regression benchmark, with a CV-tuned threshold
and a tiny ridge-regularized debias step to stay numerically stable
under near-collinear basis functions. The contribution is empirical:
on this specific 50-point noisy problem, SINDy with a trig +
polynomial + Gaussian-windowed library hits **0.0009**, an order of
magnitude under target.

### Honest positioning
On a problem where the underlying function happens to live (or be
well-approximated) by a hand-chosen library, this approach is hard to
beat for sample efficiency. A genetic-programming symbolic regressor
would likely produce a more interpretable expression but would not
necessarily achieve a lower test MSE on this benchmark.

## Glossary
- **SINDy**: Sparse Identification of Nonlinear Dynamics.
- **STLSQ**: Sequentially Thresholded Least Squares.
- **LassoCV**: Cross-validated L1-penalized linear regression.
- **OLS**: Ordinary Least Squares.
- **MSE**: Mean Squared Error.

## Files
- `solution.py` — model fit + `f(x)` predictor (called by evaluator).
- `make_figures.py` — generates `figures/narrative.png` and `figures/results.png`.
- `run.sh` — reproduces eval from scratch.
- `figures/narrative.png` — qualitative behavior: data + baseline + fit + residuals + selected coefs.
- `figures/results.png` — quantitative: bar of test MSE, threshold-CV sweep, prediction overlay.

## References
- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). *Discovering
  governing equations from data by sparse identification of nonlinear
  dynamical systems*. PNAS 113(15), 3932–3937.
  https://www.pnas.org/doi/10.1073/pnas.1517384113
- Tibshirani, R. (1996). *Regression shrinkage and selection via the
  lasso*. JRSS-B 58(1), 267–288.
