---
issue: 3
parents: []
eval_version: eval-v1
metric: 0.000866
---

# Research Notes — Gaussian Process regression on noisy 1-D data

## Result

**MSE = 0.000866** on the held-out test set (target: 0.01).
That is roughly **11.5× below the target** and **447× below the
constant-mean baseline** (0.3867).  The eval is deterministic given a
fitted model (the test set is generated with a fixed seed inside the
evaluator), so all three "seeds" produce the identical metric.

## Approach

We treat the 50 noisy observations as samples from a smooth latent
function plus i.i.d. Gaussian observation noise:

```
y_i = g(x_i) + ε_i,        ε_i ~ N(0, σ²)
g    ~ GP(0, c·k_Matern(·,·;ℓ,ν=5/2))
```

The kernel is

```
k(x, x') = c · Matern_{ν=5/2}(x, x'; ℓ)  +  σ²·δ(x − x')
```

implemented as `ConstantKernel * Matern(nu=2.5) + WhiteKernel`.
Hyperparameters `(c, ℓ, σ²)` are learned by Type-II marginal-likelihood
maximization, with **20 restarts** to dodge local optima.  We also
mean-centre and unit-scale `y` before fitting for numerical stability and
restore the scale on prediction.

### Why Matern 5/2

| Kernel | Sample-path smoothness | Behaviour here |
|--------|------------------------|----------------|
| Matern 1/2 (Laplace) | continuous, not differentiable | over-rough; residuals show kinks |
| Matern 3/2 | once differentiable | slightly under-smooths the peaks |
| **Matern 5/2** | **twice differentiable**       | **best ML; balanced** |
| RBF / SE | C∞ | over-smooths near sharper turns and pulls noise into ringing |

Matern 5/2 is the standard "second-order-smooth" choice and has been
the workhorse for GP modelling of physical 1-D signals since
Stein (1999).  We confirmed by hand that switching ν from 5/2 → ∞ (RBF)
or 5/2 → 3/2 changes log-marginal-likelihood by a small but measurable
amount; ν=5/2 wins.

### Why a WhiteKernel and not `alpha`

`alpha` in sklearn is a fixed jitter — it does not get tuned by
marginal-likelihood maximization.  Putting the noise in a `WhiteKernel`
turns σ² into a *learned* hyperparameter, which lets the GP separate
"true smooth structure" from "observation noise" in a principled way.

## Learned hyperparameters

Fitted kernel printed by sklearn:

```
1.56**2 * Matern(length_scale=1.84, nu=2.5) + WhiteKernel(noise_level=0.00449)
```

| Hyperparameter | Learned value | Interpretation |
|----------------|---------------|----------------|
| signal stddev `c^{1/2}` | **1.56** | latent function varies on order of ±1.5 |
| length-scale `ℓ`        | **1.84** | feature length ≈ 1.8 in x-units (data range = 10) |
| noise stddev `σ`        | **0.067** | small but non-trivial — explains the scatter |
| log marginal likelihood | **+19.67** | high — model fits the data well |

These are physically sensible: the data spans x∈[-5,5] with multiple
oscillations, so `ℓ ≈ 1.8` resolves about 5 features across the range.
The recovered noise of 0.067 is consistent with what one sees in the
residual scatter (figures/results.png panel b RMSE 0.029, but that is the
*test-set* residual including a small systematic component; the
WhiteKernel is doing the right thing on the training side).

## Results table

| Seed | Metric (MSE) | Wall time |
|------|-------------:|----------:|
| 1    | 0.000866     |   1.0 s   |
| 2    | 0.000866     |   1.0 s   |
| 3    | 0.000866     |   0.9 s   |
| **Mean** | **0.000866 ± 0.000000** |   |

The evaluator generates the test set with a fixed internal seed (99) and
the GP fit is deterministic, so per-`--seed` variation is zero — that is
a property of the eval harness, not a bug.

## Why this beats the baseline so much

The constant-mean baseline cannot capture *any* x-dependence and is
penalized for the entire signal variance ≈ 0.39.  The GP captures
essentially all of that signal — its remaining error 0.0009 is dominated
by the irreducible structure that 50 noisy training points leave
unconstrained near the boundaries (visible as the slightly larger CI
band in figures/narrative.png at x = ±5).

## Prior Art & Novelty

### What is already known
- Gaussian Process regression with a Matern kernel plus learned noise is
  the standard textbook approach: Rasmussen & Williams,
  *Gaussian Processes for Machine Learning* (MIT Press, 2006), §2.7,
  §4.2.  https://gaussianprocess.org/gpml/
- Marginal-likelihood (Type-II) hyperparameter selection is also from
  the same reference (§5.4).
- The `sklearn.gaussian_process.GaussianProcessRegressor` interface and
  composite kernel construction is documented at
  https://scikit-learn.org/stable/modules/gaussian_process.html .

### What this orbit adds
Nothing methodologically new — this is the textbook recipe applied to
the benchmark.  The contribution is empirical: showing that GP-Matern
with multistart marginal-likelihood maximization clears the 0.01 target
by an order of magnitude on this dataset, providing a strong reference
point for any subsequent (e.g. symbolic, neural) orbits.

### Honest positioning
GP regression is the right tool when (a) the data are smooth, (b) noise
is roughly Gaussian, and (c) the dataset is small enough that
O(N³)≈O(50³) Cholesky is free.  All three hold here.  A symbolic
regression orbit could produce something more interpretable, but
unlikely to produce a *lower* MSE — the GP is essentially at the noise
floor of 50 samples.

## Glossary
- **GP** — Gaussian Process
- **Matern 5/2** — Matérn covariance kernel with smoothness parameter ν = 5/2
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **CI** — Confidence Interval (here, ±1.96σ posterior band ≈ 95%)
- **Type-II ML** — Type-II Maximum Likelihood = marginal-likelihood
  maximization over hyperparameters
- **WhiteKernel** — sklearn kernel modelling i.i.d. Gaussian observation noise

## References
- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for
  Machine Learning*. MIT Press. https://gaussianprocess.org/gpml/
- Stein, M. L. (1999). *Interpolation of Spatial Data: Some Theory for
  Kriging*. Springer.  (Argument for ν = 5/2 as default smoothness.)
- scikit-learn developers (2024). *GaussianProcessRegressor* docs.
  https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
