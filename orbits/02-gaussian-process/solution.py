"""Gaussian Process regression solution for noisy symbolic regression.

Approach
--------
We treat the noisy training points as samples from a smooth latent function
plus i.i.d. Gaussian observation noise.  A Gaussian Process with a Matern 5/2
kernel times a constant scale, plus a WhiteKernel for the noise level, is a
flexible non-parametric prior whose hyperparameters can be learned by
marginal-likelihood maximization (sklearn does this internally with several
restarts to avoid local optima).

Why Matern 5/2:
- Matern 5/2 is twice differentiable in mean-square sense (smoother than 3/2,
  rougher than RBF), giving a good bias for physical signals while not being
  as over-smoothing as the RBF/SE kernel which can ring near sharper turns.
- length_scale and noise_level are learned from the data; we don't fix them.
- ConstantKernel learns the prior signal variance.

The fit happens at module import; predictions just call the trained GP.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)

# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DATA = os.path.join(_REPO, "research", "eval", "train_data.csv")

_train = np.loadtxt(_DATA, delimiter=",", skiprows=1)
_X = _train[:, 0:1]                 # shape (N, 1)
_y = _train[:, 1].astype(float)     # shape (N,)

# Normalize y for fitting stability (we restore the scale on predict)
_Y_MEAN = float(_y.mean())
_Y_STD = float(_y.std())
_y_norm = (_y - _Y_MEAN) / _Y_STD

# ---------------------------------------------------------------------------
# Build kernel:  C * Matern(5/2)  +  White
# ---------------------------------------------------------------------------
# Reasonable initial values; bounds are wide enough that marginal-likelihood
# maximization can move freely.
_kernel = (
    ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
    * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _gp = GaussianProcessRegressor(
        kernel=_kernel,
        normalize_y=False,        # we handled normalization manually
        n_restarts_optimizer=20,  # multistart -> robust hyperparameters
        random_state=0,
        alpha=0.0,                # noise modeled by WhiteKernel, not alpha
    )
    _gp.fit(_X, _y_norm)

LEARNED_KERNEL = str(_gp.kernel_)
LOG_MARGINAL_LIKELIHOOD = float(_gp.log_marginal_likelihood(_gp.kernel_.theta))


def f(x):
    """Posterior mean of the GP at query points x.

    Parameters
    ----------
    x : array-like, shape (M,) or (M, 1)
        Query x-values in the training range [-5, 5].

    Returns
    -------
    y : np.ndarray, shape (M,)
        Predicted function values.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x_in = x[:, None]
    else:
        x_in = x
    y_norm = _gp.predict(x_in)
    return y_norm * _Y_STD + _Y_MEAN


def predict_with_std(x):
    """Posterior mean and stddev (handy for plotting CI bands)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x_in = x[:, None]
    else:
        x_in = x
    y_norm, sd_norm = _gp.predict(x_in, return_std=True)
    return y_norm * _Y_STD + _Y_MEAN, sd_norm * _Y_STD


if __name__ == "__main__":
    # Quick self-check
    print("Learned kernel :", LEARNED_KERNEL)
    print("log marginal L :", LOG_MARGINAL_LIKELIHOOD)
    xt = np.linspace(-5, 5, 7)
    print("preds          :", f(xt))
