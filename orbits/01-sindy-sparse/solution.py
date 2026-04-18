"""
SINDy-style sparse symbolic regression on noisy 1D data.

Approach
--------
Build a rich library of candidate basis functions (polynomial, trig at
several frequencies, products like x*sin(wx), x*cos(wx), and Gaussian-
modulated trig terms sin/cos(wx)*exp(-alpha*x^2)). Then fit a sparse
linear combination via:

  1. Standardize column features so Lasso sees a fair scale.
  2. LassoCV across a wide alpha grid -> initial sparse fit.
  3. STLSQ (Sequentially Thresholded Least Squares) refinement: at each
     step, drop columns whose standardized coefficient is below
     threshold, refit by OLS on the surviving columns, repeat. Use a
     held-out validation MSE (5-fold CV proxy) to pick the threshold
     that gives the best generalization.
  4. Debias the surviving sparse support with a tiny ridge fit on the
     unstandardized columns so coefficients are unbiased but stable
     even if the support contains near-collinear basis functions.

All training is done at import time on `train_data.csv` (50 noisy
points). `f(x)` is just a fast evaluation against the cached coefficient
vector and basis builder.

Runtime: well under 60s on a single CPU; LassoCV over ~50 columns and
50 samples is tiny.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
from sklearn.linear_model import LassoCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Lasso convergence warnings on tiny problems are noise.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Objective did not converge")

# ------------------------------------------------------------------
# Locate training data — no peeking at test data.
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TRAIN_CSV = os.path.join(_REPO, "research", "eval", "train_data.csv")


def _load_train():
    data = np.loadtxt(_TRAIN_CSV, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


# ------------------------------------------------------------------
# Feature library.
# ------------------------------------------------------------------
# Frequency grid covers slow drifts up through ~3x cycles in [-5,5].
_OMEGAS = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
# Gaussian envelope widths.
_ALPHAS = np.array([0.05, 0.1, 0.2, 0.3])
# Polynomial degrees (no constant — intercept absorbed separately).
_POLY_DEGREES = np.arange(1, 6)


def _build_features(x: np.ndarray):
    """Return (Phi, names) where Phi has shape (n_samples, n_features).

    Does NOT include a constant column; the model carries an explicit
    intercept term outside Phi.
    """
    x = np.asarray(x, dtype=float).ravel()
    cols = []
    names = []

    for d in _POLY_DEGREES:
        cols.append(x ** d)
        names.append(f"x^{d}")

    for w in _OMEGAS:
        s = np.sin(w * x)
        c = np.cos(w * x)
        cols.append(s);            names.append(f"sin({w:g}x)")
        cols.append(c);            names.append(f"cos({w:g}x)")
        cols.append(x * s);        names.append(f"x*sin({w:g}x)")
        cols.append(x * c);        names.append(f"x*cos({w:g}x)")

    for a in _ALPHAS:
        env = np.exp(-a * x * x)
        for w in _OMEGAS:
            cols.append(np.sin(w * x) * env)
            names.append(f"sin({w:g}x)*exp(-{a:g}x^2)")
            cols.append(np.cos(w * x) * env)
            names.append(f"cos({w:g}x)*exp(-{a:g}x^2)")

    Phi = np.column_stack(cols)
    return Phi, names


# ------------------------------------------------------------------
# STLSQ refinement with CV-selected threshold.
# ------------------------------------------------------------------
def _stlsq(Phi: np.ndarray, y: np.ndarray, threshold: float, n_iter: int = 20):
    """Sequentially thresholded least squares.

    At each step, set coefficients with |c| < threshold to zero, refit
    by OLS on the remaining columns, repeat until support stabilizes.
    Returns (coef, active_mask).
    """
    p = Phi.shape[1]
    # Use a small ridge for the initial fit to handle p>n / collinearity.
    ridge = Ridge(alpha=1e-3, fit_intercept=False).fit(Phi, y)
    coef = ridge.coef_.copy()
    active = np.ones(p, dtype=bool)
    for _ in range(n_iter):
        small = np.abs(coef) < threshold
        new_active = active & ~small
        if not new_active.any():
            return np.zeros(p), np.zeros(p, dtype=bool)
        if np.array_equal(new_active, active):
            break
        active = new_active
        coef = np.zeros(p)
        # Tiny ridge for numerical stability on the surviving block.
        block = Ridge(alpha=1e-6, fit_intercept=False).fit(Phi[:, active], y)
        coef[active] = block.coef_
    return coef, active


def _cv_score_threshold(Phi_s: np.ndarray, y_centered: np.ndarray,
                         threshold: float, n_splits: int = 5,
                         seed: int = 0) -> float:
    """K-fold CV mean MSE for STLSQ at a given threshold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    mses = []
    for tr, va in kf.split(Phi_s):
        coef, active = _stlsq(Phi_s[tr], y_centered[tr], threshold)
        if not active.any():
            yhat = np.zeros(len(va))
        else:
            yhat = Phi_s[va] @ coef
        mses.append(np.mean((y_centered[va] - yhat) ** 2))
    return float(np.mean(mses))


def _fit():
    x_tr, y_tr = _load_train()
    Phi, names = _build_features(x_tr)

    # Standardize columns: zero mean, unit std. Intercept handled separately.
    scaler = StandardScaler()
    Phi_s = scaler.fit_transform(Phi)
    y_mean = float(np.mean(y_tr))
    y_centered = y_tr - y_mean

    # ---- LassoCV: just to confirm sparse regression is feasible and to
    # initialize a sane threshold scale on the standardized features. ----
    alphas = np.logspace(-4, 0, 60)
    lasso = LassoCV(
        alphas=alphas,
        cv=5,
        fit_intercept=False,
        max_iter=200_000,
        tol=1e-7,
        n_jobs=1,
    ).fit(Phi_s, y_centered)
    lasso_coef = lasso.coef_
    lasso_alpha = float(lasso.alpha_)

    # ---- Sweep STLSQ thresholds, pick the best by 5-fold CV MSE. ----
    # Range from very loose (almost no thresholding) to aggressive.
    max_init = max(np.max(np.abs(lasso_coef)), 1e-3)
    thresholds = np.logspace(np.log10(max_init * 1e-3),
                             np.log10(max_init * 0.5), 25)
    cv_mses = np.array([_cv_score_threshold(Phi_s, y_centered, t)
                        for t in thresholds])
    best_idx = int(np.argmin(cv_mses))
    best_threshold = float(thresholds[best_idx])

    # ---- Final STLSQ on all training data with best threshold. ----
    coef_s, active = _stlsq(Phi_s, y_centered, best_threshold)

    if not active.any():
        # Fall back to dense ridge if nothing survived.
        coef_raw = np.zeros(Phi.shape[1])
        intercept_raw = y_mean
    else:
        # Debias on the unstandardized surviving support with tiny ridge
        # to stay stable under near-collinearity, plus explicit intercept.
        Phi_active = Phi[:, active]
        A = np.column_stack([Phi_active, np.ones(len(y_tr))])
        # Ridge in closed form: (A.T A + lam I) beta = A.T y, with no
        # penalty on intercept column.
        lam = 1e-4
        AtA = A.T @ A
        reg = np.eye(A.shape[1]) * lam
        reg[-1, -1] = 0.0  # don't penalize intercept
        beta = np.linalg.solve(AtA + reg, A.T @ y_tr)
        coef_raw = np.zeros(Phi.shape[1])
        coef_raw[active] = beta[:-1]
        intercept_raw = float(beta[-1])

    y_hat_tr = Phi @ coef_raw + intercept_raw
    train_mse = float(np.mean((y_tr - y_hat_tr) ** 2))

    selected = [
        (names[i], float(coef_raw[i]))
        for i in range(len(coef_raw))
        if abs(coef_raw[i]) > 0
    ]
    selected.sort(key=lambda t: -abs(t[1]))

    return {
        "coef": coef_raw,
        "intercept": intercept_raw,
        "names": names,
        "selected": selected,
        "train_mse": train_mse,
        "lasso_alpha": lasso_alpha,
        "best_threshold": best_threshold,
        "thresholds": thresholds,
        "cv_mses": cv_mses,
        "active_mask": active,
        "x_train": x_tr,
        "y_train": y_tr,
    }


# Fit once at import time.
_MODEL = _fit()


def f(x):
    """Predict y given x (numpy array or scalar)."""
    x_arr = np.asarray(x, dtype=float)
    flat = x_arr.ravel()
    Phi, _ = _build_features(flat)
    y = Phi @ _MODEL["coef"] + _MODEL["intercept"]
    return y.reshape(x_arr.shape)


if __name__ == "__main__":
    print(f"LassoCV alpha            = {_MODEL['lasso_alpha']:.4g}")
    print(f"Best STLSQ threshold     = {_MODEL['best_threshold']:.4g}")
    print(f"In-sample train MSE      = {_MODEL['train_mse']:.6f}")
    print(f"Number of active terms   = {len(_MODEL['selected'])}")
    print("Selected basis functions (top 20 by |coef|):")
    for name, c in _MODEL["selected"][:20]:
        print(f"  {c:+.4f} * {name}")
    print(f"  intercept = {_MODEL['intercept']:+.4f}")
