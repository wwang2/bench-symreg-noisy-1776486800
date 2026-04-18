"""Generate behavior + results figures for the GP orbit."""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "research", "eval"))

import solution as sol  # noqa: E402  (relies on path insert above)
from generate_data import generate_test_data  # noqa: E402

# ---- style -----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

COL_GP        = "#4C72B0"   # GP mean
COL_BAND      = "#4C72B0"   # 95% CI fill
COL_TRAIN     = "#222222"   # training points
COL_TRUE      = "#888888"   # ground-truth (only used in results panel)
COL_BASELINE  = "#C44E52"   # mean-y baseline
COL_RES_POS   = "#4C72B0"
COL_RES_NEG   = "#DD8452"

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- data ------------------------------------------------------------------
train = np.loadtxt(os.path.join(REPO, "research", "eval", "train_data.csv"),
                   delimiter=",", skiprows=1)
x_train, y_train = train[:, 0], train[:, 1]

x_test, y_test = generate_test_data(n_points=500, seed=99)

x_grid = np.linspace(-5, 5, 600)
y_mean, y_std = sol.predict_with_std(x_grid)

# Predictions for metric reporting
y_pred_test = sol.f(x_test)
mse_gp = float(np.mean((y_test - y_pred_test) ** 2))
mse_baseline = float(np.mean((y_test - y_train.mean()) ** 2))


# ===========================================================================
# narrative.png — qualitative artifact: GP mean + 95% CI vs training points
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 6.2))

ax.fill_between(x_grid,
                y_mean - 1.96 * y_std,
                y_mean + 1.96 * y_std,
                color=COL_BAND, alpha=0.18, linewidth=0,
                label="95% posterior CI")
ax.plot(x_grid, y_mean, color=COL_GP, lw=2.2, label="GP posterior mean")
ax.scatter(x_train, y_train, s=42, color=COL_TRAIN, zorder=5,
           edgecolor="white", linewidth=0.7, label="Training data (n=50)")

ax.set_xlim(-5.05, 5.05)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(
    f"Gaussian Process fit — Matern 5/2 kernel    "
    f"(MSE on held-out test = {mse_gp:.4f})",
    loc="left",
)

ax.annotate(
    f"learned length-scale = "
    f"{sol._gp.kernel_.k1.k2.length_scale:.2f}\n"
    f"learned noise σ = "
    f"{np.sqrt(sol._gp.kernel_.k2.noise_level):.3f}\n"
    f"log marginal L = {sol.LOG_MARGINAL_LIKELIHOOD:.2f}",
    xy=(4.9, 0.55), xycoords="data",
    ha="right", va="top",
    fontsize=10, color="#333",
)

ax.legend(loc="lower left")
fig.savefig(os.path.join(OUT, "narrative.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ===========================================================================
# results.png — quantitative comparison panel
#   (a) GP prediction vs test ground truth
#   (b) residual histogram + per-seed metric bars
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4),
                         gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})

# --- (a) GP mean vs test set --------------------------------------------------
axA = axes[0]
axA.scatter(x_test, y_test, s=10, color=COL_TRUE, alpha=0.55,
            label="Held-out test (n=500)")
axA.plot(x_grid, y_mean, color=COL_GP, lw=2.2, label="GP posterior mean")
axA.set_xlim(-5.05, 5.05)
axA.set_xlabel("x")
axA.set_ylabel("y")
axA.set_title("(a) Prediction vs held-out test set", loc="left")
axA.legend(loc="lower right")
axA.text(-0.13, 1.05, "(a)", transform=axA.transAxes,
         fontsize=14, fontweight="bold")

# --- (b) Residual distribution ------------------------------------------------
axB = axes[1]
resid = y_pred_test - y_test
colors = np.where(resid >= 0, COL_RES_POS, COL_RES_NEG)
axB.scatter(x_test, resid, s=14, color=colors, alpha=0.7)
axB.axhline(0.0, color="#444", lw=0.8, linestyle="--")
axB.set_xlabel("x")
axB.set_ylabel("residual  (pred − truth)")
rmse = float(np.sqrt(np.mean(resid ** 2)))
axB.set_title(f"(b) Residuals (RMSE = {rmse:.4f})", loc="left")
axB.text(-0.13, 1.05, "(b)", transform=axB.transAxes,
         fontsize=14, fontweight="bold")

# --- (c) Metric comparison ----------------------------------------------------
axC = axes[2]
labels = ["mean-y\nbaseline", "GP\n(this orbit)", "target"]
values = [mse_baseline, mse_gp, 0.01]
bar_colors = [COL_BASELINE, COL_GP, "#888888"]
bars = axC.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.8)
axC.set_yscale("log")
axC.set_ylim(3e-4, 1.2)
axC.set_ylabel("MSE on test set (log)")
axC.set_title("(c) Metric vs baselines / target", loc="left")
for b, v in zip(bars, values):
    axC.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.4f}",
             ha="center", va="bottom", fontsize=10, color="#222")
axC.grid(False)
axC.text(-0.13, 1.05, "(c)", transform=axC.transAxes,
         fontsize=14, fontweight="bold")

fig.suptitle("Gaussian Process orbit — quantitative results", y=1.04)
fig.savefig(os.path.join(OUT, "results.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"MSE GP        = {mse_gp:.6f}")
print(f"MSE baseline  = {mse_baseline:.6f}")
print(f"Saved figures to {OUT}/")
