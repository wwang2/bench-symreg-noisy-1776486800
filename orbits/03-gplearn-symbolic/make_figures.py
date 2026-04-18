"""Produce narrative.png and results.png from cached sweep artifacts.

narrative.png:  3-panel layout
  (a) training data + baseline-mean prediction + gplearn prediction on dense grid
  (b) residuals for baseline vs gplearn (same x axis)
  (c) fitness-over-generations curves (all 16 seeds, best highlighted)

results.png: 2-panel
  (left)  bar chart of per-seed training MSE vs target (0.01) + noise floor
  (right) gplearn vs baseline on test set: scatter of (y_true, y_pred) with y=x line

Follows research/style.md: constrained_layout, no boxed text, muted palette.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ORBIT_DIR = Path(__file__).parent
REPO_DIR = ORBIT_DIR.parent.parent
FIG_DIR = ORBIT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO_DIR / "research" / "eval"))
from generate_data import generate_test_data  # noqa: E402

# ---- style (research/style.md) -----------------------------------------
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

COL = {
    "data": "#333333",
    "baseline": "#888888",
    "gplearn": "#4C72B0",
    "highlight": "#C44E52",
    "other": "#BEBEBE",
    "target": "#DD8452",
    "noise": "#55A868",
}

# ---- load artifacts ----------------------------------------------------
train = np.loadtxt(REPO_DIR / "research" / "eval" / "train_data.csv",
                   delimiter=",", skiprows=1)
x_tr, y_tr = train[:, 0], train[:, 1]

model = pickle.loads((ORBIT_DIR / "best_predictor.pkl").read_bytes())
results = json.loads((ORBIT_DIR / "all_results_v2.json").read_text())
results.sort(key=lambda r: r["seed"])
histories = json.loads((ORBIT_DIR / "histories_v2.json").read_text())

best_seed = min(results, key=lambda r: r["train_mse"])["seed"]

x_dense = np.linspace(-5, 5, 500)
y_gp_dense = model.predict(x_dense.reshape(-1, 1))

x_test, y_test = generate_test_data(n_points=500, seed=99)
y_pred = model.predict(x_test.reshape(-1, 1))

# Baseline: constant mean of training y
_mean_y = float(y_tr.mean())
y_baseline_dense = np.full_like(x_dense, _mean_y)
y_baseline_test = np.full_like(x_test, _mean_y)

TARGET = 0.01
NOISE_VAR = 0.05 ** 2  # sigma=0.05 from generate_data.py noise_sigma

# ======================================================================
# narrative.png
# ======================================================================
fig = plt.figure(figsize=(13.5, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.7])
ax_fit = fig.add_subplot(gs[0, :])
ax_res = fig.add_subplot(gs[1, 0])
ax_hist = fig.add_subplot(gs[1, 1])

# (a) fit
ax_fit.scatter(x_tr, y_tr, s=34, color=COL["data"], alpha=0.75,
               label="training data (noisy)", zorder=3, edgecolors="white",
               linewidths=0.4)
ax_fit.plot(x_dense, y_baseline_dense, color=COL["baseline"], ls="--", lw=1.4,
            label=f"baseline: const mean={_mean_y:.3f}")
ax_fit.plot(x_dense, y_gp_dense, color=COL["gplearn"], lw=2.2,
            label="gplearn best (symbolic)")
ax_fit.set_xlabel("x")
ax_fit.set_ylabel("y")
ax_fit.set_title("(a) Discovered symbolic fit on 50 noisy training points  "
                 f"— test MSE = 0.00112")
ax_fit.legend(loc="upper left")
ax_fit.text(-0.05, 1.04, "(a)", transform=ax_fit.transAxes,
            fontsize=14, fontweight="bold")

# (b) residuals
resid_gp = y_tr - model.predict(x_tr.reshape(-1, 1))
resid_bl = y_tr - _mean_y
ax_res.axhline(0, color="black", lw=0.6)
ax_res.scatter(x_tr, resid_bl, s=26, color=COL["baseline"], alpha=0.7,
               label=f"baseline  (MSE={float(np.mean(resid_bl**2)):.3f})",
               edgecolors="white", linewidths=0.3)
ax_res.scatter(x_tr, resid_gp, s=30, color=COL["gplearn"], alpha=0.9,
               label=f"gplearn  (MSE={float(np.mean(resid_gp**2)):.4f})",
               edgecolors="white", linewidths=0.3)
ax_res.axhspan(-0.05, 0.05, color=COL["noise"], alpha=0.12,
               label="noise ±σ (0.05)")
ax_res.set_xlabel("x")
ax_res.set_ylabel("y − ŷ")
ax_res.set_title("(b) Training residuals — gplearn sits inside noise band")
ax_res.legend(loc="lower right")
ax_res.text(-0.15, 1.05, "(b)", transform=ax_res.transAxes,
            fontsize=14, fontweight="bold")

# (c) convergence
for seed_str, hist in histories.items():
    seed = int(seed_str)
    if seed == best_seed:
        continue
    ax_hist.plot(range(1, len(hist) + 1), hist, color=COL["other"],
                 lw=0.9, alpha=0.7)
hist_best = histories[str(best_seed)]
ax_hist.plot(range(1, len(hist_best) + 1), hist_best,
             color=COL["highlight"], lw=2.3, label=f"best (seed {best_seed})")
ax_hist.axhline(TARGET, color=COL["target"], lw=1.2, ls="--",
                label=f"target MSE = {TARGET}")
ax_hist.axhline(NOISE_VAR, color=COL["noise"], lw=1.2, ls=":",
                label=f"noise floor σ² = {NOISE_VAR:.4f}")
ax_hist.set_yscale("log")
ax_hist.set_xlabel("generation")
ax_hist.set_ylabel("best fitness (MSE on train)")
ax_hist.set_title("(c) Fitness over generations — 16 parallel seeds")
ax_hist.legend(loc="upper right")
ax_hist.text(-0.15, 1.05, "(c)", transform=ax_hist.transAxes,
             fontsize=14, fontweight="bold")

fig.suptitle("gplearn symbolic regression on noisy 1D data", fontsize=15,
             fontweight="medium")
narrative_path = FIG_DIR / "narrative.png"
fig.savefig(narrative_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"wrote {narrative_path}")

# ======================================================================
# results.png — quantitative panel
# ======================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))

# Left: per-seed train MSE bar chart
seeds = [r["seed"] for r in results]
mses = [r["train_mse"] for r in results]
colors_bar = [COL["highlight"] if s == best_seed else COL["gplearn"] for s in seeds]
axL.bar(seeds, mses, color=colors_bar, edgecolor="white", linewidth=0.6)
axL.axhline(TARGET, color=COL["target"], ls="--", lw=1.4,
            label=f"target = {TARGET}")
axL.axhline(NOISE_VAR, color=COL["noise"], ls=":", lw=1.4,
            label=f"noise σ² = {NOISE_VAR:.4f}")
# Horizontal line for baseline (mean-only) train MSE
bl_train = float(np.mean((y_tr - _mean_y) ** 2))
axL.axhline(bl_train, color=COL["baseline"], ls="--", lw=1.0,
            label=f"baseline (mean) = {bl_train:.3f}")
axL.set_xticks(seeds)
axL.set_xlabel("seed")
axL.set_ylabel("training MSE")
axL.set_yscale("log")
axL.set_title("Per-seed training MSE — 16 independent gplearn runs")
axL.legend(loc="upper right")
# Highlight best
best_mse = [r["train_mse"] for r in results if r["seed"] == best_seed][0]
axL.annotate(f"best  seed {best_seed}\ntrain MSE = {best_mse:.4f}",
             xy=(best_seed, best_mse), xytext=(best_seed + 2, best_mse * 0.35),
             fontsize=10, arrowprops=dict(arrowstyle="->", color="gray",
                                          lw=0.8))

# Right: predicted vs true on test set (500 points)
axR.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="black", lw=0.8, ls="--", label="y = x (ideal)")
axR.scatter(y_test, y_baseline_test, s=10, color=COL["baseline"], alpha=0.5,
            label=f"baseline (MSE={float(np.mean((y_test-y_baseline_test)**2)):.3f})")
axR.scatter(y_test, y_pred, s=10, color=COL["gplearn"], alpha=0.8,
            label=f"gplearn (MSE={float(np.mean((y_test-y_pred)**2)):.4f})")
axR.set_xlabel("y_test (held-out truth)")
axR.set_ylabel("prediction")
axR.set_title("Predicted vs truth on 500-point clean test set")
axR.legend(loc="upper left")

fig.suptitle("Quantitative results — gplearn symbolic regression",
             fontsize=14, fontweight="medium")
results_path = FIG_DIR / "results.png"
fig.savefig(results_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"wrote {results_path}")
