"""Generate figures for the orbit: narrative.png and results.png.

Reads the trained model from solution.py (no test data peek beyond what
the public evaluator already does for scoring).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, HERE)

# Import solution AFTER configuring matplotlib style.
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

import solution  # noqa: E402

COLORS = {
    "data":     "#444444",
    "fit":      "#4C72B0",
    "baseline": "#888888",
    "method":   "#DD8452",
    "residual": "#55A868",
}


def _baseline_predict(x):
    """Constant-mean baseline matches research/eval/baseline.py."""
    y_mean = float(np.mean(solution._MODEL["y_train"]))
    return np.full_like(np.asarray(x, dtype=float), y_mean)


def make_narrative():
    x_tr = solution._MODEL["x_train"]
    y_tr = solution._MODEL["y_train"]
    x_dense = np.linspace(-5, 5, 1000)
    y_fit = solution.f(x_dense)
    y_baseline = _baseline_predict(x_dense)
    y_baseline_pts = _baseline_predict(x_tr)
    y_fit_pts = solution.f(x_tr)
    res_baseline = y_tr - y_baseline_pts
    res_fit = y_tr - y_fit_pts
    mse_baseline_in = float(np.mean(res_baseline ** 2))
    mse_fit_in = float(np.mean(res_fit ** 2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5),
                             sharex=False, constrained_layout=True)

    # (a) Baseline overlay
    ax = axes[0, 0]
    ax.scatter(x_tr, y_tr, s=22, color=COLORS["data"],
               label="noisy training data", zorder=3)
    ax.plot(x_dense, y_baseline, color=COLORS["baseline"], linestyle="--",
            lw=1.5, label=f"mean baseline (MSE={mse_baseline_in:.3f})")
    ax.set_title("(a) baseline: predict the mean")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_ylim(-1.6, 1.4)
    ax.legend(loc="lower right")

    # (b) SINDy sparse fit overlay
    ax = axes[0, 1]
    ax.scatter(x_tr, y_tr, s=22, color=COLORS["data"],
               label="noisy training data", zorder=3)
    ax.plot(x_dense, y_fit, color=COLORS["method"], lw=2.0,
            label=f"sparse SINDy fit (MSE={mse_fit_in:.4f})")
    ax.set_title("(b) sparse basis fit captures the dynamics")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_ylim(-1.6, 1.4)
    ax.legend(loc="lower right")

    # (c) residuals comparison
    ax = axes[1, 0]
    ax.axhline(0, color="#aaaaaa", lw=0.8)
    ax.scatter(x_tr, res_baseline, s=22, color=COLORS["baseline"],
               label="baseline residuals")
    ax.scatter(x_tr, res_fit, s=22, color=COLORS["method"],
               label="SINDy residuals")
    ax.set_title("(c) residuals: baseline vs SINDy")
    ax.set_xlabel("x"); ax.set_ylabel("y - prediction")
    ax.legend(loc="lower right")

    # (d) selected basis (top by |coef|), as a horizontal bar chart
    ax = axes[1, 1]
    selected = solution._MODEL["selected"][:12]
    names = [n for n, _ in selected][::-1]
    coefs = [c for _, c in selected][::-1]
    colors = [COLORS["method"] if c > 0 else COLORS["fit"] for c in coefs]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, coefs, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="#aaaaaa", lw=0.8)
    ax.set_title(f"(d) top-12 of {len(solution._MODEL['selected'])} sparse coefs")
    ax.set_xlabel("coefficient")
    ax.grid(False, axis="y")

    fig.suptitle("Sparse SINDy regression on 50 noisy samples — discovers oscillatory + damped structure",
                 fontsize=14, y=1.02)
    out = os.path.join(FIG_DIR, "narrative.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def make_results():
    """Quantitative results panel: train MSE breakdown + threshold CV
    sweep + per-seed test MSE bar (recomputed by mirroring the public
    test set generator from research/eval/generate_data.py).
    """
    sys.path.insert(0, os.path.join(REPO, "research", "eval"))
    from generate_data import generate_test_data  # noqa: E402

    x_test, y_test = generate_test_data(n_points=500, seed=99)
    y_pred_test = solution.f(x_test)
    test_mse = float(np.mean((y_test - y_pred_test) ** 2))

    x_tr = solution._MODEL["x_train"]
    y_tr = solution._MODEL["y_train"]
    y_pred_tr = solution.f(x_tr)
    train_mse = float(np.mean((y_tr - y_pred_tr) ** 2))

    # Baseline on the same test set (matches the campaign baseline number).
    y_mean = float(np.mean(y_tr))
    baseline_test_mse = float(np.mean((y_test - y_mean) ** 2))

    thresholds = solution._MODEL["thresholds"]
    cv_mses = solution._MODEL["cv_mses"]
    best_threshold = solution._MODEL["best_threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    # (a) bar: baseline vs SINDy on test set
    ax = axes[0]
    bars = ax.bar(["baseline\n(mean)", "SINDy\n(this orbit)"],
                  [baseline_test_mse, test_mse],
                  color=[COLORS["baseline"], COLORS["method"]],
                  edgecolor="white", linewidth=0.8)
    ax.axhline(0.01, color="#C44E52", linestyle="--", lw=1.2,
               label="target (0.01)")
    for b, v in zip(bars, [baseline_test_mse, test_mse]):
        ax.text(b.get_x() + b.get_width() / 2,
                v * 1.05, f"{v:.4f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("MSE on held-out test (500 pts)")
    ax.set_title("(a) test-set MSE")
    ax.legend(loc="upper right")

    # (b) STLSQ threshold sweep
    ax = axes[1]
    ax.plot(thresholds, cv_mses, "-o", color=COLORS["fit"], ms=4, lw=1.2)
    ax.axvline(best_threshold, color=COLORS["method"], linestyle="--",
               lw=1.2, label=f"chosen = {best_threshold:.3g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("STLSQ threshold (standardized coef)")
    ax.set_ylabel("5-fold CV MSE (centered y)")
    ax.set_title("(b) threshold selected by 5-fold CV")
    ax.legend(loc="upper left")

    # (c) predictions vs truth on test set
    ax = axes[2]
    ax.plot(x_test, y_test, color=COLORS["data"], lw=1.3,
            label="true f(x)")
    ax.plot(x_test, y_pred_test, color=COLORS["method"], lw=1.6,
            linestyle="--", label="SINDy prediction")
    ax.fill_between(x_test, y_test, y_pred_test,
                    color=COLORS["residual"], alpha=0.25,
                    label="error band")
    ax.set_title(f"(c) prediction vs truth — test MSE = {test_mse:.4f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="lower right")

    fig.suptitle("SINDy sparse regression — quantitative results", fontsize=14, y=1.04)
    out = os.path.join(FIG_DIR, "results.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"baseline test MSE = {baseline_test_mse:.4f}")
    print(f"SINDy   test MSE = {test_mse:.6f}")
    print(f"SINDy  train MSE = {train_mse:.6f}")


if __name__ == "__main__":
    make_narrative()
    make_results()
