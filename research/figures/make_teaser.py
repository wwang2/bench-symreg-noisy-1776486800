"""Generate teaser figure showing the noisy training data."""
import numpy as np
import matplotlib.pyplot as plt
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "eval", "train_data.csv")
OUT = os.path.join(os.path.dirname(__file__), "teaser.png")

data = np.loadtxt(DATA, delimiter=",", skiprows=1)
x, y = data[:, 0], data[:, 1]

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.scatter(x, y, s=40, color="#2E86AB", edgecolor="white", linewidth=0.8, zorder=3, label="training data (noisy)")
ax.axhline(0, color="#999", linewidth=0.5, linestyle="--", zorder=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Symbolic Regression Benchmark — 50 noisy points, x ∈ [-5, 5]")
ax.grid(alpha=0.3, zorder=0)
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
