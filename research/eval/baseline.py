"""Trivial constant baseline: predict mean of training y."""
import numpy as np
import os

DATA = os.path.join(os.path.dirname(__file__), "train_data.csv")
_data = np.loadtxt(DATA, delimiter=",", skiprows=1)
_mean_y = float(_data[:, 1].mean())

def f(x):
    x = np.asarray(x)
    return np.full_like(x, _mean_y, dtype=float)
