"""Symbolic regression solution — loads cached gplearn SymbolicRegressor.

The model was trained offline by ``train.py`` (8 seeds in parallel, gplearn
0.4.3 with function set {add, sub, mul, div, sin, cos, log, sqrt}).
We pickle the best fitted SymbolicRegressor and load it here, so f(x)
runs in milliseconds rather than re-training.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKL_PATH = os.path.join(_HERE, "best_predictor.pkl")

with open(_PKL_PATH, "rb") as _fh:
    _model = pickle.load(_fh)


def f(x):
    """Predict y for given x (numpy array or scalar)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return _model.predict(x)


if __name__ == "__main__":
    xs = np.linspace(-5, 5, 11)
    print("x      f(x)")
    for xi, yi in zip(xs, f(xs)):
        print(f"{xi:6.2f} {yi:7.4f}")
