"""Train gplearn SymbolicRegressor on noisy 1D data, multi-seed.

Outputs:
  - best_program.txt   (sympy/gplearn expression as string)
  - best_predictor.pkl (pickled SymbolicRegressor of the best seed)
  - all_results.json   (per-seed train MSE, expression, depth, length)
  - figures/training_curves.png (fitness over generations across seeds)

Workflow:
  1. Load train_data.csv.
  2. Run SymbolicRegressor across N seeds in parallel (one process per seed
     via concurrent.futures.ProcessPoolExecutor — gplearn is single-threaded).
  3. Pick the seed with lowest training MSE; persist its program + model.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from gplearn.genetic import SymbolicRegressor

ORBIT_DIR = Path(__file__).parent
DATA_PATH = ORBIT_DIR.parent.parent / "research" / "eval" / "train_data.csv"
FIG_DIR = ORBIT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def fit_one(seed: int,
            generations: int = 80,
            population_size: int = 3000,
            parsimony_coefficient: float = 0.001,
            p_crossover: float = 0.7,
            p_subtree_mutation: float = 0.1,
            p_hoist_mutation: float = 0.05,
            p_point_mutation: float = 0.1,
            tournament_size: int = 20,
            init_depth=(2, 6),
            max_samples: float = 1.0):
    """Fit SymbolicRegressor for one seed; return (seed, regressor, mse, history)."""
    data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
    X = data[:, [0]]
    y = data[:, 1]

    sr = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        function_set=("add", "sub", "mul", "div", "sin", "cos", "log", "sqrt"),
        metric="mse",
        parsimony_coefficient=parsimony_coefficient,
        p_crossover=p_crossover,
        p_subtree_mutation=p_subtree_mutation,
        p_hoist_mutation=p_hoist_mutation,
        p_point_mutation=p_point_mutation,
        tournament_size=tournament_size,
        init_depth=init_depth,
        max_samples=max_samples,
        const_range=(-2.0, 2.0),
        feature_names=("x",),
        verbose=0,
        random_state=seed,
        n_jobs=1,
        warm_start=False,
    )
    t0 = time.time()
    sr.fit(X, y)
    elapsed = time.time() - t0
    y_pred = sr.predict(X)
    mse = float(np.mean((y - y_pred) ** 2))
    history = [float(v) for v in sr.run_details_["best_fitness"]]
    program = str(sr._program)
    length = int(sr._program.length_)
    depth = int(sr._program.depth_)
    return {
        "seed": seed,
        "train_mse": mse,
        "elapsed": elapsed,
        "program": program,
        "length": length,
        "depth": depth,
        "history": history,
    }, sr


def _worker(args):
    seed, kw = args
    res, sr = fit_one(seed, **kw)
    # Drop the generational history — it's tens of megabytes of past
    # populations we don't need for prediction. _program is the winning tree.
    sr._programs = None
    return res, pickle.dumps(sr)


def run_sweep(seeds, suffix="", **fit_kwargs):
    """Run a parameter sweep across seeds; persist best to *_<suffix>.* files."""
    args = [(s, fit_kwargs) for s in seeds]
    print(f"Sweep '{suffix or 'main'}': seeds={list(seeds)} kw={fit_kwargs}")
    results = []
    pickled = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(seeds), os.cpu_count() or 4)) as ex:
        futures = {ex.submit(_worker, a): a[0] for a in args}
        for fut in as_completed(futures):
            res, pkl = fut.result()
            print(f"  seed={res['seed']:>2}  mse={res['train_mse']:.5f}  "
                  f"len={res['length']:>3}  depth={res['depth']:>2}  "
                  f"time={res['elapsed']:.1f}s")
            results.append(res)
            pickled[res["seed"]] = pkl
    print(f"  wall: {time.time() - t0:.1f}s")
    results.sort(key=lambda r: r["train_mse"])
    best = results[0]
    print(f"  BEST seed={best['seed']} train_mse={best['train_mse']:.6f}")
    print(f"        expr={best['program']}")
    sfx = f"_{suffix}" if suffix else ""
    (ORBIT_DIR / f"best_program{sfx}.txt").write_text(best["program"] + "\n")
    (ORBIT_DIR / f"best_predictor{sfx}.pkl").write_bytes(pickled[best["seed"]])
    (ORBIT_DIR / f"all_results{sfx}.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "history"} for r in results],
                   indent=2)
    )
    (ORBIT_DIR / f"histories{sfx}.json").write_text(
        json.dumps({str(r["seed"]): r["history"] for r in results}, indent=2)
    )
    return results, best


def main(seeds=tuple(range(8)), **fit_kwargs):
    print(f"Training gplearn across seeds {list(seeds)}...")
    print(f"Hyperparams: {fit_kwargs}")
    args = [(s, fit_kwargs) for s in seeds]
    results = []
    pickled = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(seeds), os.cpu_count() or 4)) as ex:
        futures = {ex.submit(_worker, a): a[0] for a in args}
        for fut in as_completed(futures):
            seed = futures[fut]
            res, pkl = fut.result()
            print(f"  seed={res['seed']:>2}  mse={res['train_mse']:.5f}  "
                  f"len={res['length']:>3}  depth={res['depth']:>2}  "
                  f"time={res['elapsed']:.1f}s")
            results.append(res)
            pickled[res["seed"]] = pkl
    print(f"Total wall time: {time.time() - t0:.1f}s")

    # Best by training MSE
    results.sort(key=lambda r: r["train_mse"])
    best = results[0]
    print("\n=== BEST ===")
    print(f"seed={best['seed']}  train_mse={best['train_mse']:.6f}  "
          f"len={best['length']}  depth={best['depth']}")
    print(f"expression: {best['program']}")

    # Persist
    (ORBIT_DIR / "best_program.txt").write_text(best["program"] + "\n")
    (ORBIT_DIR / "best_predictor.pkl").write_bytes(pickled[best["seed"]])
    (ORBIT_DIR / "all_results.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "history"} for r in results],
                   indent=2)
    )
    # Save histories separately (for plot)
    (ORBIT_DIR / "histories.json").write_text(
        json.dumps({str(r["seed"]): r["history"] for r in results}, indent=2)
    )
    return results


if __name__ == "__main__":
    main()
