#!/usr/bin/env python3
"""How often does Theorist find the actual optimum, not just a good answer?

Every function here is small enough to solve by exhaustive search, so "solved"
is exact rather than a threshold someone picked. That matters: an earlier
version of this benchmark scored Rastrigin 0/60 against a hand-chosen cutoff of
8.0, when the true grid optimum is 16.0 and Theorist was finding it at the
median. The cutoff was unreachable and the "failure" was the measurement's.

Two hazards worth copying if you write your own:

  1. Theorist persists its brain to `brain_path` and reuses it. Two runs with
     the same path are not independent, and an A/B silently compares
     contaminated state -- the same configuration scored 16.00 and then 20.35
     on identical seeds. Every run below gets a fresh temporary directory which
     is deleted afterwards.

  2. Compare against a real baseline, not just random. Random search is a floor
     almost anything clears. A plain coordinate sweep is the honest opponent
     and it is much harder to beat on separable problems.

Usage:  python3 benchmarks/solved_vs_exhaustive.py [--seeds 80] [--n 20]
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import shutil
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import theorist


def sphere(c):
    """Smooth bowl. Optimum sits near the middle of every axis."""
    return {"m": sum((v - 0.5) ** 2 for v in c.values())}


def corner(c):
    """Separable, optimum at every axis's extreme. Each param is independently best at 1.0."""
    return {"m": sum((v - 1.0) ** 2 for v in c.values())}


def rastrigin(c):
    """Many local minima. Punishes anything that only ever mutates its current best."""
    return {"m": sum(10 + (v * 4 - 2) ** 2 - 10 * math.cos(2 * math.pi * (v * 4 - 2))
                     for v in c.values())}


FUNCTIONS = [("sphere", sphere), ("corner", corner), ("rastrigin", rastrigin)]


def grid(dims: int, points: int) -> dict:
    return {f"x{i}": [round(j / (points - 1), 3) for j in range(points)] for i in range(dims)}


def exhaustive_optimum(fn, space: dict) -> float:
    keys = list(space)
    return min(fn(dict(zip(keys, combo)))["m"]
               for combo in itertools.product(*[space[k] for k in keys]))


def run_theorist(fn, space: dict, n: int, seed: int) -> float:
    path = tempfile.mkdtemp(prefix="theorist-bench-")   # fresh brain, always
    try:
        random.seed(seed)
        results = theorist.Theorist(domain="benchmark", brain_path=path).optimize(
            fn, space, n=n, metric="m", minimize=True, verbose=False)
        best = getattr(results, "best_metric", None)
        return best if isinstance(best, (int, float)) else float("nan")
    finally:
        shutil.rmtree(path, ignore_errors=True)


def run_random(fn, space: dict, n: int, seed: int) -> float:
    random.seed(seed)
    return min(fn({k: random.choice(v) for k, v in space.items()})["m"] for _ in range(n))


def run_coordinate_sweep(fn, space: dict, n: int, seed: int) -> float:
    """One param at a time, every value once, keep what improves. No randomness."""
    current = {k: v[len(v) // 2] for k, v in space.items()}
    best = fn(current)["m"]
    used = 1
    for param in space:
        for value in space[param]:
            if used >= n:
                return best
            if value == current[param]:
                continue
            trial = dict(current)
            trial[param] = value
            metric = fn(trial)["m"]
            used += 1
            if metric < best:
                best, current = metric, trial
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=80)
    ap.add_argument("--n", type=int, default=20, help="experiment budget per run")
    ap.add_argument("--dims", type=int, default=4)
    ap.add_argument("--points", type=int, default=6, help="values per param")
    a = ap.parse_args()

    space = grid(a.dims, a.points)
    cells = a.points ** a.dims
    print(f"{a.dims} params x {a.points} values = {cells} configurations, "
          f"budget {a.n}, {a.seeds} seeds\n")
    print(f"{'function':11s} {'optimum':>9s} | "
          + "  ".join(f"{name:>14s}" for name in ("theorist", "coord-sweep", "random")))

    for name, fn in FUNCTIONS:
        opt = exhaustive_optimum(fn, space)
        cells_out = []
        for solver in (run_theorist, run_coordinate_sweep, run_random):
            results = [solver(fn, space, a.n, s) for s in range(a.seeds)]
            solved = sum(1 for r in results if r <= opt + 1e-9)
            cells_out.append(f"{solved:4d}/{a.seeds}  {statistics.median(results):7.3f}")
        print(f"{name:11s} {opt:9.4f} | " + "  ".join(cells_out))

    print("\nsolved = matched exhaustive search exactly; median = typical best found")


if __name__ == "__main__":
    main()
