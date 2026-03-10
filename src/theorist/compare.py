"""Side-by-side: Theorist vs blind random search.

The killer demo. Same budget, same search space.
Shows how prediction-error learning beats random mutation.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional

from .core import Theorist
from .types import Results


def compare(
    fn: Callable[[Dict[str, Any]], Any],
    search_space: Dict[str, list],
    n: int = 20,
    metric: str = "metric",
    minimize: bool = True,
    domain: str = "general",
    brain_path: str = "~/.theorist",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the same task with Theorist and blind random search, print comparison.

    Returns dict with 'theorist' and 'random' result sets.
    """
    # --- Theorist run ---
    if verbose:
        print("=" * 60)
        print("THEORIST vs RANDOM SEARCH")
        print("=" * 60)
        print(f"\nRunning Theorist ({n} experiments)...")

    t = Theorist(domain=domain, brain_path=brain_path)
    theorist_results = t.optimize(
        fn=fn, search_space=search_space,
        n=n, metric=metric, minimize=minimize, verbose=False,
    )

    # --- Random search run ---
    if verbose:
        print(f"Running Random Search ({n} experiments)...")

    random_best_metric: Optional[float] = None
    random_best_config: Dict[str, Any] = {}
    random_experiments: list = []
    random_best_at: int = 0

    for i in range(n):
        config = {k: random.choice(v) for k, v in search_space.items()}
        result = fn(config)
        actual = result[metric] if isinstance(result, dict) else float(result)

        is_better = (
            random_best_metric is None
            or (minimize and actual < random_best_metric)
            or (not minimize and actual > random_best_metric)
        )
        if is_better:
            random_best_metric = actual
            random_best_config = config.copy()
            random_best_at = i

        random_experiments.append({"config": config, "actual": actual, "best_so_far": random_best_metric})

    # --- Comparison ---
    theorist_best = theorist_results.best_metric
    random_best = random_best_metric or 0.0

    theorist_keeps = sum(1 for e in theorist_results.experiments if e.status == "best")
    random_keeps = sum(
        1 for i, e in enumerate(random_experiments)
        if i == 0 or (
            (minimize and e["actual"] < random_experiments[i - 1]["best_so_far"])
            or (not minimize and e["actual"] > random_experiments[i - 1]["best_so_far"])
        )
    )

    theorist_best_at = next(
        (e.exp_id for e in reversed(theorist_results.experiments) if e.actual == theorist_best),
        n - 1,
    )

    # Prediction calibration (theorist only)
    errors = [abs(e.predicted - e.actual) for e in theorist_results.experiments]
    first_half_mae = sum(errors[:n // 2]) / max(len(errors[:n // 2]), 1)
    second_half_mae = sum(errors[n // 2:]) / max(len(errors[n // 2:]), 1)

    if verbose:
        print(f"\n{'':>24} {'Theorist':>12} {'Random':>12}")
        print("-" * 50)
        print(f"{'Best metric':>24} {theorist_best:>12.4f} {random_best:>12.4f}")
        print(f"{'Found at experiment':>24} {theorist_best_at:>12d} {random_best_at:>12d}")
        print(f"{'Improvements (keeps)':>24} {theorist_keeps:>12d} {random_keeps:>12d}")
        print(f"{'Keep rate':>24} {theorist_keeps / n:>11.0%} {random_keeps / n:>11.0%}")
        print()
        print(f"{'Prediction calibration':>24} {'(theorist only)':>12}")
        print(f"{'  First half MAE':>24} {first_half_mae:>12.4f}")
        print(f"{'  Second half MAE':>24} {second_half_mae:>12.4f}")
        if first_half_mae > 0:
            improvement = (1 - second_half_mae / first_half_mae) * 100
            print(f"{'  Improvement':>24} {improvement:>11.0f}%")

        op = "<" if minimize else ">"
        if (minimize and theorist_best < random_best) or (not minimize and theorist_best > random_best):
            delta = abs(theorist_best - random_best)
            print(f"\nTheorist wins by {delta:.4f}")
        elif theorist_best == random_best:
            print(f"\nTie.")
        else:
            delta = abs(theorist_best - random_best)
            print(f"\nRandom search wins by {delta:.4f} (noise or small n?)")

    return {
        "theorist": theorist_results,
        "random": {
            "best_config": random_best_config,
            "best_metric": random_best,
            "experiments": random_experiments,
            "best_at": random_best_at,
        },
    }
